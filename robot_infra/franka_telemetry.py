"""Thread-safe Franka telemetry snapshots and transport helpers."""

from dataclasses import dataclass
import ipaddress
import logging
import socket
import struct
import threading
import time

if __package__:
    from .franka_telemetry_protocol import (
        MAGIC,
        PACKET_BYTES,
        SCHEMA_VERSION,
        FIELD_ORDER_ID,
        FrankaTelemetryState,
        encode_packet,
    )
else:
    from franka_telemetry_protocol import (
        MAGIC,
        PACKET_BYTES,
        SCHEMA_VERSION,
        FIELD_ORDER_ID,
        FrankaTelemetryState,
        encode_packet,
    )


_TELEMETRY_ARRAY_FIELDS = (
    "q",
    "q_d",
    "dq",
    "dq_d",
    "ddq_d",
    "tau_J",
    "tau_J_d",
    "tau_ext_hat_filtered",
    "O_T_EE",
    "K_F_ext_hat_K",
    "O_F_ext_hat_K",
)
_FRAME_FIELDS = ("F_T_EE", "F_T_NE", "NE_T_EE", "EE_T_K")
_PAYLOAD_ARRAY_FIELDS = (
    "F_x_Cee",
    "I_ee",
    "F_x_Cload",
    "I_load",
    "F_x_Ctotal",
    "I_total",
)
_PAYLOAD_SCALAR_FIELDS = ("m_ee", "m_load", "m_total")
_TELEMETRY_WARNING_LOCK = threading.Lock()
_LAST_TELEMETRY_WARNING_NS = 0


@dataclass(frozen=True)
class _FrankaStateSample:
    received_monotonic_ns: int
    ros_stamp_sec: int
    ros_stamp_nsec: int
    franka_time_sec: float
    arrays: tuple
    control_command_success_rate: float
    robot_mode: int
    frames: tuple
    payload_arrays: tuple
    payload_scalars: tuple
    current_errors: tuple
    last_motion_errors: tuple

    def array(self, name):
        return self.arrays[_TELEMETRY_ARRAY_FIELDS.index(name)][1]


@dataclass(frozen=True)
class _JacobianSample:
    received_monotonic_ns: int
    zero_jacobian: tuple


@dataclass(frozen=True)
class SendSnapshot:
    state: FrankaTelemetryState
    source_state_age_us: int
    source_jacobian_age_us: int


def safe_telemetry_update(
    update,
    msg,
    monotonic_ns=time.monotonic_ns,
    logger=logging.warning,
):
    """Run a telemetry-only callback without affecting legacy ROS updates."""

    global _LAST_TELEMETRY_WARNING_NS
    try:
        update(msg)
        return True
    except Exception as exc:
        now_ns = monotonic_ns()
        should_log = False
        with _TELEMETRY_WARNING_LOCK:
            if (
                now_ns - _LAST_TELEMETRY_WARNING_NS
                >= 5_000_000_000
            ):
                _LAST_TELEMETRY_WARNING_NS = now_ns
                should_log = True
        if should_log:
            logger("Franka telemetry snapshot update failed: %s", exc)
        return False


def _as_seconds(value):
    to_sec = getattr(value, "to_sec", None)
    return float(to_sec() if callable(to_sec) else value)


def _active_errors(errors):
    if errors is None:
        return ()
    names = getattr(errors, "__slots__", ())
    return tuple(
        name
        for name in names
        if not name.startswith("_") and bool(getattr(errors, name, False))
    )


class TelemetryStateStore:
    """Copies ROS callbacks into immutable, lock-protected snapshots."""

    def __init__(self, monotonic_ns=time.monotonic_ns):
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._franka_state = None
        self._jacobian = None

    def update_franka_state(self, msg, received_monotonic_ns=None):
        received_ns = (
            self._monotonic_ns()
            if received_monotonic_ns is None
            else int(received_monotonic_ns)
        )
        sample = _FrankaStateSample(
            received_monotonic_ns=received_ns,
            ros_stamp_sec=int(msg.header.stamp.secs),
            ros_stamp_nsec=int(msg.header.stamp.nsecs),
            franka_time_sec=_as_seconds(msg.time),
            arrays=tuple(
                (name, tuple(float(value) for value in getattr(msg, name)))
                for name in _TELEMETRY_ARRAY_FIELDS
            ),
            control_command_success_rate=float(
                msg.control_command_success_rate
            ),
            robot_mode=int(msg.robot_mode),
            frames=tuple(
                (name, tuple(float(value) for value in getattr(msg, name)))
                for name in _FRAME_FIELDS
            ),
            payload_arrays=tuple(
                (name, tuple(float(value) for value in getattr(msg, name)))
                for name in _PAYLOAD_ARRAY_FIELDS
            ),
            payload_scalars=tuple(
                (name, float(getattr(msg, name)))
                for name in _PAYLOAD_SCALAR_FIELDS
            ),
            current_errors=_active_errors(
                getattr(msg, "current_errors", None)
            ),
            last_motion_errors=_active_errors(
                getattr(msg, "last_motion_errors", None)
            ),
        )
        with self._lock:
            self._franka_state = sample

    def update_jacobian(self, msg, received_monotonic_ns=None):
        received_ns = (
            self._monotonic_ns()
            if received_monotonic_ns is None
            else int(received_monotonic_ns)
        )
        sample = _JacobianSample(
            received_monotonic_ns=received_ns,
            zero_jacobian=tuple(
                float(value) for value in msg.zero_jacobian
            ),
        )
        with self._lock:
            self._jacobian = sample

    def _samples(self):
        with self._lock:
            return self._franka_state, self._jacobian

    def snapshot_for_send(self, now_monotonic_ns, max_age_ns):
        franka, jacobian = self._samples()
        if franka is None or jacobian is None:
            return None

        state_age_ns = max(
            0, int(now_monotonic_ns) - franka.received_monotonic_ns
        )
        jacobian_age_ns = max(
            0, int(now_monotonic_ns) - jacobian.received_monotonic_ns
        )
        if state_age_ns > max_age_ns or jacobian_age_ns > max_age_ns:
            return None

        arrays = dict(franka.arrays)
        state = FrankaTelemetryState(
            ros_stamp_sec=franka.ros_stamp_sec,
            ros_stamp_nsec=franka.ros_stamp_nsec,
            franka_time_sec=franka.franka_time_sec,
            zero_jacobian=jacobian.zero_jacobian,
            control_command_success_rate=(
                franka.control_command_success_rate
            ),
            robot_mode=franka.robot_mode,
            **arrays,
        )
        return SendSnapshot(
            state=state,
            source_state_age_us=state_age_ns // 1_000,
            source_jacobian_age_us=jacobian_age_ns // 1_000,
        )

    def status_snapshot(self, now_monotonic_ns=None):
        now_ns = (
            self._monotonic_ns()
            if now_monotonic_ns is None
            else int(now_monotonic_ns)
        )
        franka, jacobian = self._samples()
        if franka is None:
            reason = "waiting_for_franka_state"
        elif jacobian is None:
            reason = "waiting_for_zero_jacobian"
        else:
            reason = "ready"

        source = {"reason": reason}
        robot = {}
        frames = {}
        payload = {}
        if franka is not None:
            source["franka_state_age_us"] = max(
                0, now_ns - franka.received_monotonic_ns
            ) // 1_000
            robot = {
                "robot_mode": franka.robot_mode,
                "control_command_success_rate": (
                    franka.control_command_success_rate
                ),
                "current_errors": list(franka.current_errors),
                "last_motion_errors": list(franka.last_motion_errors),
            }
            frames = {
                name: list(values) for name, values in franka.frames
            }
            payload = dict(franka.payload_scalars)
            payload.update(
                {
                    name: list(values)
                    for name, values in franka.payload_arrays
                }
            )
        if jacobian is not None:
            source["zero_jacobian_age_us"] = max(
                0, now_ns - jacobian.received_monotonic_ns
            ) // 1_000

        return {
            "ready": franka is not None and jacobian is not None,
            "source": source,
            "robot": robot,
            "frames": frames,
            "payload": payload,
        }


def build_status_payload(
    store,
    publisher_stats,
    server_commit,
    server_started_at,
    entrypoint,
    now_monotonic_ns=None,
):
    """Build the stable JSON object returned by ``/telemetry/status``."""

    status = store.status_snapshot(now_monotonic_ns=now_monotonic_ns)
    now_ns = (
        time.monotonic_ns()
        if now_monotonic_ns is None
        else int(now_monotonic_ns)
    )
    udp = dict(publisher_stats or {"enabled": False})
    host = udp.pop("host", "")
    port = udp.pop("port", None)
    hz = udp.pop("hz", None)
    last_send_ns = udp.pop("last_send_monotonic_ns", None)
    udp["destination"] = (
        f"{host}:{port}" if host and port is not None else None
    )
    udp["target_hz"] = hz
    udp["last_send_age_ms"] = (
        max(0, now_ns - last_send_ns) / 1_000_000.0
        if last_send_ns is not None
        else None
    )
    return {
        "ready": status["ready"],
        "schema": {
            "magic": MAGIC.decode("ascii"),
            "version": SCHEMA_VERSION,
            "packet_bytes": PACKET_BYTES,
            "field_order": FIELD_ORDER_ID,
        },
        "server": {
            "started_at": server_started_at,
            "commit": server_commit,
            "entrypoint": entrypoint,
        },
        "udp": udp,
        "source": status["source"],
        "robot": status["robot"],
        "frames": status["frames"],
        "payload": status["payload"],
    }


class FrankaUdpPublisher:
    """Best-effort UDP publisher isolated from the robot control path."""

    def __init__(
        self,
        store,
        host,
        port=5010,
        hz=200,
        socket_factory=None,
        monotonic_ns=time.monotonic_ns,
    ):
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise ValueError("telemetry port must be in range 1-65535")
        if (
            not isinstance(hz, int)
            or isinstance(hz, bool)
            or not 1 <= hz <= 200
        ):
            raise ValueError("telemetry hz must be in range 1-200")
        if host:
            try:
                ipaddress.IPv4Address(host)
            except ipaddress.AddressValueError as exc:
                raise ValueError(
                    "telemetry host must be an IPv4 literal"
                ) from exc

        self.store = store
        self.host = host
        self.port = port
        self.hz = hz
        self.enabled = bool(host)
        self._monotonic_ns = monotonic_ns
        self._max_age_ns = round(5_000_000_000 / hz)
        self._period_ns = round(1_000_000_000 / hz)
        self._stop_event = threading.Event()
        self._thread = None
        self._sequence = 0
        self._stats_lock = threading.Lock()
        self._stats = {
            "enabled": self.enabled,
            "host": host,
            "port": port,
            "hz": hz,
            "running": False,
            "sent_packets": 0,
            "stale_skips": 0,
            "encode_errors": 0,
            "send_errors": 0,
            "last_send_monotonic_ns": None,
            "last_sequence": None,
            "last_error": None,
        }
        self._last_warning_ns = 0
        if self.enabled:
            factory = socket_factory or (
                lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            )
            self._socket = factory()
        else:
            self._socket = None

    def _record(self, field, now_ns, error=None):
        with self._stats_lock:
            self._stats[field] += 1
            if error is not None:
                self._stats["last_error"] = str(error)
        if (
            error is not None
            and now_ns - self._last_warning_ns >= 5_000_000_000
        ):
            logging.warning("Franka UDP telemetry error: %s", error)
            self._last_warning_ns = now_ns

    def send_once(self, now_monotonic_ns=None):
        if not self.enabled:
            return False
        now_ns = (
            self._monotonic_ns()
            if now_monotonic_ns is None
            else int(now_monotonic_ns)
        )
        snapshot = self.store.snapshot_for_send(
            now_monotonic_ns=now_ns,
            max_age_ns=self._max_age_ns,
        )
        if snapshot is None:
            self._record("stale_skips", now_ns)
            return False

        try:
            packet = encode_packet(
                snapshot.state,
                sequence=self._sequence,
                send_monotonic_ns=now_ns,
                source_state_age_us=snapshot.source_state_age_us,
            )
        except (TypeError, ValueError, OverflowError, struct.error) as exc:
            self._record("encode_errors", now_ns, error=exc)
            return False

        try:
            sent_bytes = self._socket.sendto(
                packet,
                (self.host, self.port),
            )
            if sent_bytes != len(packet):
                raise OSError(
                    f"short UDP send: {sent_bytes}/{len(packet)} bytes"
                )
        except OSError as exc:
            self._record("send_errors", now_ns, error=exc)
            return False

        with self._stats_lock:
            self._stats["sent_packets"] += 1
            self._stats["last_send_monotonic_ns"] = now_ns
            self._stats["last_sequence"] = self._sequence
            self._stats["last_error"] = None
        self._sequence = (self._sequence + 1) & (2**64 - 1)
        return True

    def _run(self):
        deadline_ns = self._monotonic_ns()
        while not self._stop_event.is_set():
            now_ns = self._monotonic_ns()
            remaining_ns = deadline_ns - now_ns
            if remaining_ns > 0:
                self._stop_event.wait(remaining_ns / 1_000_000_000)
                continue
            self.send_once(now_monotonic_ns=now_ns)
            deadline_ns += self._period_ns
            if now_ns - deadline_ns > self._period_ns:
                deadline_ns = now_ns + self._period_ns

    def start(self):
        if not self.enabled or (
            self._thread is not None and self._thread.is_alive()
        ):
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="franka-udp-telemetry",
            daemon=True,
        )
        with self._stats_lock:
            self._stats["running"] = True
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        with self._stats_lock:
            self._stats["running"] = False

    def stats_snapshot(self):
        with self._stats_lock:
            return dict(self._stats)
