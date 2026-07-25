"""Thread-safe Franka telemetry snapshots and transport helpers."""

from dataclasses import dataclass
import threading
import time

if __package__:
    from .franka_telemetry_protocol import (
        MAGIC,
        PACKET_BYTES,
        SCHEMA_VERSION,
        FrankaTelemetryState,
    )
else:
    from franka_telemetry_protocol import (
        MAGIC,
        PACKET_BYTES,
        SCHEMA_VERSION,
        FrankaTelemetryState,
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
    entrypoint,
    now_monotonic_ns=None,
):
    """Build the stable JSON object returned by ``/telemetry/status``."""

    status = store.status_snapshot(now_monotonic_ns=now_monotonic_ns)
    return {
        "ready": status["ready"],
        "schema": {
            "magic": MAGIC.decode("ascii"),
            "version": SCHEMA_VERSION,
            "packet_bytes": PACKET_BYTES,
        },
        "server": {
            "commit": server_commit,
            "entrypoint": entrypoint,
        },
        "udp": dict(publisher_stats or {"enabled": False}),
        "source": status["source"],
        "robot": status["robot"],
        "frames": status["frames"],
        "payload": status["payload"],
    }
