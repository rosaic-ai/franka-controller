"""Fixed-size Franka state telemetry protocol.

Schema v1 uses network byte order and remains below the project's 1,200-byte
UDP payload budget.
"""

from dataclasses import dataclass
import math
import struct
from typing import Tuple


MAGIC = b"FRS1"
SCHEMA_VERSION = 1

HEADER_STRUCT = struct.Struct("!4sHHHHQqIdQI")
PAYLOAD_STRUCT = struct.Struct("!127dB")
HEADER_BYTES = HEADER_STRUCT.size
PACKET_BYTES = HEADER_STRUCT.size + PAYLOAD_STRUCT.size

FIELD_LENGTHS = (
    ("q", 7),
    ("q_d", 7),
    ("dq", 7),
    ("dq_d", 7),
    ("ddq_d", 7),
    ("tau_J", 7),
    ("tau_J_d", 7),
    ("tau_ext_hat_filtered", 7),
    ("O_T_EE", 16),
    ("zero_jacobian", 42),
    ("K_F_ext_hat_K", 6),
    ("O_F_ext_hat_K", 6),
)

assert HEADER_BYTES == 52
assert PAYLOAD_STRUCT.size == 1017
assert PACKET_BYTES == 1069
assert PACKET_BYTES <= 1200


@dataclass(frozen=True)
class FrankaTelemetryState:
    ros_stamp_sec: int
    ros_stamp_nsec: int
    franka_time_sec: float
    q: Tuple[float, ...]
    q_d: Tuple[float, ...]
    dq: Tuple[float, ...]
    dq_d: Tuple[float, ...]
    ddq_d: Tuple[float, ...]
    tau_J: Tuple[float, ...]
    tau_J_d: Tuple[float, ...]
    tau_ext_hat_filtered: Tuple[float, ...]
    O_T_EE: Tuple[float, ...]
    zero_jacobian: Tuple[float, ...]
    K_F_ext_hat_K: Tuple[float, ...]
    O_F_ext_hat_K: Tuple[float, ...]
    control_command_success_rate: float
    robot_mode: int


@dataclass(frozen=True)
class DecodedFrankaPacket:
    state: FrankaTelemetryState
    sequence: int
    send_monotonic_ns: int
    source_state_age_us: int


def _validate_uint(name, value, maximum):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} is out of range")


def _validate_state(state):
    if not isinstance(state.ros_stamp_sec, int) or isinstance(
        state.ros_stamp_sec, bool
    ):
        raise ValueError("ros_stamp_sec must be an integer")
    if not -(2**63) <= state.ros_stamp_sec < 2**63:
        raise ValueError("ros_stamp_sec is out of range")
    _validate_uint("ros_stamp_nsec", state.ros_stamp_nsec, 999_999_999)
    _validate_uint("robot_mode", state.robot_mode, 255)

    scalar_values = (
        ("franka_time_sec", state.franka_time_sec),
        (
            "control_command_success_rate",
            state.control_command_success_rate,
        ),
    )
    for name, value in scalar_values:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    for name, expected_length in FIELD_LENGTHS:
        values = getattr(state, name)
        if len(values) != expected_length:
            raise ValueError(f"{name} must contain {expected_length} values")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"{name} values must be finite")


def encode_packet(
    state,
    sequence,
    send_monotonic_ns,
    source_state_age_us,
):
    """Encode one immutable state snapshot as a schema-v1 datagram."""

    _validate_state(state)
    _validate_uint("sequence", sequence, 2**64 - 1)
    _validate_uint("send_monotonic_ns", send_monotonic_ns, 2**64 - 1)
    _validate_uint("source_state_age_us", source_state_age_us, 2**32 - 1)

    header = HEADER_STRUCT.pack(
        MAGIC,
        SCHEMA_VERSION,
        HEADER_BYTES,
        PACKET_BYTES,
        0,
        sequence,
        state.ros_stamp_sec,
        state.ros_stamp_nsec,
        float(state.franka_time_sec),
        send_monotonic_ns,
        source_state_age_us,
    )
    values = []
    for name, _ in FIELD_LENGTHS:
        values.extend(float(value) for value in getattr(state, name))
    values.append(float(state.control_command_success_rate))
    payload = PAYLOAD_STRUCT.pack(*values, state.robot_mode)
    packet = header + payload
    if len(packet) != PACKET_BYTES:
        raise AssertionError(f"encoded packet is {len(packet)} bytes")
    return packet


def decode_packet(packet):
    """Decode and validate one schema-v1 datagram."""

    if len(packet) != PACKET_BYTES:
        raise ValueError(
            f"packet must contain exactly {PACKET_BYTES} bytes, got {len(packet)}"
        )

    (
        magic,
        version,
        header_bytes,
        packet_bytes,
        _reserved,
        sequence,
        ros_stamp_sec,
        ros_stamp_nsec,
        franka_time_sec,
        send_monotonic_ns,
        source_state_age_us,
    ) = HEADER_STRUCT.unpack_from(packet)

    if magic != MAGIC:
        raise ValueError(f"invalid magic {magic!r}")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {version}")
    if header_bytes != HEADER_BYTES:
        raise ValueError(f"invalid header size {header_bytes}")
    if packet_bytes != PACKET_BYTES:
        raise ValueError(f"invalid declared packet size {packet_bytes}")

    unpacked = PAYLOAD_STRUCT.unpack_from(packet, HEADER_BYTES)
    double_values = unpacked[:-1]
    robot_mode = unpacked[-1]
    if not all(math.isfinite(value) for value in double_values):
        raise ValueError("payload values must be finite")

    offset = 0
    state_fields = {}
    for name, length in FIELD_LENGTHS:
        state_fields[name] = tuple(double_values[offset : offset + length])
        offset += length

    state = FrankaTelemetryState(
        ros_stamp_sec=ros_stamp_sec,
        ros_stamp_nsec=ros_stamp_nsec,
        franka_time_sec=franka_time_sec,
        control_command_success_rate=double_values[offset],
        robot_mode=robot_mode,
        **state_fields,
    )
    _validate_state(state)
    return DecodedFrankaPacket(
        state=state,
        sequence=sequence,
        send_monotonic_ns=send_monotonic_ns,
        source_state_age_us=source_state_age_us,
    )
