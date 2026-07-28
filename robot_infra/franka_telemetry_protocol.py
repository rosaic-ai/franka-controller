"""Fixed-size Franka state telemetry protocol (schema v1/v2 + 미래 버전 관용).

버전 정책 (2026-07-28):
- v1: reserved16 은 항상 0. jacobian 유효성 표현 불가(디코드 시 None).
- v2: 동일 레이아웃, reserved16 자리가 flags16 (bit0 = zero_jacobian_valid).
- 미래 버전(v>2): 선언 header/packet 길이를 신뢰해 알려진 prefix 만 파싱.
- 진화 계약: 기존 필드 오프셋 불변, 확장은 헤더/페이로드 꼬리에만 붙인다.

Network byte order, 1,200-byte UDP payload budget 유지.
"""

from dataclasses import dataclass
import math
import struct
from typing import Tuple


MAGIC = b"FRS1"
SCHEMA_VERSION = 2
KNOWN_VERSIONS = (1, 2)
FLAG_ZERO_JACOBIAN_VALID = 0x0001
FIELD_ORDER_ID = "franka_state_v2_127d_flags"
MAX_TOLERATED_PACKET_BYTES = 4096

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


def verify_layout():
    """Wire 레이아웃 불변식 명시 검사 — ``python -O`` 에서도 동작해야 한다."""
    checks = (
        ("HEADER_BYTES", HEADER_BYTES, 52),
        ("payload bytes", PAYLOAD_STRUCT.size, 1017),
        ("PACKET_BYTES", PACKET_BYTES, 1069),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise RuntimeError(
                f"telemetry wire layout drifted: {name}={actual}, "
                f"expected {expected}"
            )
    if PACKET_BYTES > 1200:
        raise RuntimeError("PACKET_BYTES exceeds the 1,200-byte UDP budget")


verify_layout()


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
    schema_version: int
    zero_jacobian_valid: "bool | None"


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
    *,
    zero_jacobian_valid,
):
    """Encode one immutable state snapshot as a schema-v2 datagram.

    ``zero_jacobian_valid`` 는 필수다 — 호출부가 jacobian 신선도를 판단해
    명시해야 하며, 무효면 수신측이 zero_jacobian 열을 신뢰하지 않는다.
    """

    if not isinstance(zero_jacobian_valid, bool):
        raise ValueError("zero_jacobian_valid must be a bool")
    _validate_state(state)
    _validate_uint("sequence", sequence, 2**64 - 1)
    _validate_uint("send_monotonic_ns", send_monotonic_ns, 2**64 - 1)
    _validate_uint("source_state_age_us", source_state_age_us, 2**32 - 1)

    header = HEADER_STRUCT.pack(
        MAGIC,
        SCHEMA_VERSION,
        HEADER_BYTES,
        PACKET_BYTES,
        FLAG_ZERO_JACOBIAN_VALID if zero_jacobian_valid else 0,
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
    """Decode and validate one telemetry datagram (v1/v2 엄격, 미래 버전 관용)."""

    if len(packet) < HEADER_BYTES:
        raise ValueError(
            f"packet must contain at least {HEADER_BYTES} header bytes, "
            f"got {len(packet)}"
        )

    (
        magic,
        version,
        header_bytes,
        packet_bytes,
        flags,
        sequence,
        ros_stamp_sec,
        ros_stamp_nsec,
        franka_time_sec,
        send_monotonic_ns,
        source_state_age_us,
    ) = HEADER_STRUCT.unpack_from(packet)

    if magic != MAGIC:
        raise ValueError(f"invalid magic {magic!r}")
    if version in KNOWN_VERSIONS:
        if len(packet) != PACKET_BYTES:
            raise ValueError(
                f"schema v{version} packet must contain exactly "
                f"{PACKET_BYTES} bytes, got {len(packet)}"
            )
        if header_bytes != HEADER_BYTES:
            raise ValueError(f"invalid header size {header_bytes}")
        if packet_bytes != PACKET_BYTES:
            raise ValueError(f"invalid declared packet size {packet_bytes}")
        payload_offset = HEADER_BYTES
    elif version > SCHEMA_VERSION:
        # 미래 버전 관용 경로 — 선언 길이를 신뢰하고 알려진 prefix 만 읽는다.
        if len(packet) > MAX_TOLERATED_PACKET_BYTES:
            raise ValueError(
                f"packet of {len(packet)} bytes exceeds tolerated maximum "
                f"{MAX_TOLERATED_PACKET_BYTES}"
            )
        if packet_bytes != len(packet):
            raise ValueError(
                f"declared packet size {packet_bytes} does not match "
                f"datagram of {len(packet)} bytes"
            )
        if header_bytes < HEADER_BYTES:
            raise ValueError(
                f"declared header size {header_bytes} is smaller than the "
                f"known prefix {HEADER_BYTES}"
            )
        if header_bytes + PAYLOAD_STRUCT.size > len(packet):
            raise ValueError(
                "packet too small for the known payload prefix after its "
                f"declared {header_bytes}-byte header"
            )
        payload_offset = header_bytes
    else:
        raise ValueError(f"unsupported schema version {version}")

    unpacked = PAYLOAD_STRUCT.unpack_from(packet, payload_offset)
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
        schema_version=version,
        zero_jacobian_valid=(
            None if version == 1 else bool(flags & FLAG_ZERO_JACOBIAN_VALID)
        ),
    )
