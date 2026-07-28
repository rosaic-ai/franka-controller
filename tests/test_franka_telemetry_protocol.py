import ast
import dataclasses
import inspect
import struct
import subprocess
import sys
import unittest

import tests  # noqa: F401  # installs test-only optional dependency stubs

from robot_infra.franka_telemetry_protocol import (
    PACKET_BYTES,
    FrankaTelemetryState,
    decode_packet,
    encode_packet,
)


def make_state():
    values7 = tuple(float(i) for i in range(7))
    return FrankaTelemetryState(
        ros_stamp_sec=123,
        ros_stamp_nsec=456_000_000,
        franka_time_sec=7.5,
        q=values7,
        q_d=tuple(v + 10.0 for v in values7),
        dq=tuple(v + 20.0 for v in values7),
        dq_d=tuple(v + 30.0 for v in values7),
        ddq_d=tuple(v + 40.0 for v in values7),
        tau_J=tuple(v + 50.0 for v in values7),
        tau_J_d=tuple(v + 60.0 for v in values7),
        tau_ext_hat_filtered=tuple(v + 70.0 for v in values7),
        O_T_EE=tuple(float(i) for i in range(16)),
        zero_jacobian=tuple(float(i) / 10.0 for i in range(42)),
        K_F_ext_hat_K=tuple(float(i) for i in range(6)),
        O_F_ext_hat_K=tuple(float(i + 6) for i in range(6)),
        control_command_success_rate=0.999,
        robot_mode=2,
    )


class ProtocolTest(unittest.TestCase):
    def test_packet_is_exact_size_and_round_trips(self):
        packet = encode_packet(
            make_state(),
            sequence=9,
            send_monotonic_ns=88,
            source_state_age_us=700,
            zero_jacobian_valid=True,
        )
        self.assertEqual(len(packet), 1069)
        self.assertEqual(PACKET_BYTES, 1069)
        decoded = decode_packet(packet)
        self.assertEqual(decoded.sequence, 9)
        self.assertEqual(decoded.send_monotonic_ns, 88)
        self.assertEqual(decoded.source_state_age_us, 700)
        self.assertEqual(decoded.state, make_state())

    def test_rejects_wrong_shape_and_non_finite_value(self):
        state = make_state()
        with self.assertRaisesRegex(ValueError, "q must contain 7"):
            encode_packet(
                dataclasses.replace(state, q=(1.0,)),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            encode_packet(
                dataclasses.replace(state, q=(float("nan"),) + state.q[1:]),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )

    def test_rejects_invalid_header_and_size(self):
        packet = bytearray(
            encode_packet(
                make_state(),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )
        )
        packet[0:4] = b"BAD!"
        with self.assertRaisesRegex(ValueError, "magic"):
            decode_packet(bytes(packet))

        truncated = encode_packet(
            make_state(),
            sequence=0,
            send_monotonic_ns=0,
            source_state_age_us=0,
            zero_jacobian_valid=True,
        )[:-1]
        with self.assertRaisesRegex(ValueError, "1069"):
            decode_packet(truncated)

        packet = bytearray(
            encode_packet(
                make_state(),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )
        )
        packet[4:6] = b"\x00\x00"
        with self.assertRaisesRegex(ValueError, "version"):
            decode_packet(bytes(packet))

    def test_rejects_invalid_scalar_ranges(self):
        state = make_state()
        with self.assertRaisesRegex(ValueError, "robot_mode"):
            encode_packet(
                dataclasses.replace(state, robot_mode=256),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )
        with self.assertRaisesRegex(ValueError, "ros_stamp_nsec"):
            encode_packet(
                dataclasses.replace(state, ros_stamp_nsec=1_000_000_000),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
                zero_jacobian_valid=True,
            )


class SchemaV2Test(unittest.TestCase):
    def _encode(self, valid):
        return encode_packet(
            make_state(),
            sequence=1,
            send_monotonic_ns=2,
            source_state_age_us=3,
            zero_jacobian_valid=valid,
        )

    def test_v2_round_trip_carries_jacobian_validity(self):
        for valid in (True, False):
            decoded = decode_packet(self._encode(valid))
            self.assertEqual(decoded.schema_version, 2)
            self.assertIs(decoded.zero_jacobian_valid, valid)

    def test_decode_accepts_v1_packet_with_unknown_validity(self):
        # v1 = 현행 배포본 하위호환 — flags 는 reserved(0), validity 미지(None)
        packet = bytearray(self._encode(True))
        struct.pack_into("!H", packet, 4, 1)
        struct.pack_into("!H", packet, 10, 0)
        decoded = decode_packet(bytes(packet))
        self.assertEqual(decoded.schema_version, 1)
        self.assertIsNone(decoded.zero_jacobian_valid)

    def test_future_version_decodes_known_prefix_and_ignores_tail(self):
        base = bytearray(self._encode(True))
        extra = 16
        struct.pack_into("!H", base, 4, 7)
        struct.pack_into("!H", base, 8, 1069 + extra)
        packet = bytes(base) + b"\xee" * extra
        decoded = decode_packet(packet)
        self.assertEqual(decoded.schema_version, 7)
        self.assertEqual(decoded.state, make_state())
        self.assertIs(decoded.zero_jacobian_valid, True)

    def test_future_version_rejects_declared_size_mismatch(self):
        base = bytearray(self._encode(True))
        struct.pack_into("!H", base, 4, 7)
        struct.pack_into("!H", base, 8, 1069 + 1)
        with self.assertRaises(ValueError):
            decode_packet(bytes(base))


class LayoutGuardTest(unittest.TestCase):
    def test_layout_guard_survives_python_optimize_mode(self):
        import robot_infra.franka_telemetry_protocol as protocol

        protocol.verify_layout()
        module_ast = ast.parse(inspect.getsource(protocol))
        self.assertEqual(
            [n for n in module_ast.body if isinstance(n, ast.Assert)], []
        )
        # 패키지 __init__(gym 의존)을 우회해 모듈 파일을 직접 로드한다 —
        # gym 없는 파이썬(Control PC 기본 env)에서도 -O 내성만 검증되도록.
        module_file = inspect.getsourcefile(protocol)
        loader_code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('ftp', {module_file!r}); "
            "m = importlib.util.module_from_spec(spec); "
            "sys.modules['ftp'] = m; "
            "spec.loader.exec_module(m); "
            "m.verify_layout()"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", loader_code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
