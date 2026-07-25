import dataclasses
import unittest

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
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            encode_packet(
                dataclasses.replace(state, q=(float("nan"),) + state.q[1:]),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
            )

    def test_rejects_invalid_header_and_size(self):
        packet = bytearray(
            encode_packet(
                make_state(),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
            )
        )
        packet[0:4] = b"BAD!"
        with self.assertRaisesRegex(ValueError, "magic"):
            decode_packet(bytes(packet))
        with self.assertRaisesRegex(ValueError, "1069"):
            decode_packet(bytes(packet[:-1]))

        packet = bytearray(
            encode_packet(
                make_state(),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
            )
        )
        packet[4:6] = b"\x00\x02"
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
            )
        with self.assertRaisesRegex(ValueError, "ros_stamp_nsec"):
            encode_packet(
                dataclasses.replace(state, ros_stamp_nsec=1_000_000_000),
                sequence=0,
                send_monotonic_ns=0,
                source_state_age_us=0,
            )


if __name__ == "__main__":
    unittest.main()
