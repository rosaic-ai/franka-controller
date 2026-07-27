import unittest

import tests  # noqa: F401  # installs test-only optional dependency stubs

from robot_infra.franka_telemetry import FrankaUdpPublisher
from robot_infra.franka_telemetry_protocol import decode_packet
from robot_infra.franka_telemetry import TelemetryStateStore
from tests.test_franka_telemetry_store import make_franka_msg, ready_store


class FakeSocket:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []
        self.closed = False

    def sendto(self, data, destination):
        if self.fail:
            raise OSError("network down")
        self.sent.append((data, destination))
        return len(data)

    def close(self):
        self.closed = True


def fail_factory():
    raise AssertionError("disabled publisher must not open a socket")


class UdpPublisherTest(unittest.TestCase):
    def test_disabled_publisher_opens_no_socket(self):
        publisher = FrankaUdpPublisher(
            store=ready_store(),
            host="",
            socket_factory=fail_factory,
        )
        self.assertFalse(publisher.enabled)
        self.assertFalse(
            publisher.send_once(now_monotonic_ns=1_000_000)
        )
        publisher.start()
        publisher.stop()

    def test_send_once_emits_exact_packet_and_increments_sequence(self):
        sock = FakeSocket()
        publisher = FrankaUdpPublisher(
            store=ready_store(),
            host="20.42.0.54",
            port=5010,
            hz=200,
            socket_factory=lambda: sock,
        )
        self.assertTrue(
            publisher.send_once(now_monotonic_ns=1_000_000)
        )
        self.assertTrue(
            publisher.send_once(now_monotonic_ns=1_000_000)
        )
        packet, destination = sock.sent[0]
        self.assertEqual(len(packet), 1069)
        self.assertEqual(destination, ("20.42.0.54", 5010))
        self.assertEqual(decode_packet(packet).sequence, 0)
        self.assertEqual(decode_packet(sock.sent[1][0]).sequence, 1)
        self.assertEqual(publisher.stats_snapshot()["sent_packets"], 2)
        self.assertEqual(publisher.stats_snapshot()["last_sequence"], 1)
        publisher.stop()
        self.assertTrue(sock.closed)

    def test_stale_and_socket_error_never_raise_to_control_path(self):
        stale = FrankaUdpPublisher(
            store=ready_store(state_ns=0, jacobian_ns=0),
            host="20.42.0.54",
            socket_factory=lambda: FakeSocket(),
        )
        self.assertFalse(
            stale.send_once(now_monotonic_ns=25_000_001)
        )
        self.assertEqual(stale.stats_snapshot()["stale_skips"], 1)

        broken = FrankaUdpPublisher(
            store=ready_store(),
            host="20.42.0.54",
            socket_factory=lambda: FakeSocket(fail=True),
        )
        self.assertFalse(
            broken.send_once(now_monotonic_ns=1_000_000)
        )
        self.assertEqual(broken.stats_snapshot()["send_errors"], 1)

    def test_jacobian_validity_flag_reaches_wire_and_counts(self):
        sock = FakeSocket()
        publisher = FrankaUdpPublisher(
            store=ready_store(),
            host="20.42.0.54",
            socket_factory=lambda: sock,
        )
        self.assertTrue(publisher.send_once(now_monotonic_ns=1_000_000))
        self.assertIs(
            decode_packet(sock.sent[0][0]).zero_jacobian_valid, True
        )
        self.assertEqual(
            publisher.stats_snapshot()["jacobian_invalid_sends"], 0
        )

        state_only_sock = FakeSocket()
        state_only = TelemetryStateStore(monotonic_ns=lambda: 1_000_000)
        state_only.update_franka_state(
            make_franka_msg(), received_monotonic_ns=900_000
        )
        state_only_publisher = FrankaUdpPublisher(
            store=state_only,
            host="20.42.0.54",
            socket_factory=lambda: state_only_sock,
        )
        self.assertTrue(
            state_only_publisher.send_once(now_monotonic_ns=1_000_000)
        )
        decoded = decode_packet(state_only_sock.sent[0][0])
        self.assertIs(decoded.zero_jacobian_valid, False)
        self.assertEqual(decoded.state.zero_jacobian, (0.0,) * 42)
        self.assertEqual(
            state_only_publisher.stats_snapshot()["jacobian_invalid_sends"],
            1,
        )

    def test_rejects_invalid_configuration_before_opening_socket(self):
        with self.assertRaisesRegex(ValueError, "IPv4"):
            FrankaUdpPublisher(store=ready_store(), host="not-an-ip")
        with self.assertRaisesRegex(ValueError, "1-200"):
            FrankaUdpPublisher(
                store=ready_store(),
                host="20.42.0.54",
                hz=201,
            )
        with self.assertRaisesRegex(ValueError, "1-65535"):
            FrankaUdpPublisher(
                store=ready_store(),
                host="20.42.0.54",
                port=0,
            )


if __name__ == "__main__":
    unittest.main()
