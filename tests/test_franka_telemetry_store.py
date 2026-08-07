import unittest
from types import SimpleNamespace

import tests  # noqa: F401  # installs test-only optional dependency stubs

from robot_infra.franka_telemetry import (
    TelemetryStateStore,
    build_health_payload,
    build_status_payload,
    safe_telemetry_update,
)


F_T_EE = tuple(float(i) for i in range(16))
F_T_NE = tuple(float(i + 16) for i in range(16))
NE_T_EE = tuple(float(i + 32) for i in range(16))
EE_T_K = tuple(float(i + 48) for i in range(16))


class FakeErrors:
    __slots__ = ("joint_reflex", "communication_constraints_violation")

    def __init__(
        self,
        joint_reflex=False,
        communication_constraints_violation=False,
    ):
        self.joint_reflex = joint_reflex
        self.communication_constraints_violation = (
            communication_constraints_violation
        )


def make_franka_msg():
    values7 = tuple(float(i) for i in range(7))
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(secs=123, nsecs=456_000_000)
        ),
        time=7.5,
        q=values7,
        q_d=tuple(v + 10.0 for v in values7),
        dq=tuple(v + 20.0 for v in values7),
        dq_d=tuple(v + 30.0 for v in values7),
        ddq_d=tuple(v + 40.0 for v in values7),
        tau_J=tuple(v + 50.0 for v in values7),
        tau_J_d=tuple(v + 60.0 for v in values7),
        tau_ext_hat_filtered=tuple(v + 70.0 for v in values7),
        O_T_EE=tuple(float(i) for i in range(16)),
        K_F_ext_hat_K=tuple(float(i) for i in range(6)),
        O_F_ext_hat_K=tuple(float(i + 6) for i in range(6)),
        control_command_success_rate=0.999,
        robot_mode=2,
        F_T_EE=F_T_EE,
        F_T_NE=F_T_NE,
        NE_T_EE=NE_T_EE,
        EE_T_K=EE_T_K,
        m_ee=0.1,
        F_x_Cee=(0.01, 0.02, 0.03),
        I_ee=tuple(float(i) / 100.0 for i in range(9)),
        m_load=0.96,
        F_x_Cload=(0.0, 0.01, 0.03),
        I_load=tuple(float(i) / 10.0 for i in range(9)),
        m_total=1.06,
        F_x_Ctotal=(0.0, 0.02, 0.04),
        I_total=tuple(float(i) / 5.0 for i in range(9)),
        current_errors=FakeErrors(joint_reflex=True),
        last_motion_errors=FakeErrors(
            communication_constraints_violation=True
        ),
    )


def make_jacobian_msg():
    return SimpleNamespace(
        zero_jacobian=tuple(float(i) / 10.0 for i in range(42))
    )


def ready_store(state_ns=900_000, jacobian_ns=950_000):
    store = TelemetryStateStore(monotonic_ns=lambda: 1_000_000)
    store.update_franka_state(
        make_franka_msg(),
        received_monotonic_ns=state_ns,
    )
    store.update_jacobian(
        make_jacobian_msg(),
        received_monotonic_ns=jacobian_ns,
    )
    return store


class TelemetryStoreTest(unittest.TestCase):
    def test_health_fails_closed_for_stale_state_and_reports_faults(self):
        store = ready_store(state_ns=900_000, jacobian_ns=950_000)
        healthy = build_health_payload(
            store=store,
            controller_active=True,
            observed_at="2026-08-07T08:00:00+00:00",
            now_monotonic_ns=1_000_000,
        )
        self.assertTrue(healthy["stateFresh"])
        self.assertTrue(healthy["controllerActive"])
        self.assertEqual(healthy["robotMode"], 2)
        self.assertEqual(healthy["fault"], {
            "active": True,
            "codes": ["joint_reflex"],
        })

        stale = build_health_payload(
            store=store,
            controller_active=False,
            observed_at="2026-08-07T08:00:03+00:00",
            now_monotonic_ns=2_001_000_000,
        )
        self.assertFalse(stale["stateFresh"])
        self.assertFalse(stale["controllerActive"])

    def test_state_only_snapshot_sends_with_invalid_jacobian(self):
        # 컨트롤러 전환/position 모드에서 jacobian publisher 가 멈춰도
        # state 텔레메트리는 계속 흘러야 한다 — 무효 플래그로 명시 (v2).
        store = TelemetryStateStore(monotonic_ns=lambda: 1_000_000)
        self.assertIsNone(
            store.snapshot_for_send(
                now_monotonic_ns=1_000_000,
                max_age_ns=25_000_000,
            )
        )
        store.update_franka_state(
            make_franka_msg(),
            received_monotonic_ns=900_000,
        )
        snapshot = store.snapshot_for_send(
            now_monotonic_ns=1_000_000,
            max_age_ns=25_000_000,
        )
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.zero_jacobian_valid)
        self.assertEqual(snapshot.state.zero_jacobian, (0.0,) * 42)
        self.assertEqual(snapshot.source_jacobian_age_us, 2**32 - 1)
        store.update_jacobian(
            make_jacobian_msg(),
            received_monotonic_ns=950_000,
        )
        snapshot = store.snapshot_for_send(
            now_monotonic_ns=1_000_000,
            max_age_ns=25_000_000,
        )
        self.assertTrue(snapshot.zero_jacobian_valid)
        self.assertEqual(
            snapshot.state.zero_jacobian,
            make_jacobian_msg().zero_jacobian,
        )
        self.assertEqual(snapshot.source_state_age_us, 100)
        self.assertEqual(snapshot.source_jacobian_age_us, 50)

    def test_stale_state_suppresses_but_stale_jacobian_sends_invalid(self):
        stale_state = ready_store(state_ns=0, jacobian_ns=1_000_000)
        self.assertIsNone(
            stale_state.snapshot_for_send(
                now_monotonic_ns=25_000_001,
                max_age_ns=25_000_000,
            )
        )
        stale_jacobian = ready_store(state_ns=1_000_000, jacobian_ns=0)
        snapshot = stale_jacobian.snapshot_for_send(
            now_monotonic_ns=25_000_001,
            max_age_ns=25_000_000,
        )
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.zero_jacobian_valid)
        self.assertEqual(snapshot.state.zero_jacobian, (0.0,) * 42)
        # 존재하되 신선하지 않은 jacobian 은 실제 나이를 보고한다
        self.assertEqual(snapshot.source_jacobian_age_us, 25_000)

    def test_status_reports_applied_payload_frames_and_active_errors(self):
        status = build_status_payload(
            store=ready_store(),
            publisher_stats={
                "enabled": True,
                "host": "20.42.0.54",
                "port": 5010,
                "hz": 200,
                "sent_packets": 10,
                "send_errors": 0,
                "stale_skips": 0,
                "last_sequence": 9,
                "last_send_monotonic_ns": 900_000,
            },
            server_commit="abc123",
            server_started_at="2026-07-25T12:00:00+09:00",
            entrypoint="robot_infra/franka_server.py",
            now_monotonic_ns=1_000_000,
        )
        self.assertTrue(status["ready"])
        self.assertEqual(
            status["schema"]["field_order"],
            "franka_state_v2_127d_flags",
        )
        self.assertEqual(
            status["server"]["started_at"],
            "2026-07-25T12:00:00+09:00",
        )
        self.assertEqual(status["udp"]["destination"], "20.42.0.54:5010")
        self.assertEqual(status["udp"]["target_hz"], 200)
        self.assertEqual(status["udp"]["last_sequence"], 9)
        self.assertEqual(status["udp"]["last_send_age_ms"], 0.1)
        self.assertEqual(status["payload"]["m_load"], 0.96)
        self.assertEqual(status["frames"]["F_T_EE"], list(F_T_EE))
        self.assertEqual(
            status["robot"]["current_errors"],
            ["joint_reflex"],
        )
        self.assertEqual(
            status["robot"]["last_motion_errors"],
            ["communication_constraints_violation"],
        )

    def test_status_explains_which_input_is_missing(self):
        empty = TelemetryStateStore(monotonic_ns=lambda: 1_000_000)
        status = empty.status_snapshot(now_monotonic_ns=1_000_000)
        self.assertFalse(status["ready"])
        self.assertEqual(status["source"]["reason"], "waiting_for_franka_state")

        empty.update_franka_state(
            make_franka_msg(),
            received_monotonic_ns=900_000,
        )
        status = empty.status_snapshot(now_monotonic_ns=1_000_000)
        self.assertEqual(
            status["source"]["reason"],
            "waiting_for_zero_jacobian",
        )

    def test_safe_update_contains_telemetry_conversion_errors(self):
        def broken_update(_msg):
            raise ValueError("bad telemetry field")

        self.assertFalse(
            safe_telemetry_update(
                broken_update,
                object(),
                logger=lambda *_args: None,
            )
        )


if __name__ == "__main__":
    unittest.main()
