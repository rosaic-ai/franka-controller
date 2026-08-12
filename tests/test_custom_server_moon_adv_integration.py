import ast
from pathlib import Path
import unittest


SERVER = (
    Path(__file__).parents[1]
    / "robot_infra"
    / "custom_server_moon_adv.py"
)


def parsed_server():
    return ast.parse(SERVER.read_text(encoding="utf-8"))


def route_map(tree):
    routes = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "route"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                continue
            methods = None
            for keyword in decorator.keywords:
                if keyword.arg == "methods":
                    methods = [
                        item.value
                        for item in keyword.value.elts
                        if isinstance(item, ast.Constant)
                    ]
            routes[decorator.args[0].value] = methods
    return routes


class MoonAdvIntegrationTest(unittest.TestCase):
    def test_systemd_service_uses_safe_start_and_restarts(self):
        unit = (Path(__file__).parents[1] / "systemd" / "franka-gateway.service").read_text()
        self.assertIn("--safe_start=true", unit)
        self.assertIn("Restart=always", unit)

    def test_keeps_operational_routes_and_adds_status(self):
        routes = route_map(parsed_server())
        expected_post_routes = {
            "/startimp",
            "/stopimp",
            "/getpos",
            "/getvel",
            "/getforce",
            "/gettorque",
            "/getq",
            "/getdq",
            "/getjacobian",
            "/get_gripper",
            "/jointreset",
            "/open_gripper",
            "/close_gripper",
            "/control_gripper",
            "/clearerr",
            "/pose",
            "/getstate",
            "/start_joint_controller",
            "/precision_mode",
            "/compliance_mode",
            "/diag_static",
        }
        self.assertEqual(
            routes,
            {
                **{route: ["POST"] for route in expected_post_routes},
                "/health": ["POST"],
                "/telemetry/status": ["GET"],
            },
        )

    def test_defines_telemetry_flags_and_callback_wiring(self):
        tree = parsed_server()
        flag_names = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"DEFINE_string", "DEFINE_integer"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertTrue(
            {"telemetry_host", "telemetry_port", "telemetry_hz"}
            <= flag_names
        )

        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        for method_name, update_name in (
            ("_set_currpos", "update_franka_state"),
            ("_set_jacobian", "update_jacobian"),
        ):
            first = methods[method_name].body[0]
            self.assertIsInstance(first, ast.Expr)
            self.assertIsInstance(first.value, ast.Call)
            self.assertIsInstance(first.value.func, ast.Name)
            self.assertEqual(first.value.func.id, "safe_telemetry_update")
            callback = first.value.args[0]
            self.assertIsInstance(callback, ast.Attribute)
            self.assertEqual(callback.attr, update_name)

    def test_safe_start_defaults_off_and_guards_startup_motion(self):
        tree = parsed_server()
        safe_flag = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "DEFINE_bool"
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "safe_start"
        )
        self.assertIs(safe_flag.args[1].value, False)

        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        safe_branch = next(
            node
            for node in main.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "safe_start"
        )
        guard = safe_branch.orelse
        guarded_calls = {
            node.func.attr
            for statement in guard
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(
            {"start_impedance", "home_gripper", "update_configuration", "move"}
            <= guarded_calls
        )
        self.assertEqual(
            [
                node.func.attr
                for statement in safe_branch.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "robot_server"
            ],
            ["start_state_backend"],
        )

    def test_main_owns_publisher_lifecycle_and_reports_real_entrypoint(self):
        tree = parsed_server()
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        called_names = {
            node.func.id
            for node in calls
            if isinstance(node.func, ast.Name)
        }
        called_attributes = {
            node.func.attr
            for node in calls
            if isinstance(node.func, ast.Attribute)
        }
        self.assertIn("FrankaUdpPublisher", called_names)
        self.assertIn("start", called_attributes)
        self.assertIn("stop", called_attributes)
        self.assertTrue(
            any(
                isinstance(node, ast.Try) and node.finalbody
                for node in ast.walk(tree)
            )
        )

        entrypoints = {
            keyword.value.value
            for node in calls
            for keyword in node.keywords
            if keyword.arg == "entrypoint"
            and isinstance(keyword.value, ast.Constant)
        }
        self.assertEqual(
            entrypoints,
            {"robot_infra/custom_server_moon_adv.py"},
        )

    def test_signal_shutdown_unwinds_and_publisher_is_registered_early(self):
        tree = parsed_server()
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        handler = methods["_shutdown_handler"]
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "_exit"
                for node in ast.walk(handler)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and isinstance(node.exc.func, ast.Name)
                and node.exc.func.id == "SystemExit"
                for node in ast.walk(handler)
            )
        )

        main = methods["main"]
        atexit_register_lines = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "atexit"
            and node.func.attr == "register"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
            and isinstance(node.args[0].value, ast.Name)
            and node.args[0].value.id == "telemetry_publisher"
            and node.args[0].attr == "stop"
        ]
        controller_start_lines = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "robot_server"
            and node.func.attr == "start_impedance"
        ]
        self.assertEqual(len(atexit_register_lines), 1)
        self.assertTrue(controller_start_lines)
        self.assertLess(
            atexit_register_lines[0],
            min(controller_start_lines),
        )


if __name__ == "__main__":
    unittest.main()
