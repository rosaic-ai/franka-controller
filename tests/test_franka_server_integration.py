import ast
from pathlib import Path
import unittest


SERVER = Path(__file__).parents[1] / "robot_infra" / "franka_server.py"


def parsed_server():
    return ast.parse(SERVER.read_text(encoding="utf-8"))


class ServerIntegrationTest(unittest.TestCase):
    def test_server_keeps_legacy_routes_and_adds_get_status_route(self):
        routes = {}
        for node in ast.walk(parsed_server()):
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

        self.assertEqual(routes["/getstate"], ["POST"])
        self.assertEqual(routes["/pose"], ["POST"])
        self.assertEqual(routes["/get_gripper"], ["POST"])
        self.assertEqual(routes["/telemetry/status"], ["GET"])

    def test_server_defines_telemetry_flags_and_callback_wiring(self):
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
            self.assertEqual(first.value.func.attr, update_name)

    def test_main_owns_publisher_lifecycle(self):
        tree = parsed_server()
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
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
            any(isinstance(node, ast.Try) and node.finalbody for node in ast.walk(tree))
        )


if __name__ == "__main__":
    unittest.main()
