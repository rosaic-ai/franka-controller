"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`

[moon variant - Option MAX-A : 회전 우선 + 안전]
- K_t=1500, clip_t=0.018 → max F_EE = 27 N (병진 손목 부하 4.05 N·m)
- K_r=160,  clip_r=0.035 → max τ_EE = 5.6 N·m (회전 손목 부하 5.6 N·m, woo의 75%)
- spring 최대 손목 토크 ≈ 9.65 N·m
  motion 중 damping ~2 N·m 추가 → peak ~11.65 N·m
  HW 12 N·m 한계 안에 마진 확보 (yaml soft threshold 11 N·m)
- nullspace_stiffness=5.0  → joint drift 억제 (노란 LED 빈도 ↓)
- joint1_nullspace_stiffness=10.0  → joint 1 별도 보정
- D_t=77 (critical for K_t=1500), D_r=8 (slight overdamped for K_r=160)
- Ki=0  (접촉 task 적분 누적 방지)
"""
import _bootlocale
_bootlocale.getpreferredencoding = lambda *args: 'UTF-8'

# (이 아래부터 원래 코드가 시작됩니다...)
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, HomingActionGoal

from serl_franka_controllers.msg import ZeroJacobian
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_float("gripper_dist", 0.05,
                   "Gripper open distance: 0.09 for single-object task, 0.075 for multi-object task")

flags.DEFINE_bool("force_base_frame", False, "Use base frame for force/torque")

JOINT_RESET_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]

class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self):
        self.grippermovepub = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self.grippergrasppub = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.gripperhomingpub = rospy.Publisher(
            "/franka_gripper/homing/goal", HomingActionGoal, queue_size=1
        )
        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.gripper_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self._update_gripper
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        time.sleep(2)
        # Initialize state variables to prevent AttributeErrors on startup
        self.pos = np.zeros(7)
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.vel = np.zeros(6)
        self.jacobian = np.zeros((6, 7))
        self.gripper_pos = 0.0

        self.state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states", FrankaState, self._set_currpos
        )

    def start_joint_controller(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.stop_impedance()
            self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        self.clear()

        rospy.set_param("/target_joint_positions", JOINT_RESET_TARGET)

        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                "serl_franka_controllers",
                "joint_gumi.launch",
                "robot_ip:=" + FLAGS.robot_ip,
                f"load_gripper:=true",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        # Wait until target joint angles are reached
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(JOINT_RESET_TARGET) - np.array(self.q),
            0,
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

    def set_joint(self):
        rospy.set_param("/target_joint_positions", JOINT_RESET_TARGET)
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(JOINT_RESET_TARGET) - np.array(self.q),
            0,
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        # Stop joint controller
        print("RESET DONE")

    def start_impedance(self):
        """Starts the impedance controller"""
        self.imp = subprocess.Popen([
            "roslaunch", "serl_franka_controllers", "impedance_gumi.launch",
            "robot_ip:=" + FLAGS.robot_ip, "load_gripper:=true"
        ])
        time.sleep(10)

    def stop_impedance(self):
        """Stops the impedance controller"""
        self.imp.terminate()
        time.sleep(1)

    def clear(self):
        """Clears any errors"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop impedance
        try:
            self.stop_impedance()
            self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        self.clear()

        # Launch joint controller reset
        rospy.set_param("/target_joint_positions", JOINT_RESET_TARGET)

        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                "serl_franka_controllers",
                "joint_gumi.launch",
                "robot_ip:=" + FLAGS.robot_ip,
                f"load_gripper:=true",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        # Wait until target joint angles are reached
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(JOINT_RESET_TARGET) - np.array(self.q),
            0,
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        # Stop joint controller
        print("RESET DONE")
        self.joint_controller.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)

        # Restart impedece controller
        self.start_impedance()
        print("impedance STARTED")

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "fr3_link0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        force_torque = msg.O_F_ext_hat_K if FLAGS.force_base_frame else msg.K_F_ext_hat_K
        self.force = np.array(list(force_torque)[:3])
        self.torque = np.array(list(force_torque)[3:])
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)
            rospy.logwarn("Jacobian not set, end-effector velocity temporarily not available")

    def _set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian

    def _update_gripper(self, msg):
        self.gripper_pos = np.sum(msg.position)


    def close(self):
        print("close")
        msg = MoveActionGoal()
        msg.goal.width = 0.0
        msg.goal.speed = 0.2 # 속도 보완
        self.grippermovepub.publish(msg)
        return 'Closed'

    def home_gripper(self):
        print("homing gripper")
        self.gripperhomingpub.publish(HomingActionGoal())
        return 'Homing'

    def open(self):
        print("open")
        msg = MoveActionGoal()
        msg.goal.width=FLAGS.gripper_dist
        msg.goal.speed=0.2 # 속도 보완
        self.grippermovepub.publish(msg)
        return 'Opened'

    def control_gripper(self, width):
        print("control gripper")
        msg = MoveActionGoal()
        msg.goal.width=width
        msg.goal.speed=0.15
        self.grippermovepub.publish(msg)
        return "Control Gripper"

###############################################################################


def main(_):
    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen("roscore")
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    # Start ros node
    rospy.init_node("franka_control_api")

    """Starts impedance controller"""
    robot_server = FrankaServer()
    robot_server.start_impedance()

    # 그리퍼 homing (franka_gripper 노드가 뜬 직후 자동 homing이 안 됐을 경우 대비)
    robot_server.home_gripper()
    time.sleep(5)  # homing 완료 대기

    reconf_client = ReconfClient(
        "/cartesian_impedance_controller/dynamic_reconfigure_compliance_param_node"
    )

    # ==========================================================================
    # [INIT] Option MAX-A — 회전 우선 + 안전
    #   손목 토크 = K_t × clip_t × lever(0.15m) + K_r × clip_r
    #             = 1500 × 0.018 × 0.15 + 160 × 0.035
    #             = 4.05 + 5.60 = 9.65 N·m  (spring 최대)
    #   motion 중 damping ~2 N·m 추가 → peak ~11.65 N·m  ≤ yaml soft 11
    # ==========================================================================
    print("[INIT] Resetting to MAX-A gains "
          "(K_t=1500, clip_t=0.018, K_r=160, clip_r=0.035, nullspace=5.0)")
    reconf_client.update_configuration({"translational_stiffness": 1500})
    reconf_client.update_configuration({"translational_damping": 77})    # 2√1500 ≈ 77 (critical)
    reconf_client.update_configuration({"rotational_stiffness": 160})
    reconf_client.update_configuration({"rotational_damping": 8})        # slight overdamped
    # nullspace 5.0: joint drift 억제 (노란 LED 빈도 ↓, 흔들림 ↓)
    # joint1_ns 10 × ns 5 = 50 effective (joint1 별도 보정)
    reconf_client.update_configuration({"nullspace_stiffness": 5.0})
    reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
        reconf_client.update_configuration({"translational_clip_" + direction: 0.018})
        reconf_client.update_configuration({"rotational_clip_"+ direction: 0.035})
    # Ki를 켜기 전에 현재 위치를 equilibrium으로 전송 → error_i.setZero() 트리거
    # (누적된 적분항이 갑작스런 힘을 유발하는 것을 방지)
    robot_server.move(robot_server.pos.tolist())
    time.sleep(0.2)
    reconf_client.update_configuration({"translational_Ki": 0})
    reconf_client.update_configuration({"rotational_Ki": 0})
    print("[INIT] Impedance parameters reset complete (MAX-A)")


    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": robot_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        robot_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        robot_server.close()
        return "Closed"

    @webapp.route("/control_gripper", methods=["POST"])
    def control_gripper():
        print('set gripper width')
        data = request.get_json()
        width = data.get('width', 0.04)
        print('data: ', data)
        print(f'set gripper width: {width}')
        robot_server.control_gripper(width)

        return "Control Gripper"


    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        print("Moving to", pos)
        robot_server.move(pos)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        try:
            state = {
                "pose": np.array(robot_server.pos).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper": robot_server.gripper_pos,
            }
            return jsonify(state)
        except Exception as e:
            print(f"Error in get_state: {e}")
            return jsonify({"error": str(e)}), 500

    @webapp.route("/start_joint_controller", methods=["POST"])
    def change_controller():
        robot_server.start_joint_controller()
        return "Controller Changed"

    # ==========================================================================
    # /precision_mode — Option MAX-A (INIT과 동일)
    # spring 최대 손목 토크 9.65 N·m + damping ~2 N·m → peak ~11.65 N·m
    # ==========================================================================
    @webapp.route("/precision_mode", methods=["POST"])
    def precision_mode():
        # 1. stiffness / damping (MAX-A)
        reconf_client.update_configuration({"translational_stiffness": 1500})
        reconf_client.update_configuration({"translational_damping": 77})
        reconf_client.update_configuration({"rotational_stiffness": 160})
        reconf_client.update_configuration({"rotational_damping": 8})
        # 2. nullspace: drift 억제 (노란 LED 빈도 ↓, 흔들림 ↓)
        reconf_client.update_configuration({"nullspace_stiffness": 5.0})
        reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
        # 3. clip (MAX-A — 손목 11 N·m soft threshold 안에 cap)
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.018})
            reconf_client.update_configuration({"rotational_clip_"+ direction: 0.035})
        # 4. Ki 켜기 직전 현재 pos 를 equilibrium 으로 전송 → error_i.setZero() 트리거
        robot_server.move(robot_server.pos.tolist())
        time.sleep(0.2)
        # 5. Ki = 0 유지 (접촉 task 적분 누적 방지)
        reconf_client.update_configuration({"translational_Ki": 0})
        reconf_client.update_configuration({"rotational_Ki": 0})
        return 'Precision'

    # ==========================================================================
    # /compliance_mode — 현재 eval_robot.py 흐름에서 호출 안 됨 (dead route)
    # 사용 시점에 검토 필요. 보관용으로만 유지.
    # ==========================================================================
    @webapp.route("/compliance_mode", methods=["POST"])
    def compliance_mode():
        # 1. stiffness / damping
        reconf_client.update_configuration({"translational_stiffness": 800})
        reconf_client.update_configuration({"translational_damping": 80})
        reconf_client.update_configuration({"rotational_stiffness": 150})
        reconf_client.update_configuration({"rotational_damping": 7})
        # 2. nullspace
        reconf_client.update_configuration({"nullspace_stiffness": 2.0})
        reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
        # 3. clip
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.05})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.07})
        # 4. Ki 켜기 직전 현재 pos 를 equilibrium 으로 전송 → error_i.setZero() 트리거
        robot_server.move(robot_server.pos.tolist())
        time.sleep(0.2)
        # 5. Ki = 0 유지
        reconf_client.update_configuration({"translational_Ki": 0})
        reconf_client.update_configuration({"rotational_Ki": 0})
        return 'Compliance'

    webapp.run(host="0.0.0.0")


if __name__ == "__main__":
    app.run(main)
