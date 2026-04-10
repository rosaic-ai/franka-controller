"""
Franka 로봇 제어 서버 (Position Controller 버전)
- Flask 서버로 pose/state 요청 처리
- CartesianPoseReplayController 사용 (진짜 position control)
- 빠르고 정확한 위치 제어
"""
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
import signal
import sys
import threading
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "172.16.0.2", "IP address of the franka robot's controller box")
flags.DEFINE_string("arm_id", "panda", "Arm ID (panda or fr3)")
flags.DEFINE_float("gripper_dist", 0.05, "Gripper open distance")
flags.DEFINE_bool("force_base_frame", False, "Use base frame for force/torque")

JOINT_RESET_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]

# 전역 종료 플래그
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print('\n[SHUTDOWN] Ctrl+C 감지, 서버 종료 중...')
    shutdown_flag = True
    if rospy.is_initialized():
        rospy.signal_shutdown('User interrupted')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class FrankaServerPosition:
    """Franka 로봇 제어 서버 (Position Controller)"""

    def __init__(self):
        # 상태 변수 초기화
        self.pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.vel = np.zeros(6)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.gripper_pos = 0.0

        self.grippermovepub = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal, queue_size=1)
        self.grippergrasppub = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1)

        # CartesianPoseReplayController의 target_pose topic 사용
        self.eepub = rospy.Publisher(
            "/cartesian_pose_replay_controller/target_pose",
            geom_msg.PoseStamped,
            queue_size=10
        )

        self.resetpub = rospy.Publisher("/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1)
        self.gripper_sub = rospy.Subscriber("/franka_gripper/joint_states", JointState, self._update_gripper)
        time.sleep(2)
        self.state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self._set_currpos)

    def start_controller(self):
        """Position 컨트롤러 시작"""
        print(f"[CONTROLLER] Starting CartesianPoseReplayController (robot_ip={FLAGS.robot_ip}, arm_id={FLAGS.arm_id})")

        # Launch 파일로 franka_control + controller 시작
        launch_cmd = f"source /home/demo-panda/catkin_ws/devel/setup.bash && roslaunch serl_franka_controllers cartesian_pose_gumi.launch robot_ip:={FLAGS.robot_ip} load_gripper:=true"
        self.controller_proc = subprocess.Popen(
            launch_cmd,
            shell=True,
            executable="/bin/bash"
        )

        # Controller가 시작될 때까지 대기
        print("[CONTROLLER] Waiting for controller to start...")
        time.sleep(10)
        print("[CONTROLLER] Controller started!")

    def stop_controller(self):
        """Position 컨트롤러 중지"""
        if hasattr(self, 'controller_proc'):
            self.controller_proc.terminate()
        time.sleep(1)

    def clear(self):
        """에러 복구"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def move(self, pose: list):
        """로봇 이동: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = FLAGS.arm_id + "_link0"
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
        self.vel = np.zeros(6)

    def _update_gripper(self, msg):
        self.gripper_pos = np.sum(msg.position)

    def close(self):
        grasp = GraspActionGoal()
        grasp.goal.width = 0.02
        grasp.goal.speed = 0.15
        grasp.goal.epsilon.inner = 1
        grasp.goal.epsilon.outer = 1
        grasp.goal.force = 0.2
        self.grippergrasppub.publish(grasp)

    def open(self):
        msg = MoveActionGoal()
        msg.goal.width = FLAGS.gripper_dist
        msg.goal.speed = 0.15
        self.grippermovepub.publish(msg)

    def control_gripper(self, width):
        msg = MoveActionGoal()
        msg.goal.width = width
        msg.goal.speed = 0.15
        self.grippermovepub.publish(msg)


###############################################################################

def main(_):
    webapp = Flask(__name__)

    print("[INIT] Starting position controller server...")

    # 먼저 controller launch (roscore 포함)
    print("[INIT] Starting controller (this will take ~10 seconds)...")

    # FrankaServerPosition을 나중에 초기화 (ROS 없이)
    # 먼저 launch만 실행 (workspace를 source한 환경에서)
    launch_cmd = f"source /home/demo-panda/catkin_ws/devel/setup.bash && roslaunch serl_franka_controllers cartesian_pose_gumi.launch robot_ip:={FLAGS.robot_ip} load_gripper:=true"
    controller_proc = subprocess.Popen(
        launch_cmd,
        shell=True,
        executable="/bin/bash"
    )

    print("[INIT] Waiting for ROS master and controller to start...")
    time.sleep(12)

    # 이제 ROS node 초기화
    rospy.init_node("franka_control_api_position")
    print("[INIT] ROS node initialized")

    # Robot server 초기화 (controller는 이미 실행 중이므로 start_controller 호출 안 함)
    robot_server = FrankaServerPosition()
    robot_server.controller_proc = controller_proc  # 나중에 종료할 수 있게 저장

    print("[INIT] CartesianPoseReplayController started (빠르고 정확한 위치 제어)")

    # 최신 target만 유지
    latest_target = {"pos": None, "time": 0}
    target_lock = threading.Lock()

    # Flask Routes
    @webapp.route("/startimp", methods=["POST"])
    def start_controller():
        robot_server.clear()
        robot_server.start_controller()
        return "Started controller"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_controller():
        robot_server.stop_controller()
        return "Stopped controller"

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
        return jsonify({"jacobian": np.zeros((6, 7)).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": robot_server.gripper_pos})

    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        robot_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        robot_server.close()
        return "Closed"

    @webapp.route("/control_gripper", methods=["POST"])
    def control_gripper():
        data = request.get_json()
        width = data.get('width', 0.04)
        robot_server.control_gripper(width)
        return "Control Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        nonlocal latest_target
        recv_time = time.time()
        data = request.json
        pos = np.array(data["arr"])

        # 자동 에러 복구
        robot_server.clear()

        # 최신 target으로 덮어쓰기
        with target_lock:
            latest_target = {"pos": pos, "time": recv_time, "data": data}

        return "Queued"

    @webapp.route("/pose_sync", methods=["POST"])
    def pose_sync():
        """Sync pose: 로봇이 도달할 때까지 블로킹 후 현재 pose 반환"""
        data = request.json
        pos = np.array(data["arr"])
        thresh = data.get("reach_threshold", 0.0002)
        timeout_s = data.get("reach_timeout_ms", 300) / 1000.0

        robot_server.clear()

        current_pos = robot_server.pos[:3]
        distance = np.linalg.norm(pos[:3] - current_pos)
        if distance > 0.05:
            estimated_time = (distance / 0.050) + 1.0
            timeout_s = max(timeout_s, estimated_time)
            print(f"[POSE_SYNC] 큰 이동 감지 ({distance*1000:.1f}mm), timeout 자동 연장 → {timeout_s:.1f}s")

        start_pos = robot_server.pos[:3].copy()
        robot_server.move(pos)

        start_t = time.time()
        moved = False
        status = "timeout"

        while time.time() - start_t < timeout_s:
            curr = robot_server.pos[:3]

            if not moved:
                if np.linalg.norm(curr - start_pos) > 0.001:
                    moved = True
                else:
                    time.sleep(0.002)
                    continue

            pos_err = np.linalg.norm(curr - pos[:3])
            if pos_err < thresh:
                status = "reached"
                break
            time.sleep(0.002)

        curr_pose = robot_server.pos
        pos_err_final = np.linalg.norm(curr_pose[:3] - pos[:3])
        err_xyz = (curr_pose[:3] - pos[:3]) * 1000
        print(f"[POSE_SYNC] {status.upper()}: err_total={pos_err_final*1000:.2f}mm [x={err_xyz[0]:+.2f}, y={err_xyz[1]:+.2f}, z={err_xyz[2]:+.2f}], elapsed={time.time()-start_t:.3f}s")

        return jsonify({
            "status": status,
            "pose": np.array(curr_pose).tolist()
        })

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify({
            "pose": np.array(robot_server.pos).tolist(),
            "vel": np.array(robot_server.vel).tolist(),
            "force": np.array(robot_server.force).tolist(),
            "torque": np.array(robot_server.torque).tolist(),
            "q": np.array(robot_server.q).tolist(),
            "dq": np.array(robot_server.dq).tolist(),
            "jacobian": np.zeros((6, 7)).tolist(),
            "gripper": robot_server.gripper_pos,
        })

    @webapp.route("/precision_mode", methods=["POST"])
    def precision_mode():
        return 'Position controller (always precise)'

    @webapp.route("/compliance_mode", methods=["POST"])
    def compliance_mode():
        return 'Position controller (no compliance mode)'

    # Pose executor thread with smooth interpolation
    def pose_executor():
        nonlocal latest_target
        last_exec_time = 0
        current_target = None
        interpolation_steps = 5  # 5 steps interpolation for smooth motion

        while not shutdown_flag:
            time.sleep(0.005)  # 200Hz

            with target_lock:
                if latest_target["pos"] is not None:
                    # 새 target 받음 - interpolation 시작
                    new_target = latest_target["pos"].copy()
                    latest_target["pos"] = None

                    # 현재 로봇 위치에서 새 target까지 interpolation
                    start_pos = robot_server.pos.copy()

                    # Interpolation: 현재 위치 → 새 target (부드럽게)
                    for i in range(1, interpolation_steps + 1):
                        alpha = i / interpolation_steps
                        # Linear interpolation for position
                        interp_pos = start_pos[:3] * (1 - alpha) + new_target[:3] * alpha
                        # Slerp for quaternion (simplified: just linear for now)
                        interp_quat = start_pos[3:] * (1 - alpha) + new_target[3:] * alpha
                        # Normalize quaternion
                        quat_norm = np.linalg.norm(interp_quat)
                        if quat_norm > 0:
                            interp_quat = interp_quat / quat_norm

                        interp_full = np.concatenate([interp_pos, interp_quat])
                        robot_server.move(interp_full)
                        time.sleep(0.005)  # 5ms between interpolation steps

                    current_target = new_target
                    last_exec_time = time.time()

    executor_thread = threading.Thread(target=pose_executor, daemon=True)
    executor_thread.start()

    print("[SERVER] Flask 서버 시작 (Position Controller, Ctrl+C로 종료)")
    webapp.run(host="0.0.0.0", threaded=True)


if __name__ == "__main__":
    app.run(main)
