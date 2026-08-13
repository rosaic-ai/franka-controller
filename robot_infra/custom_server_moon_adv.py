"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python custom_server_moon_adv.py`

[moon_adv variant] — MAX-A gains + session logging + q_drift hypothesis verification
─────────────────────────────────────────────────────────────────────────────
Same impedance gains as custom_server_moon.py (Option MAX-A):
  K_t=1500, c_t=0.018 | K_r=160, c_r=0.035 | nullspace=5.0 | Ki=0

Additional logging for diagnostics:
  - Silenced Flask access log + /pose throttling (1/100)
  - Discrete event logs: [MODE], [RECOVERY], [RESET], [POSE_MILESTONE]
  - Background STATUS thread (every 5s): pose, q_drift, force, torque
  - Session log directory: ~/franka_logs/session_YYYYMMDD_HHMMSS/
      ├ status.csv          — time series for plotting (pandas-ready)
      ├ events.log          — event timeline
      ├ pose_milestones.log — pose milestones every 100 steps
      └ summary.txt         — auto-generated on shutdown

Hypothesis under verification:
  controller's q_d_nullspace_ is captured ONCE at starting() time.
  Over time, current q drifts from q_d_nullspace_ → nullspace tug-of-war
  → performance degradation → only server restart resolves it.
  q_drift = ‖current q - start_q‖ is the direct measurement of this mismatch.
"""
import _bootlocale
_bootlocale.getpreferredencoding = lambda *args: 'UTF-8'

from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import os
import csv
import atexit
import signal
import threading
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_gripper.msg import GraspActionGoal, MoveActionGoal, HomingActionGoal

from serl_franka_controllers.msg import ZeroJacobian
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

if __package__:
    from .franka_telemetry import (
        FrankaUdpPublisher,
        TelemetryStateStore,
        build_health_payload,
        build_status_payload,
        safe_telemetry_update,
    )
else:
    from franka_telemetry import (
        FrankaUdpPublisher,
        TelemetryStateStore,
        build_health_payload,
        build_status_payload,
        safe_telemetry_update,
    )

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_float("gripper_dist", 0.05,
                   "Gripper open distance: 0.09 for single-object task, 0.075 for multi-object task")

flags.DEFINE_bool("force_base_frame", False, "Use base frame for force/torque")
flags.DEFINE_string(
    "telemetry_host",
    "",
    "ROSAIC UDP telemetry destination; empty disables",
)
flags.DEFINE_integer(
    "telemetry_port",
    5010,
    "ROSAIC UDP telemetry port",
)
flags.DEFINE_integer(
    "telemetry_hz",
    200,
    "Franka UDP telemetry rate, 1-200 Hz",
)
flags.DEFINE_bool(
    "safe_start",
    False,
    "Start HTTP/telemetry without controller, homing, gain, or pose commands",
)

JOINT_RESET_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]

# ============================================================================
# Session Logging Setup
# ============================================================================
SESSION_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
SESSION_DIR = os.path.expanduser(f"~/franka_logs/session_{SESSION_TS}")
os.makedirs(SESSION_DIR, exist_ok=True)

STATUS_CSV_PATH = os.path.join(SESSION_DIR, "status.csv")
EVENTS_LOG_PATH = os.path.join(SESSION_DIR, "events.log")
POSE_LOG_PATH   = os.path.join(SESSION_DIR, "pose_milestones.log")
SUMMARY_PATH    = os.path.join(SESSION_DIR, "summary.txt")

# Open files (line-buffered)
_status_csv_file = open(STATUS_CSV_PATH, "w", buffering=1)
_status_writer = csv.writer(_status_csv_file)
_status_writer.writerow(["timestamp", "t_sec",
                         "pose_x", "pose_y", "pose_z",
                         "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                         "q_drift", "force_mag", "torque_mag"])
_status_csv_file.flush()

_events_file = open(EVENTS_LOG_PATH, "w", buffering=1)
_pose_milestones_file = open(POSE_LOG_PATH, "w", buffering=1)

# Session-level state
SESSION_START_TIME = time.time()
START_Q = None  # captured after impedance controller stabilizes

# Event counters for summary
_event_counts = {
    "MODE_precision": 0,
    "MODE_compliance": 0,
    "RECOVERY": 0,
    "JOINTRESET": 0,
    "POSE": 0,
}

# In-memory STATUS samples — used by _write_summary so it works even on SIGKILL
# (CSV is also written, but in-memory list is the source of truth for summary)
_status_samples = []
_status_lock = threading.Lock()
_shutdown_requested = threading.Event()
_exit_code = 0


def _log_event(tag, msg):
    """Log a discrete event to events.log and stdout."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{tag}] {msg}"
    print(line)
    try:
        _events_file.write(line + "\n")
        _events_file.flush()
    except Exception:
        pass


def _write_summary(verbose=False, final=False):
    """Generate summary.txt from in-memory samples + event counts.

    Called periodically (verbose=False) from STATUS thread AND on shutdown
    (verbose=True, final=True). Using in-memory samples means summary survives
    even on SIGKILL (atexit/signal handlers might not run, but periodic writes
    ensure the latest summary up to the last STATUS sample is always on disk).
    """
    if final:
        try:
            _events_file.close()
            _status_csv_file.close()
            _pose_milestones_file.close()
        except Exception:
            pass

    try:
        with _status_lock:
            samples = list(_status_samples)

        if not samples:
            with open(SUMMARY_PATH, "w") as f:
                f.write("No status samples collected yet.\n")
            return

        q_drifts = [s["q_drift"] for s in samples]
        force_mags = [s["force_mag"] for s in samples]
        torque_mags = [s["torque_mag"] for s in samples]

        q_drift_max = max(q_drifts)
        q_drift_mean = sum(q_drifts) / len(q_drifts)
        q_drift_final = q_drifts[-1]
        q_drift_max_idx = q_drifts.index(q_drift_max)
        q_drift_max_time = samples[q_drift_max_idx]["timestamp"]

        force_max = max(force_mags)
        force_mean = sum(force_mags) / len(force_mags)
        force_max_idx = force_mags.index(force_max)
        force_max_time = samples[force_max_idx]["timestamp"]
        torque_max = max(torque_mags)

        first, last = samples[0], samples[-1]
        start_q_arr = first["q"]
        final_q_arr = last["q"]
        delta_q = final_q_arr - start_q_arr
        most_drift_idx = int(np.argmax(np.abs(delta_q)))

        duration_sec = last["t_sec"]
        h = int(duration_sec // 3600)
        m = int((duration_sec % 3600) // 60)
        s = int(duration_sec % 60)
        duration_str = f"{h}h {m}m {s}s"

        # Hypothesis verdict heuristic
        verdict = "INCONCLUSIVE (need more data)"
        if q_drift_max > 0.5 and q_drift_max > max(q_drifts[0], 0.01) * 5:
            verdict = "SUPPORTS hypothesis (q_drift grew significantly)"
        elif q_drift_max < 0.2:
            verdict = "REFUTES hypothesis (q_drift stayed small)"

        status_marker = "[FINAL]" if final else "[LIVE — updated every STATUS tick]"

        summary = f"""═══════════════════════════════════════════════════════════
  Session Summary {status_marker} — {SESSION_TS}
═══════════════════════════════════════════════════════════

[Timing]
Start:     {first['timestamp']}
End:       {last['timestamp']}
Duration:  {duration_str}
Samples:   {len(samples)} (every 5s)

[q drift analysis]   ← Hypothesis verification core
start_q:       {np.round(start_q_arr, 3).tolist()}
final_q:       {np.round(final_q_arr, 3).tolist()}
q_drift max:   {q_drift_max:.3f} rad  @ {q_drift_max_time}
q_drift mean:  {q_drift_mean:.3f} rad
q_drift final: {q_drift_final:.3f} rad

Per-joint drift (final - start):
  J1: {delta_q[0]:+.3f} rad
  J2: {delta_q[1]:+.3f} rad
  J3: {delta_q[2]:+.3f} rad
  J4: {delta_q[3]:+.3f} rad
  J5: {delta_q[4]:+.3f} rad
  J6: {delta_q[5]:+.3f} rad
  J7: {delta_q[6]:+.3f} rad
Most-drifted joint: J{most_drift_idx + 1} ({delta_q[most_drift_idx]:+.3f} rad)

[Force / Torque]
force_mag max:   {force_max:.2f} N    @ {force_max_time}
force_mag mean:  {force_mean:.2f} N
torque_mag max:  {torque_max:.2f} N·m

[Events]
MODE precision_mode:   {_event_counts['MODE_precision']}
MODE compliance_mode:  {_event_counts['MODE_compliance']}
RECOVERY (/clearerr):  {_event_counts['RECOVERY']}
JOINT resets:          {_event_counts['JOINTRESET']}
Total /pose commands:  {_event_counts['POSE']}

[Hypothesis verdict]
Verdict: {verdict}

[Files]
Session dir:    {SESSION_DIR}
Time series:    status.csv
Event timeline: events.log
Pose markers:   pose_milestones.log

═══════════════════════════════════════════════════════════
"""
        # Atomic-ish write: write to temp then rename (avoid partial summary on crash)
        tmp_path = SUMMARY_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(summary)
        os.replace(tmp_path, SUMMARY_PATH)

        if verbose:
            print("\n" + summary)
            print(f"📊 Summary saved: {SUMMARY_PATH}\n")

    except Exception as e:
        if verbose:
            print(f"[summary error] {e}")
            import traceback
            traceback.print_exc()


def _shutdown_handler(*_):
    _shutdown_requested.set()
    raise SystemExit(_exit_code)


def _watch_backend(backend):
    global _exit_code
    returncode = backend.wait()
    if _shutdown_requested.is_set():
        return
    _exit_code = 1
    _log_event(
        "BACKEND",
        f"state backend exited with code {returncode}; restarting Gateway",
    )
    os.kill(os.getpid(), signal.SIGTERM)


atexit.register(lambda: _write_summary(verbose=True, final=True))
signal.signal(signal.SIGINT, _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)
# SIGHUP — terminal close (without this, closing terminal would skip summary)
try:
    signal.signal(signal.SIGHUP, _shutdown_handler)
except AttributeError:
    pass  # Windows


def _status_thread_fn(robot_server):
    """Background thread — writes STATUS samples every 5 seconds.

    Also keeps in-memory sample list and periodically updates summary.txt
    so that even SIGKILL (e.g. robot_kill) leaves a near-current summary.
    """
    global START_Q
    while True:
        time.sleep(5)
        try:
            if START_Q is None:
                continue
            t = time.time() - SESSION_START_TIME
            q = np.array(robot_server.q)
            q_drift = float(np.linalg.norm(q - START_Q))
            f_mag = float(np.linalg.norm(robot_server.force))
            t_mag = float(np.linalg.norm(robot_server.torque))
            pose = robot_server.pos[:3]
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 1) Write CSV row (existing behavior)
            _status_writer.writerow([
                ts, f"{t:.0f}",
                f"{pose[0]:.3f}", f"{pose[1]:.3f}", f"{pose[2]:.3f}",
                *[f"{qi:.3f}" for qi in q],
                f"{q_drift:.3f}",
                f"{f_mag:.2f}", f"{t_mag:.2f}",
            ])
            _status_csv_file.flush()

            # 2) Append to in-memory sample list (used by _write_summary)
            with _status_lock:
                _status_samples.append({
                    "timestamp": ts,
                    "t_sec": t,
                    "pose": pose.copy(),
                    "q": q.copy(),
                    "q_drift": q_drift,
                    "force_mag": f_mag,
                    "torque_mag": t_mag,
                })

            # 3) Overwrite summary.txt with current state (survives SIGKILL)
            _write_summary(verbose=False, final=False)

            # 4) Console output
            print(f"[STATUS t={t:.0f}s] "
                  f"pose={np.round(pose, 3).tolist()} "
                  f"q_drift={q_drift:.2f}rad "
                  f"force={f_mag:.1f}N")
        except Exception as e:
            print(f"[STATUS error] {e}")


# ============================================================================
# FrankaServer (동일, moon.py 기반)
# ============================================================================
class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self):
        self.imp = None
        self.backend = None
        self.telemetry_store = TelemetryStateStore()
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(JOINT_RESET_TARGET) - np.array(self.q),
            0, atol=1e-2, rtol=1e-2,
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
            0, atol=1e-2, rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break
        print("RESET DONE")

    def start_impedance(self):
        if self.imp is not None and self.imp.poll() is None:
            return False
        command = (
            ["rosrun", "controller_manager", "spawner", "cartesian_impedance_controller"]
            if FLAGS.safe_start
            else [
                "roslaunch", "serl_franka_controllers", "impedance_gumi.launch",
                "robot_ip:=" + FLAGS.robot_ip, "load_gripper:=true"
            ]
        )
        self.imp = subprocess.Popen(command)
        time.sleep(10)
        return True

    def start_state_backend(self):
        self.backend = subprocess.Popen([
            "roslaunch", "franka_control", "franka_control_gumi.launch",
            "robot_ip:=" + FLAGS.robot_ip, "load_gripper:=true"
        ])

    def stop_impedance(self):
        if self.imp is None or self.imp.poll() is not None:
            return False
        self.imp.terminate()
        time.sleep(1)
        return True

    def clear(self):
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def reset_joint(self):
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(JOINT_RESET_TARGET) - np.array(self.q),
            0, atol=1e-2, rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        print("RESET DONE")
        self.joint_controller.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)

        self.start_impedance()
        print("impedance STARTED")

    def move(self, pose: list):
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "fr3_link0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        safe_telemetry_update(
            self.telemetry_store.update_franka_state,
            msg,
        )
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
        safe_telemetry_update(
            self.telemetry_store.update_jacobian,
            msg,
        )
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian

    def _update_gripper(self, msg):
        self.gripper_pos = np.sum(msg.position)

    def close(self):
        print("close")
        msg = MoveActionGoal()
        msg.goal.width = 0.0
        msg.goal.speed = 0.2
        self.grippermovepub.publish(msg)
        return 'Closed'

    def home_gripper(self):
        print("homing gripper")
        self.gripperhomingpub.publish(HomingActionGoal())
        return 'Homing'

    def open(self):
        print("open")
        msg = MoveActionGoal()
        msg.goal.width = FLAGS.gripper_dist
        msg.goal.speed = 0.2
        self.grippermovepub.publish(msg)
        return 'Opened'

    def control_gripper(self, width):
        print("control gripper")
        msg = MoveActionGoal()
        msg.goal.width = width
        msg.goal.speed = 0.15
        self.grippermovepub.publish(msg)
        return "Control Gripper"


###############################################################################


def _read_server_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(_):
    # Silence Flask access log (the noisy 127.0.0.1 - - POST ... lines)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen("roscore")
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    rospy.init_node("franka_control_api")

    robot_server = FrankaServer()
    telemetry_publisher = FrankaUdpPublisher(
        store=robot_server.telemetry_store,
        host=FLAGS.telemetry_host,
        port=FLAGS.telemetry_port,
        hz=FLAGS.telemetry_hz,
    )
    atexit.register(telemetry_publisher.stop)
    server_commit = _read_server_commit()
    server_started_at = datetime.now(timezone.utc).astimezone().isoformat()
    reconf_client = None
    if FLAGS.safe_start:
        robot_server.start_state_backend()
        threading.Thread(
            target=_watch_backend,
            args=(robot_server.backend,),
            daemon=True,
        ).start()
    else:
        robot_server.start_impedance()
        robot_server.home_gripper()
        time.sleep(5)
        reconf_client = ReconfClient(
            "/cartesian_impedance_controller/dynamic_reconfigure_compliance_param_node"
        )

        # ======================================================================
        # [INIT] Option MAX-A
        # ======================================================================
        _log_event("INIT", "Session dir: " + SESSION_DIR)
        _log_event("INIT", "Applying MAX-A gains (K_t=1500, c_t=0.018, K_r=160, c_r=0.035, nullspace=5.0)")
        reconf_client.update_configuration({"translational_stiffness": 1500})
        reconf_client.update_configuration({"translational_damping": 77})
        reconf_client.update_configuration({"rotational_stiffness": 160})
        reconf_client.update_configuration({"rotational_damping": 8})
        reconf_client.update_configuration({"nullspace_stiffness": 5.0})
        reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.018})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.035})
        robot_server.move(robot_server.pos.tolist())
        time.sleep(0.2)
        reconf_client.update_configuration({"translational_Ki": 0})
        reconf_client.update_configuration({"rotational_Ki": 0})
        _log_event("INIT", "Impedance parameters reset complete (MAX-A)")

        # Capture baseline q for q_drift measurement
        time.sleep(1)
        global START_Q
        START_Q = np.array(robot_server.q).copy()
        _log_event("INIT", f"start_q captured: {START_Q.round(3).tolist()}")
        _log_event("INIT", "q_drift = ‖current q − start_q‖  (hypothesis verification metric)")

    # Start STATUS thread
    threading.Thread(target=_status_thread_fn, args=(robot_server,), daemon=True).start()
    _log_event("INIT", "STATUS thread started (every 5s)")

    # ==========================================================================
    # Routes
    # ==========================================================================

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        nonlocal reconf_client
        global START_Q
        _log_event("CTRL", "/startimp called")
        robot_server.clear()
        started = robot_server.start_impedance()
        if FLAGS.safe_start and started:
            robot_server.home_gripper()
            time.sleep(5)
            reconf_client = ReconfClient(
                "/cartesian_impedance_controller/dynamic_reconfigure_compliance_param_node"
            )
            precision_mode()
            START_Q = np.array(robot_server.q).copy()
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        nonlocal reconf_client
        _log_event("CTRL", "/stopimp called")
        robot_server.stop_impedance()
        reconf_client = None
        return "Stopped impedance"

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

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": robot_server.gripper_pos})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        cur_q = np.array(robot_server.q).round(2).tolist()
        cur_drift = float(np.linalg.norm(robot_server.q - START_Q)) if START_Q is not None else 0.0
        _log_event("RESET", f"/jointreset called, current q={cur_q}, q_drift={cur_drift:.2f}")
        _event_counts["JOINTRESET"] += 1
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        _log_event("GRIPPER", "open")
        robot_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        _log_event("GRIPPER", "close")
        robot_server.close()
        return "Closed"

    @webapp.route("/control_gripper", methods=["POST"])
    def control_gripper():
        data = request.get_json()
        width = data.get('width', 0.04)
        _log_event("GRIPPER", f"control width={width}")
        robot_server.control_gripper(width)
        return "Control Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        cur_drift = float(np.linalg.norm(robot_server.q - START_Q)) if START_Q is not None else 0.0
        _log_event("RECOVERY", f"/clearerr called, q_drift={cur_drift:.2f}")
        _event_counts["RECOVERY"] += 1
        robot_server.clear()
        return "Clear"

    # /pose throttled to log every 100th call
    _pose_counter = [0]

    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        _pose_counter[0] += 1
        _event_counts["POSE"] = _pose_counter[0]
        if _pose_counter[0] % 100 == 0:
            ts = datetime.now().strftime('%H:%M:%S')
            line = f"[{ts}] [POSE #{_pose_counter[0]}] {pos[:3].round(3).tolist()}"
            print(line)
            try:
                _pose_milestones_file.write(line + "\n")
                _pose_milestones_file.flush()
            except Exception:
                pass
        robot_server.move(pos)
        return "Moved"

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

    @webapp.route("/telemetry/status", methods=["GET"])
    def telemetry_status():
        return jsonify(
            build_status_payload(
                store=robot_server.telemetry_store,
                publisher_stats=telemetry_publisher.stats_snapshot(),
                server_commit=server_commit,
                server_started_at=server_started_at,
                entrypoint="robot_infra/custom_server_moon_adv.py",
                now_monotonic_ns=time.monotonic_ns(),
            )
        )

    @webapp.route("/health", methods=["POST"])
    def health():
        impedance = getattr(robot_server, "imp", None)
        return jsonify(
            build_health_payload(
                store=robot_server.telemetry_store,
                controller_active=(
                    impedance is not None and impedance.poll() is None
                ),
                observed_at=datetime.now(timezone.utc).isoformat(),
                now_monotonic_ns=time.monotonic_ns(),
            )
        )

    @webapp.route("/start_joint_controller", methods=["POST"])
    def change_controller():
        _log_event("CTRL", "/start_joint_controller called")
        robot_server.start_joint_controller()
        return "Controller Changed"

    # ==========================================================================
    # Mode endpoints (MAX-A gains, with event logging)
    # ==========================================================================

    @webapp.route("/precision_mode", methods=["POST"])
    def precision_mode():
        cur_drift = float(np.linalg.norm(robot_server.q - START_Q)) if START_Q is not None else 0.0
        _log_event("MODE", f"precision_mode applied, q_drift={cur_drift:.2f}")
        _event_counts["MODE_precision"] += 1
        reconf_client.update_configuration({"translational_stiffness": 1500})
        reconf_client.update_configuration({"translational_damping": 77})
        reconf_client.update_configuration({"rotational_stiffness": 160})
        reconf_client.update_configuration({"rotational_damping": 8})
        reconf_client.update_configuration({"nullspace_stiffness": 5.0})
        reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.018})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.035})
        robot_server.move(robot_server.pos.tolist())
        time.sleep(0.2)
        reconf_client.update_configuration({"translational_Ki": 0})
        reconf_client.update_configuration({"rotational_Ki": 0})
        return 'Precision'

    @webapp.route("/compliance_mode", methods=["POST"])
    def compliance_mode():
        cur_drift = float(np.linalg.norm(robot_server.q - START_Q)) if START_Q is not None else 0.0
        _log_event("MODE", f"compliance_mode applied, q_drift={cur_drift:.2f}")
        _event_counts["MODE_compliance"] += 1
        reconf_client.update_configuration({"translational_stiffness": 800})
        reconf_client.update_configuration({"translational_damping": 80})
        reconf_client.update_configuration({"rotational_stiffness": 150})
        reconf_client.update_configuration({"rotational_damping": 7})
        reconf_client.update_configuration({"nullspace_stiffness": 5.0})
        reconf_client.update_configuration({"joint1_nullspace_stiffness": 10.0})
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.05})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.07})
        robot_server.move(robot_server.pos.tolist())
        time.sleep(0.2)
        reconf_client.update_configuration({"translational_Ki": 0})
        reconf_client.update_configuration({"rotational_Ki": 0})
        return 'Compliance'

    # ==========================================================================
    # Diagnostic endpoint — for hypothesis verification at static state
    # ==========================================================================
    @webapp.route("/diag_static", methods=["POST"])
    def diag_static():
        """Call when robot is held still (policy off) to measure nullspace mismatch."""
        q = np.array(robot_server.q)
        q_drift = float(np.linalg.norm(q - START_Q)) if START_Q is not None else 0.0
        force_mag = float(np.linalg.norm(robot_server.force))
        torque_mag = float(np.linalg.norm(robot_server.torque))
        _log_event("DIAG", f"static measurement: q_drift={q_drift:.3f}rad "
                            f"force={force_mag:.2f}N torque={torque_mag:.2f}Nm")
        return jsonify({
            "q_drift": q_drift,
            "current_q": q.tolist(),
            "start_q": START_Q.tolist() if START_Q is not None else None,
            "force_mag": force_mag,
            "torque_mag": torque_mag,
        })

    try:
        telemetry_publisher.start()
        webapp.run(host="0.0.0.0")
    finally:
        telemetry_publisher.stop()


if __name__ == "__main__":
    app.run(main)
