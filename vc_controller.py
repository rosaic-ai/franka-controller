import numpy as np
import time
import requests
from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC

control = FrankaVC()

# limit range
# z_min = 0.015 / z_max = 0.4
# x_min = 0.30  / x_max = 0.6
# y_min = -0.30 / y_max = 0.30


def setup_initial_pose(control):
    state = control.get_ee_pose()
    initial_pose = state['pose']
    print("Initial robot state:", initial_pose)
    return initial_pose


def move_through_waypoints(control, curr_pose, target_pose, steps):
    curr_pose = np.array(curr_pose)
    target_pose = np.array(target_pose)
    waypoints = []

    for step in range(1, steps + 1):
        ratio = step / steps
        waypoint = curr_pose + ratio * (target_pose - curr_pose)
        waypoints.append(waypoint.tolist())
    
    for waypoint in waypoints:
        control.move_to_pos(waypoint)
        time.sleep(0.01)

    # 최종 목표 위치에 도달
    control.move_to_pos(target_pose)


# control.compliance_mode()
control.precision_mode()

# state = control.get_ee_pose()
# quat = state['pose'][3:]
# euler = control.quat_2_euler(quat)
# print(euler)

initial_pose = setup_initial_pose(control)

target_trans = [0.5, 0, 0.35]
target_euler = [0,0,0] #roll, pitch, yaw

reset = False
if reset:
    target_quat = [0.9957491616373879, -0.0311666836346376, 0.04373945910063208, 0.07482716516908729]
else:
    target_quat = control.euler_2_quat(target_euler[0], target_euler[1], target_euler[2])
    
target_pose = np.concatenate([target_trans, target_quat])
# control.move_to_pos(target_pose)

# 범위 내로 조정
move_through_waypoints(control, initial_pose, target_pose, 5)
