import numpy as np
import time
import requests
from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC
from robot_infra.state_machine import RobotStateMachine
import argparse
from modules.config import Config

def get_argparser():
    parser = argparse.ArgumentParser(description="Control parameters for robot simulation.")
    parser.add_argument('--config_robot', type=str,
                        default='robot_infra/configs/robot_params.yaml',
                        help='PATH/TO/CONFIG/FILE/.yaml')
    parser.add_argument('--reset', type=int, default=0, help="0:False, 1:True")
    parser.add_argument('--gripper', type=str, default="open", choices=['open', 'close'],
                        help="Control the state of the gripper: open or close")
    return parser

def lerp(start, end, t):
    return start + t * (end - start)

def calculate_distance(start_pose, target_pose):
    return np.linalg.norm(np.array(target_pose) - np.array(start_pose))

def cal_waypoints(start_pose, target_pose, steps, distance_threshold=0.1):
    waypoints = []
    distance = calculate_distance(start_pose, target_pose)

    # 거리에 비례하여 t를 조절
    for i in range(steps):
        if distance > distance_threshold:
            t = i / (steps - 1)
        else:
            t = (i / (steps - 1)) * (distance / distance_threshold)  # 가까울수록 t를 줄임

        waypoint = lerp(start_pose, target_pose, t)
        waypoints.append(waypoint)

    return waypoints

def move_to_waypoint(control, waypoints, gripper_state, current_index):
    if current_index < len(waypoints):
        control.move_to_pos(waypoints[current_index])
        current_index += 1
    
    return current_index

def main():
    last_state = None
    waypoints = []
    start_pos = None
    current_index = 0  # 웨이포인트의 현재 인덱스
    last_target_pose = None
    while True:
        # Get current end-effector pose
        curr_ee_pose = control.get_ee_pose()
        curr_ee_trans = curr_ee_pose['pose'][:3]
        curr_ee_quat = curr_ee_pose['pose'][3:]

        curr_force = control.get_ee_ft()
        target_pose, current_state, gripper_state = state_machine.update_state(curr_ee_trans, curr_ee_quat, curr_force)
        
        if last_target_pose is None or not np.array_equal(target_pose, last_target_pose):
            start_pos = curr_ee_pose['pose'][:]
            waypoints = cal_waypoints(start_pos, target_pose, 20)
            current_index = 0
            last_target_pose = target_pose
            
        current_index = move_to_waypoint(control, waypoints, gripper_state, current_index)
                

if __name__ == "__main__":
    
    parser = get_argparser()
    args = parser.parse_args()
    config_robot = Config(args.config_robot).get_config()
    
    # Define the Franka controller
    control = FrankaVC(config_robot=config_robot, hz=50, start_gripper=1) # if 1, keep close
    state_machine = RobotStateMachine(device='cpu', config_robot=config_robot)
    
    if not args.reset:
        main()
    elif args.reset:
        state_machine.reset()
            
    if control.currgrip == 0 and state_machine.gripper_state == 'close':
        control.close_gripper()
    elif control.currgrip == 1 and state_machine.gripper_state == 'open':
        control.open_gripper()        