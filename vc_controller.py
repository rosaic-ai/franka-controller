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
    return parser

def setup_initial_pose(control):
    state = control.get_ee_pose()
    initial_pose = state['pose']
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
    control.move_to_pos(target_pose)
    

def main():
    # Get current end-effector pose
    curr_ee_pose = control.get_ee_pose()
    curr_ee_trans = curr_ee_pose['pose'][:3]
    curr_ee_quat = curr_ee_pose['pose'][3:]

    curr_force = control.get_ee_ft()
    
    initial_pose = setup_initial_pose(control)

    # Update state and get next target pose and gripper state
    target_pose, gripper_values, current_state = state_machine.update_state(curr_ee_trans, curr_ee_quat, curr_force)
    
    # Move to the next waypoint calculated by the state machine
    move_through_waypoints(control, initial_pose, target_pose, 20)
    control.set_gripper(gripper_values)
    
    time.sleep(0.1)

if __name__ == "__main__":
    
    parser = get_argparser()
    args = parser.parse_args()
    config_robot = Config(args.config_robot).get_config()
    
    # Define the Franka controller
    control = FrankaVC(config_robot=config_robot)
    state_machine = RobotStateMachine(device='cpu', config_robot=config_robot)
    
    state_machine.reset()
    
    if not args.reset:
        while True:
            main()