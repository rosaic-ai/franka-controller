'''
This code is converting 3d mouse value([x,y,z,a,b,c],[button0(-y), button1(+y)])
to panda robot action()


1. 3d mouse value:
    Spacemouse action: [x,y,z,a,b,c],
    buttons: [button0(-y), button1(+y)]
2. panda robot action:

'''

from robot_infra.spacemouse.spacemouse_custom import SpaceMouseExpert
from robot_infra.envs.franka_vc_env import FrankaVC

from robot_infra.utils.rotations import euler_2_quat

from modules.config import Config
import argparse
import time
import numpy as np
    
def get_argparser():
    parser = argparse.ArgumentParser(description="Control parameters for robot simulation.")
    parser.add_argument('--config_robot', type=str,
                        default='robot_infra/configs/robot_params.yaml',
                        help='PATH/TO/CONFIG/FILE/.yaml')
    parser.add_argument('--reset', type=int, default=0, help="0:False, 1:True")
    parser.add_argument('--gripper', type=str, default="open", choices=['open', 'close'],
                        help="Control the state of the gripper: open or close")
    return parser

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([x, y, z, w])

if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    config_robot = Config(args.config_robot).get_config()
    if args.gripper == 'close':
        start_gripper = 0
    else:
        start_gripper = 1
        
    # Define the Franka controller
    control = FrankaVC(config_robot=config_robot, hz=50, start_gripper=start_gripper) # if 1, keep close    
    spacemouse = SpaceMouseExpert()

    translation_speed = 0.02    # cm
    rotation_speed = 0.07     # unit
    scaled_delta_action = np.zeros(7)
    robot_action = np.zeros(7)

    initial_state = np.array(control.get_ee_pose()['pose'])

    while True:
        # 3d mouse
        mouse_action, buttons = spacemouse.get_action()
        # print(f"Spacemouse action: {mouse_action}, buttons: {buttons}")

        # robot_action[:3] += mouse_action[:3] * translation_speed
        
        robot_action = initial_state.copy()
        
        robot_action[:3] += mouse_action[:3] * translation_speed
        
        mouse_action[3] *= rotation_speed
        mouse_action[4] *= rotation_speed
        mouse_action[5] *= -rotation_speed
        scaled_delta_quat = euler_2_quat(mouse_action[3:6])
        
        robot_action[3:7] = quaternion_multiply(robot_action[3:7], scaled_delta_quat)

        # 로봇의 새로운 위치로 이동합니다.
        control.move_to_pos(robot_action)
        
        # 현재 상태를 업데이트 합니다.
        initial_state = robot_action.copy()
        
        if buttons[0]:
            control.close_gripper()
        elif buttons[1]:
            control.open_gripper()

        # time.sleep(0.1)
