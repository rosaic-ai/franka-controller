import numpy as np
import time
import requests
import cv2
import argparse
from modules.config import Config

from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC, GetImageThread, ImageDisplayer
from robot_infra.state_machine import RobotStateMachine, State
from robot_infra.camera.rs_capture import RSCapture
from robot_infra.camera.video_capture import VideoCapture

import logging
import collections

last_camera_fetch_time = 0
last_pose_fetch_time = 0
last_force_fetch_time = 0

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

    for i in range(steps):
        if distance > distance_threshold:
            t = i / (steps - 1)
        else:
            t = (i / (steps - 1)) * (distance / distance_threshold)  # 가까울수록 t를 줄임

        waypoint = lerp(start_pose, target_pose, t)
        waypoints.append(waypoint)

    return waypoints

def spiral_search(center, radius_increment, angle_increment, spiral_step):
    waypoints = []
    radius = spiral_step * radius_increment  # 스파이럴 스텝에 따라 반경을 계산합니다
    angle = spiral_step * angle_increment  # 스파이럴 스텝에 따라 각도를 계산합니다
    
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    waypoints.append([x, y])
    
    return waypoints

def move_to_waypoint(control, waypoints, gripper_state, current_index):
    if current_index < len(waypoints):
        control.move_to_pos(waypoints[current_index])
        current_index += 1
    
    return current_index

def _get_observation():
    # Fetch the latest image
    get_img = camera_thread.fetch_images()

    # Get current end-effector pose
    curr_ee_pose = control.get_ee_pose()
    curr_ee_trans = curr_ee_pose['pose'][:3]
    curr_ee_quat = curr_ee_pose['pose'][3:]

    # Get current end-effector force
    curr_force = control.get_ee_force()['force']  
    curr_torque = control.get_ee_torque()['torque']  
    force_array = np.array(curr_force)
    torque_array = np.array(curr_torque)
    curr_ee_ft = np.concatenate((force_array, torque_array))    
    
    curr_joint_pos = control.get_joint_pos()

    return get_img, curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_ee_ft, curr_joint_pos

import time

def _get_observation_for_debugging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    global last_camera_fetch_time, last_pose_fetch_time, last_force_fetch_time

    # Fetch the latest image
    start_time = time.time()
    get_img = camera_thread.fetch_images()
    if get_img.any():
        interval = start_time - last_camera_fetch_time
        camera_freq = 1 / interval if interval else 0
        last_camera_fetch_time = start_time
        logging.debug(f"Camera sampling frequency: {camera_freq} Hz")

    # Get current end-effector pose
    start_time = time.time()
    curr_ee_pose = control.get_ee_pose()
    if curr_ee_pose:
        interval = start_time - last_pose_fetch_time
        pose_freq = 1 / interval if interval else 0
        last_pose_fetch_time = start_time
        logging.debug(f"Pose sampling frequency: {pose_freq} Hz")

    # Get current end-effector force
    start_time = time.time()
    curr_force = control.get_ee_force()['force']  
    curr_torque = control.get_ee_torque()['torque']  
    force_array = np.array(curr_force)
    torque_array = np.array(curr_torque)
    
    curr_ee_ft = np.concatenate((force_array, torque_array))
    
    if curr_ee_ft.any():
        interval = start_time - last_force_fetch_time
        force_freq = 1 / interval if interval else 0
        last_force_fetch_time = start_time
        logging.debug(f"Force sampling frequency: {force_freq} Hz")
    
    # Get current joint angles
    start_time = time.time()
    curr_joint_pos = control.get_joint_pos()
    if curr_joint_pos:
        interval = start_time - last_pose_fetch_time
        pose_freq = 1 / interval if interval else 0
        last_pose_fetch_time = start_time
        logging.debug(f"Joint sampling frequency: {pose_freq} Hz")
    
    return get_img, curr_ee_pose, curr_ee_pose['pose'][:3], curr_ee_pose['pose'][3:], curr_ee_ft, curr_joint_pos


def fetch_all_observation(get_img, curr_joint_pos, curr_contact, curr_pose):
    timestamp = time.time()  # Get current time for all observations to maintain uniformity
    obs_with_timestamp = {
        'camera_data':  {'image': get_img, 'timestamp': timestamp},
        'joint_data':   {'joint': curr_joint_pos, 'timestamp': timestamp},
        'contact_data': {'contact': curr_contact, 'timestamp': timestamp},
        'pose_data': {'pose': curr_pose, 'timestamp': timestamp},
    }
    return obs_with_timestamp

def synchronize_data(observation_data):
    camera_data = observation_data['camera_data']    
    joint_data = observation_data['joint_data']
    contact_data = observation_data['contact_data']
    pose_data = observation_data['pose_data']

    # Extract timestamps for each data type
    camera_timestamp = camera_data['timestamp']
    joint_timestamp = joint_data['timestamp']
    force_timestamp = contact_data['timestamp']
    pose_timestamp = pose_data['timestamp']

    # Find the minimum time difference to synchronize data
    timestamps = [camera_timestamp, joint_timestamp, force_timestamp, pose_timestamp]
    avg_timestamp = sum(timestamps) / len(timestamps)  # Calculate average timestamp

    # Find data entries closest to the average timestamp
    closest_to_avg = min(timestamps, key=lambda x: abs(x - avg_timestamp))

    # Prepare the synchronized dataset
    synced_data = {}
    if abs(camera_timestamp - closest_to_avg) < 0.05:
        synced_data['camera_data'] = camera_data['image']
    if abs(joint_timestamp - closest_to_avg) < 0.05:
        synced_data['joint_data'] = joint_data['joint']['q']
    if abs(force_timestamp - closest_to_avg) < 0.05:
        synced_data['contact_data'] = contact_data['contact']
    if abs(pose_timestamp - closest_to_avg) < 0.05:
        synced_data['pose_data'] = pose_data['pose']['pose']

    return synced_data

def _model_inference(image_buffers, robot_data_buffers):
    return None
    
def main():
    last_state = None
    waypoints = []
    start_pos = None
    current_index = 0
    last_target_pose = None
    spiral_step = 0
    loop_counter = 0 
    sequence_length = 5
    step_num_image = 4
    
    image_buffers = collections.deque(maxlen=sequence_length)
    robot_data_buffers = collections.deque(maxlen=sequence_length * step_num_image)
    
    last_time = time.time()
    loop_freq = 0
    manage_state_internally = False

    while True:
        # 루프 시작 시간 측정
        current_time = time.time()
        if loop_counter > 0:
            loop_interval = current_time - last_time
            loop_freq = 1 / loop_interval if loop_interval else 0
            print(f"Loop Frequency: {loop_freq:.2f} Hz")
        last_time = current_time
                
        get_img, curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_contact, curr_joint = _get_observation() #_get_observation(), _get_observation_for_debugging()
        obs_with_timestamp = fetch_all_observation(get_img, curr_joint, curr_contact, curr_ee_pose)
        synced_obs = synchronize_data(obs_with_timestamp)
        
        synced_img = synced_obs['camera_data']
        synced_contact = np.array(synced_obs['contact_data'])
        synced_joint = np.array(synced_obs['joint_data'])
        synced_pose = np.array(synced_obs['pose_data'])        
    
        if not manage_state_internally:
            target_pose, current_state, gripper_state = state_machine.update_state(synced_pose[:3], synced_pose[3:], synced_contact, spiral_step)
            print(f"Current State: {State(current_state).name}")

            if current_state == State.CONTACT_AND_MOVE.value:
                manage_state_internally = True
        else:
            print("Managing state internally")
            if current_state == State.CONTACT_AND_MOVE.value: #MOVE_TO_PREDICTED_POSE, CONTACT_AND_MOVE
                if loop_counter % 1 == 0:
                    image_buffers.append(synced_img)
                    robot_data_buffers.append((synced_contact, synced_joint))
                    
                if len(image_buffers) == sequence_length and len(robot_data_buffers) == step_num_image * sequence_length:
                    print("Model input ready")
                    pred_action = _model_inference(image_buffers, robot_data_buffers)
                    ########################################################
                    ############### ADD MODEL INFERENCE HERE ###############
                    ########################################################
        
        if current_state == State.CONTACT_AND_MOVE.value:
            control.move_to_pos(target_pose)
            if loop_counter % 4 == 0:
                spiral_step += 1
        else:
            if last_target_pose is None or not np.array_equal(target_pose, last_target_pose):
                start_pos = curr_ee_pose['pose'][:]
                waypoints = cal_waypoints(start_pos, target_pose, 15)
                current_index = 0
                last_target_pose = target_pose
                
            current_index = move_to_waypoint(control, waypoints, gripper_state, current_index)
        
        loop_counter += 1

if __name__ == "__main__":
    
    parser = get_argparser()
    args = parser.parse_args()
    config_robot = Config(args.config_robot).get_config()
    if args.gripper == 'close':
        start_gripper = 0
    else:
        start_gripper = 1
    
    state_target_trans = config_robot.get('state_target_trans', {})
    state_target_ori   = config_robot.get('state_target_ori', {})
    state_pose_thres   = config_robot.get('state_pose_thres', {})
    
    # Define the Franka controller
    control = FrankaVC(config_robot=config_robot, hz=30, start_gripper=start_gripper) # if 1, keep close
    state_machine = RobotStateMachine(device='cpu', config_robot=config_robot)
    
    camera_thread = GetImageThread(serial_number='130322270132', dim=(848, 480), fps=30, depth=False)
    displayer_thread = ImageDisplayer(camera_thread.img_queue)
    displayer_thread.start()
    
    if not args.reset:
        main()
    elif args.reset:
        # state_machine.reset()
        state_machine.reset_tmp()


# 추가해야할거
# camera view 수정

# Model inference
# Go to model output