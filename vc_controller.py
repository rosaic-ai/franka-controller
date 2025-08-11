import numpy as np
import time
import argparse
from modules.config import Config
import signal
import sys

from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC, GetImageThread, ImageDisplayer, USBImageCapture, USBDisplayer, RecorderThread
from robot_infra.state_machine import RobotStateMachine, State
from robot_infra.camera.rs_capture import RSCapture
from robot_infra.camera.video_capture import VideoCapture

import socket
import struct
import pickle
import time
import os
import cv2
from datetime import datetime

import logging
import collections
import torchvision.transforms as transforms
from PIL import Image as Im

import rospy
from geometry_msgs.msg import WrenchStamped, Wrench

rospy.init_node('robot_force_torque_publisher')
force_torque_pub = rospy.Publisher('ee_force_torque', WrenchStamped, queue_size=5)

last_camera_fetch_time = 0
last_pose_fetch_time = 0
last_force_fetch_time = 0

def signal_handler(sig, frame):
    print('\n프로그램을 안전하게 종료합니다...')
    
    # Clean up camera resources
    realsense_thread.stop()
    realsense_displayer.stop()
    
    if record:
        usb_camera.stop()
        # usb_displayer.stop()
        realsense_recorder.stop()
        usb_recorder.stop()
    
    # Clean up robot control
    control.stop()
    
    # Clean up ROS node
    rospy.signal_shutdown('User interrupted')
    
    sys.exit(0)

def get_argparser():
    parser = argparse.ArgumentParser(description="Control parameters for robot simulation.")
    parser.add_argument('--config_robot', type=str,
                        default='robot_infra/configs/robot_params.yaml',
                        help='PATH/TO/CONFIG/FILE/.yaml')
    parser.add_argument('--reset', type=int, default=0, help="0:False, 1:True")
    parser.add_argument('--gripper', type=str, default="open", choices=['open', 'close'],
                        help="Control the state of the gripper: open or close")
    parser.add_argument('--port', type=int, default=83, help="Port number for communication")
    parser.add_argument('--record', type=int, default=0, choices=[0, 1],
                        help="Enable image recording: 0 for False, 1 for True")
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
            t = (i / (steps - 1)) * (distance / distance_threshold)

        waypoint = lerp(start_pose, target_pose, t)
        waypoints.append(waypoint)

    return waypoints

def spiral_search(center, radius_increment, angle_increment, spiral_step):
    waypoints = []
    radius = spiral_step * radius_increment 
    angle = spiral_step * angle_increment
    
    x = center[0] + radius * np.cos(angle)
    y = center[1] + radius * np.sin(angle)
    waypoints.append([x, y])
    
    return waypoints

def move_to_waypoint(control, waypoints, gripper_state, current_index):
    if current_index < len(waypoints):
        control.move_to_pos(waypoints[current_index])
        current_index += 1
    
    return current_index

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),            
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Im.fromarray(np.uint8(image)).convert('RGB')
    processed_image = transform(image)
    return processed_image

def initialize_force_torque_sensors():
    initial_force = control.get_ee_force()['force']
    initial_torque = control.get_ee_torque()['torque']
    initial_force_torque = np.concatenate((initial_force, initial_torque))
    return initial_force_torque

def _publish_ft_topic(initial_force_torque):
    curr_force = control.get_ee_force()['force']
    curr_torque = control.get_ee_torque()['torque']

    force_torque_msg = WrenchStamped()
    force_torque_msg.header.stamp = rospy.Time.now()
    force_torque_msg.header.frame_id = 'ee_frame'
    
    force_torque_msg.wrench.force.x  = (curr_force[0] - initial_force_torque[0])
    force_torque_msg.wrench.force.y  = (curr_force[1] - initial_force_torque[1])
    force_torque_msg.wrench.force.z  = (curr_force[2] - initial_force_torque[2])
    force_torque_msg.wrench.torque.x = (curr_torque[0] - initial_force_torque[3])
    force_torque_msg.wrench.torque.y = (curr_torque[1] - initial_force_torque[4])
    force_torque_msg.wrench.torque.z = (curr_torque[2] - initial_force_torque[5])
        
    force_torque_pub.publish(force_torque_msg)

def _get_observation(initial_force_torque, time_step, black_pixel_prob=0.6, add_noise=True):
    # Fetch the latest image
    get_img = realsense_thread.fetch_images()

    if add_noise:
        if get_img is not None:
            img_height, img_width, _ = get_img.shape

            # 특정 확률로 이미지에 노이즈 추가
            if np.random.rand() < black_pixel_prob:
                print(f"[INFO] Black pixels applied at time step {time_step}")
                get_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)  # 검은 이미지

    # Get current end-effector pose
    curr_ee_pose = control.get_ee_pose()
    curr_ee_trans = curr_ee_pose['pose'][:3]
    curr_ee_quat = curr_ee_pose['pose'][3:]

    # Get current end-effector force
    curr_force = control.get_ee_force()['force']  
    curr_torque = control.get_ee_torque()['torque']

    force_array = -(np.array(curr_force) - initial_force_torque[:3])
    torque_array = -(np.array(curr_torque) - initial_force_torque[3:])
    curr_ee_ft = np.concatenate((force_array, torque_array))
    
    curr_joint_pos = control.get_joint_pos()

    return get_img, curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_ee_ft, curr_joint_pos

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
    processed_img = process_image(get_img)

    timestamp = time.time()  # Get current time for all observations to maintain uniformity
    obs_with_timestamp = {
        'camera_data':  {'image': processed_img, 'timestamp': timestamp},
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

def send_data(sock, data):
    """데이터를 전송하는 함수"""
    serialized_data = pickle.dumps(data)
    sock.sendall(struct.pack('>I', len(serialized_data)))
    sock.sendall(serialized_data)

def receive_result(sock):
    """서버로부터 결과를 받는 함수"""
    result_length = struct.unpack('>I', sock.recv(4))[0]
    result = sock.recv(result_length)
    return pickle.loads(result)

def main():
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    last_state = None
    waypoints = []
    start_pos = None
    current_index = 0
    last_target_pose = None
    spiral_step = 0
    loop_counter = 0 
    sequence_length = 1
    initial_force_torque = np.zeros(6)
    current_state = None
    previous_pos = None
    previous_target_pose = None
    
    host = '172.27.190.155'
    port = args.port
    
    image_buffers = collections.deque(maxlen=sequence_length)
    robot_data_buffers = collections.deque(maxlen=sequence_length)
    
    last_time = time.time()
    loop_freq = 0
    manage_state_internally = False

    # 삽입 시간 측정을 위한 변수
    start_insertion_timer = None
    end_insertion_timer = None

    # 설정값
    force_threshold = 5  # 힘의 임계값
    stuck_duration_threshold = 5  # 걸린 상태 감지 시간 (초)
    last_force_check_time = time.time()  # 힘 검사를 시작한 마지막 시간

    state_machine.reset_tmp()
    control.precision_mode()

    collect_ft = []
    while True:
        _publish_ft_topic(initial_force_torque)

        # 루프 시작 시간 측정
        current_time = time.time()
        if loop_counter > 0:
            loop_interval = current_time - last_time
            loop_freq = 1 / loop_interval if loop_interval else 0
        last_time = current_time

        # 타임스텝을 _get_observation에 전달
        get_img, curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_contact, curr_joint = _get_observation(
            initial_force_torque, time_step=loop_counter, black_pixel_prob=0.6, add_noise=False)
        obs_with_timestamp = fetch_all_observation(get_img, curr_joint, curr_contact, curr_ee_pose)
        synced_obs = synchronize_data(obs_with_timestamp)
        
        synced_img = synced_obs['camera_data']
        synced_contact = np.array(synced_obs['contact_data'])
        # (Pdb) synced_contact
        #array([ 0.17053409, -0.35093738,  3.66809306, -0.45032463,  0.23181329, 0.10818459])

        # save synced_contact as csv
        
        synced_joint = np.array(synced_obs['joint_data'])
        synced_pose = np.array(synced_obs['pose_data'])       

        # OpenCV로 이미지 시각화
        cv2.imshow("Image with Noise", get_img)
        cv2.waitKey(1)
        
        # Reset sensors
        if current_state == State.ZERO_FORCE.value: 
            initial_force_torque = initialize_force_torque_sensors()
    
        if not manage_state_internally:
            target_pose, current_state, gripper_state = state_machine.update_state(synced_pose[:3], synced_pose[3:], synced_contact, spiral_step)
            print(f"Current State: {State(current_state).name}")

            if current_state == State.MOVE_TO_PREDICTED_POSE.value: 
                start_insertion_timer = time.time()

                manage_state_internally = True
                start_curr_pos = synced_pose[:3]
                start_curr_quat = synced_pose[3:]
                start_curr_euler = R.from_quat(start_curr_quat).as_euler('xyz', degrees=True)
                
                curr_pos = start_curr_pos
                curr_euler = start_curr_euler
                
        else:
            if current_state == State.MOVE_TO_PREDICTED_POSE.value:
                if loop_counter % 1 == 0:               
                    
                    collect_ft.append(synced_contact)
                    
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((host, port))

                        data = {
                            "images": synced_img,
                            "ft_list": synced_contact,
                            "proprio_list": synced_joint
                        }

                        send_data(s, data)
                        result = receive_result(s)
                        pred_action = result['action']
                    
                    position_scale = 1.5 # 2 arrow, 1.5 , 2pin 1.2
                    orientation_scale = 1 # 1.2 arrow, 2pin 1
                    
                    # Get predicted action
                    pred_delta_pos = [x * position_scale for x in pred_action[:2]]
                    perturbation = np.random.normal(0, 0.0004, size=2)
                    pred_delta_pos = [x + p for x, p in zip(pred_delta_pos, perturbation)]

                    force_threshold = 5
                    is_inserting = False

                    # Insert
                    print(curr_ee_trans[2])
                    if curr_ee_trans[2] < state_target_trans['insertion_done'][2]:
                        target_pos[2] -= 0.003
                        print("Insertion")
                        is_inserting = True
                        # save ft as csv
                        import csv
                        with open('ft_data.csv', mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"])
                            writer.writerows(collect_ft)
                                
                    elif abs(synced_contact[0]) > force_threshold or abs(synced_contact[1]) > force_threshold:
                            print("High force detected. Adjusting target position upwards.")
                            target_pos[2] += 0.0007
                    else:
                        if pred_delta_pos[0] < 0:
                            pred_delta_pos[0] -= 0.001
                        target_pos = curr_pos[:2] + pred_delta_pos
                        target_pos = np.array([*target_pos, curr_pos[2]])
                        target_pos[2] = state_target_trans['contact_and_move'][2]

                    if not is_inserting:
                        pred_delta_magnitued = np.linalg.norm(pred_delta_pos)
                        if pred_delta_magnitued > 0.0013:
                            scale = 0.1
                        else:
                            scale = 0.02 #0.02
                        angle_magnitude = abs(curr_euler[2])

                        if curr_euler[2] < 0:
                            scaling_factor = 1 + angle_magnitude / scale
                        else:
                            scaling_factor = 1 + angle_magnitude / scale

                        pred_delta_euler_z = pred_action[3] * orientation_scale * scaling_factor
                        perturbation = np.random.normal(-2, 2, size=1)
                        pred_delta_euler_z += perturbation[0]
                        
                        target_euler_z = curr_euler[2] + pred_delta_euler_z
                        target_euler_new = np.array([*curr_euler[:2], target_euler_z])                    
                        target_quat_new = R.from_euler('xyz', target_euler_new, degrees=True).as_quat()

                    target_pose = np.concatenate((target_pos, target_quat_new))
                    curr_pos = target_pos

                    if curr_ee_trans[2] < state_target_trans['done'][2]:
                        end_insertion_timer = time.time()
                        insertion_duration = end_insertion_timer - start_insertion_timer
                        print(f"Insertion completed in {insertion_duration:.2f} seconds.")
                        break

        if current_state == State.CONTACT_AND_MOVE.value \
            or current_state == State.MOVE_TO_PREDICTED_POSE.value \
            or current_state == State.LOWER_TO_HOLE.value \
            or current_state == State.MOVE_TO_CONTACT.value:
            control.move_to_pos(target_pose)
            if loop_counter % 1 == 0:
                spiral_step += 1
        else:
            if last_target_pose is None or not np.array_equal(target_pose, last_target_pose):
                start_pos = curr_ee_pose['pose'][:]
                waypoints = cal_waypoints(start_pos, target_pose, 10)
                current_index = 0
                last_target_pose = target_pose
                
            current_index = move_to_waypoint(control, waypoints, gripper_state, current_index)
        
        loop_counter += 1

if __name__ == "__main__":
    try:
        parser = get_argparser()
        args = parser.parse_args()
        config_robot = Config(args.config_robot).get_config()
        record = args.record

        if args.gripper == 'close':
            start_gripper = 0
        else:
            start_gripper = 1
        
        state_target_trans = config_robot.get('state_target_trans', {})
        state_target_ori   = config_robot.get('state_target_ori', {})
        state_pose_thres   = config_robot.get('state_pose_thres', {})
        
        # Define the Franka controller
        control = FrankaVC(config_robot=config_robot, hz=30, start_gripper=start_gripper)
        state_machine = RobotStateMachine(device='cpu', config_robot=config_robot)
        
        # Get Images
        realsense_thread = GetImageThread(serial_number='427622270633', dim=(848, 480), fps=30, depth=False)
        realsense_displayer = ImageDisplayer(realsense_thread.img_queue)
        realsense_displayer.start()
        
        if record:
            usb_camera = USBImageCapture(device_index=6, dim=(1280, 720))
            usb_camera.start()
            usb_displayer = USBDisplayer(usb_camera.img_queue)
            
            BASE_SAVE_PATH = "./record_save_repo"
            os.makedirs(BASE_SAVE_PATH, exist_ok=True)

            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_path = os.path.join(BASE_SAVE_PATH, start_time)
            os.makedirs(session_path, exist_ok=True)

            realsense_path = os.path.join(session_path, "realsense")
            usb_cam_path = os.path.join(session_path, "usb_cam")
            os.makedirs(realsense_path, exist_ok=True)
            os.makedirs(usb_cam_path, exist_ok=True)

            realsense_timestamps = os.path.join(realsense_path, "timestamps.txt")
            usb_cam_timestamps = os.path.join(usb_cam_path, "timestamps.txt")
            
            realsense_recorder = RecorderThread(realsense_thread.img_queue, realsense_path, realsense_timestamps, fps=30)
            usb_recorder = RecorderThread(usb_camera.img_queue, usb_cam_path, usb_cam_timestamps, fps=30)
            
            realsense_recorder.start()
            usb_recorder.start()

        if not args.reset:
            main()
        elif args.reset:
            state_machine.reset_tmp()
            
    except KeyboardInterrupt:
        print('\n프로그램이 사용자에 의해 중단되었습니다.')
    finally:
        # Cleanup code in case of any exception
        if 'realsense_thread' in locals():
            realsense_thread.stop()
        if 'realsense_displayer' in locals():
            realsense_displayer.stop()
        
        if record:
            if 'usb_camera' in locals():
                usb_camera.stop()
            if 'realsense_recorder' in locals():
                realsense_recorder.stop()
            if 'usb_recorder' in locals():
                usb_recorder.stop()
        
        if 'control' in locals():
            control.stop()
        
        if rospy.is_initialized():
            rospy.signal_shutdown('Program ended')