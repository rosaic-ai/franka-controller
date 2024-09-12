import numpy as np
import time
import argparse
from modules.config import Config

from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC, GetImageThread, ImageDisplayer
from robot_infra.state_machine import RobotStateMachine, State
from robot_infra.camera.rs_capture import RSCapture
from robot_infra.camera.video_capture import VideoCapture

import socket
import struct
import pickle
import time

import logging
import collections

import rospy
from geometry_msgs.msg import WrenchStamped, Wrench

rospy.init_node('robot_force_torque_publisher')
force_torque_pub = rospy.Publisher('ee_force_torque', WrenchStamped, queue_size=5)

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
    # force_torque_msg.wrench.force.x = curr_force[0]
    # force_torque_msg.wrench.force.y = curr_force[1]
    # force_torque_msg.wrench.force.z = curr_force[2]
    # force_torque_msg.wrench.torque.x = curr_torque[0]
    # force_torque_msg.wrench.torque.y = curr_torque[1]
    # force_torque_msg.wrench.torque.z = curr_torque[2]
    
    force_torque_msg.wrench.force.x = -(curr_force[0] - initial_force_torque[0])
    force_torque_msg.wrench.force.y = -(curr_force[1] - initial_force_torque[1])
    force_torque_msg.wrench.force.z = -(curr_force[2] - initial_force_torque[2])
    force_torque_msg.wrench.torque.x = -(curr_torque[0] - initial_force_torque[3])
    force_torque_msg.wrench.torque.y = -(curr_torque[1] - initial_force_torque[4])
    force_torque_msg.wrench.torque.z = -(curr_torque[2] - initial_force_torque[5])
        
    force_torque_pub.publish(force_torque_msg)

def _get_observation(initial_force_torque):
    # Fetch the latest image
    get_img = camera_thread.fetch_images()

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

    # force_array = np.array(curr_force)
    # torque_array = np.array(curr_torque)
    # curr_ee_ft = np.concatenate((force_array, torque_array))    
    
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
    last_state = None
    waypoints = []
    start_pos = None
    current_index = 0
    last_target_pose = None
    spiral_step = 0
    loop_counter = 0 
    sequence_length = 5
    initial_force_torque = np.zeros(6)
    current_state = None
    
    host = '172.27.190.155'
    port = 73
    
    image_buffers = collections.deque(maxlen=sequence_length)
    robot_data_buffers = collections.deque(maxlen=sequence_length)
    
    last_time = time.time()
    loop_freq = 0
    manage_state_internally = False

    state_machine.reset_tmp()

    while True:
        _publish_ft_topic(initial_force_torque)

        # 루프 시작 시간 측정
        current_time = time.time()
        if loop_counter > 0:
            loop_interval = current_time - last_time
            loop_freq = 1 / loop_interval if loop_interval else 0
            print(f"Loop Frequency: {loop_freq:.2f} Hz")
        last_time = current_time
        
        get_img, curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_contact, curr_joint = _get_observation(initial_force_torque) #_get_observation(), _get_observation_for_debugging()
        obs_with_timestamp = fetch_all_observation(get_img, curr_joint, curr_contact, curr_ee_pose)
        synced_obs = synchronize_data(obs_with_timestamp)
        
        synced_img = synced_obs['camera_data']
        synced_contact = np.array(synced_obs['contact_data'])
        synced_joint = np.array(synced_obs['joint_data'])
        synced_pose = np.array(synced_obs['pose_data'])        
                    
        # Reset sensors
        if current_state == State.ZERO_FORCE.value: 
            initial_force_torque = initialize_force_torque_sensors()
    
        if not manage_state_internally:
            target_pose, current_state, gripper_state = state_machine.update_state(synced_pose[:3], synced_pose[3:], synced_contact, spiral_step)
            print(f"Current State: {State(current_state).name}")

                
            if current_state == State.MOVE_TO_PREDICTED_POSE.value: 
                manage_state_internally = True
                start_curr_pos = synced_pose[:3]
                start_curr_quat = synced_pose[3:]
                start_curr_euler = R.from_quat(start_curr_quat).as_euler('xyz', degrees=True)
                
                curr_pos = start_curr_pos
                curr_euler = start_curr_euler
                
        else:
            print("Managing state internally")
        
            if current_state == State.MOVE_TO_PREDICTED_POSE.value: #MOVE_TO_PREDICTED_POSE, CONTACT_AND_MOVE, LOWER_TO_HOLE
                if loop_counter % 4 == 0:
                    image_buffers.append(synced_img)
                    robot_data_buffers.append((synced_contact, synced_joint))
                    
                if len(image_buffers) == sequence_length:
                    # print("Model input ready")
                    
                    # agrregate dataset
                    image_list = np.stack(list(image_buffers)[-5:]) # N_img, H, W, C
                    ft_list, proprio_list = [], []
                    for ft, proprio in list(robot_data_buffers)[-5:]:
                        ft_list.append(ft)
                        proprio_list.append(proprio)
                    
                    ft_list = np.stack(ft_list)
                    proprio_list = np.stack(proprio_list)
                    
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.connect((host, port))
                        # print(f"서버 {host}:{port}에 연결되었습니다.")

                        data = {
                            "images": image_list,
                            "ft_list": ft_list,
                            "proprio_list": proprio_list
                        }

                        send_data(s, data)
                        # print("데이터 전송이 완료되었습니다.")

                        result = receive_result(s)
                        pred_action = result['action']
                        
                    # Get predicted action
                    pred_delta_pos = [x * 2 for x in pred_action[:2]]
                    # pred_delta_pos = pred_action[:2]
                    perturbation = np.random.normal(0, 0.0004, size=2)  # 2D perturbation for x, y
                    pred_delta_pos = [x + p for x, p in zip(pred_delta_pos, perturbation)]
                    # Update the target position
                    
                    if curr_ee_trans[2] < state_target_trans['insertion_done'][2]:
                        target_pos[2] -= 0.001
                        print("Insertion")
                    else:
                        target_pos = curr_pos[:2] + pred_delta_pos  
                        target_pos = np.array([*target_pos, curr_pos[2]])
                        target_pos[2] = state_target_trans['contact_and_move'][2]
                    
                    # # Absolute quartertion
                    # pred_abs_quat = pred_action[3:7]  # absolute quaternion
                    # pred_abs_euler = R.from_quat(pred_abs_quat).as_euler('xyz', degrees=True)
                    # pred_abs_euler[2] = pred_abs_euler[2]*2
                    # target_quat = R.from_euler('xyz', pred_abs_euler, degrees=True).as_quat()
                    
                    # Delta Euler angles
                    pred_delta_euler_z = pred_action[3]*3 # delta euler angles # [0,0, pred_z]를 예상
                    target_euler_z = curr_euler[2] + pred_delta_euler_z # update the target euler angles
                    target_euler = np.array([*curr_euler[:2], target_euler_z])
                    target_quat = R.from_euler('xyz', target_euler, degrees=True).as_quat() # change euler to quaternion

                    # Concat the target position and orientation
                    target_pose = np.concatenate((target_pos, target_quat))

                    # Update the current position and orientation
                    curr_pos = target_pos
                    # curr_euler = target_euler

                    if curr_ee_trans[2] < state_target_trans['done'][2]:
                        print("Done moving to predicted pose")
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