"""
구미 로봇 제어 서버
- 로봇을 초기 자세로 이동 후 클라이언트의 명령을 받아 제어
"""

import numpy as np
import time
import argparse
from modules.config import Config
import signal
import sys
import socket
import json
from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC
import rospy
from geometry_msgs.msg import WrenchStamped
import select 

# 이 부분을 전역 혹은 함수 밖에 한 번만 정의해 두세요.
T_offset = np.eye(4)
T_offset[2, 3] = 0.05  # Z축으로 −5cm 이동

def transform_pose(orig_pose, T_offset):
    # orig_pose: [px, py, pz, w, x, y, z]
    # 1) position
    pos = orig_pose[0:3]
    # 2) reorder quaternion
    w, x, y, z = orig_pose[3:7]
    quat_xyzw = np.array([x, y, z, w])

    # 3) 4×4 행렬로 변환
    T_orig = np.eye(4)
    T_orig[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T_orig[:3,  3] = pos

    # 4) offset 합성
    T_new = T_offset @ T_orig

    # 5) 다시 pose 벡터로
    new_pos  = T_new[:3, 3]
    new_quat = R.from_matrix(T_new[:3, :3]).as_quat()  # [x,y,z,w]
    # move_to_pos expects [w, x, y, z] 순서라면 다시 뒤집기
    new_quat_wxyz = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    return np.hstack([new_pos, new_quat_wxyz])

def signal_handler(sig, frame):
    print('\n프로그램을 종료합니다.')
    if 'control' in globals():
        control.stop()
    if rospy.is_initialized():
        rospy.signal_shutdown('Program ended')
    sys.exit(0)

def gripper_control(self, gripper_state='open'):
    print(self.control.currgrip, gripper_state)
    if self.control.currgrip == 0 and gripper_state == 'close':
        self.control.close_gripper()
    elif self.control.currgrip == 1 and gripper_state == 'open':
        self.control.open_gripper()

def _get_observation(control, initial_force_torque):
    """로봇의 현재 상태를 관측"""
    curr_ee_pose = control.get_ee_pose()
    curr_ee_trans = curr_ee_pose['pose'][:3]
    curr_ee_quat = curr_ee_pose['pose'][3:]

    curr_force = control.get_ee_force()['force']  
    curr_torque = control.get_ee_torque()['torque']
    force_array = -(np.array(curr_force) - initial_force_torque[:3])
    torque_array = -(np.array(curr_torque) - initial_force_torque[3:])
    curr_ee_ft = np.concatenate((force_array, torque_array))
    
    curr_joint_pos = control.get_joint_pos()
    return curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_ee_ft, curr_joint_pos

def initialize_force_torque_sensors(control):
    initial_force = control.get_ee_force()['force']
    initial_torque = control.get_ee_torque()['torque']
    initial_force_torque = np.concatenate((initial_force, initial_torque))
    return initial_force_torque

def _publish_ft_topic(control, initial_force_torque, force_torque_pub):
    curr_force = control.get_ee_force()['force']
    curr_torque = control.get_ee_torque()['torque']

    force_torque_msg = WrenchStamped()
    force_torque_msg.header.stamp = rospy.Time.now()
    force_torque_msg.header.frame_id = 'ee_frame'
    
    # force_torque_msg.wrench.force.x  = (curr_force[0] - initial_force_torque[0])
    # force_torque_msg.wrench.force.y  = (curr_force[1] - initial_force_torque[1])
    # force_torque_msg.wrench.force.z  = (curr_force[2] - initial_force_torque[2])
    # force_torque_msg.wrench.torque.x = (curr_torque[0] - initial_force_torque[3])
    # force_torque_msg.wrench.torque.y = (curr_torque[1] - initial_force_torque[4])
    # force_torque_msg.wrench.torque.z = (curr_torque[2] - initial_force_torque[5])
        
    force_torque_msg.wrench.force.x  = (curr_force[0])
    force_torque_msg.wrench.force.y  = (curr_force[1])
    force_torque_msg.wrench.force.z  = (curr_force[2])
    force_torque_msg.wrench.torque.x = (curr_torque[0])
    force_torque_msg.wrench.torque.y = (curr_torque[1])
    force_torque_msg.wrench.torque.z = (curr_torque[2])
    
    force_torque_pub.publish(force_torque_msg)

def move_to_initial_pose(control):
    """로봇을 초기 위치로 이동"""
    initial_pose = np.array([
        0.45,  # x
        0.0,    # y
        0.2,  # z
        1.0,    # qx
        0.0,    # qy
        0.0,    # qz
        0.0     # qw
    ])
    
    # print("\n=== 로봇 초기화 중 ===")
    # print(f"목표 초기 위치: {initial_pose[:3]}")
    # print(f"목표 초기 자세: {initial_pose[3:]}")
    
    control.move_to_pos(initial_pose)
    time.sleep(2.0)  # 안정화를 위한 대기 시간 증가
    
    curr_pose = control.get_ee_pose()['pose']
    print("\n=== 초기화 완료 ===")
    print(f"현재 위치: {curr_pose[:3]}")
    print(f"현재 자세: {curr_pose[3:]}")
    print("===================\n")
    
    return initial_pose

def run_robot_server(control, force_torque_pub, host='0.0.0.0', port=5000):
    """로봇 제어 서버 실행"""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"서버가 {host}:{port}에서 대기 중입니다.")
        
        initial_force_torque = initialize_force_torque_sensors(control)

        client_socket = None
        buffer = ""
        rospy_rate = rospy.Rate(100)  # 100Hz로 루프 유지

        while not rospy.is_shutdown():
            # 1. Force-Torque 지속 퍼블리시
            _publish_ft_topic(control, initial_force_torque, force_torque_pub)

            # 2. 클라이언트 새로 연결
            if client_socket is None:
                readable, _, _ = select.select([server_socket], [], [], 0.001)
                if readable:
                    client_socket, addr = server_socket.accept()
                    print(f"{addr}에서 연결됨")
                    buffer = ""

            # 3. 클라이언트 명령 수신
            if client_socket:
                try:
                    client_socket.setblocking(False)
                    data = client_socket.recv(1024).decode('utf-8')
                    if not data:
                        print("클라이언트 연결 종료")
                        client_socket.close()
                        client_socket = None
                        continue
                    
                    buffer += data
                    while '\n' in buffer:
                        message, buffer = buffer.split('\n', 1)
                        try:
                            received_data = json.loads(message)
                            target_pose = np.array(received_data['target_pose'])
                            
                            # new_target = transform_pose(target_pose, T_offset)

                            if 'gripper_command' in received_data:
                                gripper_state = received_data['gripper_command']
                                if gripper_state == 'close' and control.currgrip == 0:
                                    control.close_gripper()
                                elif gripper_state == 'open' and control.currgrip == 1:
                                    control.open_gripper()

                            control.move_to_pos(target_pose)
                        except json.JSONDecodeError as e:
                            print(f"JSON 디코딩 오류: {e}")

                except BlockingIOError:
                    pass  # 수신 데이터가 없으면 그냥 넘어감
                except ConnectionResetError:
                    print("클라이언트 연결이 끊어졌습니다.")
                    client_socket.close()
                    client_socket = None

            rospy_rate.sleep()

    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        if client_socket:
            client_socket.close()
        server_socket.close()
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_robot', type=str,
                    default='robot_infra/configs/robot_params.yaml',
                    help='로봇 설정 파일 경로')
    parser.add_argument('--port', type=int, default=4999,
                    help='서버 포트 번호')
    parser.add_argument('--gripper', type=str, default="open", choices=['open', 'close'],
                        help="Control the state of the gripper: open or close")
    args = parser.parse_args()

    if args.gripper == 'close':
        start_gripper = 0
    else:
        start_gripper = 1

    # 로봇 초기화
    config_robot = Config(args.config_robot).get_config()
    control = FrankaVC(config_robot=config_robot, hz=100, start_gripper=start_gripper)
    control.precision_mode()
    # control.compliance_mode()

    try:
        # 로봇을 초기 자세로 이동
        # move_to_initial_pose(control)
        
        curr_pose = control.get_ee_pose()['pose']
        print(f"현재 위치: {curr_pose[:3]}")
        print(f"현재 자세: {curr_pose[3:]}")    

        # ROS 노드 초기화
        rospy.init_node('robot_force_torque_publisher')
        force_torque_pub = rospy.Publisher('ee_force_torque', WrenchStamped, queue_size=5)

        # 서버 실행
        run_robot_server(control, force_torque_pub, port=args.port)
 
        
    except KeyboardInterrupt:
        print('\n프로그램이 사용자에 의해 중단되었습니다.')
    finally:
        if 'control' in locals():
            control.stop()
        if rospy.is_initialized():
            rospy.signal_shutdown('Program ended')

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()