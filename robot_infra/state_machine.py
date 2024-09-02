"""
State machine for visuo-contact peg-in-hole task
"""

import torch
import numpy as np
from enum import Enum
import time
from scipy.spatial.transform import Rotation as R
from .envs.franka_vc_env import FrankaVC, GetImageThread, ImageDisplayer

import sys
from copy import deepcopy

# 상태 정의
class State(Enum):
    INITIAL = 0
    APPROACH_PEG = 1
    LOWER_TO_PEG = 2
    GRASP_PEG = 3
    MOVE_TO_HOLE_ABOVE = 4
    LOWER_TO_HOLE = 5
    MOVE_TO_PREDICTED_POSE = 6
    INSERT_TO_HOLE = 7
    POSE_RECOVERY = 8
    RELEASE_PEG = 9
    CONTACT = 10
    CONTACT_AND_MOVE = 11
    DONE = 12

class RobotStateMachine:
    def __init__(self, device='cpu', num_envs=1, config_robot=None):
        self.device = device
        self.state_target_trans = config_robot.get('state_target_trans', {})
        self.state_target_ori   = config_robot.get('state_target_ori', {})
        self.state_pose_thres   = config_robot.get('state_pose_thres', {})
        self.control = FrankaVC(config_robot=config_robot)        
        self.current_state = State.APPROACH_PEG.value # Initial state
        self.previous_state = None
        self.gripper_state = None
        # self.camera_thread = GetImageThread(serial_number='130322270132', dim=(848, 480), fps=15, depth=False)
        # displayer_thread = ImageDisplayer(self.camera_thread.img_queue)
        # displayer_thread.start()
    
    def calculate_distance(self, pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        distance = np.linalg.norm(pos1 - pos2)
        return distance

    def get_curr_pose(self):
        curr_ee_pose = self.control.get_ee_pose()
        curr_ee_pos = curr_ee_pose['pose'][:3]
        curr_ee_quat = curr_ee_pose['pose'][3:]
        return curr_ee_pos, curr_ee_quat
    
    def reset_tmp(self):
        print("Resetting to initial state...")
        self.control.compliance_mode()
        curr_pos, curr_quat = self.get_curr_pose()
                
        # Extract 5cm
        raised_pos = np.array(curr_pos)
        raised_pos[2] += 0.05
        raised_pose = np.concatenate([raised_pos, curr_quat])
        
        print("---" * 20)
        print("curr_pos   :", np.round(curr_pos,    3))
        print("curr_quat  :", np.round(curr_quat,   3))
        print("raised_pose:", np.round(raised_pose, 3))
        print("---" * 20)
        
        self.control.move_to_pos(raised_pose)
        time.sleep(1)
        
        self.gripper_control(gripper_state='open')
        # self.gripper_control(gripper_state='close')

        self.target_trans = self.state_target_trans['reset_pose']
        self.target_euler = self.state_target_ori['reset_euler']
        self.target_quat = self.control.euler_2_quat(*self.target_euler)
                
        initial_pose = np.concatenate([self.target_trans, self.target_quat])
        
        print("---" * 20)
        print("targ trans:", np.round(self.target_trans, 3))
        print("targ euler:", np.round(self.target_euler, 3))
        print("targ quat :", np.round(self.target_quat,  3))
        print("targ pose :", np.round(initial_pose,      3))   
        print("---" * 20)     
        self.control.move_to_pos(initial_pose)
        
        print("Reset complete. Starting with APPROACH_PEG state.")
        
    def reset(self):
        print("Resetting to initial state...")
        self.control.compliance_mode()
        curr_pos, curr_quat = self.get_curr_pose()
        
        # Extract 5cm
        raised_pos = np.array(curr_pos)
        raised_pos[2] += 0.05
        raised_pose = np.concatenate([raised_pos, curr_quat])
        self.control.move_to_pos(raised_pose)
        time.sleep(1)
        
        self.gripper_control(gripper_state='open')

        self.target_trans = self.state_target_trans['reset_pose']
        self.target_euler = self.state_target_ori['reset_euler']
        self.target_quat = self.control.euler_2_quat(*self.target_euler)
        
        initial_pose = np.concatenate([self.target_trans, self.target_quat])
        self.control.move_to_pos(initial_pose)
        
        print("Reset complete. Starting with APPROACH_PEG state.")
        
    def spiral_search(self, center, radius_increment, angle_increment, spiral_step):
        waypoints = []
        radius = spiral_step * radius_increment  # 스파이럴 스텝에 따라 반경을 계산합니다
        angle = spiral_step * angle_increment  # 스파이럴 스텝에 따라 각도를 계산합니다
        
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        waypoints.append([x, y])
        
        return waypoints

    def gripper_control(self, gripper_state='open'):
        print(self.control.currgrip, gripper_state)
        if self.control.currgrip == 0 and gripper_state == 'close':
            self.control.close_gripper()
        elif self.control.currgrip == 1 and gripper_state == 'open':
            self.control.open_gripper()
    
    def update_state(self, curr_ee_trans, curr_ee_quat, curr_ft, sprial_step, config = None):
        ##################################
        ########## APPROACH_PEG ##########
        ##################################
        approach = self.current_state == State.APPROACH_PEG.value
        if approach:
            self.control.compliance_mode()
            # self.gripper_control(gripper_state='open')
            self.target_trans = self.state_target_trans['approach_to_peg']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)
            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.MOVE_TO_HOLE_ABOVE.value #MOVE_TO_HOLE_ABOVE, LOWER_TO_PEG


        ####################################
        ########### LOWER_TO_PEG ###########
        ####################################
        lower_to_peg = self.current_state == State.LOWER_TO_PEG.value
        if lower_to_peg:
            self.target_trans = self.state_target_trans['lower_to_peg']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)

            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.GRASP_PEG.value
              
        
        ####################################
        ############ GRASP_PEG #############
        ####################################
        grasp_peg = self.current_state == State.GRASP_PEG.value
        if grasp_peg:
            self.gripper_control(gripper_state='close')
            time.sleep(2)
            # After the delay, transition to the next state
            self.current_state = State.MOVE_TO_HOLE_ABOVE.value
            
        
        ####################################
        ######## MOVE_TO_HOLE_ABOVE ########
        ####################################
        move_to_hole_above = self.current_state == State.MOVE_TO_HOLE_ABOVE.value
        if move_to_hole_above:
            self.target_trans = self.state_target_trans['move_to_hole_above']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)

            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.LOWER_TO_HOLE.value
    
    
        ####################################
        ########### LOWER_TO_HOLE ##########
        ####################################
        lower_to_hole = self.current_state == State.LOWER_TO_HOLE.value
        if lower_to_hole:
            self.target_trans = self.state_target_trans['lower_to_hole']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)
            curr_z_force = curr_ft[2]
            
            # Get distance between target pose, current pose
            if curr_z_force > 2:
                self.current_state = State.CONTACT_AND_MOVE.value 

            # if get_pose_error < self.state_pose_thres['trans_thres']:
            #     self.current_state = State.CONTACT_AND_MOVE.value
            
        ####################################
        ###### MOVE_TO_PREDICTED_POSE ######
        ####################################
        move_to_pred = self.current_state == State.MOVE_TO_PREDICTED_POSE.value
        if move_to_pred:
            print("Moving to predicted pose...")

        
        ####################################
        ########## INSERT_TO_HOLE ##########
        ####################################
        insert = self.current_state == State.INSERT_TO_HOLE.value
        if insert:
            self.control.compliance_mode()


        ####################################
        ############ RELEASE_PEG ###########
        ####################################
        release_peg = self.current_state == State.RELEASE_PEG.value
        if release_peg:
            self.gripper_values = 0 # 1:open, 0:close
        
        
        ###################################
        ############# CONTACT #############
        ###################################
        contact = self.current_state == State.CONTACT.value
        if contact:
            self.target_trans = self.state_target_trans['contact']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)

            curr_z_force = curr_ft['force'][2]

            # Get distance between target pose, current pose
            if curr_z_force > 5:
                self.current_state = State.CONTACT_AND_MOVE.value 
        
                
        ####################################
        ######### CONTACT_AND_MOVE #########
        ####################################
        contact_and_move = self.current_state == State.CONTACT_AND_MOVE.value
        if contact_and_move:
            self.target_trans = self.state_target_trans['contact_and_move']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)
            
            center = self.target_trans[:2]
            radius_increment = 0.000001  # 반경 증가량을 반으로 줄임
            angle_increment = np.pi / 36  # 각도 증가량을 절반으로 줄임
            
            spiral_points = self.spiral_search(center, radius_increment, angle_increment, sprial_step)
            
            self.target_trans[:2] = spiral_points[0]
            
            if curr_ee_trans[2] < 0.225:
                self.current_state = State.RELEASE_PEG.value 

            # Get distance between target pose, current pose
            # if get_pose_error < self.state_pose_thres['trans_thres']:
                # self.current_state = State.RELEASE_PEG.value 
        
        return np.concatenate([self.target_trans, self.target_quat]), self.current_state, self.gripper_state
