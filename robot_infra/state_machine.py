"""
State machine for visuo-contact peg-in-hole task
"""

import torch
import numpy as np
from enum import Enum
import time
from scipy.spatial.transform import Rotation as R
from .envs.franka_vc_env import FrankaVC

import sys

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
    
    def reset(self):
        print("Resetting to initial state...")
        self.control.precision_mode()
        self.control.set_gripper(1)  # Assuming 1:open

        self.target_trans = self.state_target_trans['reset_pose']
        self.target_euler = self.state_target_ori['reset_euler']
        self.target_quat = self.control.euler_2_quat(*self.target_euler)
        
        initial_pose = np.concatenate([self.target_trans, self.target_quat])
        self.control.move_to_pos(initial_pose)
        
        time.sleep(2)
        print("Reset complete. Starting with APPROACH_PEG state.")
        
    def update_state(self, curr_ee_trans, curr_ee_quat, curr_ft, config = None):
        
        ##################################
        ########## APPROACH_PEG ##########
        ##################################
        approach = self.current_state == State.APPROACH_PEG.value
        if approach:
            print("Approach to Peg")
            self.control.precision_mode()
            self.gripper_values = 0 # 0:open, 1:close
            
            self.target_trans = self.state_target_trans['approach_to_peg']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)
            print(get_pose_error)
            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.LOWER_TO_PEG.value

        ####################################
        ########### LOWER_TO_PEG ###########
        ####################################
        lower_to_peg = self.current_state == State.LOWER_TO_PEG.value
        if lower_to_peg:
            print("Lower to Peg")
            self.target_trans = self.state_target_trans['lower_to_peg']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            self.gripper_values = 1 # 0:open, 1:close
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)

            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.GRASP_PEG.value
              
        ####################################
        ############ GRASP_PEG #############
        ####################################
        grasp_peg = self.current_state == State.GRASP_PEG.value
        if grasp_peg:
            print("Grasp Peg")
            self.target_trans = self.state_target_trans['lower_to_peg']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            self.gripper_values = 1 # 0:open, 1:close
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)
            
            if get_pose_error < self.state_pose_thres['trans_thres']:
                self.current_state = State.CONTACT.value
              
            
        ####################################
        ######## MOVE_TO_HOLE_ABOVE ########
        ####################################
        move_to_hole_above = self.current_state == State.MOVE_TO_HOLE_ABOVE.value
        if move_to_hole_above:
            print("Move to Hole Above")
    
    
        ####################################
        ########### LOWER_TO_HOLE ##########
        ####################################
        lower_to_hole = self.current_state == State.LOWER_TO_HOLE.value
        if lower_to_hole:
            print("Lower to Hole")
            
            
        ####################################
        ###### MOVE_TO_PREDICTED_POSE ######
        ####################################
        move_to_pred = self.current_state == State.MOVE_TO_PREDICTED_POSE.value
        if move_to_pred:
            print("Move to Pred Pose")
            self.control.compliance_mode()

        
        ####################################
        ########## INSERT_TO_HOLE ##########
        ####################################
        insert = self.current_state == State.INSERT_TO_HOLE.value
        if insert:
            print("Insert!!!")
            self.control.compliance_mode()


        ####################################
        ############ RELEASE_PEG ###########
        ####################################
        release_peg = self.current_state == State.RELEASE_PEG.value
        if release_peg:
            print("Success & Release Peg")
            self.gripper_values = 0 # 1:open, 0:close
        
        
        ###################################
        ############# CONTACT #############
        ###################################
        contact = self.current_state == State.CONTACT.value
        if contact:
            print("Contact")
            self.control.compliance_mode()

            self.target_trans = self.state_target_trans['contact']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            self.gripper_values = 1 # 1:open, 0:close
            
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
            print("Contact & Move")
            self.control.compliance_mode()

            self.target_trans = self.state_target_trans['contact_and_move']
            self.target_euler = self.state_target_ori['target_euler']
            self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
            self.gripper_values = 1 # 1:open, 0:close
            
            get_pose_error = self.calculate_distance(curr_ee_trans, self.target_trans)

            # Get distance between target pose, current pose
            # if get_pose_error < self.state_pose_thres['trans_thres']:
                # self.current_state = State.GRASP_PEG.value 
        
        
        return np.concatenate([self.target_trans, self.target_quat]), self.gripper_values, self.current_state