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
    DONE = 10

# configs/robot_parm

class RobotStateMachine:
    def __init__(self, device='cpu', num_envs=1, config_robot=None):
        self.device = device
        self.sim_params = config_robot.get('sim_params', {})
        self.control = FrankaVC(config_robot=config_robot)        
        self.current_state = torch.tensor([State.APPROACH_PEG.value], dtype=torch.int64, device=self.device)

    def update_state(self, curr_ee_pos, curr_ee_quat, config = None):
        ##################################
        ########## APPROACH_PEG ##########
        ##################################
        approach = self.current_state == State.APPROACH_PEG.value
        self.control.precision_mode()
        
        self.target_trans = [0.4, 0.0, 0.4]
        self.target_euler = [0,0,10]
        self.target_quat = self.control.euler_2_quat(self.target_euler[0], self.target_euler[1], self.target_euler[2])
        self.target_pose = np.concatenate([self.target_trans, self.target_quat])
        self.gripper_values = 0
        
        # Get distance between target pose & current pose
        # if curr_ee_pos 
        
        ####################################
        ########### LOWER_TO_PEG ###########
        ####################################
        lower_to_peg = self.current_state == State.LOWER_TO_PEG.value
        
              
        ####################################
        ############ GRASP_PEG #############
        ####################################
        grasp_peg = self.current_state == State.GRASP_PEG.value
        
                
        ####################################
        ######## MOVE_TO_HOLE_ABOVE ########
        ####################################
        approach = self.current_state == State.MOVE_TO_HOLE_ABOVE.value
       
    
        ####################################
        ########### LOWER_TO_HOLE ##########
        ####################################
        approach = self.current_state == State.LOWER_TO_HOLE.value
        
        ####################################
        ###### MOVE_TO_PREDICTED_POSE ######
        ####################################
        approach = self.current_state == State.MOVE_TO_PREDICTED_POSE.value

        
        ####################################
        ########## INSERT_TO_HOLE ##########
        ####################################
        approach = self.current_state == State.INSERT_TO_HOLE.value

        ####################################
        ############ RELEASE_PEG ###########
        ####################################
        approach = self.current_state == State.RELEASE_PEG.value
          
        return np.concatenate([self.target_trans, self.target_quat]), self.gripper_values, self.current_state