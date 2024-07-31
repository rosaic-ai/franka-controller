'''Gym Interface for Franka'''
import threading
import numpy as np
import gym
from gym import core, spaces
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import cv2
import copy
import time
import requests
import queue


class FrankaVC(gym.Env):
    def __init__(self, 
                 randomReset=np.zeros(6), 
                 hz = 100,
                 img_dim=(480, 640), # H x W
                 start_gripper=0,
                 ):
        

        self.resetpos = np.zeros(7)
        self.resetpos[:3] = np.array([0.5, 0.1, 0.2])
        self.reset_yaw=np.pi/2
        self.resetpos[3:] = self.euler_2_quat(np.pi, 0, self.reset_yaw )
        self.nextpos=self.resetpos
        self.currpos = self.resetpos[:].copy()
        self.currvel = np.zeros((6,))
        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.currforce = np.zeros((3,))
        self.currtorque = np.zeros((3,))
        self.currjacobian = np.zeros((6,7))
        self.currgrip = start_gripper
        self.lastsent = time.time()
        self.randomreset = randomReset
        self.actionnoise = 0
        self.hz = hz

        ## NUC
        self.ip = '127.0.0.1'
        self.url = 'http://'+self.ip+':5000/'

       # Bouding box
        self.xyz_bounding_box = gym.spaces.Box(
            np.array((0.35, -0.3, 0.02)),
            np.array((0.82, 0.3, 0.4)),
            dtype=np.float64
        )
        self.rpy_bounding_box = gym.spaces.Box(
            np.array((2*np.pi/3, -np.pi/3, -np.pi/2)),
            np.array((np.pi, np.pi/3, 5*np.pi/6)),
            dtype=np.float64
            )
        ## Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.array((-0.06, -0.06, -0.06, -0.25, -0.25, -0.25, 0-1e-8)),
            np.array((0.06, 0.06, 0.06, 0.25 , 0.25, 0.25, 1+1e-8))
        )
        self.img_dim = img_dim
        self.observation_space = spaces.Dict({
                                'side_1': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'side_1_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 1), dtype=np.uint16),
                                'side_1_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),

                                'side_2': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'side_2_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'side_2_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                'wrist_1': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'wrist_1_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'wrist_1_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                'wrist_2': spaces.Box(low=0, high=225, shape=(256, 256, 3), dtype=np.uint8),
                                'wrist_2_depth': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint16),
                                'wrist_2_full': spaces.Box(low=0, high=225, shape=(img_dim[0], img_dim[1], 4), dtype=np.uint8),
                                
                                
                                'tcp_pose': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'tcp_vel': spaces.Box(-np.inf, np.inf, shape=(6,)),
                                'gripper_pose': spaces.Box(-1, 1, shape=(1,), dtype=np.int8),
                                'q': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'dq': spaces.Box(-np.inf, np.inf, shape=(7,)),
                                'tcp_force': spaces.Box(-np.inf, np.inf, shape=(3,)),
                                'tcp_torque': spaces.Box(-np.inf, np.inf, shape=(3,)),
                                'jacobian': spaces.Box(-np.inf, np.inf, shape=((6,7))),
                                'gripper_dist': spaces.Box(-np.inf, np.inf, shape=(1,)),
                            })

        print("Initialized Franka")
        if start_gripper==0:
            requests.post(self.url + 'open_gripper')
            
    def _get_obs(self):
        state_observation = self._get_state()

        return copy.deepcopy(state_observation)

    def _get_state(self):
        state_observation = {
            'tcp_pose': self.currpos,
            'tcp_vel': self.currvel,
            'gripper_pose': self.currgrip,
            'q': self.q,
            'dq': self.dq,
            'tcp_force': self.currforce,
            'tcp_torque': self.currtorque,
            'jacobian': self.currjacobian,
            'gripper_dist': self.gripper_dist,
        }
        return state_observation

    def _send_pos_command(self, pos):
        self.recover()
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist()}
        requests.post(self.url + 'pose', json=data)

    def move_to_pos(self, pos):
        start_time = time.time()
        if len(pos[3:]) == 3:
            trans = pos[:3]
            quat = self.euler_2_quat(pos[3], pos[4], pos[5])
            pos = np.concatenate([trans, quat])
        self._send_pos_command([pos])
        dl = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dl))
        self.update_currpos()
        obs = self._get_obs()
        return obs

    def get_current_state(self):
        """ Retrieve the current state of the robot. """
        return requests.get(f'{self.url}getstate').json()

    def set_gripper(self, position):
        if position != self.currgrip:
            if position == 1:
                st = 'close_gripper'
                self.currgrip = 1
            else:
                st = 'open_gripper'
                self.currgrip = 0
        else:
            return

        ### IMPORTANT, IF FRANKA GRIPPER GETS OPEN/CLOSE COMMANDS TOO QUICKLY IT WILL FREEZE
        delta = time.time() - self.lastsent
        time.sleep(max(0, 1 - delta))

        requests.post(self.url + st)
        if st == 'close_gripper':
            time.sleep(1.2)
        else:
            time.sleep(0.6)
        self.lastsent = time.time()

    def precision_mode(self):
        requests.post(self.url+ 'precision_mode')

    def compliance_mode(self):
        requests.post(self.url+ 'compliance_mode')

    def open_gripper(self):
        """ Open the gripper. """
        requests.post(f'{self.url}open_gripper')

    def close_gripper(self):
        """ Close the gripper. """
        requests.post(f'{self.url}close_gripper')

    def update_currpos(self):
        ps = requests.post(self.url + 'getstate').json()
        self.currpos[:] = np.array(ps['pose'])
        self.currvel[:] = np.array(ps['vel'])
        self.currforce[:] = np.array(ps['force'])
        self.currtorque[:] = np.array(ps['torque'])
        self.currjacobian[:] = np.reshape(np.array(ps['jacobian']), (6,7))
        self.q[:] = np.array(ps['q'])
        self.dq[:] = np.array(ps['dq'])
        self.gripper_dist = np.array(ps['gripper'])

    def reset(self, jpos=None, gripper=0, require_input=True):
        requests.post(self.url+ 'precision_mode')
        self.set_gripper(gripper)
        self.update_currpos()
        if jpos == None:
            jpos = (np.abs(self.q[0])>0.3)

        success = self.go_to_rest(jpos=jpos)
        self.curr_path_length = 0
        self.recover()
        if jpos==True:
            self.go_to_rest(jpos=False)
            self.recover()

        if require_input:
            input('Reset Environment, Press Enter Once Complete: ')
        self.update_currpos()
        # self.last_quat = self.currpos[3:]
        o = self._get_obs()
        requests.post(self.url+ 'compliance_mode')

        return o

    def get_ee_pose(self):
        """ Request the current robot state """
        response = requests.post(self.url + 'getpos')
        state = response.json()
        return state

    def recover(self):
        """ Clear any errors and recover the robot. """
        requests.post(f'{self.url}clearerr')

    def update_position(self):
        """ Update the internal representation of the robot's position. """
        self.currpos = self.get_current_state()['pose']

    def quat_2_euler(self, quat):
        # calculates and returns: yaw, pitch, roll from given quaternion
        if not isinstance(quat, Quaternion):
            quat = Quaternion(quat)
        yaw, pitch, roll = quat.yaw_pitch_roll
        return yaw + np.pi, pitch, roll


    def euler_2_quat(self, yaw=np.pi/2, pitch=0.0, roll=np.pi):
        yaw = np.pi - yaw
        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw), np.cos(yaw), 0.0], [0, 0, 1.0]])
        pitch_matrix = np.array([[np.cos(pitch), 0., np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0, np.cos(pitch)]])
        roll_matrix = np.array([[1.0, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
        return Quaternion(matrix=rot_mat).elements
