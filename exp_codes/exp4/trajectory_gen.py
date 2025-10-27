#!/usr/bin/env python3
# This is the robot-side code, may not be executed on other robots.

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rospy
from std_msgs.msg import Float32MultiArray, Int16
import time
from bezier import Bezier2  
import csv
import sys
import os
from pathlib import Path
import torch

# Add hexapod scripts to system path for importing RobotInterface
path = Path(sys.path[0])
src_folder = path.parent.parent
sys.path.append(os.path.join(src_folder, 'hexapod', 'scripts'))
print('hexapod_path = ', sys.path)
from robot_interface import RobotInterface  # Custom interface for robot control

# Constants for optimization
MAXITER = 10  # Maximum iterations for optimization (if used elsewhere)
TOLERANCE = 0.00005  # Tolerance for optimization convergence (if used elsewhere)

# Class to generate trajectories for a hexapod robot
class TrajGenerator(object):
    def __init__(self, omega_0, Hz):
        # Initialize robot interface
        self.interface = RobotInterface()
        # Angular velocity (rad/s) and control frequency (Hz)
        self.omega_0 = omega_0
        self.Hz = Hz
        # Duty factor (fraction of cycle time in stance phase)
        self.mu = 0.5
        # Operator scaling factor for displacement
        self.operator = 3
        # Displacement for right and left legs
        self.disp_r = 0.02 * self.operator
        self.disp_l = -self.disp_r
        # Calculate initial velocity based on angular velocity and displacement
        self.v = self.omega_0 * self.disp_r * 2 / (2 * math.pi * self.mu)
        # Dictionary of Bezier2 objects for each leg (0 to 5)
        self.beziers = {str(i): Bezier2() for i in range(6)}
        # Initialize position arrays for x, y, z coordinates
        self.x = [0] * 6
        self.y = [0] * 6
        self.z = [0] * 6
        # Height of robot in stance (negative due to coordinate system)
        self.standheight = -self.interface.rest_pose[0, 2]
        # Dictionary for joint commands for each leg joint
        self.joint_command = {
            'j_c1_lf': 0, 'j_c1_lm': 0.0, 'j_c1_lr': 0,
            'j_c1_rf': 0, 'j_c1_rm': 0.0, 'j_c1_rr': 0,
            'j_thigh_lf': 0, 'j_thigh_lm': 0, 'j_thigh_lr': 0,
            'j_thigh_rr': 0, 'j_thigh_rm': 0, 'j_thigh_rf': 0,
            'j_tibia_lf': 0, 'j_tibia_lm': 0, 'j_tibia_lr': 0,
            'j_tibia_rr': 0, 'j_tibia_rm': 0, 'j_tibia_rf': 0
        }
        # Track start time of stance phase for each leg
        self.stance_start_time = [0] * 6
        # Track leg state: 0 (swing), 1 (touchdown), 2 (stance)
        self.last_state = [0] * 6
        # Dictionaries to store trajectory data for each leg (keys: '1' to '6')
        self.leg_traj_x = {str(i): [] for i in range(1, 7)}
        self.leg_traj_y = {str(i): [] for i in range(1, 7)}
        self.leg_traj_z = {str(i): [] for i in range(1, 7)}
        self.cycle_time_traj = {str(i): [] for i in range(1, 7)}
        self.duration_traj = {str(i): [] for i in range(1, 7)}
        self.t_traj = {str(i): [] for i in range(1, 7)}
        # Subscribe to gait phase data from ROS topic
        self.gait_sub = rospy.Subscriber('/g_cpg_gait', Float32MultiArray, self.gait_cb)
        # Track phase for each leg
        self.phase = [0] * 6
        # Track last update time for each leg
        self.time_last = [0] * 6
        # Touchdown lock for leg groups
        self.touchdown_lock = {'0': False, '1': False}
        # Define leg groups: left (0,1,2) and right (3,4,5)
        self.side_legs = {'0': [0, 1, 2], '1': [3, 4, 5]}
        # Track touchdown completion for each leg
        self.touchdown_end = [False] * 6
        # Flag to indicate gait transition
        self.at_transition = False
        # Track touchdown start time for each leg
        self.td_start_time = [1] * 6
        # Count touch-down events for each leg
        self.touchdown_counter = [0] * 6
        # Command array for legs
        self.command = [0] * 6
        # Define all legs and operating legs (can exclude defective legs)
        self.all_legs = np.array([0, 1, 2, 3, 4, 5])
        self.operating_legs = np.array([0, 1, 2, 3, 4, 5])

    # Calculate stance phase velocity (forward motion)
    def calculate_stance_v(self):
        self.v = -self.omega_0 * self.disp_r * 2 / (2 * math.pi * self.mu)
        return self.v

    # Calculate stance phase velocity for reverse motion
    def calculate_stance_v_reverse(self):
        self.v_reverse = self.omega_0 * self.disp_r * 2 / (2 * math.pi * self.mu)
        return self.v_reverse

    # Generate leg poses based on phase values
    def leg_pose_from_phase(self, phase):
        # Initialize target pose tensor for 6 legs (x, y, z coordinates)
        e_tar = torch.ones(6, 3, dtype=torch.float32, device='cpu')
        # Identify defective legs (not in operating_legs)
        defect_legs = np.setdiff1d(self.all_legs, self.operating_legs)

        # Handle defective legs by setting them to rest pose
        if defect_legs.size == 0:
            print('None of the legs is defected')
        else:
            print('Defected legs: ', defect_legs)
            for leg_index in defect_legs:
                x_tar = self.interface.rest_pose[leg_index, 0]
                y_tar = self.interface.rest_pose[leg_index, 1]
                z_tar = -self.standheight + 0.12  # Fixed height for defective legs
                e_tar[leg_index, 0] = x_tar
                e_tar[leg_index, 1] = y_tar
                e_tar[leg_index, 2] = z_tar

        # Process each operating leg
        for i, leg_index in enumerate(self.operating_legs):
            cycle_time = phase[i]
            # Stance phase (cycle_time < mu)
            if cycle_time < self.mu:
                # On touchdown, reset time and set initial y-position
                if self.last_state[leg_index] == 1:
                    self.time_last[leg_index] = time.time()
                    self.y[leg_index] = self.disp_r if leg_index <= 2 else self.disp_l
                # Calculate velocity based on leg group (left or right)
                v = self.calculate_stance_v() if leg_index <= 2 else self.calculate_stance_v_reverse()
                # Use fixed time step based on control frequency
                dt = 1 / self.Hz
                dy = v * dt
                self.y[leg_index] += dy
                self.z[leg_index] = 0  # Leg on ground during stance
                self.last_state[leg_index] = 0  # Update state to swing
                self.time_last[leg_index] = time.time()
            # Swing phase (cycle_time >= mu)
            else:
                # On swing start, set up Bezier curve control points
                if self.last_state[leg_index] == 0:
                    if leg_index <= 2:
                        x1 = self.y[leg_index]
                        x2 = (self.y[leg_index] + self.disp_r) / 2
                        x3 = self.disp_r
                    else:
                        x1 = self.y[leg_index]
                        x2 = (self.y[leg_index] + self.disp_l) / 2
                        x3 = self.disp_l
                    y1 = 0
                    y2 = 0.08  # Peak height of swing trajectory
                    y3 = 0
                    x_vec = [x1, x2, x3]
                    y_vec = [y1, y2, y3]
                    # Set Bezier curve control points
                    self.beziers[str(leg_index)].setPoint(x_vec, y_vec)
                # Calculate position using Bezier curve for swing phase
                t = (cycle_time - self.mu) / (1 - self.mu)  # Normalize t for swing phase
                self.y[leg_index], self.z[leg_index] = self.beziers[str(leg_index)].getPos(t)
                self.last_state[leg_index] = 1  # Update state to touchdown
            # Compute target pose relative to rest pose
            x_tar = self.interface.rest_pose[leg_index, 0]
            y_tar = self.interface.rest_pose[leg_index, 1] + self.y[leg_index]
            z_tar = -self.standheight + self.z[leg_index]
            # Store trajectory data
            self.leg_traj_x[str(leg_index + 1)].append(x_tar)
            self.leg_traj_y[str(leg_index + 1)].append(y_tar)
            self.leg_traj_z[str(leg_index + 1)].append(z_tar)
            # Update target pose tensor
            e_tar[leg_index, 0] = x_tar
            e_tar[leg_index, 1] = y_tar
            e_tar[leg_index, 2] = z_tar
        return e_tar



    # Callback for ROS gait phase subscriber
    def gait_cb(self, data):
        self.phase = data.data  # Update phase values from ROS topic

