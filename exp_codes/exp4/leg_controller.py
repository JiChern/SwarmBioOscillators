#!/usr/bin/env python3
# This is the robot-side code, may not be executed on other robots.

import math
import torch
import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, Float32MultiArray, MultiArrayDimension
import time
import csv
import os
from kinematic import Kinematic  # Custom module for kinematic calculations
from hex_cfg import HexCfg  # Custom module for hexapod configuration
from trajectory_gen import TrajGenerator  # Custom module for trajectory generation

# Constants for iterative inverse kinematics
MAXITER = 5  # Maximum iterations for IK convergence
TOLERANCE = 0.001  # Convergence tolerance for IK error

# Class to control hexapod leg movements using inverse kinematics
class LegController(object):
    def __init__(self):
        # Set device to CPU for tensor operations
        device = 'cpu'
        # Load hexapod configuration
        cfg = HexCfg('config')
        # Initialize kinematic model with link lengths from config
        self.kin = Kinematic(cfg.link.l1, cfg.link.l2, cfg.link.l3, device)
        # Reference to trajectory generator (assumes tg is defined externally)
        self.tg = tg
        # Current joint angles for 6 legs (3 joints per leg)
        self.q_cur = torch.zeros(6, 3, dtype=torch.float32, device=device)
        # End-effector positions for 6 legs (x, y, z)
        self.e_pos = torch.ones(6, 3, dtype=torch.float32, device=device)
        # List of leg names for right (RF, RM, RB) and left (LF, LM, LB) legs
        leg_name_list = ['RF', 'RM', 'RB', 'LF', 'LM', 'LB']
        # Lists for ROS publishers, subscribers, and messages
        self.q_des_pub_list = []
        self.q_cur_sub_list = []
        self.q_des_msg_list = []
        # Default joint angles for resetting legs (6 legs, 3 joints each)
        self.reset_angles = np.array([
            [0.35, 0.69, -2.14],  # RF
            [0.0, 0.69, -2.14],   # RM
            [-0.35, 0.69, -2.14], # RB
            [-0.35, 0.69, -2.14], # LF
            [0.0, 0.69, -2.14],   # LM
            [0.35, 0.69, -2.14]   # LB
        ])
        # Initialize ROS publishers and subscribers for each leg
        for i, leg_name in enumerate(leg_name_list):
            self.q_des_pub_list.append(rospy.Publisher(
                '/' + leg_name + '/sita_des', Float64MultiArray, queue_size=10))
            self.q_cur_sub_list.append(rospy.Subscriber(
                '/' + leg_name + '/sita_cur', Float64MultiArray, self.UpdateQState, callback_args=i))
            self.q_des_msg_list.append(Float64MultiArray())
        # Publisher for operating legs data
        self.leg_pub = rospy.Publisher('operating_legs', Float32MultiArray, queue_size=10)
        # Subscriber for gait phase data
        self.gait_all_sub = rospy.Subscriber('/g_cpg_gait_all', Float32MultiArray, self.all_gait_cb)
        # Store gait phase data for all legs
        self.all_gait = np.zeros(6)

    # Callback to update gait phase data from ROS topic
    def all_gait_cb(self, data):
        self.all_gait = data.data

    # Callback to update current joint angles for a specific leg
    def UpdateQState(self, q_state_msgs: Float64MultiArray, index):
        self.q_cur[index, 0:3] = torch.tensor(
            q_state_msgs.data[0:3], dtype=torch.float32, device='cpu')

    # Inverse kinematics solver for a single leg
    def IKSolve(self, leg, target):
        converged = False
        diff = 100
        iter = 0
        # Compute forward kinematics and Jacobian for the leg
        joint_angles, trans, _, jacobian = self.hexapod.fk(leg['interface'], leg['joint_names'])
        while iter < MAXITER:
            # Compute position error
            error = (target - trans).reshape(3, 1)
            jacobian = jacobian[0:3, :]  # Use position components of Jacobian
            cart_vel = error
            # Compute pseudo-inverse of Jacobian
            pseudo = np.linalg.pinv(jacobian)
            # Calculate joint velocity
            q_dot = np.matmul(pseudo, cart_vel)
            # Update joint angles with scaled velocity
            joint_angles[0] += q_dot[0] * 0.5
            joint_angles[1] += q_dot[1] * 0.5
            joint_angles[2] += q_dot[2] * 0.5
            # Recalculate forward kinematics and Jacobian
            trans, _, _ = leg['interface'].forward_kinematics(joint_angles)
            jacobian = leg['interface'].jacobian(joint_angles)
            # Compute Euclidean distance error
            diff = math.sqrt(
                math.pow(target[0] - trans[0], 2) +
                math.pow(target[1] - trans[1], 2) +
                math.pow(target[2] - trans[2], 2))
            # Check for convergence
            if diff < TOLERANCE:
                converged = True
                break
            iter += 1
        return converged, joint_angles

    # Move all legs to target end-effector positions
    def move_to(self, e_tar):
        # Check if current joint angles are non-zero (valid data)
        if torch.all(self.q_cur != 0):
            converged = False
            diff = 100
            iter = 0
            trans = torch.ones(6, 3, dtype=torch.float32, device='cpu')
            joint_angles = self.q_cur
            # Compute forward kinematics for current joint angles
            self.kin.ForwardKin(self.q_cur, trans)
            # Compute damped inverse Jacobian
            damp_inv_jac_env = self.kin.DampInvJac(self.q_cur)
            while iter < MAXITER:
                # Compute position error
                pos_err = (e_tar - trans).unsqueeze(-1)  # Shape: (6, 3, 1)
                # Calculate joint velocity
                q_dot = damp_inv_jac_env @ pos_err
                coeff = 1
                # Update joint angles
                joint_angles += coeff * q_dot.squeeze(-1)
                # Recalculate forward kinematics and Jacobian
                self.kin.ForwardKin(joint_angles, trans)
                damp_inv_jac_env = self.kin.DampInvJac(joint_angles)
                # Compute total error
                tensor_1d = (e_tar - trans).flatten()
                sum_squares = (tensor_1d ** 2).sum()
                diff = torch.sqrt(sum_squares).item()
                # Check for convergence
                if diff < TOLERANCE:
                    converged = True
                    print("converged!!")
                    break
                iter += 1
            # Publish desired joint angles for each leg
            msg = Float64MultiArray()
            msg.data = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
            for i in range(6):
                msg.data[1] = joint_angles[i, 0]
                msg.data[2] = joint_angles[i, 1]
                msg.data[3] = joint_angles[i, 2]
                self.q_des_pub_list[i].publish(msg)
        else:
            # If no valid joint data, use reset angles
            msg = Float64MultiArray()
            msg.data = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
            print('no angle data')
            for i in range(6):
                msg.data[1] = self.reset_angles[i, 0]
                msg.data[2] = self.reset_angles[i, 1]
                msg.data[3] = self.reset_angles[i, 2]
                self.q_des_pub_list[i].publish(msg)

    # Update operating legs and gait parameters
    def defect_legs(self, operating_legs, target_phase, mu):
        self.tg.operating_legs = operating_legs
        self.tg.mu = mu
        # Combine operating legs and phase data for publishing
        combined_data = operating_legs + target_phase
        msg = Float32MultiArray()
        msg.data = combined_data
        # Define message layout for ROS
        dim1 = MultiArrayDimension(label="array1", size=len(operating_legs), stride=2)
        dim2 = MultiArrayDimension(label="array2", size=len(target_phase), stride=1)
        msg.layout.dim = [dim1, dim2]
        self.leg_pub.publish(msg)

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('leg_control')
    hz = 50  # Control frequency (Hz)
    omega = 2 * np.pi  # Angular velocity (rad/s)
    # Initialize trajectory generator
    tg = TrajGenerator(omega_0=omega, Hz=hz)
    # Initialize leg controller
    leg_controller = LegController()
    step = 0
    r = rospy.Rate(hz)
    # Open CSV file for logging gait data
    cwd = os.getcwd()
    csvfile = open(cwd + '/data/' + 'gait_all.csv', 'w')
    writer = csv.writer(csvfile)
    tg.mu = 0.5  # Initial duty factor
    start_time = time.time()
    # Main control loop
    while not rospy.is_shutdown():
        loop_start_time = time.time()
        # Skip if operating legs and phase data lengths mismatch
        if len(tg.operating_legs) != len(tg.phase):
            continue
        # Generate target end-effector positions
        e_tar = tg.leg_pose_from_phase(tg.phase)
        # Move legs to target positions
        leg_controller.move_to(e_tar)
        step += 1
        # Simulate defective legs at specific steps
        if 800 < step < 1400:
            arr1 = [0, 2, 3, 4, 5]  # Operating legs
            arr2 = [0, 1/5, 2/5, 3/5, 4/5]  # Desired phase lags for five legs
            leg_controller.defect_legs(arr1, arr2, 0.8)
        elif 1400 <= step < 1900:
            arr1 = [0, 2, 3, 5]  # Operating legs
            arr2 = [0, 1/2, 1/2, 0]  # Desired phase lags for four legs
            leg_controller.defect_legs(arr1, arr2, 0.8)
        elif step >= 1900:
            break
        # Log gait data
        print('time: ', time.time() - start_time, ' gait: ', leg_controller.all_gait)
        writer.writerow([time.time() - start_time] + list(leg_controller.all_gait))
        print('step: ', step)
        r.sleep()