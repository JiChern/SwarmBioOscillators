#!/usr/bin/env python3
# This is the robot-side code, may not be executed on other robots.

import math

# Configuration class for a hexapod robot, defining initial state, control parameters, and physical properties
class HexCfg:
    def __init__(self, yaml_file: str):
        # Render mode for visualization (default: human-readable display)
        self.render_mode = "human"
        # Alternative render mode (commented out, likely for non-visual simulation)
        # self.render_mode = "headless"
        # Name of the foot end-effector link
        self.foot_end_name = "toe"
        # Maximum suction force for foot attachment (e.g., for climbing or gripping)
        self.max_suck_force = 100.0

    # Nested class to define initial state of the robot
    class init_state:
        # Initial position of the robot's body (x, y, z) in meters
        pos = [0, 0, 0.3]
        # Initial rotation as a quaternion (x, y, z, w)
        rot = [0.0, 0.0, 0.0, 1.0]
        # Initial linear velocity (x, y, z) in m/s
        lin_vel = [0.0, 0.0, 0.0]
        # Initial angular velocity (x, y, z) in rad/s
        ang_vel = [0.0, 0.0, 0.0]
        # List of leg identifiers (left back, left front, left middle, right back, right front, right middle)
        _tao = ['lb', 'lf', 'lm', 'rb', 'rf', 'rm']
        # List of joint names for each leg
        _q_name = ['thigh', 'knee', 'ankle', 'foot', 'ball1', 'ball2', 'suck']
        # Default joint angle initialization
        _joint = 0.0
        # Dictionary to store default joint angles for each leg and joint
        default_joint_angles = {}
        for t in _tao:
            for qn in _q_name:
                if qn == 'thigh':
                    # Set thigh angles based on leg position
                    if t == 'rf' or t == 'lb':
                        _joint = 0.35  # Positive angle for right front and left back
                    elif t == 'lf' or t == 'rb':
                        _joint = -0.35  # Negative angle for left front and right back
                    else:
                        _joint = 0.0  # Neutral for middle legs
                elif qn == 'knee':
                    _joint = 0.7  # Fixed knee angle
                elif qn == 'ankle':
                    _joint = -2.14  # Fixed ankle angle
                elif qn == 'foot':
                    _joint = -0.13  # Fixed foot angle
                else:
                    _joint = 0.0  # Zero for ball1, ball2, suck joints
                # Store joint angle in dictionary with key format 'j_<leg>_<joint>'
                default_joint_angles['j_' + t + '_' + qn] = _joint
        # Commented-out code to align foot parallel to robot base
        # default_joint_angles['foot'] = -(default_joint_angles['knee'] + default_joint_angles['ankle']) - math.pi / 2.0
        # Default position of the foot end-effector (x, y, z) in meters
        foot_end_pos = [0.19, 0, -0.08]

    # Nested class for control parameters
    class control:
        # Proportional gain for motor control
        motor_kp = 100.0
        # Derivative gain for motor control
        motor_kd = 1.0
        # Torque limits for motors (min, max) in Nm
        motor_torque_limits = [-27.0, 27.0]
        # Proportional gain for servo control
        servo_kp = 20.0
        # Derivative gain for servo control
        servo_kd = 0.1
        # Torque limits for servos (min, max) in Nm
        servo_torque_limits = [-5.0, 5.0]

    # Nested class for link lengths of the robot's legs
    class link:
        # Length of first link (thigh) in meters
        l1 = 0.072
        # Length of second link (knee to ankle) in meters
        l2 = 0.13
        # Length of third link (ankle to toe) in meters
        l3 = 0.17

    # Nested class for maximum velocities and adjustments
    class max_vec:
        # Maximum linear velocity in x-direction (m/s)
        x = 0.4
        # Maximum linear velocity in y-direction (m/s)
        y = 0.5
        # Maximum linear velocity in z-direction (m/s)
        z = 0.4
        # Adjustment factor for angular velocity (rad/s)
        omega_adj = 0.15
        # Maximum angular velocity for movement (rad/s)
        omega_move = 1.2

    # Nested class for body dimensions
    class body_shape:
        # Body length in x-direction (m)
        x = 0.1
        # Body width in y-direction (m)
        y = 0.22