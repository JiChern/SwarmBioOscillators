import pybullet as p  
import time  
import numpy as np  
from cent_env import CentipedeEnv  
import rospy  
from argparse import ArgumentParser  
import PyKDL as kdl  



# Function to calculate desired foot-tip position based on side, phase, and duty cycle (mu)
def calculate_x_des(side, phase, mu):
    # Set nominal leg pose based on side (right or left)
    if side == 'right':
        leg_norm_pose = np.array([0.71, 0, 0.48])
        side_factor = 1  # Right is outer for right turn
    else:
        leg_norm_pose = np.array([0.71, 0, -0.48])
        side_factor = -1  # Left is outer for left turn

    # Extract nominal positions
    x_nom = leg_norm_pose[0]
    y_nom = leg_norm_pose[1]
    z_nom = leg_norm_pose[2]

    # Adjust amplitude based on turn
    # a_adj = (-A/(N-1))*seg + A + A/(N-1)
    # A_side = a_adj * (1 + turn_rate * side_factor) 
    # A_side = A * (1 + turn_rate * side_factor)  # e.g., for turn_rate=0.2, right A increases, left decreases  # (Commented out: alternative amplitude adjustments)
    A = 0.3  # Amplitude for y-motion
    H = 0.2  # Height for swing lift

    A_side = A  # Side-specific amplitude (currently fixed)

    # Calculate desired position based on phase
    if phase < mu:
        # Stance: linear backward motion in y (propulsion)
        prog = phase / mu  # Progress through stance phase
        y_des = A_side - 2 * A_side * prog  # From +A_side to -A_side
        x_des = x_nom  # No lift in x during stance
    else:
        # Swing: forward reset in y, with lift
        prog = (phase - mu) / (1 - mu)  # Progress through swing phase
        y_des = -A_side + 2 * A_side * prog  # From -A_side to +A_side
        x_des = x_nom - H * np.sin(np.pi * prog)  # Lift (x decreases) using sine for smooth trajectory

    z_des = z_nom  # Z remains nominal
    return np.array([x_des, y_nom + y_des, z_des])  # Return desired [x, y, z]


# Main execution block
if __name__ == "__main__":
    # Set up argument parser for command-line options
    parser = ArgumentParser()
    parser.add_argument('--seg_num', type=int, default=6)  # Argument for number of body segments

    args = parser.parse_args()  # Parse arguments
    params = vars(args)  # Convert arguments to dictionary
    
    # Initialize ROS node for the centipede simulation
    rospy.init_node('cent_sim')


    seg_num = params['seg_num']  # Retrieve number of segments
    
    # Init the environment
    env = CentipedeEnv(N=seg_num)  # Create CentipedeEnv instance with specified segments
    env.set_env()  # Set up PyBullet environment (load URDF, set gravity, etc.)
    env.set_solvers()  # Set up KDL chains, FK, and Jacobian solvers for legs

    hz = 100  # Simulation frequency (Hz)
    r = rospy.Rate(hz)  # Rate controller for loop frequency

    dt = 1.0 / hz  # Time step for 100 Hz loop  
    K = np.diag([1, 1, 1])*50  # Proportional gain for position error of tip-pose's Cartesian controller(tune as needed)  # Gain matrix for PD control

    start_t = time.time()  # Record start time (unused in code)
    # Main simulation loop: runs until ROS shutdown
    while not rospy.is_shutdown():
        loop_start_time = time.time()  # Start time for loop iteration

        # Spring parameters (tune these; start low)
        rest_angle = 0.0  # Equilibrium position (radians, e.g., 0 for straight)
        k = 0.0001          # Stiffness (higher = stiffer spring)
        d = 0.0           # Damping (higher = more resistance to motion)
        max_torque = 50.0 # Optional cap to prevent excessive forces

        # Initial setup: Make body joints passive
        for joint_index in env.passive_joint_indices:  # Loop through passive (body) joints
            p.setJointMotorControl2(
                bodyUniqueId=env.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )  # Set to velocity control with zero force (passive)

        # Apply spring-damper torques to passive body joints
        for joint_index in env.passive_joint_indices:
            # Get current joint state
            joint_state = p.getJointState(env.robot_id, joint_index)
            current_angle = joint_state[0]  # Position
            angular_velocity = joint_state[1]  # Velocity

            # Normalize angle error for bofy-like joints (optional but recommended for continuous rotation)  # (Typo: 'bofy' likely 'body')
            angle_error = current_angle - rest_angle
            angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi  # To [-pi, pi]  # Normalize error to [-pi, pi]

            # Compute spring-damper torque
            torque = -k * angle_error - d * angular_velocity

            # Optional: Clamp torque for stability
            torque = np.clip(torque, -max_torque, max_torque)

            # Apply torque
            p.setJointMotorControl2(
                bodyUniqueId=env.robot_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=torque  # For revolute joints, 'force' is torque
            )

        # leg motion planning based on gait signals
        for idx, leg in enumerate(env.legs):  # Loop through each leg chain
            leg_phase = env.gait[idx]  # Get phase for this leg from gait data
            
            # Get current joint angles for this leg
            current_q = kdl.JntArray(3)
            for i, pb_j in enumerate(env.pb_joint_indicies[idx]):
                current_q[i] = p.getJointState(env.robot_id, pb_j)[0]
            
            # Compute current end-effector (foot-tip) position using forward kinematics
            end_frame = kdl.Frame()
            status = env.fk_solvers[idx].JntToCart(current_q, end_frame)
            x_curr = np.array([end_frame.p.x(), end_frame.p.y(), end_frame.p.z()])  # Extract Cartesian position

            # Determine this leg is at which side
            if idx < seg_num:
                side = 'right'  # First half of legs are right-side
            else:
                side = 'left'  # Second half are left-side
            
            # Calculate the desired foot-tip position based on the side and the gait signal of this leg  
            x_des = env.calculate_x_des(side, leg_phase)  # Get desired position

            # Desired velocity dot_x = K * position error
            pos_error = x_des - x_curr  # Compute position error
            dot_x = K @ pos_error   # Compute desired Cartesian velocity

            # Compute Jacobian J at current q (3x3)
            jac = kdl.Jacobian(3)
            env.jaco_solvers[idx].JntToJac(current_q, jac)  # Compute Jacobian for this leg

            # Convert KDL Jacobian to NumPy array
            J = np.zeros((3, 3))
            for row in range(3):
                for col in range(3):
                    J[row, col] = jac[row, col]
            
            # Pseudo-inverse J_pinv (handles singularities better than direct inverse)
            J_pinv = np.linalg.pinv(J)  # Compute pseudo-inverse of Jacobian

            # Compute joint velocities from Cartesian velocity
            q_dot = J_pinv @ dot_x 
            
            # Drive the joints on this leg  
            for i, pb_j in enumerate(env.pb_joint_indicies[idx]):  # Loop through joints in this leg
                p.setJointMotorControl2(
                    bodyUniqueId=env.robot_id,
                    jointIndex=pb_j,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=current_q[i] + q_dot[i]*dt,  # Integrate velocity to get target position
                    force=500,  # Adjust force as needed for your robot (e.g., max joint torque)
                    maxVelocity=10.0  # Optional: limit velocity
                )
        
        # Execute one step of pybullet simulation  
        p.stepSimulation()  # Advance the simulation by one step
        r.sleep()  # Sleep to maintain loop rate
        print('dt: ', time.time()-loop_start_time)  # Print loop execution time for debugging
    