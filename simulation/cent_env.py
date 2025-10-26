from gen_urdf import CentipedeURDFGenerator
import pybullet as p
import pybullet_data
import time
import os
import matplotlib.pyplot as plt
import PyKDL as kdl
import xml.etree.ElementTree as ET
import numpy as np
from std_msgs.msg import Float32MultiArray,Float32
import rospy

# Class for simulating a centipede robot environment using PyBullet, KDL, and ROS integration
class CentipedeEnv(object):
    # Constructor to initialize the environment with a given number of segments
    def __init__(self,N) -> None:
        self.robot_id = None  # ID of the loaded robot in PyBullet
        self.num_segments = N  # Number of body segments in the centipede
        self.num_joints = None  # Total number of joints in the robot
        self.joint_names = []  # List of all joint names
        self.link_names = []  # List of all link names
        self.urdf_path = "centipede_robot.urdf"  # Path to the generated URDF file
        # ROS subscriber for gait data (multi-array of floats)
        self.gait_sub = rospy.Subscriber('/gait', Float32MultiArray, self.gait_cb, queue_size=10)
        # ROS subscriber for turning command (single float)
        self.turning_sub = rospy.Subscriber('/turning', Float32, self.turning_cb, queue_size=10)
        self.gait = None  # Current gait data received from ROS
        self.turning = 0  # Current turning value (range [-1,1])

    # Callback function for gait topic; stores the received gait data
    def gait_cb(self,msg):
        self.gait = msg.data

    # Callback function for turning topic; validates and stores the turning value
    def turning_cb(self,msg):
        if msg.data > 1 or msg.data<-1:
            raise ValueError('The turning should be in range [-1,1]')
        
        self.turning = msg.data
        

    # Initialize the pybullet environment with robot and ground
    def set_env(self):
        cwd = os.getcwd()  # Get current working directory for file paths
        # Placeholder paths for Mesh files (replace with actual paths)
        body_step = cwd+"/mesh/body11.STL"
        hip_step = cwd+"/mesh/hip13.STL"
        thigh_step = cwd+"/mesh/thigh11.STL"
        tibia_step = cwd+"/mesh/tibia11.STL"

        # Create URDF generator with mesh files and scale factor
        generator = CentipedeURDFGenerator(body_step, hip_step, thigh_step, tibia_step, scale_factor=0.01)  # Scale down by 100 times

        # Generate URDF file for the centipede with the specified number of segments
        generator.generate_urdf(N=self.num_segments, output_file=self.urdf_path)  # Generate with 5 segments
    
        # Connect to PyBullet in GUI mode with a white background
        physicsClient = p.connect(p.GUI, options="--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0")
        # clean GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)  # Disable GUI controls for a cleaner view

        # Set starting orientation (rotated 90 degrees around Y-axis)
        start_orientation = p.getQuaternionFromEuler([0, 1.57, 0])
        # Load the generated URDF
        self.robot_id = p.loadURDF(self.urdf_path, [0, 0, 0.5],start_orientation)  # Start position above the plane
        
        # Get the total number of joints
        self.num_joints = p.getNumJoints(self.robot_id)

        self.passive_joint_indices = []  # List to store indices of passive (body) joints

        # Get lists of joint and body names
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id,i)
            self.joint_names.append(joint_info[1].decode('utf-8'))  # Store joint name
            self.link_names.append(joint_info[12].decode('utf-8'))  # Store link name
            if joint_info[1].decode('utf-8').startswith('body_joint'):  # Identify passive body joints
                self.passive_joint_indices.append(i)

        # Change the color of the robot using different colors for body and legs (from Nature palette)
        body_color = [0.518, 0.569, 0.62, 1]  # RGBA for #84919e (grey for body segments)
        leg_color = [0.302, 0.733, 0.835, 1]  # RGBA for #4dbbd5 (cyan for legs)
        p.changeVisualShape(self.robot_id, -1, rgbaColor=body_color) # set the color for the first body sigment
        # Set colors for body segments and leg parts
        for joint_index, link_name in enumerate(self.link_names): # set the color for another body segments and leg segments
            if link_name.startswith('body_'):
                p.changeVisualShape(self.robot_id, joint_index, rgbaColor=body_color)
            elif link_name.startswith('hip_') or link_name.startswith('thigh_') or link_name.startswith('tibia_'):
                p.changeVisualShape(self.robot_id, joint_index, rgbaColor=leg_color)

        # Load the plane
        p.loadURDF("plane/plane1.urdf", [0, 0, 0])  # Load a ground plane

        # Set initial states of the robot
        hip_angle = 0  # Initial angle for hip joints
        thigh_angle = 0.7  # Initial angle for thigh joints
        tibia_angle = 1.2  # Initial angle for tibia joints
        body_angle = 0.0  # Initial angle for body joints

        # Reset all joints to initial angles
        for joint_index, joint_name in enumerate(self.joint_names):


            if joint_name.startswith('hip_joint_'):
                p.resetJointState(self.robot_id, joint_index, hip_angle)
            elif joint_name.startswith('thigh_joint_'):
                p.resetJointState(self.robot_id, joint_index, thigh_angle)
            elif joint_name.startswith('tibia_joint_'):
                p.resetJointState(self.robot_id, joint_index, tibia_angle)
            elif joint_name.startswith('body_joint_'):
                # body_angle += 0.02  # (Commented out: incremental adjustment for body angles)
                p.resetJointState(self.robot_id, joint_index, body_angle)
        
        # Set gravity
        p.setGravity(0, 0, -9.81)  # Set gravity in the negative Z direction

    # Helper to get PyBullet joint index from joint name
    def get_joint_index(self, joint_name):
        # Iterate through all joints to find the matching name
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[1].decode('utf-8') == joint_name:
                return j
        raise ValueError(f"Joint {joint_name} not found")
    
    # Function to build KDL chain from start to end link using XML parsing
    def build_kdl_chain(self, start_link_name, end_link_name):
        # Parse the URDF file as XML
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        segments = []  # List to store KDL segments
        pb_joint_indices = []  # List to store PyBullet joint indices
        current_link = end_link_name  # Start from the end link and traverse backwards
        
        # Traverse from end link to start link, building segments
        while current_link != start_link_name:
            # Find the joint with child link == current_link
            joint_elem = next(j for j in root.findall('joint') if j.find('child').attrib['link'] == current_link)
            joint_name = joint_elem.attrib['name']
            joint_type = joint_elem.attrib['type']
            
            # Determine KDL joint type
            if joint_type == 'revolute':
                axis_elem = joint_elem.find('axis')
                axis_str = axis_elem.attrib.get('xyz', '0 0 0')
                print(axis_str)  # Debug: print axis string
                axis = [float(x) for x in axis_str.split() if x.strip()]  # Skip empty splits

                # Map axis to KDL rotation type
                if axis == [1, 0, 0]:
                    kdl_type = kdl.Joint.RotX
                elif axis == [0, 1, 0]:
                    kdl_type = kdl.Joint.RotY
                elif axis == [0, 0, 1]:
                    kdl_type = kdl.Joint.RotZ
                else:
                    raise ValueError(f"Unsupported axis {axis} for revolute joint")
            elif joint_type == 'fixed':
                kdl_type = kdl.Joint.Fixed
            else:
                raise ValueError(f"Unsupported joint type: {joint_type}")
            
            # Create KDL joint
            kdl_joint = kdl.Joint(joint_name, kdl_type)
            
            # Get origin (transform from parent to child link / joint frame)
            origin_elem = joint_elem.find('origin')

            if origin_elem is None:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
            else:
                try:
                    xyz_str = origin_elem.attrib.get('xyz', '0 0 0')
                    xyz = [float(x) for x in xyz_str.split() if x.strip()]  # Skip empty splits
                    if len(xyz) != 3:
                        raise ValueError("Invalid length")
                except ValueError:
                    xyz = [0.0, 0.0, 0.0]
                
                try:
                    rpy_str = origin_elem.attrib.get('rpy', '0 0 0')
                    rpy = [float(r) for r in rpy_str.split() if r.strip()]
                    if len(rpy) != 3:
                        raise ValueError("Invalid length")
                except ValueError:
                    rpy = [0.0, 0.0, 0.0]
            
            
            print('joint_name: ',joint_name, ' xyz:', xyz, ' rpy: ', rpy)  # Debug: print joint details

            # Create KDL frame from rotation and translation
            rot = kdl.Rotation.RPY(rpy[0], rpy[1], rpy[2])
            trans = kdl.Vector(xyz[0], xyz[1], xyz[2])
            frame = kdl.Frame(rot, trans)

            # Create KDL segment with zero inertia (for kinematics only)
            segment = kdl.Segment(kdl_joint, frame, kdl.RigidBodyInertia(0))
            segments.append(segment)
            
            # Collect PyBullet joint index if not fixed
            if joint_type != 'fixed':
                pb_joint_indices.append(self.get_joint_index(joint_name))
            
            # Move to parent link
            current_link = joint_elem.find('parent').attrib['link']
        
        # Reverse to build chain from root to tip

        # Add a fixed tip frame (e.g., for end-effector offset)
        rot = kdl.Rotation.RPY(0, 0, 0)
        trans = kdl.Vector(0, 0, 0.36)
        frame = kdl.Frame(rot, trans)
        kdl_joint = kdl.Joint('fix_j', kdl.Joint.Fixed)
        # Create segment (f_tip is identity assuming child frame at joint; inertia zero for IK)
        segment = kdl.Segment(kdl_joint, frame, kdl.RigidBodyInertia(0))

        # Build the KDL chain by adding segments in reverse order
        chain = kdl.Chain()

        for seg in reversed(segments):
            chain.addSegment(seg)
        
        chain.addSegment(segment)  # Add the fixed tip segment

        pb_joint_indices = list(reversed(pb_joint_indices))  # Reverse joint indices to match chain order
        
        return chain, pb_joint_indices  # Return the chain and corresponding PyBullet indices
    

    # Function to get current joint angles for the KDL chain from PyBullet
    def get_chain_joint_states(self, joint_indices):
        num_joints = len(joint_indices)
        q = kdl.JntArray(num_joints)  # Create KDL joint array
        # Populate with current joint positions from PyBullet
        for i, pb_j in enumerate(joint_indices):
            q[i] = p.getJointState(self.robot_id, pb_j)[0]  # [0] is position (angle for revolute)
        return q
    
    # Debug function to check and print joint types in a KDL chain
    def check_joint_types(self, chain):
        num_segments = chain.getNrOfSegments()
        print(f"Chain has {num_segments} segments and {chain.getNrOfJoints()} joints (DOFs).")
        
        # Iterate through segments and print joint details
        for i in range(num_segments):
            segment = chain.getSegment(i)
            joint = segment.getJoint()
            joint_name = joint.getName()
            joint_type = joint.getType()
            joint_type_name = joint.getTypeName()
            
            if joint_type == kdl.Joint.RotX:
                print(f"Segment {i} joint '{joint_name}' is RotX ({joint_type_name}).")
            elif joint_type == kdl.Joint.RotY:
                print(f"Segment {i} joint '{joint_name}' is RotY ({joint_type_name}).")
            elif joint_type == kdl.Joint.RotZ:
                print(f"Segment {i} joint '{joint_name}' is RotZ ({joint_type_name}).")
            else:
                print(f"Segment {i} joint '{joint_name}' is other type: {joint_type_name} (enum: {joint_type}).")

    # Set up KDL chains, forward kinematics (FK), and Jacobian solvers for all legs
    def set_solvers(self):
        """
        The legs are numbered from top-right to bottom leg
        """
        self.legs =  [None for i in range(self.num_segments*2)]  # chain for each legs, note that chain cannot be intermediate variable as the solver must access it
        self.fk_solvers = [None for i in range(self.num_segments*2)]  # fk solvers for each legs
        self.jaco_solvers = [None for i in range(self.num_segments*2)] # jacobian solvers for each legs
        self.pb_joint_indicies = [None for i in range(self.num_segments*2)] # pybullet joint indecies for each legs

        # Initialize leg0 and leg 0+N from the first robot segment
        self.legs[0], self.pb_joint_indicies[0] = self.build_kdl_chain(start_link_name='body_1', end_link_name='tibia_right_1')
        self.legs[0+self.num_segments], self.pb_joint_indicies[0+self.num_segments] = self.build_kdl_chain(start_link_name='body_1', end_link_name='tibia_left_1')


        # Create FK solver for right and left legs of first segment
        self.fk_solvers[0] = kdl.ChainFkSolverPos_recursive(self.legs[0])
        self.fk_solvers[0+self.num_segments] = kdl.ChainFkSolverPos_recursive(self.legs[0+self.num_segments])

        # Create Jacobian solver for right and left legs of first segment
        self.jaco_solvers[0] = kdl.ChainJntToJacSolver(self.legs[0])
        self.jaco_solvers[0+self.num_segments] = kdl.ChainJntToJacSolver(self.legs[0+self.num_segments])

        # Set the solvers and pybullet joint indicies for the legs on another bodies 
        # Loop through remaining segments (2 to N)
        for i in range(2, self.num_segments+1):

            # Build chains for right and left legs of current segment
            self.legs[i-1], self.pb_joint_indicies[i-1] = self.build_kdl_chain(start_link_name='body_'+str(i), end_link_name='tibia_right_'+str(i))
            self.legs[i-1+self.num_segments], self.pb_joint_indicies[i-1+self.num_segments] = self.build_kdl_chain(start_link_name='body_'+str(i), end_link_name='tibia_left_'+str(i))

            # Create FK solvers for current legs
            self.fk_solvers[i-1] = kdl.ChainFkSolverPos_recursive(self.legs[i-1])
            self.fk_solvers[i-1+self.num_segments] = kdl.ChainFkSolverPos_recursive(self.legs[i-1+self.num_segments])

            # Create Jacobian solvers for current legs
            self.jaco_solvers[i-1] = kdl.ChainJntToJacSolver(self.legs[i-1])
            self.jaco_solvers[i-1+self.num_segments] = kdl.ChainJntToJacSolver(self.legs[i-1+self.num_segments])

    # Foot-tip trajectory planning according to its side, gait signal and duty cycle
    def calculate_x_des(self, side, phase):
        # Mean duty cycle for stance/swing phases
        mu_mean = 0.7
        mu = mu_mean
        # Adjustment to duty cycle based on turning (for steering)
        mu_div = self.turning*0.2

        # Set nominal pose and adjust duty cycle based on side (right or left)
        if side == 'right':
            leg_norm_pose = np.array([0.71, 0, 0.48])
            mu = mu_mean + mu_div
        else:
            leg_norm_pose = np.array([0.71, 0, -0.48])
            mu = mu_mean - mu_div

        # Extract nominal positions
        x_nom = leg_norm_pose[0]
        y_nom = leg_norm_pose[1]
        z_nom = leg_norm_pose[2]

        A = 0.3  # Amplitude for y-motion
        H = 0.2  # Height for swing lift

        A_side = A  # Side-specific amplitude (currently same as A)

        # Calculate desired position based on phase
        if phase < mu:
            # Stance: linear backward motion in y (propulsion)
            prog = phase / mu
            y_des = A_side - 2 * A_side * prog  # From +A_side to -A_side
            x_des = x_nom
        else:
            # Swing: forward reset in y, with lift
            prog = (phase - mu) / (1 - mu)
            y_des = -A_side + 2 * A_side * prog  # From -A_side to +A_side
            x_des = x_nom - H * np.sin(np.pi * prog)  # Lift (x decreases)

        z_des = z_nom  # Z remains nominal
        return np.array([x_des, y_nom + y_des, z_des])  # Return desired [x, y, z]