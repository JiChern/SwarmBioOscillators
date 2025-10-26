import xml.etree.ElementTree as ET
import pybullet as p
import pybullet_data
import time
import os

# Class to generate URDF files for a centipede-like robot model
class CentipedeURDFGenerator:
    """
    A class to generate URDF files for a centipede robot.
    Each segment consists of a body with two legs: one left and one right.
    Each leg is composed of hip, thigh, and tibia in series.
    Segments are connected in series to form the robot.
    """

    # Constructor to initialize the generator with mesh file paths and scaling
    def __init__(self, body_step, hip_step, thigh_step, tibia_step, scale_factor=1.0):
        """
        Initialize the generator with paths to mesh files.

        Args:
        - body_step: Path to body mesh file.
        - hip_step: Path to hip mesh file.
        - thigh_step: Path to thigh mesh file.
        - tibia_step: Path to tibia mesh file.
        - scale_factor: Factor to scale linear dimensions (positions, mesh scales) by. Default 1.0.
        """
        self.body_step = body_step  # Path to body mesh
        self.hip_step = hip_step  # Path to hip mesh
        self.thigh_step = thigh_step  # Path to thigh mesh
        self.tibia_step = tibia_step  # Path to tibia mesh
        self.scale_factor = scale_factor  # Scaling factor for dimensions

    # Helper method to scale XYZ coordinates
    def _scale_xyz(self, xyz_str):
        """
        Scale an xyz string by the scale_factor.

        Args:
        - xyz_str: Space-separated xyz string, e.g., '1 0 0'

        Returns:
        - Scaled space-separated xyz string.
        """
        xyz = [float(val) * self.scale_factor for val in xyz_str.split()]  # Scale each component
        return ' '.join(map(str, xyz))  # Return as space-separated string

    # Helper method to scale inertia values (scales with size^5 assuming constant density)
    def _scale_inertia(self, inertia_str, mass):
        """
        Scale inertia values. Assuming density constant, inertia scales with mass * size^2,
        but since mass scales with size^3, overall inertia scales with size^5.

        Args:
        - inertia_str: Space-separated ixx ixy ixz iyy iyz izz (but only ixx iyy izz used).
        - mass: Original mass (used to compute scaled mass implicitly via scale).

        Returns:
        - Scaled inertia string.
        """
        scale_inertia = self.scale_factor ** 5  # size^5 for inertia if density constant
        inertia = [float(val) * scale_inertia for val in inertia_str.split()]  # Scale each inertia component
        return ' '.join(map(str, inertia))  # Return as space-separated string

    # Helper method to scale mass (scales with size^3 for volume)
    def _scale_mass(self, mass):
        """
        Scale mass by size^3 (volume).

        Args:
        - mass: Original mass string.

        Returns:
        - Scaled mass string.
        """
        scale_mass = self.scale_factor ** 3  # Volume scaling
        return str(float(mass) * scale_mass)  # Return scaled mass as string

    # Create a link element with mesh geometry for visual and collision
    def _create_link(self, name, mesh_file, mass='1.0', inertia='1 0 0 1 0 1', origin_xyz='0 0 0', origin_rpy='0 0 0', visual=True, collision=True):
        """
        Create a link element with mesh.

        Args:
        - name: Name of the link.
        - mesh_file: Path to the mesh file.
        - mass: Mass of the link.
        - inertia: Inertia values as space-separated string (ixx ixy ixz iyy iyz izz, but only ixx iyy izz used).
        - origin_xyz: XYZ origin.
        - origin_rpy: RPY origin.
        - visual: Whether to include visual element.
        - collision: Whether to include collision element.

        Returns:
        - ET.Element: The link element.
        """
        scaled_mass = self._scale_mass(mass)  # Scale mass
        scaled_inertia = self._scale_inertia(inertia, mass)  # Scale inertia
        scaled_origin_xyz = self._scale_xyz(origin_xyz)  # Scale origin XYZ
        scale_str = ' '.join([str(self.scale_factor)] * 3)  # Uniform scale for mesh

        link = ET.Element('link', name=name)  # Create link element
        
        # Add inertial properties
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'origin', xyz=scaled_origin_xyz, rpy=origin_rpy)
        ET.SubElement(inertial, 'mass', value=scaled_mass)
        ET.SubElement(inertial, 'inertia', ixx=scaled_inertia.split()[0], ixy='0', ixz='0', iyy=scaled_inertia.split()[3], iyz='0', izz=scaled_inertia.split()[5])
        
        # Add visual element if enabled
        if visual:
            visual_elem = ET.SubElement(link, 'visual')
            ET.SubElement(visual_elem, 'origin', xyz=scaled_origin_xyz, rpy=origin_rpy)
            geometry = ET.SubElement(visual_elem, 'geometry')
            ET.SubElement(geometry, 'mesh', filename=mesh_file, scale=scale_str)
            ET.SubElement(visual_elem, 'material', name='silver')
        
        # Add collision element if enabled
        if collision:
            collision_elem = ET.SubElement(link, 'collision')
            ET.SubElement(collision_elem, 'origin', xyz=scaled_origin_xyz, rpy=origin_rpy)
            geometry = ET.SubElement(collision_elem, 'geometry')
            ET.SubElement(geometry, 'mesh', filename=mesh_file, scale=scale_str)
        
        return link


    # Create a joint element between two links
    def _create_joint(self, name, type, parent, child, origin_xyz, origin_rpy, axis='0 0 1', limit_effort='100', limit_velocity='10', limit='-3.14 3.14'):
        """
        Create a joint element.

        Args:
        - name: Name of the joint.
        - type: Type of the joint (e.g., 'revolute', 'fixed').
        - parent: Parent link name.
        - child: Child link name.
        - origin_xyz: XYZ origin.
        - origin_rpy: RPY origin.
        - axis: Axis for non-fixed joints.
        - limit_effort: Effort limit.
        - limit_velocity: Velocity limit.
        - limit: Lower and upper limits as space-separated string.

        Returns:
        - ET.Element: The joint element.
        """
        scaled_origin_xyz = self._scale_xyz(origin_xyz)  # Scale origin XYZ
        joint = ET.Element('joint', name=name, type=type)  # Create joint element
        ET.SubElement(joint, 'parent', link=parent)  # Set parent link
        ET.SubElement(joint, 'child', link=child)  # Set child link
        ET.SubElement(joint, 'origin', xyz=scaled_origin_xyz, rpy=origin_rpy)  # Set origin
        if type != 'fixed':  # For non-fixed joints, add axis and limits
            ET.SubElement(joint, 'axis', xyz=axis)
            limit_elem = ET.SubElement(joint, 'limit', effort=limit_effort, velocity=limit_velocity)
            limit_values = limit.split()
            limit_elem.set('lower', limit_values[0])
            limit_elem.set('upper', limit_values[1])
        return joint

    # Main method to generate the URDF file for N segments
    def generate_urdf(self, N, output_file='centipede_robot.urdf'):
        """
        Generate a URDF file for the centipede robot with N segments.

        Args:
        - N: Number of body segments.
        - output_file: Path to save the URDF file.
        """
        # Create root element
        robot = ET.Element('robot', name='centipede_robot')  # Root robot element

        # Define materials (optional, for visualization)
        material_silver = ET.SubElement(robot, 'material', name='silver')  # Silver material for meshes
        ET.SubElement(material_silver, 'color', rgba='0.7 0.7 0.7 1')

        # Create body links and connect them
        previous_body = None  # Track previous body for connection
        for i in range(1, N + 1):  # Loop through each segment
            body_name = f'body_{i}'  # Name for current body
            body_link = self._create_link(body_name, self.body_step, mass='5.0', inertia='1 0 0 1 0 1')  # Create body link
            robot.append(body_link)  # Add to robot
            
            if previous_body:  # If not the first body, connect with a joint
                # Connect to previous body with a revolute joint (assuming yaw rotation)
                joint_name = f'body_joint_{i-1}_to_{i}'
                body_joint = self._create_joint(joint_name, 'revolute', previous_body, body_name, '0 -50 0', '0 0 0', axis='1 0 0', limit='-0.52 0.52')  # Assume bodies are connected along x-axis
                # body_joint = self._create_joint(joint_name, 'revolute', previous_body, body_name, '0 -50 0', '0 0 0', axis='0 0 1', limit='-0.52 0.52')  # Assume bodies are connected along x-axis  # (Alternative axis commented out)

                robot.append(body_joint)  # Add joint to robot
            
            previous_body = body_name  # Update previous body
            
            # Add legs to this body: left and right
            leg_sides = ['left', 'right']  # Sides for legs

            # Define hip positions and rotations for left and right legs
            leg_positions_hip = {
                'left': '0 0 -30',   # Example offsets, adjust as needed
                'right': '0 0 30'
            }


            leg_rotations_hip = {
                'left': '3.14 0 0',   # Example offsets, adjust as needed
                'right': '0 0 0'
            }
            
            for side in leg_sides:  # Loop through left and right legs
                # Hip
                # if side == 'left':
                #     continue  # (Commented out: skip left leg for testing)
                hip_name = f'hip_{side}_{i}'  # Hip name
                hip_link = self._create_link(hip_name, self.hip_step, mass='1.0', inertia='0.1 0 0 0.1 0 0.1')  # Create hip link
                robot.append(hip_link)  # Add to robot
                hip_joint_name = f'hip_joint_{side}_{i}'  # Hip joint name
                hip_joint = self._create_joint(hip_joint_name, 'revolute', body_name, hip_name, leg_positions_hip[side], leg_rotations_hip[side], axis='1 0 0')  # Sideways rotation
                robot.append(hip_joint)  # Add joint
                
                # # # Thigh
                thigh_name = f'thigh_{side}_{i}'  # Thigh name
                thigh_link = self._create_link(thigh_name, self.thigh_step, mass='0.8', inertia='0.08 0 0 0.08 0 0.08')  # Create thigh link
                robot.append(thigh_link)  # Add to robot
                thigh_joint_name = f'thigh_joint_{side}_{i}'  # Thigh joint name
                thigh_joint = self._create_joint(thigh_joint_name, 'revolute', hip_name, thigh_name, '-20 0 36', '0 0 0', axis='0 1 0')  # Forward rotation
                robot.append(thigh_joint)  # Add joint
                
                # # # Tibia
                tibia_name = f'tibia_{side}_{i}'  # Tibia name
                tibia_link = self._create_link(tibia_name, self.tibia_step, mass='0.5', inertia='0.05 0 0 0.05 0 0.05')  # Create tibia link
                robot.append(tibia_link)  # Add to robot
                tibia_joint_name = f'tibia_joint_{side}_{i}'  # Tibia joint name
                tibia_joint = self._create_joint(tibia_joint_name, 'revolute', thigh_name, tibia_name, '0 0 31', '0 0 0', axis='0 1 0')  # Forward rotation
                robot.append(tibia_joint)  # Add joint


        # Write to file
        tree = ET.ElementTree(robot)  # Create ElementTree
        ET.indent(tree, space='  ', level=0)  # Pretty print (requires Python 3.9+)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)  # Write URDF to file
        print(f'URDF file generated: {output_file}')  # Confirmation message


# Example usage to generate URDF and visualize with PyBullet
if __name__ == "__main__":
    cwd = os.getcwd()  # Get current working directory
    # Placeholder paths for mesh files (replace with actual paths)
    body_step = cwd+"/mesh/body11.STL"
    hip_step = cwd+"/mesh/hip13.STL"
    thigh_step = cwd+"/mesh/thigh11.STL"
    tibia_step = cwd+"/mesh/tibia11.STL"
    
    generator = CentipedeURDFGenerator(body_step, hip_step, thigh_step, tibia_step, scale_factor=0.01)  # Scale down by 100 times
    urdf_file = "centipede_robot.urdf"  # Output URDF file name
    generator.generate_urdf(N=2, output_file=urdf_file)  # Generate with 5 segments  # (Note: N=2, but comment says 5; likely typo)
    
    # Now visualize with PyBullet
    physicsClient = p.connect(p.GUI)  # Connect to PyBullet GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For plane.urdf
    
    # Load the plane
    p.loadURDF("plane.urdf", [0, 0, 0])  # Load ground plane
    
    start_orientation = p.getQuaternionFromEuler([0, 1.57, 0])  # Set starting orientation (90 degrees around Y)
    # Load the generated URDF
    robot_id = p.loadURDF(urdf_file, [0, 0, 0.5],start_orientation)  # Start position above the plane
    
    # Set gravity
    p.setGravity(0, 0, -9.81)  # Negative Z gravity
    
    # Run simulation for visualization
    for _ in range(10000):  # Run for a while
        p.stepSimulation()  # Step the simulation
        time.sleep(1./240.)  # Slow down for visualization
    
