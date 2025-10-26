import torch  
import torch_geometric  
import sys, os  
import numpy as np  
import rospy, time 
import math  # 
from std_msgs.msg import Float32MultiArray  # Import ROS message type for publishing gait data

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path

# Import custom modules: GCPG-Polic, CPG environment, utility functions
from agent.networks import Policy
from environment.env import CPGEnv
from utils import rearrange_state_vector_hopf
import matplotlib.pyplot as plt  
from utils import generate_edge_idx  # functin for calculate adjacency matrix for a graph structure
from argparse import ArgumentParser  # Import ArgumentParser for command-line arguments


# Set device to GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Load pre-trained GCPG model checkpoint
checkpoint = torch.load(parent_dir+'/model_params/model-8-64.pt', weights_only=True)



# Function to calculate phase vector from oscillator states
def cal_phase(x, y):
    """
    Calculate the phase vector \in [0,1]^N to as the input of the leg trajectory generator
    Input Args: 
        x: vector (numpy array) of first state of each oscillation unit
        y: vector (numpy array) of second state of each oscillation unit   
    """
    # Use np.arctan2 for vectorized quadrant-aware angle calculation in (-pi, pi]
    theta = np.arctan2(y, x)
    
    # Shift negative angles to [0, 2pi) for consistency with the intended range
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    
    # Compute phase in [0, 1)
    phase = theta / (2 * np.pi)
    
    # Handle cases where x == 0 and y == 0 (undefined phase), set to np.nan to match original intent of None for scalars
    phase = np.where((x == 0) & (y == 0), np.nan, phase)
    
    return phase



# Main execution block
if __name__ == '__main__':
    # Set up argument parser for command-line options
    parser = ArgumentParser()
    parser.add_argument('--cell_num', type=int, default=6)  # Argument for number of oscillator cells

    args = parser.parse_args()  # Parse arguments
    params = vars(args)  # Convert arguments to dictionary

    cell_num = params['cell_num']  # Retrieve number of cells

    # Initialize ROS node for gait generation
    rospy.init_node('gait_generator')



    
    cate = np.arange(0,1,2/cell_num)

    # Caterpillar gait for this robot (a moving wave from tail to head)
    gait = np.concatenate((cate,cate))  # Concatenate to form full gait array
    

    # Number of attention heads and the feature space dimension used in policy
    heads = 8  
    fd = 64  

    # Initialize GCPG policy network
    model = Policy(state_dim=16+8, action_dim=8, heads=heads, feature_dim=fd).to(device)  # Create Policy model and move to device

    # Load model checkpoint 
    cwd = os.getcwd()  # Get current working directory (unused)
    model.load_state_dict(checkpoint['policy_state_dict'])  # Load saved policy state
    model.eval()  # Set model to evaluation mode

    hz = 100  # Simulation frequency (Hz)
    env = CPGEnv(cell_nums=cell_num,env_length=500,hz=hz)  # Initialize CPG environment
    env.omega = np.pi*2  # Set angular frequency for oscillators
    edge_index = generate_edge_idx(cell_num).to(device=device)  # Generate edge indices for graph and move to device


    env.desired_lag = gait  # Set desired phase lags for gait

    # Prepare initial state: concatenate oscillator positions and encoded desired lags
    env.z_mat = np.zeros((cell_num,2))  # Initialize oscillator state matrix
    env.z_mat[0,0] = 0.5  # Set initial value for first oscillator
    obs = env.z_mat.ravel()  # Flatten oscillator states
    env.relative_lags = env.cal_relative_lags(env.desired_lag)  # Calculate relative lags
    rl_encoding = env.encoding_angle(env.desired_lag)  # Encode desired lags as angles
    state = np.concatenate((obs,rl_encoding.ravel()))  # Combine observations and encodings into state

    # ROS relevant
    gait_pub = rospy.Publisher('/gait', Float32MultiArray, queue_size=10)  # Publisher for gait phase data


    r = rospy.Rate(hz)  # Rate controller for loop frequency


    duration = 0  # Initialize duration
    start_time = time.time()  # Record start time

    phase_list = []  # List to store phases (unused)

    rospy.loginfo('Gait Generator in on')  # Log message indicating start

    # Main loop: runs until ROS shutdown
    while not rospy.is_shutdown():
        loop_start_time = time.time()  # Start time for loop iteration
        duration = time.time()-start_time  # Update total duration

        # Rearrange state for GNN input and move to device
        gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)



        # Disable gradient computation (inference only) and get the action based on model and states
        with torch.no_grad():
            action = model(gnn_x, edge_index)  # Get action from policy
            action.clamp_(-1, 1)  # Clamp action values to [-1, 1]
            action = action.squeeze().cpu().numpy()  # Convert to NumPy array
            action = action  # (Redundant assignment)
    
        nextstate = env.step_env(action)  # Step the environment with action
        state = nextstate  # Update state

        # Calculate phase vector from current oscillator states
        phase_vec = cal_phase(env.z_mat[:,0],env.z_mat[:,1])

        # Prepare and publish ROS message with phase data
        msg = Float32MultiArray()
        msg.data = phase_vec
        gait_pub.publish(msg)


        r.sleep()  # Sleep to maintain loop rate
