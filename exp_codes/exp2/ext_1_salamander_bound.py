import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (Models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path
import csv
import time
import matplotlib.pyplot as plt
from other_models.salamander import Salamander # import diffusive CPG model
from utils import phase_distance


# Set deisred phase configuration
desired_lag = np.array([0, np.pi, 0, np.pi])
# desired_lag = np.array([0, np.pi, 3*np.pi/2, np.pi/2])

# Set deisred lag corresponding to desired phase configuration for phase distance calculation
desired_lag_norm = desired_lag/2/np.pi

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
    phase = theta
    
    # Handle cases where x == 0 and y == 0 (undefined phase), set to np.nan to match original intent of None for scalars
    phase = np.where((x == 0) & (y == 0), np.nan, phase)
    
    return phase


if __name__ == "__main__":

     # Set-up Salamander CPG env
    hz = 100
    dt = 1/hz
    cell_num = 4
    cpg = Salamander(omega=2*np.pi, cell_num=4, hz=100, desired_phase_diffs=desired_lag)

    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/salamander_converge_time_bound.csv',"w")
    writer = csv.writer(f)

    # Load the initial condition shared for all CPG models
    init_conds  = np.loadtxt(cwd+'/data/init_cond.csv', delimiter=',', dtype=float)

    try:
        # Get converge times over 1000 random trials
        for i in range(1000):
            # Reset the initial states around a circle with radius of 1
            z_x = np.array(init_conds[i,0:4])
            z_y = np.array(init_conds[i,4:8])
            cpg.theta = cal_phase(z_x,z_y)  # Random initial phases
            cpg.r = np.ones(cpg.N)  # Initial amplitudes set to 1
            cpg.dr = np.zeros(cpg.N)  # Initial amplitude derivatives set to 0
            cpg._update_output()  # Initialize output signals

            # initialize the converge time variable
            converge_time = 0

            # if converged break the loop
            done = False
            time_step = 0
            
            # Compute phase distance between the initial state and the "walk" state
            phase = cpg.get_phase_diffs()
            phase_dist = phase_distance(phase, desired_lag)

            while not done:  

                # Calculate phase divergence of all unit
                phase_norm = cpg.get_phase_diffs()/2/np.pi
                angle_diff = np.zeros(cell_num)
                for p in range(cell_num):
                    angle_diff[p] = abs((phase_norm[p]-desired_lag_norm[p]))
                    if angle_diff[p] > 0.5:
                        angle_diff[p] = 1-angle_diff[p]

                # Approx 5 degrees
                threshold = 0.014
                all_less_than_threshold = np.all(angle_diff < threshold)
                

                # Converge is considered when phase divergence of all unit less than 5 degrees
                if all_less_than_threshold:
                    print('converge time: ', converge_time)
                    writer.writerow([converge_time, phase_dist])
                    done = True
                
                # Simulate for 10s
                if converge_time > 10:
                    writer.writerow([converge_time, phase_dist])
                    done = True

                # Execute one-step evolution of the environent
                cpg.step()

                converge_time += dt
                time_step += 1

    except KeyboardInterrupt:
        print("\nScript terminated by user")
