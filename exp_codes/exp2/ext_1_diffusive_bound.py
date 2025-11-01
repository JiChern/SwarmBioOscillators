import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path
import csv
import time
import matplotlib.pyplot as plt
from other_models.diffusive import DiffusiveCPG # import diffusive CPG model

from utils import state_to_goal1, phase_distance

# Set deisred phase configuration
pi = np.pi
# walk = np.array([-pi,-pi/2,pi,pi/2])
bound = np.array([pi,-pi,pi,-pi])

# Set deisred lag corresponding to desired phase configuration for phase distance calculation
desired_lag = np.array([0,0.5,0,0.5])



if __name__ == '__main__':

    # Set-up Diffusive CPG env
    hz = 100
    dt = 1/hz
    cell_num = 4
    cpg = DiffusiveCPG(cell_num=cell_num, alpha=10, beta=10, mu=1, omega=2*np.pi, gamma=1, hz=hz)

    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/diffusive_converge_time_bound.csv',"w")
    writer = csv.writer(f)

    # Load the initial condition shared for all CPG models
    init_conds  = np.loadtxt(cwd+'/data/init_cond.csv', delimiter=',', dtype=float)

    try:
        # Get converge times over 1000 random trials
        for i in range(1000):
            # Reset the initial states around a circle with radius of 1
            cpg.set_theta(bound)
            z_x = np.array(init_conds[i,0:4])
            z_y = np.array(init_conds[i,4:8])
            cpg.pos = np.array([z_x,z_y])
            state = cpg.pos

            # initialize the converge time variable
            converge_time = 0

            # if converged break the loop
            done = False
            time_step = 0

            # Compute phase distance between the initial state and the "walk" state
            phase = state_to_goal1(state.T.ravel(), cell_num=cell_num)
            phase_dist = phase_distance(phase*2*np.pi, desired_lag* np.pi * 2)

            while not done:  

                # Calculate phase divergence of all unit
                phase = state_to_goal1(state.T.ravel(), cell_num=cell_num)
                angle_diff = np.zeros(cell_num)
                for p in range(cell_num):
                    angle_diff[p] = abs((phase[p]-desired_lag[p]))
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
                cpg.update_soft()
                state = cpg.pos

                converge_time += dt
                time_step += 1






    except KeyboardInterrupt:
        print("\nScript terminated by user")