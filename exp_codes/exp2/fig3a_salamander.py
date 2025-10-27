import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path
import csv
import time
import matplotlib.pyplot as plt
from other_models.salamander import Salamander # import diffusive CPG model

# Set deisred phase configurations for gait transition
trot = np.array([0, np.pi, np.pi, 0])
walk = np.array([0, np.pi, 3*np.pi/2, np.pi/2])


if __name__ == '__main__':

    # Set-up Salamander CPG env
    hz = 100
    cell_num = 4
    cpg = Salamander(omega=2*np.pi, cell_num=4, hz=100, desired_phase_diffs=trot)


    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/salamander_trot_walk.csv',"w")
    writer = csv.writer(f)
    # state = cpg.reset_ini_states(r=1)

    x_mat = np.array([cpg.x[0], cpg.x[1], cpg.x[2], cpg.x[3]]).reshape(-1, cell_num) 

    # Set initial states
    cpg.theta = np.zeros(cpg.N)  # Random initial phases
    cpg.theta[0] = np.pi/2
    cpg.r = np.ones(cpg.N)  # Initial amplitudes set to 1
    cpg.dr = np.zeros(cpg.N)  # Initial amplitude derivatives set to 0
    cpg._update_output()  # Initialize output signals

    # Set the initial trot gait
    cpg.set_theta(trot)

    # Set start time and time-duratio vector
    dt = 1/hz
    duration = 0
    duration_vec = [0]
    start_time = time.time()
    try:
        while True:

            # Execute trot-to-walk gait transition
            if duration>5:
                cpg.set_theta(walk)   

            # Execute one-step evolution of the environent
            cpg.step()

            # Record state vectors for plotting
            x_vec = np.array([cpg.x[0], cpg.x[1], cpg.x[2], cpg.x[3]]).reshape(-1, cell_num)
            x_mat = np.concatenate((x_mat,x_vec),axis=0)

            # Record state vectors and duration
            writer.writerow([duration,cpg.x[0], cpg.x[1], cpg.x[2], cpg.x[3]])
            duration += dt
            duration_vec.append(duration)

            # Simulate for 10s
            if duration > 10:
                break


    except KeyboardInterrupt:
        print('stopped')
    
    plt.plot(duration_vec,x_mat[:,0], label=f'x {1}')
    plt.plot(duration_vec,x_mat[:,1], label=f'x {2}')
    plt.plot(duration_vec,x_mat[:,2], label=f'x {3}')
    plt.plot(duration_vec,x_mat[:,3], label=f'x {4}')

    plt.legend()

    plt.show()

