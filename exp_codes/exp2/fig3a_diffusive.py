import sys, os
import numpy as np

from pathlib import Path  # Import Path for handling file paths (gcpg models are in the parent path)
parent_dir = str(Path(__file__).parent.parent.parent)  # Set parent directory path for importing modules and loading files
sys.path.append(parent_dir)  # Add parent directory to system path
import csv
import time
import matplotlib.pyplot as plt
from other_models.diffusive import DiffusiveCPG # import diffusive CPG model

# Set deisred phase configurations for gait transition
pi = np.pi
walk = np.array([-pi,-pi/2,pi,pi/2])
trot = np.array([pi,0,pi,0])

if __name__ == '__main__':
    
    # Set-up diffusive CPG env
    hz = 100
    cell_num = 4
    cpg = DiffusiveCPG(cell_num=cell_num, alpha=10, beta=10, mu=1, omega=2*np.pi, gamma=1, hz=hz)

    # Set-up data recorder
    cwd = os.getcwd()
    f = open(cwd+'/data/diffusive_trot_walk.csv',"w")
    writer = csv.writer(f)

    # Set initial states
    z_x = np.array([0,1,1,1])
    z_y = np.array([0,0,0,0])
    cpg.pos = np.array([z_x,z_y])

    x_mat = np.array([cpg.pos[0][0], cpg.pos[0][1], cpg.pos[0][2], cpg.pos[0][3]]).reshape(-1, cell_num)
    

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
            cpg.update_soft()
            state = cpg.pos

            # Record state vectors for plotting
            x_vec = np.array([cpg.pos[0][0], cpg.pos[0][1], cpg.pos[0][2], cpg.pos[0][3]]).reshape(-1, cell_num)
            x_mat = np.concatenate((x_mat,x_vec),axis=0)

            # Record state vectors and duration
            writer.writerow([duration,cpg.pos[0][0], cpg.pos[0][1], cpg.pos[0][2], cpg.pos[0][3]])
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

