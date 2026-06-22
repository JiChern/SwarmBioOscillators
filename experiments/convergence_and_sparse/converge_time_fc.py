import csv
import sys, os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from other_models.fully_coupled1 import FullyCoupled1 # import diffusive CPG model

from data_utils import compute_order_trace

# Set deisred phase configuration
walk = np.array([[0,-1,1,-1],
                    [-1,0,-1,1],
                    [-1,1,0,-1],
                    [1,-1,-1,0]])

GAITS_FC = {
    "WALK":  np.array([[0,-1,1,-1],
                    [-1,0,-1,1],
                    [-1,1,0,-1],
                    [1,-1,-1,0]]),
    "TROT":  np.array([[0,-1,-1,1],
                [-1,0, 1,1],
                [-1,1,0,-1],
                [1,-1,-1,0]]),

    "BOUND": np.array([[0,-1,1,-1],
                [-1,0, -1,1],
                [1,-1,0,-1],
                [-1,1,-1,0]])
}

DESIEED_LAG = {
    "WALK":  np.array([0,    0.5,  0.75, 0.25]),
    "TROT":  np.array([0,    0.5,  0.5,  0   ]),
    "BOUND": np.array([0,    0.5,  0,    0.5 ])
}

DESIRED_GAIT_NAME = "TROT"      

# Set deisred lag corresponding to desired phase configuration for order parameter calculation
desired_lag = DESIEED_LAG[DESIRED_GAIT_NAME]
ORDER_THRESHOLD = 0.999
PHASE_SIGN = "auto"
MAX_TIME = 10.0


def convergence_time_from_order(waveforms, desired_lag_cycles, dt):
    order_t, phase_sign = compute_order_trace(
        waveforms,
        desired_lag_cycles,
        phase_sign=PHASE_SIGN,
        ranking_window_size=min(100, waveforms.shape[1]),
    )
    converged_steps = np.flatnonzero(order_t >= ORDER_THRESHOLD)
    if len(converged_steps) == 0:
        return MAX_TIME, phase_sign, np.nan

    step = int(converged_steps[0])
    return step * dt, phase_sign, order_t[step]


def simulate_waveforms(cpg, init_cond, cell_num, max_steps):
    cpg.set_theta(GAITS_FC[DESIRED_GAIT_NAME])
    z_x = np.array(init_cond[0:4])
    z_y = np.array(init_cond[4:8])
    cpg.pos = np.array([z_x, z_y])

    waveforms = np.zeros((cell_num, max_steps))
    for step in range(max_steps):
        waveforms[:, step] = cpg.pos[0, :]
        cpg.update_soft()

    return waveforms

if __name__ == '__main__':

    # Set-up FC CPG env
    hz = 100
    dt = 1/hz
    cell_num = 4
    max_steps = int(MAX_TIME * hz)
    cpg = FullyCoupled1(cell_num=cell_num, alpha=10, beta=10, mu=1, omega=2*np.pi, gamma=1, hz=hz)

    # Set-up data recorder
    cwd = os.getcwd()
    init_conds = np.loadtxt(cwd+'/data/init_cond.csv', delimiter=',', dtype=float)

    
    try:
        with open(cwd+f'/data/fc1_converge_time_{DESIRED_GAIT_NAME}.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["convergence_time", "phase_sign", "order_parameter_at_convergence"])

            # Get converge times over 1000 random trials
            for i in range(1000):
                waveforms = simulate_waveforms(cpg, init_conds[i], cell_num, max_steps)
                convergence_time, phase_sign, order_at_convergence = convergence_time_from_order(
                    waveforms,
                    desired_lag,
                    dt,
                )

                writer.writerow([convergence_time, phase_sign, order_at_convergence])
                print(
                    "trial:",
                    i + 1,
                    "convergence_time:",
                    convergence_time,
                    "phase_sign:",
                    phase_sign,
                )


    except KeyboardInterrupt:
        print("\nScript terminated by user")
