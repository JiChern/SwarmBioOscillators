import csv
import sys, os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from other_models.salamander import Salamander # import diffusive CPG model

from data_utils import compute_order_trace



GAITS = {
    "WALK":  np.array([0, np.pi, 3*np.pi/2, np.pi/2]),
    "TROT":  np.array([0, np.pi, np.pi, 0]),
    "BOUND": np.array([0, np.pi, 0, np.pi])
}

DESIRED_GAIT_NAME = "BOUND"        

# Set deisred phase configuration
desired_lag = GAITS[DESIRED_GAIT_NAME]

# Set deisred lag corresponding to desired phase configuration for order parameter calculation
desired_lag_norm = desired_lag/2/np.pi
ORDER_THRESHOLD = 0.999
PHASE_SIGN = "auto"
MAX_TIME = 10.0

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
    z_x = np.array(init_cond[0:4])
    z_y = np.array(init_cond[4:8])
    cpg.theta = cal_phase(z_x, z_y)
    cpg.r = np.ones(cpg.N)
    cpg.dr = np.zeros(cpg.N)
    cpg._update_output()

    waveforms = np.zeros((cell_num, max_steps))
    for step in range(max_steps):
        waveforms[:, step] = np.cos(cpg.theta)
        cpg.step()

    return waveforms


if __name__ == "__main__":

     # Set-up Salamander CPG env
    hz = 100
    dt = 1/hz
    cell_num = 4
    max_steps = int(MAX_TIME * hz)
    cpg = Salamander(omega=2*np.pi, cell_num=4, hz=100, desired_phase_diffs=desired_lag)

    # Set-up data recorder
    cwd = os.getcwd()
    init_conds = np.loadtxt(cwd+'/data/init_cond.csv', delimiter=',', dtype=float)

    try:
        with open(cwd+f'/data/salamander_converge_time_{DESIRED_GAIT_NAME}.csv', "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["convergence_time", "phase_sign", "order_parameter_at_convergence"])

            # Get converge times over 1000 random trials
            for i in range(1000):
                waveforms = simulate_waveforms(cpg, init_conds[i], cell_num, max_steps)
                convergence_time, phase_sign, order_at_convergence = convergence_time_from_order(
                    waveforms,
                    desired_lag_norm,
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
