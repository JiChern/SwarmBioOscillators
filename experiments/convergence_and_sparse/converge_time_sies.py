import csv
from pathlib import Path
import sys,os

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import (
    compute_order_trace,
    load_policy,
    make_cds_env,
    make_edge_index,
    model_action,
    set_env_desired_lag,
    step_intrinsic_dynamics,
)

GAITS = {
    "WALK":  np.array([0,    0.5,  0.75, 0.25]),
    "TROT":  np.array([0,    0.5,  0.5,  0   ]),
    "BOUND": np.array([0,    0.5,  0,    0.5 ])
}

DESIRED_GAIT_NAME = "BOUND"        
DESIRED_LAG = GAITS[DESIRED_GAIT_NAME]


# DESIRED_LAG = np.array([0, 0.5, 0.75, 0.25])
CELL_NUM = 4
HZ = 100
DT = 1 / HZ
MAX_TIME = 10.0
MAX_STEPS = int(MAX_TIME * HZ)
ORDER_THRESHOLD = 0.999
PHASE_SIGN = "auto"
HEADS = 8
FEATURE_DIM = 64
N_TRIALS = 1000
INTRINSIC = "hopf"


def convergence_time_from_order(waveforms, desired_lag):
    order_t, phase_sign = compute_order_trace(
        waveforms,
        desired_lag,
        phase_sign=PHASE_SIGN,
        ranking_window_size=min(100, waveforms.shape[1]),
    )
    converged_steps = np.flatnonzero(order_t > ORDER_THRESHOLD)
    if len(converged_steps) == 0:
        return MAX_TIME, phase_sign, np.nan

    step = int(converged_steps[0])
    return step * DT, phase_sign, order_t[step]


def simulate_waveforms(model, edge_index, initial_state, desired_lag):
    env = make_cds_env(cell_nums=CELL_NUM, env_length=MAX_STEPS, hz=HZ)
    env.omega = np.pi * 2
    env.z_mat = np.asarray(initial_state, dtype=float).reshape(CELL_NUM, 2)

    obs = env.z_mat.ravel()
    rl_encoding = set_env_desired_lag(env, desired_lag)
    state = np.concatenate((obs, rl_encoding.ravel()))
    waveforms = np.zeros((CELL_NUM, MAX_STEPS))

    for step in range(MAX_STEPS):
        waveforms[:, step] = state[: CELL_NUM * 2].reshape(CELL_NUM, 2)[:, 0]
        action = model_action(model, state, edge_index, CELL_NUM)
        action = action * 2
        state = step_intrinsic_dynamics(env, action, INTRINSIC)

    return waveforms


def main():
    model = load_policy(heads=HEADS, feature_dim=FEATURE_DIM)
    edge_index = make_edge_index(CELL_NUM)

    data_dir = Path.cwd() / "data"
    data_dir.mkdir(exist_ok=True)
    init_conds = np.loadtxt(data_dir / "init_cond.csv", delimiter=",", dtype=float)
    output_file = data_dir / f"sies_converge_time_{DESIRED_GAIT_NAME}.csv"

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "convergence_time",
                "phase_sign",
                "order_parameter_at_convergence",
            ]
        )

        for trial_idx in range(min(N_TRIALS, len(init_conds))):
            initial_state = np.column_stack(
                (init_conds[trial_idx, 0:4], init_conds[trial_idx, 4:8])
            )
            waveforms = simulate_waveforms(
                model=model,
                edge_index=edge_index,
                initial_state=initial_state,
                desired_lag=DESIRED_LAG,
            )

            convergence_time, phase_sign, order_at_convergence = convergence_time_from_order(
                waveforms,
                DESIRED_LAG,
            )

            writer.writerow(
                [
                    convergence_time,
                    phase_sign,
                    order_at_convergence,
                ]
            )
            print(
                "trial:",
                trial_idx + 1,
                "convergence_time:",
                convergence_time,
                "phase_sign:",
                phase_sign,
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user")
