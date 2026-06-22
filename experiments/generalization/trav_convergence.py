import csv
from pathlib import Path

import numpy as np
import sys,os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import (
    compute_order_trace,
    find_convergence_step,
    get_desired_lag,
    load_policy,
    make_cds_env,
    make_edge_index,
    model_action,
    set_env_desired_lag,
    step_intrinsic_dynamics,
)

INTRINSIC = 'harmonic'
TARGET_MODE = "traveling"
MIN_CELLS = 4
MAX_CELLS = 4  # 127
EVAL_LENGTH = 500
N_TRIALS = 50
RANDOM_SEED = 0

ORDER_THRESHOLD = 0.98
WINDOW_SIZE = 100
WINDOW_CRITERION = "mean"  # "mean" or "all"
PHASE_SIGN = "auto"  # "auto", "positive", or "negative"

HEADS = 8
FEATURE_DIM = 64


def simulate_waveforms(cell_num, model, rng):
    edge_index = make_edge_index(cell_num)

    env = make_cds_env(cell_nums=cell_num, env_length=EVAL_LENGTH)
    initial_angles = rng.uniform(0, 2 * np.pi, cell_num)
    env.z_mat = np.column_stack((np.cos(initial_angles), np.sin(initial_angles)))

  

    obs = env.z_mat.ravel()

    desired_lag = get_desired_lag(cell_num, mode=TARGET_MODE)
    rl_encoding = set_env_desired_lag(env, desired_lag)
    state = np.concatenate((obs, rl_encoding.ravel()))

    waveforms = np.zeros((cell_num, EVAL_LENGTH))
    for step in range(EVAL_LENGTH):
        waveforms[:, step] = state[: cell_num * 2].reshape(cell_num, 2)[:, 0]
        action = model_action(model, state, edge_index, cell_num, clamp=False)
        state = step_intrinsic_dynamics(env, action, INTRINSIC)

    return waveforms, env.desired_lag


def main():
    model = load_policy(heads=HEADS, feature_dim=FEATURE_DIM)

    root = Path(__file__).resolve().parent
    data_dir = root / 'data'
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"1convergence_steps_random_initials_traveling_{INTRINSIC}.csv"
    rng = np.random.default_rng(RANDOM_SEED)

    cell_nums = np.arange(MIN_CELLS, MAX_CELLS + 1)
    convergence_steps = np.full((N_TRIALS, len(cell_nums)), np.nan)

    for col_idx, cell_num in enumerate(cell_nums):
        for trial_idx in range(N_TRIALS):
            waveforms, desired_lag = simulate_waveforms(cell_num, model, rng)
            order_t, phase_sign = compute_order_trace(
                waveforms,
                desired_lag,
                phase_sign=PHASE_SIGN,
                ranking_window_size=WINDOW_SIZE,
            )
            convergence_step, _, _ = find_convergence_step(
                order_t,
                threshold=ORDER_THRESHOLD,
                window_size=WINDOW_SIZE,
                criterion=WINDOW_CRITERION,
            )

            convergence_steps[trial_idx, col_idx] = convergence_step
            print(
                "cell_num:",
                cell_num,
                "trial:",
                trial_idx + 1,
                "phase_sign:",
                phase_sign,
                "convergence_step:",
                convergence_step,
            )

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(cell_num) for cell_num in cell_nums])
        writer.writerows(convergence_steps)


if __name__ == "__main__":
    main()
