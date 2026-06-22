import csv
from pathlib import Path

import numpy as np
import sys,os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from data_utils import (
    get_desired_lag,
    load_policy,
    make_cds_env,
    make_edge_index,
    model_action,
    phase_metrics_from_waveforms,
    set_env_desired_lag,
    step_intrinsic_dynamics,
    zero_initial_observation,
)

INTRINSIC = 'hopf'
RANDOM_SEED = 0
MIN_CELLS = 4
MAX_CELLS = 120
EVAL_LENGTH = 500
STABLE_WINDOW = 300
METRIC_START = None
METRIC_END = None
PHASE_SIGN = "auto"  # "auto", "positive", or "negative"


HEADS = 8
FEATURE_DIM = 64


def simulate_random_target(cell_num, model, desired_lag):
    edge_index = make_edge_index(cell_num)
    env = make_cds_env(cell_nums=cell_num, env_length=EVAL_LENGTH)
    obs = zero_initial_observation(env, cell_num)

    rl_encoding = set_env_desired_lag(env, desired_lag)
    state = np.concatenate((obs, rl_encoding.ravel()))
    waveforms = np.zeros((cell_num, EVAL_LENGTH))

    for step in range(EVAL_LENGTH):
        waveforms[:, step] = state[: cell_num * 2].reshape(cell_num, 2)[:, 0]
        action = model_action(model, state, edge_index, cell_num)
        state = step_intrinsic_dynamics(env, action, INTRINSIC)

    return phase_metrics_from_waveforms(
        waveforms=waveforms,
        desired_lag=env.desired_lag,
        stable_window=STABLE_WINDOW,
        phase_sign=PHASE_SIGN,
        metric_start=METRIC_START,
        metric_end=METRIC_END,
    )


def main():
    model = load_policy(heads=HEADS, feature_dim=FEATURE_DIM)

    root = Path(__file__).resolve().parent
    # output_file = root/ 'data' / "phase_metrics_random_harmonic.csv"
    file_name = "1phase_metrics_random_" + INTRINSIC + ".csv"

    data_dir = root / 'data'
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / file_name

    rng = np.random.default_rng(RANDOM_SEED)

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "cell_num",
                "target_id",
                "num_random_targets",
                "metric_start",
                "metric_end",
                "phase_sign",
                "order_parameter_mean",
                "order_parameter_std",
                "phase_rmse_deg_mean",
                "phase_rmse_deg_std",
                "phase_mae_deg_mean",
                "phase_mae_deg_std",
            ]
        )

        for cell_num in range(MIN_CELLS, MAX_CELLS + 1):
            num_random_targets = 2 * cell_num
            for target_id in range(num_random_targets):
                desired_lag = get_desired_lag(cell_num, mode="random", rng=rng)
                metrics = simulate_random_target(cell_num, model, desired_lag)

                writer.writerow(
                    [
                        cell_num,
                        target_id,
                        num_random_targets,
                        metrics["metric_start"],
                        metrics["metric_end"],
                        metrics["phase_sign"],
                        metrics["order_parameter_mean"],
                        metrics["order_parameter_std"],
                        metrics["phase_rmse_deg_mean"],
                        metrics["phase_rmse_deg_std"],
                        metrics["phase_mae_deg_mean"],
                        metrics["phase_mae_deg_std"],
                    ]
                )
                print(
                    "cell_num:",
                    cell_num,
                    "target:",
                    f"{target_id + 1}/{num_random_targets}",
                    "R:",
                    metrics["order_parameter_mean"],
                    "phase_rmse_deg:",
                    metrics["phase_rmse_deg_mean"],
                )


if __name__ == "__main__":
    main()
