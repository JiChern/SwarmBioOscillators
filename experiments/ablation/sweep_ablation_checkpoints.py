import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import (
    PARENT_DIR,
    load_policy,
    make_cds_env,
    make_edge_index,
    model_action,
    phase_metrics_from_waveforms,
    set_env_desired_lag,
    step_intrinsic_dynamics,
    zero_initial_observation,
)


# Select one of: full_sies, softmax, direction_unaware, state_space_aggr.
# The aliases "full sies" and "direction-unaware" are also accepted.
MODEL_NAME = "full_sies"

HEADS = 8
FEATURE_DIM = 64
CELL_NUM = 8
EVAL_LENGTH = 500
STABLE_WINDOW = 300
METRIC_START = None
METRIC_END = None
PHASE_SIGN = "auto"
INTRINSIC_DYNAMICS = "hopf"
ENV_HZ = 100

CHECKPOINT_START = 50_000
CHECKPOINT_STOP = 6_000_000
CHECKPOINT_STEP = 50_000

# Match ablation/test_model.py. Set CLAMP_ACTION=True and ACTION_SCALE=1.0
# if you want the generalization1 action convention instead.
CLAMP_ACTION = False
ACTION_SCALE = 3.0

TRAIN_DESIRED_LAGS = [
    [0, 0.5, 0.25, 0.75, 0.5, 0, 0.75, 0.25],
    [0, 0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25],
    [0, 0.5, 0.5, 0, 0, 0.5, 0.5, 0],
    [0, 0, 0.5, 0.5, 0, 0, 0.5, 0.5],
]

TEST_DESIRED_LAGS = [
    [0, 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8],
    [0, 1 / 4, 2 / 4, 3 / 4, 0, 1 / 4, 2 / 4, 3 / 4],
    [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5],
    [0, 2 / 8, 4 / 8, 6 / 8, 1 / 8, 3 / 8, 5 / 8, 7 / 8],
]

MODEL_CONFIGS = {
    "full_sies": {
        "checkpoint_dir": PARENT_DIR / "checkpoints" / "multi-goal-8-64",
        "policy_kwargs": {"signed_att": True, "direction_aware": True},
    },
    "softmax": {
        "checkpoint_dir": PARENT_DIR / "checkpoints" / "softmax-att-8-64",
        "policy_kwargs": {"signed_att": False},
    },
    "direction_unaware": {
        "checkpoint_dir": PARENT_DIR / "checkpoints" / "n-direction-8-64",
        "policy_kwargs": {"signed_att": True, "direction_aware": False},
    },
    "state_space_aggr": {
        "checkpoint_dir": PARENT_DIR / "checkpoints" / "state-space-8-64",
        "policy_kwargs": {
            "signed_att": True,
            "direction_aware": True,
            "state_space_aggr": True,
        },
    },
}

MODEL_ALIASES = {
    "full sies": "full_sies",
    "full_sies": "full_sies",
    "softmax": "softmax",
    "direction-unaware": "direction_unaware",
    "direction_unaware": "direction_unaware",
    "state space aggr": "state_space_aggr",
    "state-space-aggr": "state_space_aggr",
    "state_space_aggr": "state_space_aggr",
}


def canonical_model_name(model_name):
    key = model_name.strip().lower().replace("-", "_")
    key = key.replace(" ", "_")
    if key in MODEL_CONFIGS:
        return key
    alias_key = model_name.strip().lower()
    if alias_key in MODEL_ALIASES:
        return MODEL_ALIASES[alias_key]
    raise ValueError(
        f"Unknown MODEL_NAME={model_name!r}. "
        f"Expected one of {', '.join(MODEL_CONFIGS)}."
    )


def checkpoint_path(checkpoint_dir, checkpoint_step):
    return checkpoint_dir / f"model-{checkpoint_step}.pt"


def simulate_target(model, desired_lag, edge_index):
    env = make_cds_env(cell_nums=CELL_NUM, env_length=EVAL_LENGTH, hz=ENV_HZ)
    obs = zero_initial_observation(env, CELL_NUM)
    rl_encoding = set_env_desired_lag(env, desired_lag)
    state = np.concatenate((obs, rl_encoding.ravel()))
    waveforms = np.zeros((CELL_NUM, EVAL_LENGTH))

    for step in range(EVAL_LENGTH):
        waveforms[:, step] = state[: CELL_NUM * 2].reshape(CELL_NUM, 2)[:, 0]
        action = model_action(
            model,
            state,
            edge_index,
            CELL_NUM,
            clamp=CLAMP_ACTION,
        )
        action = ACTION_SCALE * action
        state = step_intrinsic_dynamics(env, action, INTRINSIC_DYNAMICS)

    return phase_metrics_from_waveforms(
        waveforms=waveforms,
        desired_lag=env.desired_lag,
        stable_window=STABLE_WINDOW,
        phase_sign=PHASE_SIGN,
        metric_start=METRIC_START,
        metric_end=METRIC_END,
    )


def iter_targets():
    for target_id, desired_lag in enumerate(TRAIN_DESIRED_LAGS, start=1):
        yield "train", target_id, np.asarray(desired_lag, dtype=float)

    for target_id, desired_lag in enumerate(TEST_DESIRED_LAGS, start=1):
        yield "test", target_id, np.asarray(desired_lag, dtype=float)


def write_result(writer, model_key, checkpoint_step, checkpoint_file, split, target_id, desired_lag, metrics):
    writer.writerow(
        [
            model_key,
            checkpoint_step,
            checkpoint_file,
            CELL_NUM,
            split,
            target_id,
            " ".join(f"{value:.6g}" for value in desired_lag),
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


def main():
    model_key = canonical_model_name(MODEL_NAME)
    config = MODEL_CONFIGS[model_key]
    edge_index = make_edge_index(CELL_NUM)

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{model_key}_checkpoint_phase_metrics.csv"

    checkpoint_steps = range(
        CHECKPOINT_START,
        CHECKPOINT_STOP + CHECKPOINT_STEP,
        CHECKPOINT_STEP,
    )

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "model_name",
                "checkpoint_step",
                "checkpoint_file",
                "cell_num",
                "target_split",
                "target_id",
                "desired_lag",
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

        for checkpoint_step_value in checkpoint_steps:
            checkpoint_file = checkpoint_path(
                config["checkpoint_dir"],
                checkpoint_step_value,
            )
            model = load_policy(
                heads=HEADS,
                feature_dim=FEATURE_DIM,
                checkpoint_path=checkpoint_file,
                **config["policy_kwargs"],
            )

            for split, target_id, desired_lag in iter_targets():
                metrics = simulate_target(model, desired_lag, edge_index)
                write_result(
                    writer,
                    model_key,
                    checkpoint_step_value,
                    checkpoint_file,
                    split,
                    target_id,
                    desired_lag,
                    metrics,
                )
                csvfile.flush()
                print(
                    "model:",
                    model_key,
                    "checkpoint:",
                    checkpoint_step_value,
                    "split:",
                    split,
                    "target:",
                    target_id,
                    "R:",
                    metrics["order_parameter_mean"],
                    "phase_rmse_deg:",
                    metrics["phase_rmse_deg_mean"],
                )

    print(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    main()
