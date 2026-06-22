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
TARGET_MODE = "traveling"  # "traveling", "random", or "custom"
RANDOM_SEED = 0  # Used only when TARGET_MODE = "random".
CUSTOM_DESIRED_LAG = None  # 1D array with length = cell_num when TARGET_MODE = "custom".
MIN_CELLS = 2
MAX_CELLS = 127
EVAL_LENGTH = 500
STABLE_WINDOW = 300  # Used when METRIC_START and METRIC_END are both None.
METRIC_START = None  # Example: 700
METRIC_END = None  # Example: 1500
PHASE_SIGN = "auto"  # "auto", "positive", or "negative"


def get_stable_metrics(
    cell_num,
    model,
    env,
    desired_lag_mode=TARGET_MODE,
    rng=None,
    custom_desired_lag=CUSTOM_DESIRED_LAG,
    eval_length=EVAL_LENGTH,
    stable_window=STABLE_WINDOW,
    phase_sign=PHASE_SIGN,
    metric_start=None,
    metric_end=None,
):
    length = eval_length
    edge_index = make_edge_index(cell_num)

    obs = zero_initial_observation(env, cell_num)

    # Set the target phase lags and construct the observation.
    desired_lag = get_desired_lag(
        cell_num,
        mode=desired_lag_mode,
        rng=rng,
        custom_lag=custom_desired_lag,
    )
    rl_encoding = set_env_desired_lag(env, desired_lag)
    state = np.concatenate((obs,rl_encoding.ravel()))

    waveforms = np.zeros((cell_num, length))
    
    for i in range(length):
        waveforms[:, i] = state[0:cell_num*2].reshape(cell_num, 2)[:, 0]
        action = model_action(model, state, edge_index, cell_num)
        state = step_intrinsic_dynamics(env, action, INTRINSIC)
    
    return phase_metrics_from_waveforms(
        waveforms=waveforms,
        desired_lag=env.desired_lag,
        stable_window=stable_window,
        phase_sign=phase_sign,
        metric_start=metric_start,
        metric_end=metric_end,
    )




if __name__ == '__main__':
    heads = 8
    fd = 64
    model = load_policy(heads=heads, feature_dim=fd)

    # Set-up the data recorder
    root = Path(__file__).resolve().parent
    data_dir = root / 'data'
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    output_file = data_dir / f'1phase_metrics_{TARGET_MODE}_{INTRINSIC}.csv'
    csvfile = open(output_file, 'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow([
        'cell_num',
        'target_mode',
        'metric_start',
        'metric_end',
        'phase_sign',
        'order_parameter_mean',
        'order_parameter_std',
        'phase_rmse_deg_mean',
        'phase_rmse_deg_std',
        'phase_mae_deg_mean',
        'phase_mae_deg_std',
    ])

    for i in np.arange(MIN_CELLS, MAX_CELLS + 1, 1):
        cell_num = i
        # Refersh environment for different scale of network for each trial 
        env = make_cds_env(cell_nums=cell_num,env_length=EVAL_LENGTH)
        metrics = get_stable_metrics(
            cell_num=cell_num,
            model=model,
            env=env,
            desired_lag_mode=TARGET_MODE,
            rng=rng,
            custom_desired_lag=CUSTOM_DESIRED_LAG,
            eval_length=EVAL_LENGTH,
            stable_window=STABLE_WINDOW,
            phase_sign=PHASE_SIGN,
            metric_start=METRIC_START,
            metric_end=METRIC_END,
        )

        # Record data
        writer.writerow([
            i,
            TARGET_MODE,
            metrics['metric_start'],
            metrics['metric_end'],
            metrics['phase_sign'],
            metrics['order_parameter_mean'],
            metrics['order_parameter_std'],
            metrics['phase_rmse_deg_mean'],
            metrics['phase_rmse_deg_std'],
            metrics['phase_mae_deg_mean'],
            metrics['phase_mae_deg_std'],
        ])
        print(
            'cell_num:', i,
            'phase_sign:', metrics['phase_sign'],
            'R:', metrics['order_parameter_mean'],
            'phase_rmse_deg:', metrics['phase_rmse_deg_mean'],
        )

    csvfile.close()
