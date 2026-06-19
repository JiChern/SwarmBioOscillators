import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert

PARENT_DIR = Path(
    os.environ.get("GRAPH_CPG_PROJECT_ROOT", Path(__file__).resolve().parent)
).expanduser().resolve()
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_PROJECT_DEPS = None


def _project_deps():
    global _PROJECT_DEPS
    if _PROJECT_DEPS is None:
        from agent.networks import Policy
        from environment.env import CDSEnv
        from utils import generate_edge_idx, rearrange_state_vector_hopf

        _PROJECT_DEPS = {
            "Policy": Policy,
            "CDSEnv": CDSEnv,
            "generate_edge_idx": generate_edge_idx,
            "rearrange_state_vector_hopf": rearrange_state_vector_hopf,
        }

    return _PROJECT_DEPS


def load_policy(heads=8, feature_dim=64, checkpoint_path=None, device=DEVICE, **policy_kwargs):
    Policy = _project_deps()["Policy"]
    model = Policy(heads=heads, feature_dim=feature_dim, **policy_kwargs)
    model.eval().to(device)

    if checkpoint_path is None:
        checkpoint_path = PARENT_DIR / "model_params" / f"model-{heads}-{feature_dim}.pt"

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["policy_state_dict"])
    return model


def make_cds_env(*args, **kwargs):
    return _project_deps()["CDSEnv"](*args, **kwargs)


def make_edge_index(cell_num, device=DEVICE):
    return _project_deps()["generate_edge_idx"](cell_num).to(device)


def circular_difference_cycles(a, b):
    """Return wrapped phase difference in cycles, in [-0.5, 0.5)."""
    return (a - b + 0.5) % 1.0 - 0.5


def get_desired_lag(cell_num, mode="traveling", rng=None, custom_lag=None):
    if mode == "traveling":
        return np.arange(cell_num) / cell_num

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        rand_angles = rng.integers(0, cell_num, cell_num)
        rand_angles[0] = 0
        return rand_angles / cell_num

    if mode == "custom":
        if custom_lag is None:
            raise ValueError("custom_lag must be provided when mode='custom'.")
        custom_lag = np.asarray(custom_lag, dtype=float)
        if custom_lag.shape != (cell_num,):
            raise ValueError(
                f"custom_lag must have shape ({cell_num},), got {custom_lag.shape}."
            )
        return custom_lag % 1.0

    raise ValueError(f"Unknown desired lag mode: {mode}")


def set_env_desired_lag(env, desired_lag):
    env.desired_lag = np.asarray(desired_lag, dtype=float) % 1.0
    env.relative_lags = env.cal_relative_lags(env.desired_lag)
    return env.encoding_angle(env.desired_lag)


def phase_cycles_from_waveforms(waveforms):
    centered = waveforms - np.mean(waveforms, axis=1, keepdims=True)
    return np.angle(hilbert(centered, axis=1)) / (2 * np.pi)


def phase_cycles_from_state(state, cell_num):
    """Extract oscillator phases in cycles from a flat x/y state vector."""
    node_state = np.asarray(state[: cell_num * 2], dtype=float).reshape(cell_num, 2)
    return np.mod(np.arctan2(node_state[:, 1], node_state[:, 0]) / (2 * np.pi), 1.0)


def state_to_goal_phase(state, cell_num):
    """Convert state to phases relative to the first oscillator."""
    state_2d = np.reshape(state[: cell_num * 2], (cell_num, 2))
    z_0 = state_2d[0]

    dot_products = np.dot(state_2d, z_0)
    norm_z_0 = np.linalg.norm(z_0)
    norm_z_i = np.linalg.norm(state_2d, axis=1)

    cos_angles = dot_products / (norm_z_0 * norm_z_i + 1e-10)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    cross_products = z_0[0] * state_2d[:, 1] - z_0[1] * state_2d[:, 0]
    angle_signs = np.sign(cross_products)
    return np.where(angle_signs < 0, 1 - angles / (2 * np.pi), angles / (2 * np.pi))


def phase_distance_to_target(phase_cycles, desired_lag):
    """Return unit-circle chord distance between phase vectors."""
    phase_cycles = np.asarray(phase_cycles, dtype=float)
    desired_lag = np.asarray(desired_lag, dtype=float) % 1.0

    if phase_cycles.shape != desired_lag.shape:
        raise ValueError("phase_cycles and desired_lag must have the same shape.")

    diff = 2 * np.pi * (phase_cycles - desired_lag)
    dist_sq = 2 - 2 * np.cos(diff)
    return np.sqrt(np.sum(dist_sq))


def original_phase_distance_to_target(phase_cycles, desired_lag):
    """Return the original unit-circle chord distance to target lags."""
    return phase_distance_to_target(phase_cycles, desired_lag)


def _candidate_phase_signs(phase_sign):
    if phase_sign == "auto":
        return [1, -1]
    if phase_sign == "positive":
        return [1]
    if phase_sign == "negative":
        return [-1]
    raise ValueError("phase_sign must be 'auto', 'positive', or 'negative'.")


def _metric_slice(n_steps, stable_window, metric_start, metric_end):
    if metric_start is None and metric_end is None:
        return slice(-stable_window, None), max(0, n_steps - stable_window), n_steps - 1

    if metric_start is None or metric_end is None:
        raise ValueError("metric_start and metric_end must be set together.")
    if metric_start < 0 or metric_end >= n_steps or metric_start > metric_end:
        raise ValueError(
            f"Invalid metric window [{metric_start}, {metric_end}] for {n_steps} steps."
        )

    return slice(metric_start, metric_end + 1), metric_start, metric_end


def phase_metrics_from_waveforms(
    waveforms,
    desired_lag,
    stable_window,
    phase_sign="auto",
    metric_start=None,
    metric_end=None,
):
    phase_cycles = phase_cycles_from_waveforms(waveforms)
    desired_lag = np.asarray(desired_lag, dtype=float)[:, None] % 1.0
    metric_slice, metric_start_out, metric_end_out = _metric_slice(
        waveforms.shape[1], stable_window, metric_start, metric_end
    )

    best_metrics = None
    for sign in _candidate_phase_signs(phase_sign):
        phase_error_cycles = circular_difference_cycles(sign * phase_cycles, desired_lag)
        phase_error_rad = 2 * np.pi * phase_error_cycles

        order_t = np.abs(np.mean(np.exp(1j * phase_error_rad), axis=0))
        global_phase_t = np.angle(
            np.mean(np.exp(1j * phase_error_rad), axis=0, keepdims=True)
        )
        residual = np.angle(np.exp(1j * (phase_error_rad - global_phase_t)))
        rmse_t = np.degrees(np.sqrt(np.mean(residual**2, axis=0)))
        mae_t = np.degrees(np.mean(np.abs(residual), axis=0))

        metrics = {
            "metric_start": metric_start_out,
            "metric_end": metric_end_out,
            "phase_sign": "positive" if sign == 1 else "negative",
            "order_parameter_mean": np.mean(order_t[metric_slice]),
            "order_parameter_std": np.std(order_t[metric_slice]),
            "phase_rmse_deg_mean": np.mean(rmse_t[metric_slice]),
            "phase_rmse_deg_std": np.std(rmse_t[metric_slice]),
            "phase_mae_deg_mean": np.mean(mae_t[metric_slice]),
            "phase_mae_deg_std": np.std(mae_t[metric_slice]),
        }

        if best_metrics is None or (
            metrics["order_parameter_mean"] > best_metrics["order_parameter_mean"]
        ):
            best_metrics = metrics

    return best_metrics


def compute_order_trace(
    waveforms,
    desired_lag,
    phase_sign="auto",
    ranking_window_size=100,
):
    phase_cycles = phase_cycles_from_waveforms(waveforms)
    desired_lag = np.asarray(desired_lag, dtype=float)[:, None] % 1.0

    best_order = None
    best_sign = None
    for sign in _candidate_phase_signs(phase_sign):
        phase_error = circular_difference_cycles(sign * phase_cycles, desired_lag)
        order_t = np.abs(np.mean(np.exp(1j * 2 * np.pi * phase_error), axis=0))
        if best_order is None or (
            np.mean(order_t[-ranking_window_size:])
            > np.mean(best_order[-ranking_window_size:])
        ):
            best_order = order_t
            best_sign = "positive" if sign == 1 else "negative"

    return best_order, best_sign


def find_convergence_step(order_t, threshold, window_size, criterion):
    if len(order_t) < window_size:
        raise ValueError("window_size cannot be larger than the order trace.")

    for end_idx in range(window_size - 1, len(order_t)):
        window = order_t[end_idx - window_size + 1 : end_idx + 1]
        if criterion == "mean":
            passed = np.mean(window) >= threshold
        elif criterion == "all":
            passed = np.all(window >= threshold)
        else:
            raise ValueError(f"Unknown convergence criterion: {criterion}")

        if passed:
            return end_idx, np.mean(window), np.min(window)

    return np.nan, np.nan, np.nan


def model_action(model, state, edge_index, cell_num, device=DEVICE, clamp=True):
    rearrange_state_vector_hopf = _project_deps()["rearrange_state_vector_hopf"]
    gnn_x = rearrange_state_vector_hopf(state=state, num_nodes=cell_num).to(device)
    with torch.no_grad():
        action = model(gnn_x, edge_index)
        if clamp:
            action.clamp_(-1, 1)
        return action.squeeze().cpu().numpy()


def step_intrinsic_dynamics(env, action, intrinsic):
    if intrinsic == "hopf":
        return env.step_env(action)
    if intrinsic == "vdp":
        return env.step_env_vdp_ct(action)
    return env.step_env_damped_ct(action)


def zero_initial_observation(env, cell_num, seed_value=0.1):
    env.z_mat = np.zeros((cell_num, 2))
    env.z_mat[0, 0] = seed_value
    return env.z_mat.ravel()
