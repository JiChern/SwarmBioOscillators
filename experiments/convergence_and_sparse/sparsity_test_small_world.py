import argparse
import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_utils import DEVICE, load_policy, phase_metrics_from_waveforms
from environment.env_torch import CDSEnv
from sparsity_test import DEVICE_DTYPE
from utils import rearrange_state_vector_torch


HEADS = 8
FEATURE_DIM = 64
CELL_NUM = 20
REWIRING_PROB = 0.0
GRAPH_SEED = 42
TARGET_SEED = 0
HZ = 100
ENV_LENGTH = 500
EVAL_LENGTH = 500
STABLE_WINDOW = 300


def get_phase_metrics(cell_num, edge_index, model, env, target, length):
    waveforms = np.zeros((cell_num, length), dtype=np.float32)

    env.z_mat = torch.zeros((cell_num, 2), dtype=DEVICE_DTYPE, device=DEVICE)
    env.z_mat[0, 0] = 0.1
    obs = env.z_mat.ravel()
    env.desired_lag = target
    rl_encoding = env.encoding_angle(env.desired_lag).half()
    state = torch.concatenate((obs, rl_encoding.ravel())).half()

    for step in range(length):
        waveforms[:, step] = (
            state[: cell_num * 2]
            .reshape(cell_num, 2)[:, 0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        gnn_x = rearrange_state_vector_torch(
            state=state.half(),
            num_nodes=cell_num,
        ).half()

        with torch.no_grad():
            action = model(gnn_x, edge_index).squeeze()

        state = env.step_env(action)

    return phase_metrics_from_waveforms(
        waveforms=waveforms,
        desired_lag=target.detach().float().cpu().numpy(),
        stable_window=STABLE_WINDOW,
        phase_sign="auto",
        metric_start=None,
        metric_end=None,
    )


def make_small_world_edge_index(cell_num, k, rewiring_prob, seed):
    graph = nx.connected_watts_strogatz_graph(
        n=cell_num,
        k=k,
        p=rewiring_prob,
        seed=seed,
    )
    edge_list = []
    for source, target in graph.edges():
        edge_list.append((source, target))
        edge_list.append((target, source))

    return torch.as_tensor(edge_list, dtype=torch.long, device=DEVICE).t().contiguous()


def make_random_targets(cell_num, target_num, seed):
    rng = np.random.default_rng(seed)
    target_list = torch.zeros(
        (target_num, cell_num),
        dtype=DEVICE_DTYPE,
        device=DEVICE,
    )
    for target_idx in range(target_num):
        rand_angles = rng.integers(0, cell_num, cell_num)
        rand_angles[0] = 0
        target_list[target_idx] = torch.as_tensor(
            rand_angles / cell_num,
            dtype=DEVICE_DTYPE,
            device=DEVICE,
        )
    return target_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sparsity tests on p=0 small-world ring-lattice graphs."
    )
    parser.add_argument(
        "cell_num",
        nargs="?",
        type=int,
        default=CELL_NUM,
        help=f"number of cells to test, default: {CELL_NUM}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_policy(heads=HEADS, feature_dim=FEATURE_DIM)
    model.half()

    cell_num = args.cell_num
    target_num = cell_num * 2
    target_list = make_random_targets(cell_num, target_num, TARGET_SEED)

    env = CDSEnv(cell_nums=cell_num, env_length=ENV_LENGTH, hz=HZ, device=DEVICE)
    env.to_half()
    env.omega = torch.tensor(np.pi * 2, dtype=DEVICE_DTYPE, device=DEVICE)

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"sparsity_small_world_p0_{cell_num}.csv"

    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        try:
            for k in range(2, cell_num, 2):
                edge_index = make_small_world_edge_index(
                    cell_num,
                    k,
                    REWIRING_PROB,
                    GRAPH_SEED,
                )
                order_parameter_list = np.zeros(target_num)
                phase_rmse_list = np.zeros(target_num)

                for target_idx, target in enumerate(target_list):
                    metrics = get_phase_metrics(
                        cell_num,
                        edge_index,
                        model,
                        env,
                        target,
                        EVAL_LENGTH,
                    )
                    order_parameter = metrics["order_parameter_mean"]
                    phase_rmse = metrics["phase_rmse_deg_mean"]

                    print(
                        "cell_num:",
                        cell_num,
                        "k:",
                        k,
                        "target:",
                        target_idx,
                        "order_parameter:",
                        order_parameter,
                        "phase_sign:",
                        metrics["phase_sign"],
                        "phase_rmse_deg:",
                        phase_rmse,
                    )

                    order_parameter_list[target_idx] = order_parameter
                    phase_rmse_list[target_idx] = phase_rmse

                writer.writerow(["order_parameter", k, *order_parameter_list])
                writer.writerow(["phase_rmse_deg", k, *phase_rmse_list])
                csvfile.flush()

        except KeyboardInterrupt:
            print("\nReceived Ctrl+C, exiting gracefully...")

    print(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    main()
