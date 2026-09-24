"""Train the 31D->CTBR behavior-cloning policy from expert/DAgger datasets."""

from __future__ import annotations

import argparse
import sys
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from imitation.bc_policy import Circular12BCConfig, Circular12BCPolicy
from imitation.dataset import concatenate_datasets, load_dataset


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset", action="append", type=Path, required=True,
    help="Repeat for the base expert dataset and any DAgger aggregation datasets.",
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--weight-decay", type=float, default=1.0e-5)
parser.add_argument(
    "--observation-std-floor",
    type=float,
    default=1.0e-4,
    help=(
        "Minimum per-dimension observation standard deviation used for "
        "normalization. Raising this reduces extreme OOD amplification on "
        "nearly constant state/action dimensions."
    ),
)
parser.add_argument(
    "--standardized-observation-clip",
    type=float,
    default=None,
    help=(
        "Optional symmetric clamp applied after normalization and before the "
        "BC network, e.g. 5.0 for [-5, +5] sigma. The same clamp is stored "
        "inside the checkpoint and used during inference."
    ),
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
if args.observation_std_floor <= 0.0:
    parser.error("--observation-std-floor must be positive")
if (
    args.standardized_observation_clip is not None
    and args.standardized_observation_clip <= 0.0
):
    parser.error("--standardized-observation-clip must be positive")


def _dataset_target_speed(
    dataset: dict[str, np.ndarray],
) -> float:
    value = dataset.get("target_speed_mps")
    if value is None:
        raise ValueError(
            "Circular-12 BC datasets must record target_speed_mps"
        )
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError("target_speed_mps metadata must be scalar")
    return float(array.reshape(-1)[0])


def _split_by_episode(
    episode_id: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(episode_id)
    if unique.size < 3:
        raise ValueError(
            "BC training requires at least three independent episodes"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(0.1 * unique.size)))
    n_test = max(1, int(round(0.1 * unique.size)))
    n_train = unique.size - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough episodes for train/val/test split")
    train_eps = unique[:n_train]
    val_eps = unique[n_train : n_train + n_val]
    test_eps = unique[n_train + n_val :]
    return (
        np.isin(episode_id, train_eps),
        np.isin(episode_id, val_eps),
        np.isin(episode_id, test_eps),
    )


def _metrics(
    model: Circular12BCPolicy,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        pred = model(x)
        error = pred - y
        return {
            "mae": float(error.abs().mean().item()),
            "rmse": float(torch.sqrt(error.square().mean()).item()),
            "sign_agreement": float(
                (
                    (torch.sign(pred) == torch.sign(y))
                    | (y.abs() < 0.05)
                ).float().mean().item()
            ),
            "pred_abs_mean": float(pred.abs().mean().item()),
            "target_abs_mean": float(y.abs().mean().item()),
        }


def main() -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    datasets = [load_dataset(path) for path in args.dataset]
    target_speeds = [_dataset_target_speed(dataset) for dataset in datasets]
    if max(target_speeds) - min(target_speeds) > 1.0e-6:
        raise ValueError(
            "Do not mix hidden target speeds in one BC dataset. "
            "Train sequential speed-curriculum checkpoints instead: "
            f"got {target_speeds}"
        )
    target_speed_mps = float(target_speeds[0])
    data = concatenate_datasets(*datasets)
    observation = data["observation"].astype(np.float32)
    target = data["expert_action"].astype(np.float32)
    train_mask, val_mask, test_mask = _split_by_episode(
        data["episode_id"], args.seed
    )

    obs_mean = torch.from_numpy(observation[train_mask].mean(axis=0))
    raw_obs_std = torch.from_numpy(
        observation[train_mask].std(axis=0)
    )
    obs_std = raw_obs_std.clamp_min(
        float(args.observation_std_floor)
    )
    model = Circular12BCPolicy(
        Circular12BCConfig(
            standardized_observation_clip=(
                float(args.standardized_observation_clip)
                if args.standardized_observation_clip is not None
                else None
            )
        ),
        observation_mean=obs_mean,
        observation_std=obs_std,
    ).to(device)

    x_train = torch.from_numpy(observation[train_mask]).to(device)
    y_train = torch.from_numpy(target[train_mask]).to(device)
    x_val = torch.from_numpy(observation[val_mask]).to(device)
    y_val = torch.from_numpy(target[val_mask]).to(device)
    x_test = torch.from_numpy(observation[test_mask]).to(device)
    y_test = torch.from_numpy(target[test_mask]).to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = torch.nn.SmoothL1Loss(beta=0.05)

    best_val = float("inf")
    best_state = None
    history: list[dict] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))

        val = _metrics(model, x_val, y_val)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"[bc] epoch={epoch:03d} loss={row['train_loss']:.6f} "
            f"val_mae={val['mae']:.5f} val_rmse={val['rmse']:.5f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("BC training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    train_metrics = _metrics(model, x_train, y_train)
    val_metrics = _metrics(model, x_val, y_val)
    test_metrics = _metrics(model, x_test, y_test)

    metadata = {
        "datasets": [str(path) for path in args.dataset],
        "samples": int(observation.shape[0]),
        "episodes": int(np.unique(data["episode_id"]).size),
        "target_speed_mps": target_speed_mps,
        "observation_std_floor": float(args.observation_std_floor),
        "standardized_observation_clip": (
            float(args.standardized_observation_clip)
            if args.standardized_observation_clip is not None
            else None
        ),
        "raw_observation_std_min": float(raw_obs_std.min().item()),
        "effective_observation_std_min": float(obs_std.min().item()),
        "floored_observation_dimensions": int(
            (raw_obs_std < float(args.observation_std_floor)).sum().item()
        ),
        "best_epoch": int(
            min(history, key=lambda row: row["val_rmse"])["epoch"]
        ),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "observation_mean": model.observation_mean.detach().cpu().tolist(),
        "observation_std": model.observation_std.detach().cpu().tolist(),
    }
    model.cpu().save(args.output, metadata=metadata)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
