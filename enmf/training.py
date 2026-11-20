"""Train ENMF-U, log each epoch and save the checkpoint selected on validation."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from .config import TrainConfig
from .data import MovieLensSplit, read_movielens
from .model import ENMF
from .ranking import evaluate, popularity
from .runtime import Runtime, distributed_runtime


def train_epoch(
    model: ENMF | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    split: MovieLensSplit,
    config: TrainConfig,
    runtime: Runtime,
    order: np.ndarray,
) -> float:
    model.train()
    loss_total = torch.zeros((), device=runtime.device, dtype=torch.float64)
    for start in range(0, len(order), config.batch_size):
        batch = order[start:start + config.batch_size]
        local_users = batch[runtime.rank::runtime.world_size]
        positive_rows, positive_items = split.train.batch_pairs(local_users)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = model(
            torch.as_tensor(local_users, device=runtime.device),
            torch.as_tensor(positive_rows, device=runtime.device),
            torch.as_tensor(positive_items, device=runtime.device),
            config.missing_weight,
        )
        # DDP averages gradients across ranks. Scale to the mean over actual global users.
        mean_loss = loss_sum * (runtime.world_size / len(batch))
        mean_loss.backward()
        optimizer.step()
        loss_total += loss_sum.detach()

    if runtime.backend:
        dist.all_reduce(loss_total)
    return loss_total.item() / len(order)


def save_checkpoint(
    path: Path, model: ENMF, split: MovieLensSplit, config: TrainConfig, epoch: int
) -> None:
    pending_checkpoint = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "embedding_dim": config.embedding_dim,
        "epoch": epoch,
        "user_ids": torch.from_numpy(split.user_ids),
        "item_ids": torch.from_numpy(split.item_ids),
        "seen_indptr": torch.from_numpy(split.train.indptr),
        "seen_items": torch.from_numpy(split.train.items),
    }, pending_checkpoint)
    pending_checkpoint.replace(path)


def train(config: TrainConfig, archive: Path, output: Path, device_type: str) -> None:
    with distributed_runtime(device_type, config.threads) as runtime:
        split = read_movielens(archive, config.train_fraction)
        if runtime.rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            print(json.dumps(split.stats), flush=True)
        run_training(config, split, output, runtime)


def run_training(config: TrainConfig, split: MovieLensSplit, output: Path, runtime: Runtime) -> None:
    primary_metric = f"ndcg@{config.top_k[0]}"
    torch.manual_seed(config.seed)
    model = ENMF(len(split.user_ids), len(split.item_ids), config.embedding_dim, config.dropout)
    model = model.to(runtime.device)
    training_model = model
    if runtime.backend:
        device_ids = [runtime.device.index] if runtime.device.type == "cuda" else None
        training_model = DistributedDataParallel(model, device_ids=device_ids)
    torch.manual_seed(config.seed + runtime.rank)
    shuffle_generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.Adagrad(training_model.parameters(), lr=config.learning_rate)

    history = []
    best = None
    writer = SummaryWriter(output / "tensorboard") if runtime.rank == 0 else None
    try:
        if writer is not None:
            popularity_scores = popularity(split.train, len(split.item_ids), runtime.device)
            baseline = evaluate(
                lambda users: popularity_scores.expand(len(users), -1),
                split, config.top_k, config.eval_batch_size, runtime.device,
            )
            for metric, value in baseline.items():
                writer.add_scalar(f"popularity/{metric}", value, 0)

        for epoch in range(1, config.epochs + 1):
            order = torch.randperm(len(split.user_ids), generator=shuffle_generator).numpy()
            start = perf_counter()
            loss = train_epoch(training_model, optimizer, split, config, runtime, order)
            train_seconds = perf_counter() - start
            if writer is None:
                continue

            model.eval()
            start = perf_counter()
            metrics = evaluate(
                model.scores, split, config.top_k, config.eval_batch_size, runtime.device
            )
            record = {
                "epoch": epoch, "loss": loss, "train_seconds": train_seconds,
                "validation_seconds": perf_counter() - start, **metrics,
            }
            history.append(record)
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("train/seconds", train_seconds, epoch)
            for metric, value in metrics.items():
                writer.add_scalar(f"validation/{metric}", value, epoch)
            if best is None or record[primary_metric] > best[primary_metric]:
                best = record
                save_checkpoint(output / "best.pt", model, split, config, epoch)
            print(f"epoch={epoch:02d} loss={loss:.5f} "
                  f"{primary_metric}={record[primary_metric]:.5f} "
                  f"train={train_seconds:.2f}s", flush=True)

        if writer is not None:
            results = {
                "config": asdict(config),
                "environment": {
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "numpy": np.__version__,
                    "os": platform.system(),
                    "device": runtime.device.type,
                    "gpu": (torch.cuda.get_device_name(runtime.device)
                            if runtime.device.type == "cuda" else None),
                    "world_size": runtime.world_size,
                    "backend": runtime.backend,
                },
                "data": split.stats,
                "selection_metric": primary_metric,
                "popularity": baseline,
                "best": best,
                "history": history,
            }
            (output / "metrics.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(f"Results saved to {output / 'metrics.json'}", flush=True)
    finally:
        if writer is not None:
            writer.close()
