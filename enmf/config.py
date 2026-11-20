"""Experiment parameters stored in JSON."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    train_fraction: float
    embedding_dim: int
    dropout: float
    missing_weight: float
    learning_rate: float
    epochs: int
    batch_size: int
    eval_batch_size: int
    top_k: list[int]
    seed: int
    threads: int

    @classmethod
    def read(cls, path: Path) -> "TrainConfig":
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        for name in ("embedding_dim", "epochs", "batch_size", "eval_batch_size", "threads"):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < config.train_fraction < 1 or not 0 <= config.dropout < 1:
            raise ValueError("train_fraction must be in (0, 1) and dropout in [0, 1).")
        if not 0 < config.missing_weight <= 1:
            raise ValueError("missing_weight must be in (0, 1].")
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if type(config.seed) is not int or not 0 <= config.seed <= 2**32 - 1:
            raise ValueError("seed must be an integer in [0, 2**32 - 1].")
        if (not isinstance(config.top_k, list) or not config.top_k
                or any(type(k) is not int or k < 1 for k in config.top_k)
                or len(set(config.top_k)) != len(config.top_k)):
            raise ValueError("top_k must contain distinct positive integers.")
        return config
