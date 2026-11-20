"""Full-catalogue ranking and binary NDCG for models with the same candidates."""

from collections.abc import Callable

import numpy as np
import torch

from .data import Interactions, MovieLensSplit


def top_items(scores: torch.Tensor, seen: tuple[np.ndarray, np.ndarray], k: int) -> torch.Tensor:
    rows, columns = (torch.as_tensor(indices, device=scores.device) for indices in seen)
    scores = scores.clone()
    scores[rows, columns] = -torch.inf
    # The columns follow ascending item IDs, so stable sort breaks ties by ID.
    order = scores.argsort(dim=1, descending=True, stable=True)[:, :k]
    return order.masked_fill(scores.gather(1, order).isneginf(), -1)


def ndcg_scores(hits: torch.Tensor, positive_counts: torch.Tensor, k: int) -> torch.Tensor:
    """Return one score per user, with zero for users without relevant items."""
    width = min(k, hits.shape[1])
    rank_denominators = torch.arange(2, width + 2, dtype=torch.float64, device=hits.device)
    discounts = rank_denominators.log2().reciprocal()
    dcg = (hits[:, :width] * discounts).sum(dim=1)
    ideal_by_count = torch.cat((discounts.new_zeros(1), discounts.cumsum(0)))
    idcg = ideal_by_count[positive_counts.clamp(max=width)]
    return dcg / torch.where(idcg > 0, idcg, 1.0)


def popularity(train: Interactions, num_items: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(train.items, minlength=num_items)
    return torch.as_tensor(counts, dtype=torch.float32, device=device)


@torch.inference_mode()
def evaluate(
    score_users: Callable[[torch.Tensor], torch.Tensor],
    split: MovieLensSplit,
    cutoffs: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    counts = np.diff(split.validation.indptr)
    users = np.flatnonzero(counts)
    max_k = max(cutoffs)
    totals = torch.zeros(len(cutoffs), dtype=torch.float64, device=device)
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        predictions = score_users(torch.as_tensor(batch, device=device))
        ranked = top_items(predictions, split.train.batch_pairs(batch), max_k)

        labels = torch.zeros_like(predictions, dtype=torch.bool)
        rows, columns = split.validation.batch_pairs(batch)
        labels[torch.as_tensor(rows, device=device), torch.as_tensor(columns, device=device)] = True
        hits = labels.gather(1, ranked.clamp_min(0)) & (ranked >= 0)
        positive_counts = torch.as_tensor(counts[batch], device=device)
        totals += torch.stack([ndcg_scores(hits, positive_counts, k).sum() for k in cutoffs])
    return {f"ndcg@{k}": value for k, value in zip(cutoffs, (totals / len(users)).tolist())}
