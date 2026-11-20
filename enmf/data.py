"""Load MovieLens 1M and build a global chronological split."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import file_digest
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import numpy as np
import requests


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def download_movielens(destination: Path) -> Path:
    if destination.is_file():
        _check_archive(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as target:
        temporary = Path(target.name)
    try:
        with requests.get(MOVIELENS_URL, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as target:
                shutil.copyfileobj(response.raw, target)
        _check_archive(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _check_archive(path: Path) -> None:
    with path.open("rb") as source:
        checksum = file_digest(source, "sha256").hexdigest()
    if checksum != MOVIELENS_SHA256:
        raise ValueError(f"MovieLens 1M checksum mismatch: {path}")


@dataclass(frozen=True)
class Interactions:
    """A sparse binary matrix stored as sorted item indices per user."""

    indptr: np.ndarray
    items: np.ndarray

    @classmethod
    def from_pairs(cls, pairs: np.ndarray, user_count: int) -> "Interactions":
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        counts = np.bincount(pairs[:, 0], minlength=user_count)
        return cls(np.r_[0, counts.cumsum()], pairs[order, 1])

    def batch_pairs(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        starts = self.indptr[users]
        counts = self.indptr[users + 1] - starts
        batch_rows = np.repeat(np.arange(len(users)), counts)
        # Map each position in the packed batch back to its position in self.items.
        batch_starts = counts.cumsum() - counts
        positions = np.arange(len(batch_rows)) + (starts - batch_starts)[batch_rows]
        return batch_rows, self.items[positions]


@dataclass(frozen=True)
class MovieLensSplit:
    user_ids: np.ndarray
    item_ids: np.ndarray
    titles: dict[int, str]
    train: Interactions
    validation: Interactions
    stats: dict[str, int | str]


def read_movielens(archive: Path, train_fraction: float) -> MovieLensSplit:
    if not archive.is_file():
        raise ValueError(f"Dataset is missing: {archive}. Run python -m enmf download.")
    with ZipFile(archive) as dataset:
        with dataset.open("ml-1m/ratings.dat") as source:
            events = np.array([line.split(b"::") for line in source], dtype=np.int64)
        with dataset.open("ml-1m/movies.dat") as source:
            movies = [line.decode("latin-1").split("::", 2) for line in source]
    titles = {int(item_id): title for item_id, title, _ in movies}
    return chronological_split(events[:, [0, 1, 3]], train_fraction, titles)


def chronological_split(
    events: np.ndarray, train_fraction: float, titles: dict[int, str] | None = None
) -> MovieLensSplit:
    """Use the first timestamp of each pair and keep train strictly before validation."""
    if not 0 < train_fraction < 1 or len(events) < 2:
        raise ValueError("The time split needs at least two events and a fraction in (0, 1).")

    order = np.lexsort((events[:, 2], events[:, 1], events[:, 0]))
    ordered = events[order]
    first_per_pair = np.r_[True, np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)]
    unique = ordered[first_per_pair]
    if len(unique) < 2:
        raise ValueError("The time split needs at least two distinct pairs.")

    split_index = max(1, int(len(unique) * train_fraction))
    cutoff = int(np.partition(unique[:, 2], split_index)[split_index])
    train_events = unique[unique[:, 2] < cutoff]
    future = unique[unique[:, 2] >= cutoff]
    if len(train_events) == 0:
        raise ValueError("No interactions before the time cutoff.")

    users = np.unique(train_events[:, 0])
    items = np.unique(train_events[:, 1])
    known_users = np.isin(future[:, 0], users)
    known_items = np.isin(future[:, 1], items)
    valid_events = future[known_users & known_items]
    if len(valid_events) == 0:
        raise ValueError("No validation pairs with both user and item known from train.")

    train_pairs = np.column_stack((
        np.searchsorted(users, train_events[:, 0]),
        np.searchsorted(items, train_events[:, 1]),
    ))
    valid_pairs = np.column_stack((
        np.searchsorted(users, valid_events[:, 0]),
        np.searchsorted(items, valid_events[:, 1]),
    ))
    train = Interactions.from_pairs(train_pairs, len(users))
    validation = Interactions.from_pairs(valid_pairs, len(users))
    stats = {
        "source_events": len(events),
        "unique_pairs": len(unique),
        "train_users": len(users),
        "train_items": len(items),
        "train_pairs": len(train_events),
        "train_last_timestamp": int(train_events[:, 2].max()),
        "cutoff_timestamp": cutoff,
        "cutoff_utc": datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),
        "future_pairs": len(future),
        "validation_pairs": len(valid_events),
        "validation_users": int(np.count_nonzero(np.diff(validation.indptr))),
        "excluded_unknown_user_only": int(np.count_nonzero(~known_users & known_items)),
        "excluded_unknown_item_only": int(np.count_nonzero(known_users & ~known_items)),
        "excluded_both_unknown": int(np.count_nonzero(~known_users & ~known_items)),
    }
    return MovieLensSplit(users, items, titles or {}, train, validation, stats)
