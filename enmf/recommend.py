"""Load the trained model and rank unseen movies for a MovieLens user."""

from pathlib import Path

import numpy as np
import torch

from .data import Interactions
from .model import ENMF
from .ranking import top_items


@torch.inference_mode()
def recommend(checkpoint: Path, user_id: int, k: int) -> list[dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    user_ids = state["user_ids"].tolist()
    item_ids = state["item_ids"]
    try:
        user_index = user_ids.index(user_id)
    except ValueError:
        raise ValueError(f"Unknown user ID {user_id}. Use a MovieLens user present in train.") from None

    model = ENMF(len(user_ids), len(item_ids), state["embedding_dim"], dropout=0)
    model.load_state_dict(state["model"])
    model.eval()
    train = Interactions(state["seen_indptr"].numpy(), state["seen_items"].numpy())
    users = np.array([user_index], dtype=np.int64)
    scores = model.scores(torch.from_numpy(users))
    ranked_items = top_items(scores, train.batch_pairs(users), k)[0]
    ranked_items = ranked_items[ranked_items >= 0]
    movie_ids = item_ids[ranked_items].tolist()
    movie_scores = scores[0, ranked_items].tolist()
    return [
        {"item_id": item, "score": score}
        for item, score in zip(movie_ids, movie_scores)
    ]
