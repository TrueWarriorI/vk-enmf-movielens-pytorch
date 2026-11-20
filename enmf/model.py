"""User-based ENMF with the exact whole-catalogue objective."""

import torch
from torch import nn


class ENMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.users = nn.Embedding(num_users, embedding_dim)
        self.items = nn.Embedding(num_items, embedding_dim)
        self.projection = nn.Parameter(torch.ones(embedding_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)

    def forward(
        self,
        users: torch.Tensor,
        positive_rows: torch.Tensor,
        positive_items: torch.Tensor,
        missing_weight: float,
    ) -> torch.Tensor:
        user_features = self.dropout(self.users(users)) * self.projection
        item_features = self.items.weight
        item_gram = item_features.T @ item_features
        # Sum squared scores for the whole catalogue without building the score matrix.
        squared_scores = ((user_features @ item_gram) * user_features).sum()

        positive_scores = (user_features[positive_rows] * item_features[positive_items]).sum(dim=1)
        # Replace the missing-pair penalty with the error for a positive target.
        positive_correction = (
            (positive_scores - 1).square() - missing_weight * positive_scores.square()
        ).sum()
        return missing_weight * squared_scores + positive_correction

    def scores(self, users: torch.Tensor) -> torch.Tensor:
        """Score each user against every item, with dropout disabled."""
        return (self.users(users) * self.projection) @ self.items.weight.T
