import unittest

import numpy as np
import torch

from enmf.config import TrainConfig
from enmf.data import chronological_split
from enmf.model import ENMF
from enmf.ranking import ndcg_scores, popularity, top_items
from enmf.runtime import Runtime
from enmf.training import train_epoch


class ModelTests(unittest.TestCase):
    def test_sparse_loss_and_gradients_match_dense_weighted_loss(self):
        torch.manual_seed(42)
        model = ENMF(4, 5, 3, dropout=0.5).double()
        users = torch.tensor([2, 0, 3])
        rows, items = torch.tensor([0, 0, 2]), torch.tensor([1, 4, 3])
        negative_weight = 0.3

        torch.manual_seed(12)
        sparse = model(users, rows, items, negative_weight)
        torch.manual_seed(12)
        features = model.dropout(model.users(users)) * model.projection
        scores = features @ model.items.weight.T
        labels = torch.zeros_like(scores)
        labels[rows, items] = 1
        dense = (torch.where(labels.bool(), 1.0, scores.new_tensor(negative_weight))
                 * (labels - scores).square()).sum()

        torch.testing.assert_close(sparse, dense, rtol=1e-12, atol=1e-12)
        parameters = tuple(model.parameters())
        sparse_grads = torch.autograd.grad(sparse, parameters)
        dense_grads = torch.autograd.grad(dense, parameters)
        for actual, expected in zip(sparse_grads, dense_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_empty_shard_produces_zero_gradients(self):
        model = ENMF(2, 3, 4, dropout=0)
        empty = torch.empty(0, dtype=torch.long)
        loss = model(empty, empty, empty, 0.5)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        for parameter in model.parameters():
            self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)

    def test_training_matches_dense_updates_including_short_batch(self):
        split = chronological_split(np.array([
            [1, 10, 1], [2, 20, 2], [3, 20, 2], [3, 10, 3],
        ]), 0.75)
        config = TrainConfig(0.75, 3, 0, 0.3, 0.1, 1, 2, 2, [1], 42, 1)
        model = ENMF(3, 2, 3, dropout=0)
        reference = ENMF(3, 2, 3, dropout=0)
        reference.load_state_dict(model.state_dict())
        runtime = Runtime(torch.device("cpu"), 0, 1, None)
        order = np.array([2, 0, 1])
        actual = train_epoch(model, torch.optim.Adagrad(model.parameters(), lr=0.1),
                             split, config, runtime, order)
        optimizer = torch.optim.Adagrad(reference.parameters(), lr=0.1)
        losses = []
        for batch in (order[:2], order[2:]):
            optimizer.zero_grad()
            rows, items = split.train.batch_pairs(batch)
            scores = reference.scores(torch.from_numpy(batch))
            labels = torch.zeros_like(scores)
            labels[rows, items] = 1
            loss = (torch.where(labels.bool(), 1.0, 0.3) * (labels - scores).square()).sum()
            (loss / len(batch)).backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertAlmostEqual(actual, sum(losses) / len(order), places=6)
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)


class DataAndRankingTests(unittest.TestCase):
    def test_cutoff_separates_ties_and_evaluation_uses_train_catalogue(self):
        events = np.array([
            [10, 90, 5], [20, 40, 2], [10, 90, 1], [10, 40, 3],
            [20, 90, 3], [30, 40, 3], [10, 91, 4], [30, 92, 5],
        ])
        split = chronological_split(events, 0.5)
        self.assertEqual(split.stats["train_last_timestamp"], 2)
        self.assertEqual(split.stats["cutoff_timestamp"], 3)
        self.assertEqual(split.stats["validation_pairs"], 2)
        self.assertEqual(split.stats["excluded_unknown_user_only"], 1)
        self.assertEqual(split.stats["excluded_unknown_item_only"], 1)
        self.assertEqual(split.stats["excluded_both_unknown"], 1)
        self.assertEqual(split.user_ids.tolist(), [10, 20])
        self.assertEqual(split.item_ids.tolist(), [40, 90])
        self.assertEqual(split.train.items.tolist(), [1, 0])
        self.assertEqual(split.validation.items.tolist(), [0, 1])
        self.assertEqual(popularity(split.train, 2, torch.device("cpu")).tolist(), [1, 1])

    def test_masked_items_and_deterministic_ties(self):
        scores = torch.tensor([[9.0, 3.0, 3.0], [0.0, 6.0, 3.0]])
        seen = (np.array([0, 1, 1]), np.array([0, 1, 2]))
        self.assertEqual(top_items(scores, seen, 3).tolist(), [[1, 2, -1], [0, -1, -1]])

    def test_ndcg_with_multiple_positives_and_zero_hits(self):
        hits = torch.tensor([[False, True, True], [False, False, False]])
        scores = ndcg_scores(hits, torch.tensor([3, 1]), 2)
        discount = 1 / np.log2(3)
        self.assertAlmostEqual(scores[0].item(), discount / (1 + discount), places=12)
        self.assertEqual(scores[1].item(), 0)


if __name__ == "__main__":
    unittest.main()
