"""Tests that must pass before opening the held-out test split."""
import json
import unittest

import numpy as np

from config import N_UNITS, OUTPUTS, SEED, SPLIT, CLASSES, seed_everything
from ensemble import smooth
from features import load_split, unit_features
from simulate import assignment, simulate_unit
from train_gru import GRU, Histories, training_device
from train_lgbm import features, training_params


def detector_rates():
    stats = json.loads((OUTPUTS / "detector_stats.json").read_text())["per_class"]
    return (np.array([stats[c][field] for c in CLASSES]) for field in ("recall", "fp_per_image"))


class DataTests(unittest.TestCase):
    def test_damage_never_decreases_and_stops_at_first_failure(self):
        recall, fp = detector_rates()
        # Generation checks every trajectory. This gate rechecks only train/validation units.
        seeds = np.random.SeedSequence(SEED).spawn(N_UNITS)
        splits = assignment()
        for unit in np.r_[splits["train"], splits["val"]]:
            _, remaining, damage = simulate_unit(np.random.default_rng(seeds[unit]), recall, fp)
            self.assertTrue(np.all(np.diff(damage) >= 0))
            self.assertTrue(np.all(damage[:-1] < 1))
            self.assertGreaterEqual(damage[-1], 1)
            self.assertEqual(remaining[-1], 0)

    def test_units_are_disjoint(self):
        splits = assignment()
        self.assertEqual({key: len(units) for key, units in splits.items()}, SPLIT)
        self.assertEqual(len(np.unique(np.concatenate(list(splits.values())))), N_UNITS)
        for name in ("train", "val"):
            np.testing.assert_array_equal(np.unique(load_split(name)["unit"]), np.sort(splits[name]))

    def test_truncation_does_not_change_past_features(self):
        values, _, _ = simulate_unit(np.random.default_rng(SEED), *detector_rates())
        steps = np.arange(len(values))
        full = unit_features(values, steps)
        unit = np.zeros_like(steps)
        prediction = np.random.default_rng(SEED).uniform(size=len(steps))
        for stop in (1, 10, 30, 100, len(values) // 2):
            np.testing.assert_array_equal(unit_features(values[:stop], steps[:stop]), full[:stop])
            np.testing.assert_array_equal(smooth(prediction[:stop], unit[:stop], steps[:stop]),
                                          smooth(prediction, unit, steps)[:stop])
        # The sequence view starts at the correct unit and pads only on the right.
        history = Histories({"observed": values[:110], "step": steps[:110],
                             "y": np.zeros(110), "auxiliary": np.zeros(110)},
                            {"sequence_mean": np.zeros(10), "sequence_scale": np.ones(10)})
        import torch
        seq, mask = history.batch(torch.tensor([0, 109]))
        self.assertEqual(mask.sum(dim=1).tolist(), [1, 100])
        self.assertFalse(torch.any(seq[0, 1:]))
        np.testing.assert_array_equal(seq[1, :, :10].numpy(), values[10:110].astype(np.float32))

    def test_same_seed_produces_identical_results(self):
        rates = tuple(detector_rates())
        outputs = [simulate_unit(np.random.default_rng(SEED), *rates) for _ in range(2)]
        for first, second in zip(*outputs, strict=True):
            np.testing.assert_array_equal(first, second)
        import torch
        seed_everything()
        device = training_device()
        sequences = torch.ones(2, 5, 11, device=device)
        mask = torch.ones(2, 5, dtype=torch.bool, device=device)
        predictions = []
        for _ in range(2):
            seed_everything()
            model = GRU().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            rul, damage = model(sequences, mask)
            loss = torch.nn.functional.huber_loss(rul, torch.zeros_like(rul))
            loss += 0.5 * torch.nn.functional.huber_loss(damage, torch.zeros_like(damage))
            loss.backward()
            optimizer.step()
            predictions.append(model.eval()(sequences, mask)[0].detach().cpu().numpy())
        np.testing.assert_array_equal(*predictions)

        import lightgbm as lgb
        # Sample training rows across the entire split, including the near-failure targets.
        indices = np.random.default_rng(SEED).choice(len(load_split("train")["y"]), 4096, replace=False)
        x, y = features("train")[indices], load_split("train")["y"][indices]
        params = {**training_params(), "seed": SEED}
        predictions = [lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=8).predict(x)
                       for _ in range(2)]
        np.testing.assert_array_equal(*predictions)


class ValidationTest(unittest.TestCase):
    def test_final_model_beats_time_only_on_validation(self):
        metrics = json.loads((OUTPUTS / "ensemble.json").read_text())
        self.assertLess(metrics["final_validation_mae_steps"], metrics["time_only_validation_mae_steps"],
                        "Stop: the final model did not beat the time-only baseline.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
