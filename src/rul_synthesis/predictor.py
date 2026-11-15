"""Estimate simulated remaining life from cumulative boards and defect counts."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from rul_synthesis.degradation_model import BACKGROUND_DEFECTS, DEGRADATION_DEFECT_SCALE

MODEL_PATH = Path(__file__).with_name('trained_predictor.json')
BOARDS_PER_CYCLE = 100
MIN_INSPECTIONS = 5
MIN_BOARDS = MIN_INSPECTIONS * BOARDS_PER_CYCLE
HISTORY_INSPECTIONS = 20
MAX_ESTIMATE_WIDTH_BOARDS = 4_000


def can_show_estimate(low, high):
    return (high - low) * BOARDS_PER_CYCLE <= MAX_ESTIMATE_WIDTH_BOARDS


@dataclass(frozen=True)
class SimulatedPredictor:
    wear_rates: np.ndarray
    failure_cycles: np.ndarray

    @classmethod
    def load(cls, path: Path = MODEL_PATH):
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        if data['version'] != 2 or data['synthetic'] is not True:
            raise ValueError(f'Unsupported RUL predictor artifact: {path}')
        return cls(np.array(data['wear_rates']), np.array(data['failure_cycles']))

    def expected_defects(self, observations):
        rate = self.wear_rates
        first = np.expm1(-rate * observations) / np.expm1(-rate)
        second = np.expm1(-2 * rate * observations) / np.expm1(-2 * rate)
        return BACKGROUND_DEFECTS * observations + DEGRADATION_DEFECT_SCALE * (
            observations - 2 * first + second
        )

    def remaining_cycles(self, boards, defects, recent_boards=None, recent_defects=None):
        """Return the 10th, 50th and 90th percentiles, in production cycles."""
        observations = np.atleast_1d(boards).astype(float)[:, None] / BOARDS_PER_CYCLE
        counts = np.atleast_1d(defects)[:, None]
        eligible = self.failure_cycles >= observations - 1
        supported = eligible.any(axis=1) & (observations[:, 0] >= MIN_BOARDS / BOARDS_PER_CYCLE)
        expected = self.expected_defects(observations[supported])
        observed = counts[supported]
        log_likelihood = observed * np.log(expected) - expected
        saturated = observed * np.log(np.maximum(observed, 1)) - observed
        if recent_boards is not None:
            recent_cycles = np.atleast_1d(recent_boards)[supported, None] / BOARDS_PER_CYCLE
            recent_counts = np.atleast_1d(recent_defects)[supported, None]
            earlier = self.expected_defects(observations[supported] - recent_cycles)
            # Use disjoint windows so recent counts are counted exactly once.
            earlier_counts = observed - recent_counts
            log_likelihood = (earlier_counts * np.log(np.maximum(earlier, 1e-12))
                              + recent_counts * np.log(expected - earlier) - expected)
            saturated = (earlier_counts * np.log(np.maximum(earlier_counts, 1))
                         + recent_counts * np.log(np.maximum(recent_counts, 1)) - observed)
        log_likelihood = np.where(eligible[supported], log_likelihood, -np.inf)
        best = log_likelihood.max(axis=1, keepdims=True)
        # Decline implausible counts even when the remaining candidates agree.
        plausible = 2 * (saturated - best) <= 25
        weights = np.exp(log_likelihood - best)
        cumulative = weights.cumsum(axis=1)
        percentiles = np.column_stack([
            self.failure_cycles[(cumulative >= cumulative[:, -1:] * q).argmax(axis=1)]
            for q in (0.1, 0.5, 0.9)
        ])
        estimates = np.full((len(observations), 3), np.nan)
        estimates[supported] = np.where(plausible, percentiles - observations[supported] + 1, np.nan)
        return estimates
