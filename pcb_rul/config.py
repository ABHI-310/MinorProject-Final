"""Fixed experiment configuration."""
from pathlib import Path
import os
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(__file__).resolve().parent / "outputs"
SEED = 42
N_UNITS = 900
SPLIT = {"train": 600, "val": 150, "test": 150}
LIFE_RANGE = (400, 1200)
GAMMA_SHAPE = 0.8
RUL_CAP = 150
CLASSES = ("open", "short", "mouse_bite", "spur", "spurious_copper", "missing_hole")
BASE_RATE = np.array([0.02, 0.02, 0.05, 0.05, 0.05, 0.03])
GROWTH = np.array([3.0, 3.0, 1.5, 1.5, 1.5, 2.0])
POWER = 2
SEVERITY = np.array([1.0, 1.0, 0.4, 0.3, 0.4, 0.6])
UNIT_NOISE = (0, 0.3)
WINDOWS = (10, 30, 100)
OBSERVED = (*CLASSES, "total_area", "max_area", "mean_confidence", "weighted_count")

# Five reproducible runs, with the independent member seeds derived from 42.
MEMBER_SEEDS = (SEED, *(int(s.generate_state(1)[0] % (2**31)) for s in
                        np.random.SeedSequence(SEED).spawn(4)))


def seed_everything(seed=SEED):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
