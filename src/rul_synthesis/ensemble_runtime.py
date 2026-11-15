"""Run the evaluated PCB RUL ensemble on ordered, per-image inspections."""
from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path
from threading import RLock

import numpy as np

from pcb_rul.config import CLASSES, RUL_CAP, SEVERITY

MODEL_DIR = Path(__file__).resolve().parents[2] / 'artifacts/models/pcb_rul_ensemble'
METHOD = 'pcb_rul_ensemble'


def inspection_features(detections, width, height):
    """Match the observed feature order and normalized box areas used in training."""
    counts = np.zeros(len(CLASSES), dtype=np.float32)
    areas, confidences = [], []
    for detection in detections:
        name = detection['class_name']
        counts[CLASSES.index('open' if name == 'open_circuit' else name)] += 1
        x1, y1, x2, y2 = detection['bbox']
        areas.append((min(x2, width) - min(x1, width)) * (min(y2, height) - min(y1, height)) / (width * height))
        confidences.append(detection['confidence'])
    return np.array([*counts, sum(areas), max(areas, default=0),
                     np.mean(confidences) if confidences else 0, counts @ SEVERITY], dtype=np.float32)


@lru_cache(maxsize=1)
def load_ensemble():
    if not (MODEL_DIR / 'manifest.json').is_file():
        return None
    return RULEnsemble(MODEL_DIR)


class RULEnsemble:
    def __init__(self, directory):
        import lightgbm
        import torch
        from pcb_rul.train_gru import GRU

        torch.set_num_threads(1)
        self.directory = directory
        manifest = json.loads((directory / 'manifest.json').read_text())
        members = range(len(manifest['seed_members']))
        trees = [f'lgbm_{index}.txt' for index in members]
        networks = [f'gru_{index}.pt' for index in members]
        for name in ['ensemble.json', 'scaling.npz', 'validation_example.npz', *trees, *networks]:
            with (directory / name).open('rb') as file:
                if file_digest(file, 'sha256').hexdigest() != manifest['sha256'][name]:
                    raise ValueError(f'RUL ensemble checksum mismatch for {name}')
        self.weight = manifest['validation']['weight_lightgbm']
        with np.load(directory / 'scaling.npz') as scaling:
            self.scaling = dict(scaling)
        self.trees = [lightgbm.Booster(model_file=str(directory / name)) for name in trees]
        self.networks = []
        for name in networks:
            model = GRU()
            model.load_state_dict(torch.load(directory / name, map_location='cpu', weights_only=True))
            self.networks.append(model.eval())
        self.lock = RLock()

    def predict(self, observed):
        from pcb_rul.ensemble import blend, smooth
        from pcb_rul.features import unit_features
        from pcb_rul.train_gru import Histories, predict

        steps = np.arange(len(observed))
        units = np.zeros_like(steps)
        table = unit_features(observed, steps).astype(np.float32)
        table[:] = (table - self.scaling['tabular_mean']) / self.scaling['tabular_scale']
        # Histories carries training targets, which inference never reads.
        history = Histories(dict(observed=observed, step=steps, y=units, auxiliary=units), self.scaling)
        with self.lock:
            recurrent = np.mean([predict(model, history) for model in self.networks], axis=0)
            boosted = np.mean([tree.predict(table, num_threads=1) for tree in self.trees], axis=0)
        return smooth(blend(boosted, recurrent, self.weight), units, steps)

    @lru_cache(maxsize=1)
    def demo(self):
        with np.load(self.directory / 'validation_example.npz') as example:
            predicted = self.predict(example['observed'])
            actual = np.minimum(example['true_rul'], RUL_CAP)
            return {'unit': int(example['unit']), 'synthetic': True, 'cap': RUL_CAP,
                    'timeline': [{'step': step, 'predicted_rul_steps': round(float(value), 2),
                                  'true_rul_steps': int(actual[step])}
                                 for step, value in enumerate(predicted)]}
