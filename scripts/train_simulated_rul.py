"""Fit a simulated RUL predictor and verify it on independent equipment."""
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import _bootstrap  # Sets up repository imports.
import numpy as np

from rul_synthesis.degradation_model import EquipmentDegradationModel
from rul_synthesis.predictor import (
    BOARDS_PER_CYCLE, HISTORY_INSPECTIONS, MIN_INSPECTIONS, MODEL_PATH, SimulatedPredictor, can_show_estimate,
)


def recent_inputs(machine):
    counts = np.asarray(machine.defect_progression)
    samples = np.arange(MIN_INSPECTIONS, len(counts) + 1)
    cumulative = np.r_[0, np.cumsum(counts)]
    start = np.maximum(0, samples - HISTORY_INSPECTIONS)
    return (samples * BOARDS_PER_CYCLE, cumulative[samples], (samples - start) * BOARDS_PER_CYCLE,
            cumulative[samples] - cumulative[start], len(counts) - samples)


def evaluate(machines, predictor):
    rows = []
    for machine in machines:
        boards, cumulative, recent_boards, recent_defects, target = recent_inputs(machine)
        low, median, high = predictor.remaining_cycles(boards, cumulative, recent_boards, recent_defects).T
        old = predictor.remaining_cycles(boards, cumulative)[:, 1]
        rows.extend(zip(target, old, median, low, high))
    target, old, new, low, high = np.array(rows).T
    report = {}
    for label, mask in {
        'all_supported': np.isfinite(new) & np.isfinite(old),
        'app_displays_estimate': can_show_estimate(low, high),
    }.items():
        comparable = mask & np.isfinite(old)
        report[label] = {
            'observations': int(mask.sum()),
            'share_of_observations': round(float(mask.mean()), 3),
            'cumulative_only_mae_boards': round(float(np.abs(old[comparable] - target[comparable]).mean()) * BOARDS_PER_CYCLE),
            'with_recent_timing_mae_boards': round(float(np.abs(new[mask] - target[mask]).mean()) * BOARDS_PER_CYCLE),
            'range_coverage': round(float(((target >= low) & (target <= high))[mask].mean()), 3),
        }
    return report


def main():
    train = EquipmentDegradationModel(seed=42).simulate_batch(600, 1200)
    test_seed = 942071
    test = EquipmentDegradationModel(seed=test_seed).simulate_batch(180, 1200)
    if any(machine.failure_cycle is None for machine in train + test):
        raise SystemExit('At least one simulated machine did not reach the failure point.')

    # Fit the distribution of wear rates and lifetimes from training runs only.
    sorted_train = sorted(train, key=lambda machine: machine.failure_cycle)
    rates = [machine.degradation_rate for machine in sorted_train]
    failures = [machine.failure_cycle for machine in sorted_train]
    predictor = SimulatedPredictor(np.array(rates), np.array(failures))

    report = evaluate(test, predictor)
    print(json.dumps(report, indent=2))
    overall = report['all_supported']
    if overall['with_recent_timing_mae_boards'] >= overall['cumulative_only_mae_boards']:
        raise SystemExit('The timing model did not beat the cumulative model on this test.')

    model = {
        'version': 2, 'synthetic': True, 'train_seed': 42, 'test_seed': test_seed,
        'train_equipment': len(train), 'test_equipment': len(test),
        'wear_rates': rates, 'failure_cycles': failures, 'test_results': report,
    }
    with NamedTemporaryFile(mode='w', dir=MODEL_PATH.parent, delete=False) as file:
        json.dump(model, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')
    Path(file.name).replace(MODEL_PATH)
    print(f'Saved app model to {MODEL_PATH}')


if __name__ == '__main__':
    main()
