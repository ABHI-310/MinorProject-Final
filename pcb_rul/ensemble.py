"""Select a validation weight and apply causal smoothing in inspection steps."""
import json

import numpy as np
from scipy.signal import lfilter

if __package__:
    from .config import OUTPUTS, RUL_CAP
    from .features import load_split, unit_slices
else:
    from config import OUTPUTS, RUL_CAP
    from features import load_split, unit_slices


def blend(lightgbm, gru, weight):
    return weight * lightgbm + (1 - weight) * gru


def smooth(prediction, units, step):
    raw = np.asarray(prediction, dtype=np.float64) * RUL_CAP
    result = raw.copy()
    for rows in unit_slices(units):
        failure = step[rows] + raw[rows]
        # The initial EWMA is the first estimate for this unit.
        averaged, _ = lfilter([0.2], [1.0, -0.8], failure, zi=[0.8 * failure[0]])
        result[rows] = np.where(raw[rows] < 0.9 * RUL_CAP, averaged - step[rows], raw[rows])
    return np.clip(result, 0, RUL_CAP)


def choose_weight():
    val = load_split("val")
    lightgbm, gru = (np.load(OUTPUTS / f"{name}_val.npy").mean(axis=0) for name in ("lgbm", "gru"))
    scores = {
        i / 10: float(np.abs(blend(lightgbm, gru, i / 10) - val["y"]).mean()) for i in range(11)
    }
    weight = min(scores, key=scores.__getitem__)
    combined = smooth(blend(lightgbm, gru, weight), val["unit"], val["step"])
    with np.load(OUTPUTS / "baselines_val.npz") as baseline:
        time_only = np.clip(baseline["time_only"] * RUL_CAP, 0, RUL_CAP)
    target = val["y"] * RUL_CAP
    report = {
        "weight_lightgbm": weight,
        "weight_gru": 1 - weight,
        "blend_validation_mae_normalized": scores,
        "final_validation_mae_steps": float(np.abs(combined - target).mean()),
        "time_only_validation_mae_steps": float(np.abs(time_only - target).mean()),
    }
    (OUTPUTS / "ensemble.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(choose_weight(), indent=2), flush=True)
