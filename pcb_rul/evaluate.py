"""Open the held-out synthetic test split once, after the validation gate."""
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from config import MEMBER_SEEDS, OUTPUTS, ROOT, RUL_CAP, SEED, seed_everything
from ensemble import blend, smooth
from features import N_WINDOW_FEATURES, build, load_split
from train_gru import Histories, load_model, predict, training_device

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/runtime/matplotlib"))
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt


def metrics(prediction, truth):
    residual = prediction - truth
    nasa = np.expm1(np.where(residual < 0, -residual / 13, residual / 10))
    mae = np.abs(residual).mean()
    return {
        "mae_steps": float(mae),
        "rmse_steps": float(np.sqrt(np.mean(residual**2))),
        "nasa_sum": float(nasa.sum()),
        "nasa_mean": float(nasa.mean()),
        "mae_percent_of_cap": float(100 * mae / RUL_CAP),
    }


def plots(data, prediction):
    unit, step, true = (data[key] for key in ("unit", "step", "true_rul"))
    capped = np.minimum(true, RUL_CAP)
    units = np.random.default_rng(SEED).choice(np.unique(unit), 5, replace=False)
    figure, axes = plt.subplots(5, 1, figsize=(10, 12), layout="constrained")
    for equipment, ax in zip(sorted(units), axes, strict=True):
        mask = unit == equipment
        ax.plot(step[mask], true[mask], color="0.7", label="True RUL")
        ax.plot(step[mask], capped[mask], linestyle="--", label="Capped target")
        ax.plot(step[mask], prediction[mask], label="Predicted RUL")
        ax.set(title=f"Unit {equipment}", ylabel="Steps left", xlabel="Elapsed inspection steps")
    axes[0].legend()
    figure.savefig(OUTPUTS / "five_test_units.png", dpi=150)
    plt.close(figure)

    fraction = step / (step + true)
    stage = np.minimum(9, (10 * fraction).astype(int))
    figure, ax = plt.subplots(layout="constrained")
    for target, label in ((capped, "Capped target"), (true, "Uncapped true RUL")):
        mae = [np.abs(prediction[stage == i] - target[stage == i]).mean() for i in range(10)]
        ax.plot(np.arange(5, 100, 10), mae, marker="o", label=label)
    ax.set(xlabel="Percent of simulated life elapsed", ylabel="Mean absolute error (steps)")
    ax.legend()
    figure.savefig(OUTPUTS / "error_by_life_stage.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, target, title in zip(axes, (true, capped), ("Uncapped true RUL", "Capped target"), strict=True):
        ax.scatter(target, prediction, s=2, alpha=0.1)
        ax.plot([0, RUL_CAP], [0, RUL_CAP], color="black", linestyle="--")
        ax.set(title=title, xlabel="True steps left", ylabel="Predicted steps left")
    figure.savefig(OUTPUTS / "predicted_vs_true.png", dpi=150)
    plt.close(figure)


def main():
    seed_everything()
    torch.set_num_threads(5)
    gate = subprocess.run([sys.executable, str(Path(__file__).with_name("tests.py"))])
    if gate.returncode:
        raise SystemExit("Stopped before opening the test set. See the validation check above.")

    # Fingerprint the frozen model artifacts before reading a test observation.
    selection = json.loads((OUTPUTS / "ensemble.json").read_text())
    weight = selection["weight_lightgbm"]
    frozen = {}
    artifacts = ["scaling.npz", "baselines.npz", "time_only.txt", "lightgbm_trials.json",
                 "gru_training.json", "ensemble.json"]
    artifacts.extend(f"lgbm_{i}.txt" for i in range(len(MEMBER_SEEDS)))
    artifacts.extend(f"gru_{i}.pt" for i in range(len(MEMBER_SEEDS)))
    for name in artifacts:
        with (OUTPUTS / name).open("rb") as file:
            frozen[name] = file_digest(file, "sha256").hexdigest()
    with (OUTPUTS / "test_access.json").open("x") as file:
        json.dump({"validation": selection, "seed_members": MEMBER_SEEDS, "sha256": frozen}, file, indent=2)

    data = load_split("test")
    x = build("test", data)
    device = training_device()
    with np.load(OUTPUTS / "scaling.npz") as stats:
        history = Histories(data, stats, device)
    lgbm = np.stack([lgb.Booster(model_file=str(OUTPUTS / f"lgbm_{i}.txt")).predict(x)
                     for i in range(len(MEMBER_SEEDS))])
    gru = np.stack([predict(load_model(i, device), history) for i in range(len(MEMBER_SEEDS))])
    with np.load(OUTPUTS / "baselines.npz") as baseline:
        mean = np.full(len(x), baseline["mean"])
        ridge = x[:, :N_WINDOW_FEATURES] @ baseline["ridge_coef"] + baseline["ridge_intercept"]
    time = lgb.Booster(model_file=str(OUTPUTS / "time_only.txt")).predict(data["step"][:, None])

    def steps(values):
        return np.clip(np.asarray(values, dtype=np.float64) * RUL_CAP, 0, RUL_CAP)

    def final(first, second):
        return smooth(blend(first, second, weight), data["unit"], data["step"])

    model_predictions = {
        "Train mean": (steps(mean), None),
        "Time-only LightGBM": (steps(time), None),
        "Ridge": (steps(ridge), None),
        "LightGBM": (steps(lgbm.mean(axis=0)), steps(lgbm)),
        "GRU": (steps(gru.mean(axis=0)), steps(gru)),
        "Final ensemble": (final(lgbm.mean(axis=0), gru.mean(axis=0)),
                           np.stack([final(a, b) for a, b in zip(lgbm, gru, strict=True)])),
    }
    true = data["true_rul"].astype(np.float64)
    near_failure = true <= 50
    rows, seed_rows = [], []
    for name, (prediction, members) in model_predictions.items():
        for subset, mask in (("all", np.ones(len(true), dtype=bool)), ("true RUL <= 50", near_failure)):
            for target, truth in (("true", true), ("capped", np.minimum(true, RUL_CAP))):
                result = {"model": name, "subset": subset, "target": target, "n": int(mask.sum()),
                          **metrics(prediction[mask], truth[mask])}
                if members is not None:
                    scores = pd.DataFrame([metrics(p[mask], truth[mask]) for p in members])
                    for key in scores:
                        result[f"{key}_seed_mean"] = float(scores[key].mean())
                        result[f"{key}_seed_std"] = float(scores[key].std(ddof=1))
                    for seed, row in zip(MEMBER_SEEDS, scores.to_dict("records"), strict=True):
                        seed_rows.append({"model": name, "subset": subset, "target": target, "seed": seed, **row})
                rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUTS / "test_comparison.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUTPUTS / "test_by_seed.csv", index=False)
    plots(data, model_predictions["Final ensemble"][0])
    (OUTPUTS / "test_report.json").write_text(json.dumps(
        {"synthetic": True, "test_units": int(len(np.unique(data["unit"]))), "validation": selection,
         "results": rows}, indent=2, allow_nan=False,
    ) + "\n")
    print(frame[["model", "subset", "target", "mae_steps", "rmse_steps", "nasa_sum", "nasa_mean",
                 "mae_percent_of_cap", "mae_steps_seed_mean", "mae_steps_seed_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
