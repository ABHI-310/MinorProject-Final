"""Fit the required baselines and tune/finalize LightGBM on train/validation."""
import json
import os

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.linear_model import Ridge

from config import MEMBER_SEEDS, OUTPUTS, SEED, seed_everything
from features import N_WINDOW_FEATURES, load_split

def training_params():
    backend = os.environ.get("PCB_RUL_LIGHTGBM_DEVICE", "cpu")
    params = {"objective": "huber", "alpha": 0.9, "metric": "mae", "verbosity": -1,
              "num_threads": 5, "feature_pre_filter": False, "device_type": backend}
    if backend == "gpu":
        return {**params, "gpu_use_dp": True, "gpu_device_id": 0}
    if backend == "cpu":
        return {**params, "deterministic": True, "force_col_wise": True}
    raise ValueError(f"Unsupported LightGBM device: {backend}")


def fit(train, val, params, seed):
    return lgb.train(
        {**params, "seed": seed}, train, num_boost_round=3000,
        valid_sets=[val], callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def features(split):
    return np.load(OUTPUTS / f"{split}_features.npy", mmap_mode="r")


def main():
    seed_everything()
    base = training_params()
    print(f"LightGBM backend: {base['device_type']}", flush=True)
    train, val = (load_split(split) for split in ("train", "val"))
    x_train, x_val = (features(split) for split in ("train", "val"))
    y_train, y_val = train["y"], val["y"]
    time_train = lgb.Dataset(train["step"][:, None], label=y_train)
    time_val = time_train.create_valid(val["step"][:, None], label=y_val)
    baseline = fit(time_train, time_val, {**base, "objective": "regression_l1", "verbosity": 1}, SEED)
    baseline.save_model(OUTPUTS / "time_only.txt")
    ridge = Ridge().fit(x_train[:, :N_WINDOW_FEATURES], y_train)
    np.savez(OUTPUTS / "baselines.npz", ridge_coef=ridge.coef_, ridge_intercept=ridge.intercept_,
             mean=y_train.mean())
    np.savez(OUTPUTS / "baselines_val.npz", time_only=baseline.predict(val["step"][:, None]),
             ridge=ridge.predict(x_val[:, :N_WINDOW_FEATURES]), mean=np.full(len(y_val), y_train.mean()))
    print("Baselines saved", flush=True)

    training = lgb.Dataset(x_train, label=y_train, params=base, free_raw_data=False)
    validation = training.create_valid(x_val, label=y_val)

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 10),
        }
        model = fit(training, validation, {**base, **params}, SEED)
        return model.best_score["valid_0"]["l1"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40)
    (OUTPUTS / "lightgbm_trials.json").write_text(json.dumps({
        "device_type": base["device_type"], "best_params": study.best_params, "validation_mae": study.best_value,
        "trials": [{"number": trial.number, "params": trial.params, "mae": trial.value}
                   for trial in study.trials],
    }, indent=2) + "\n")
    predictions = []
    for i, seed in enumerate(MEMBER_SEEDS):
        model = fit(training, validation, {**base, **study.best_params}, seed)
        model.save_model(OUTPUTS / f"lgbm_{i}.txt")
        predictions.append(model.predict(x_val))
        print(f"LightGBM {i + 1}/5, seed {seed}, best round {model.best_iteration}", flush=True)
    np.save(OUTPUTS / "lgbm_val.npy", np.stack(predictions))


if __name__ == "__main__":
    main()
