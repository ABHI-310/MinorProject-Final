# PCB remaining useful life experiment

RUL labels are generated from a calibrated stochastic degradation simulator. They are not measured field data.

The experiment requires Python 3.11, NumPy, pandas, scikit-learn, LightGBM, PyTorch, Optuna, Matplotlib and SciPy. Detector calibration also uses the repository's Ultralytics and OpenCV environment and its existing checkpoint. Run from the repository root.

1. `.venv/bin/python pcb_rul/simulate.py calibrate` runs the existing YOLO checkpoint on its 93-image PKU holdout, layouts 06 and 10. It scores all six classes against the original XML boxes at global confidence 0.25 after the checkpoint's original/mirror merge. Matching is one-to-one per class at IoU 0.5. Calibration is saved once in `outputs/detector_stats.json`.
2. `.venv/bin/python pcb_rul/simulate.py generate` generates the specified 900 units and saves the splits separately. The simulator starts each trajectory at step 0. It uses the given log-area distribution for all observed boxes and uses the two confidence distributions for their respective true or false detection sources. Empty frames have zero total area, max area and mean confidence. The only saved form of hidden damage is the auxiliary training target.
3. `.venv/bin/python pcb_rul/features.py` builds train and validation features and fits scaling on the train split. The Ridge baseline uses all 90 rolling window features. The time-only baseline is LightGBM with an absolute-error objective.
4. `.venv/bin/python pcb_rul/train_lgbm.py` trains the baselines, runs the 40 validation trials and fits the five members. `.venv/bin/python pcb_rul/train_gru.py` trains the five GRU members. Simulation, split selection and Optuna use seed 42. The first member uses 42 and the other four seeds are deterministically derived from 42.
5. `.venv/bin/python pcb_rul/ensemble.py` selects the blend on validation data. The smoother first converts normalized predictions to steps so it can add them to elapsed steps. It applies the causal EWMA when the current raw estimate is below 135 steps, then clips the output to 0–150.
6. `.venv/bin/python pcb_rul/evaluate.py` runs all checks in `tests.py`, including the validation gate, before it opens the test split. If the final model fails to beat the time-only baseline, it stops there. A marker records the frozen artifacts and prevents repeated test evaluation.

All generated outputs go under `outputs/`. The test comparison reports both uncapped true RUL and the capped training target. For true RUL above 150, uncapped error necessarily includes the effect of the output cap. Seed means and sample standard deviations describe the five individual fits and five paired blends. The final ensemble row also reports the prediction made after averaging the members of each model family. The fixed baselines each have one reported score.

The Kaggle T4 notebook sets `PCB_RUL_LIGHTGBM_DEVICE=gpu` to build LightGBM histograms on the GPU using double precision. The GRU uses CUDA. LightGBM's deterministic flag applies to CPU training, so the pre-test checks repeat small seeded fits on the selected devices and compare their predictions exactly.
