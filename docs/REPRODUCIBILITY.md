# Reproduce the final project

The Git checkout contains the exact weights used by the app. Training is
optional. Model hashes are checked at startup.

## Detector experiment

Download the assets from the repository's
[reproduction release](https://github.com/ABHI-310/MinorProject-Final/releases/tag/v1.0.0).
The SHA-256 file in the release covers the training input and checkpoint archives.

1. For the single-layer source detector, use
   [PCB-Defect V1](https://doi.org/10.17632/vdj74sngvn.1) and the Lazarus
   checkpoint referenced by the [50-board notebook](../notebooks/PCB_50_Board_Transfer_Kaggle.ipynb).
   Its 60-epoch result is also provided as a release asset. Run the
   [100-board upgrade cell](../notebooks/PCB_100_Board_Upgrade_Cell.py) using
   that result and the same PCB-Defect archive.
2. For double-layer adaptation, paste
   [the DefectDet cell](../notebooks/PCB_DefectDet_K50_Cell.py) into Kaggle.
   Set all `ADD_*` flags to false. Run once per budget pair
   `(TARGET_IMAGES, REPLAY_IMAGES) = (50, 50), (100, 50), (100, 100)`.
   Attach the corresponding `pcb_defectdet_*_input.zip` from the release.
   These packages include the exact split manifests and starting checkpoint.
3. For the final six-class detector, restore the cell defaults
   `(100, 100)`, `ADD_MISSING_HOLE=True`,
   `MISSING_HOLE_LAYOUT_REPLAY=True`, and the other `ADD_*` flags false.
   Attach `pcb_defectdet_k100_r100_input.zip` and
   `pcb_missing_hole_layout11_input.zip`. The latter contains the balanced
   checkpoint and the prepared missing-hole examples.
4. Compare the outputs with the [saved evaluations](../reports/revamp/).
   The paper uses the single-layer development and grouped double-layer
   comparisons. It reports missing holes separately on PKU layouts 06 and 10.

The prepared inputs use [PCB-Defect V1](https://doi.org/10.17632/vdj74sngvn.1)
by Rashid et al. and [DefectDet V1](https://doi.org/10.17632/t9d9zs3bmb.1)
by Azam et al. (CC BY 4.0), converted to YOLO labels and grouped as described
in the paper. PKU images are credited to Huang and Wei in
[the dataset paper](https://arxiv.org/abs/1901.08204), for academic research.
The base weights come from the
[Lazarus release](https://github.com/Alpha-Ldz/lazarus/releases/tag/weights-v1),
trained on [DsPCBSD+](https://doi.org/10.1038/s41597-024-03656-8).

## RUL experiment

Install `configs/requirements-research.txt` in the app environment for local
training and testing. Follow the [RUL commands](../pcb_rul/README.md).
Run `python pcb_rul/simulate.py generate` with the checked-in
[detector calibration](../pcb_rul/outputs/detector_stats.json).
To repeat calibration, obtain PKU from the
[official dataset page](https://robotics.pkusz.edu.cn/resources/dataset/)
and place the extracted `PCB_DATASET` directory under `data/raw/pku/`.
Use the [Kaggle T4 notebook](../notebooks/PCB_RUL_Kaggle_T4.ipynb) for cloud
training. To rerun the frozen Colab evaluation, upload
`pcb-rul-kaggle-v2-checkpoint.zip` from the release when prompted by the
[Colab notebook](../notebooks/PCB_RUL_Colab_Evaluation.ipynb). Retraining across
devices and library versions may produce different weights and scores.

The frozen results, seed comparisons and plots are in
[the final test report directory](../pcb_rul/outputs/colab-final-results/).

## Application checks

```sh
python scripts/manage.py app --check
python -m pytest tests/test_phase4_api.py tests/test_maintenance_trend.py tests/test_mirror_merge.py -q
```

Other test modules and scripts include earlier experimental workflows. Some
expect datasets or archived checkpoints from those experiments.
