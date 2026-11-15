# Minor Project · PCB Inspection and Maintenance Review

A single YOLO11s detector locates shorts, open circuits, mouse bites, spurs,
spurious copper and missing holes. The dashboard stores detections by equipment,
compares defect rates across inspections, and estimates simulated remaining
useful life (RUL) from inspection histories. The checked-in weights are ready
for inference.

## Run on a new machine

Use Python 3.11 and Node.js 20.19+ (or 22.12+). The app defaults to CPU and
SQLite. Use macOS, Linux, or WSL2 on Windows. LightGBM may require OpenMP
(`brew install libomp` on macOS or the `libgomp1` package on Debian/Ubuntu).

```sh
git clone https://github.com/ABHI-310/MinorProject-Final.git
cd MinorProject-Final
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/manage.py app
```

Open [the dashboard](http://127.0.0.1:8000) and upload either
[single-layer example](examples/pcb_defect_002.jpg) or
[double-layer example](examples/img62.png). Use **Settings → Run RUL demo**
for a simulated history. The app verifies model files and builds the dashboard
on startup. To build and verify without starting the server, run
`python scripts/manage.py app --check`. API documentation is at
[localhost:8000/docs](http://127.0.0.1:8000/docs).

On Linux, install `torch==2.7.1 torchvision==0.22.1` from
`https://download.pytorch.org/whl/cpu` before installing the requirements
if you want CPU-only PyTorch wheels.

## Edit or reproduce

- [Application and API](src/) and [dashboard integration](frontend/). The
  original dashboard source and lockfile are in [vendor](vendor/).
- [Production detector and RUL model files](artifacts/models/), including
  checksums and inference settings.
- [Training inputs and reproduction instructions](docs/REPRODUCIBILITY.md),
  [Kaggle and Colab notebooks](notebooks/), and [RUL training code](pcb_rul/).
- [Detector results](reports/revamp/) and
  [RUL results](pcb_rul/outputs/colab-final-results/).
- [Editable IEEE paper and figures](output/pcb_ieee_latex/).

The final detector achieved 83.65% mAP50 on the five annotated single-layer
classes and 84.46% mAP50 on the four annotated double-layer classes. Missing
holes were evaluated separately on PKU layouts 06 and 10. The RUL ensemble
had a mean absolute error of 3.28 inspection steps against the 150-step capped
target and 10.58 steps during the final 50 steps before simulated failure.

The RUL labels come from simulation. The scores describe simulated equipment
histories. Local inspection records and uploaded images are stored under
`data/application/` and excluded from Git.
