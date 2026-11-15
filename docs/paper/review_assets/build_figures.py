"""Build paper figures from verified evaluation reports and saved bounding boxes."""
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "pcb-paper-mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = HERE / "figures"
DATA = HERE / "data"
REPORTS = ROOT / "reports/revamp"
RUN = REPORTS / "pcb100_defectdet_missing_hole_layout11_verification.json"
RUL = ROOT / "pcb_rul/outputs/colab-final-results"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA
COLORS = ["#176B83", "#F09B37", "#674E9A", "#16845C", "#A44170", "#B94836"]
NAMES = ["Short", "Spur", "Spurious copper", "Open circuit", "Mouse bite", "Missing hole"]
INK = "#16243A"
BLUE = "#176B83"
GREEN = "#16845C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name, vector=True):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=.18)
    if vector:
        fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)


def table(path, rows):
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def diagram(name, heading, rows, notes):
    fig, ax = plt.subplots(figsize=(10, 4.5 if len(rows) == 3 else 3.7))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.02, .96, heading, fontsize=17, weight="bold", va="top")
    for i, (label, nodes) in enumerate(rows):
        y = .76 - i * (.58 / len(rows))
        ax.text(.02, y + .063, label, fontsize=10, weight="bold", va="bottom")
        width = (.96 - (len(nodes) - 1) * .058) / len(nodes)
        for j, node in enumerate(nodes):
            x = .02 + j * (width + .058)
            ax.add_patch(FancyBboxPatch((x, y - .06), width, .117, boxstyle="round,pad=.006,rounding_size=.014",
                                       linewidth=1, edgecolor=BLUE, facecolor="#EAF4F7" if j % 2 == 0 else "#F5F8FA"))
            ax.text(x + width / 2, y, node, ha="center", va="center", fontsize=9.7)
            if j:
                ax.annotate("", xy=(x - .01, y), xytext=(x - .048, y),
                            arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.5})
    ax.text(.02, .09, notes, va="top", fontsize=9, color="#3A5269")
    save(fig, name)


def iou(gt, pred):
    top = np.maximum(gt[:, None, :2], pred[None, :, :2])
    bottom = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    intersect = np.maximum(bottom - top, 0).prod(axis=2)
    return intersect / np.maximum((gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None]
                                  + (pred[:, 2:] - pred[:, :2]).prod(axis=1) - intersect, 1e-12)


def match(gt, pred):
    """Preserve the published same-class matches, then pair remaining class errors."""
    overlaps = iou(gt[:, 1:], pred[:, :4])
    unused = set(range(len(gt)))
    pending = []
    pairs = []
    for j in np.argsort(-pred[:, 4], kind="stable"):
        choices = [i for i in np.argsort(-overlaps[:, j], kind="stable")
                   if i in unused and gt[i, 0] == pred[j, 5] and overlaps[i, j] >= .5]
        if choices:
            unused.remove(choices[0])
            pairs.append((choices[0], j))
        else:
            pending.append(j)
    for j in pending:
        choices = [i for i in np.argsort(-overlaps[:, j], kind="stable")
                   if i in unused and overlaps[i, j] >= .5]
        if choices:
            unused.remove(choices[0])
            pairs.append((choices[0], j))
        else:
            pairs.append((None, j))
    return pairs + [(i, None) for i in sorted(unused)]


def prepare(split, title, class_ids, filename, report):
    if CACHE != DATA:
        shutil.copy2(CACHE / f"{split}.pt", DATA / f"{split}.pt")
    saved = torch.load(DATA / f"{split}.pt", weights_only=True, map_location="cpu")
    section = report["splits"][split]
    assert len(saved["targets"]) == section["images"]
    assert sum(len(gt) for gt in saved["targets"].values()) == section["labels"]
    index = {c: i for i, c in enumerate(class_ids)}
    bg = len(index)
    matrix = np.zeros((bg + 1, bg + 1), dtype=int)
    for name, truth in saved["targets"].items():
        gt = truth.numpy()
        pred = saved["predictions"]["six_class"][name].numpy()
        pred = pred[np.isin(pred[:, 5], class_ids) & (pred[:, 4] >= .325)]
        for i, j in match(gt, pred):
            matrix[bg if i is None else index[gt[i, 0]], bg if j is None else index[pred[j, 5]]] += 1
    expected = section["results"]["six_class"]["per_class"]
    labels = [NAMES[c] for c in class_ids] + ["Background"]
    for c, i in index.items():
        scores = expected[NAMES[c].lower().replace(" ", "_")]
        assert scores["tp"] == matrix[i, i]
        assert scores["fp"] == matrix[:, i].sum() - matrix[i, i]
        assert scores["fn"] == matrix[i].sum() - matrix[i, i]
    table(DATA / f"{filename}.csv", [["True / predicted", *labels], *[[n, *m] for n, m in zip(labels, matrix)]])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    table(DATA / f"{filename}_row_normalized.csv",
          [["True / predicted", *labels], *[[n, *m] for n, m in zip(labels, normalized)]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.75), layout="constrained")
    short = [n.replace("Spurious copper", "Spur. copper").replace("Background", "Bkgd.") for n in labels]
    for ax, grid, subtitle in zip(axes, (matrix, normalized), ("Detection counts", "Percent of row")):
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() if subtitle == "Detection counts" else 1)
        ax.set_xticks(range(len(labels)), short, rotation=42, ha="right")
        ax.set_yticks(range(len(labels)), short)
        ax.set(xlabel="Predicted class", ylabel="True class", title=subtitle)
        for i, j in np.ndindex(matrix.shape):
            color = "white" if grid[i, j] > grid.max() * .55 else INK
            value = str(matrix[i,j]) if subtitle == "Detection counts" else f"{grid[i,j]:.1%}"
            ax.text(j, i, "—" if i == j == bg else value, ha="center", va="center", color=color, fontsize=9)
    fig.suptitle(f"{title} • {section['images']} images | confidence 0.325 | IoU ≥ 0.50", fontsize=13)
    save(fig, filename)
    return saved


def example(saved, name, source, title, filename, class_ids):
    image = Image.open(source).convert("RGB")
    gt = saved["targets"][name].numpy()
    predictions = saved["predictions"]["six_class"][name].numpy()
    predictions = predictions[np.isin(predictions[:, 5], class_ids) & (predictions[:, 4] >= .325)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, label, boxes in zip(axes, ("Dataset annotations", "Final detector predictions"), (gt, predictions)):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(label, weight="bold", fontsize=10)
        for box in boxes:
            if len(box) == 5:
                c, x1, y1, x2, y2 = box
                score = ""
            else:
                x1, y1, x2, y2, confidence, c = box
                score = f" {confidence:.2f}"
            color = COLORS[int(c)]
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, lw=1.4, edgecolor=color))
            ax.text(x1, y1, NAMES[int(c)] + score, color="white", fontsize=7.5,
                    va="bottom", bbox={"facecolor":color, "edgecolor":"none", "pad":1.3})
    fig.suptitle(f"{title} | {name}", weight="bold")
    save(fig, filename, False)


def main():
    report = json.loads(RUN.read_text())
    checkpoint = ROOT / "artifacts/models/pcb_defectdet_k100_r100_missing_hole_layout11/model.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["result"]["checkpoint_sha256"]
    if CACHE != DATA:
        assert hashlib.sha256((CACHE / "missing_hole.pt").read_bytes()).hexdigest() == report["result"]["checkpoint_sha256"]
    diagram("01_system_flow", "PCB inspection and remaining life", [
        ("Model training", ["Lazarus weights", "100 single-layer\nPCB images", "Few-shot\nadaptation", "Add missing-hole\ntraining examples"]),
        ("Image inspection", ["Upload a\nPCB image", "YOLO11s\npredicts boxes", "Merge views and\nfilter detections", "Save boxes and\ninspection history"]),
        ("Remaining useful life", ["Comparable history\nfor one equipment ID", "Extract defect\nfeatures", "LightGBM + GRU", "Display simulated\nRUL in steps"])],
        "The RUL models are trained on simulated histories calibrated with detector error rates.")
    diagram("02_yolo_architecture", "YOLO11s detector and application inference", [
        ("Network", ["PCB image", "Backbone\nConv, C3k2, SPPF, C2PSA", "Neck\nP3 / P4 / P5", "Detection head\nboxes and classes"]),
        ("Application inference", ["Original 1280\nMirrored 1536", "Map boxes back,\nmerge, apply NMS", "640 views for\ncross-scale checks", "Filter by\nconfidence"])],
        "The same checkpoint processes every view. NMS IoU = 0.40. Default cutoff = 0.325, missing hole minimum = 0.40.")
    diagram("03_rul_flow", "Training and evaluating the RUL models", [
        ("Build the dataset", ["Calibrate YOLO\non 93 PKU images", "Simulate 900 units\nuntil failure", "Split by unit\n600 / 150 / 150"]),
        ("Fit and combine", ["Fit LightGBM on rolling features\nFit GRU on 100-step histories", "Blend predictions\n20% LightGBM / 80% GRU"]),
        ("Evaluate", ["Smooth using\ncurrent and past steps", "Clip estimates\nto 0–150 steps", "Evaluate on 150\nsimulated test units"])],
        "Five fitted members per model type. The validation split selects the blend.")
    source = prepare("source_development", "PCB-Defect single-layer development set", range(5), "08_single_layer_confusion", report)
    target = prepare("target_heldout", "DefectDet double-layer comparison set", range(4), "09_double_layer_confusion", report)
    example(source, "pcb_defect_002.jpg", ROOT / "examples/pcb_defect_002.jpg",
            "Single-layer PCB-Defect", "04_single_layer_example", range(5))
    example(target, "img62.png", ROOT / "examples/img62.png",
            "Double-layer DefectDet", "05_double_layer_example", range(4))

    scores = json.loads((REPORTS / "pcb100_defectdet_k100_r100_verification.json").read_text())
    stages = ["Source only", "50 target + 50 source", "100 target + 50 source",
              "100 target + 100 source", "Final six-class model"]
    stages_key = ["base", "k50", "k100", "balanced"]
    split_names = ("source_development", "target_heldout")
    series = [[scores[s]["results"][k]["mAP50"] * 100 for k in stages_key] +
              [report["splits"][s]["results"]["six_class"]["mAP50"] * 100] for s in split_names]
    table(DATA/"adaptation_map50.csv", [["Stage", "Single-layer mAP50 (%)", "Double-layer mAP50 (%)"],
                                      *zip(stages, *series)])
    fig, ax = plt.subplots(figsize=(9.8, 4.2), layout="constrained")
    x = np.arange(len(stages))
    for i,(label, vals, color) in enumerate(zip(("Single-layer (5 classes)", "Double-layer (4 classes)"), series, (BLUE,GREEN))):
        bars = ax.bar(x + (i-.5)*.34, vals, width=.34, label=label, color=color)
        ax.bar_label(bars, [f"{v:.2f}" for v in vals], padding=3, fontsize=8)
    ax.set(ylim=(0,108), ylabel="mAP50 (%)", title="Detection scores before and after adaptation")
    ax.set_xticks(x, [s.replace(" + ", "\n+ ") for s in stages], fontsize=8.7)
    ax.legend(loc="upper center", ncols=2, frameon=False)
    save(fig, "07_adaptation_scores")
    fig, ax = plt.subplots(figsize=(9,4.1), layout="constrained")
    per_rows = [["Split","Class","AP50","AP50-95","TP","FP","FN","Precision","Recall","F1"]]
    for j,(s, title, color) in enumerate(zip(split_names, ("Single-layer", "Double-layer"), (BLUE,GREEN))):
        classes = report["splits"][s]["results"]["six_class"]["per_class"]
        for i,n in enumerate(NAMES[:5]):
            row = classes.get(n.lower().replace(" ", "_"))
            if row is None:
                continue
            bar = ax.bar(i+(j-.5)*.35, row["AP50"]*100, width=.35, color=color, label=title if i == 0 else "")
            ax.bar_label(bar, fmt="%.1f", padding=3, fontsize=8)
            per_rows.append([title,n,*[row[k] for k in ("AP50","AP50_95","tp","fp","fn","precision","recall","f1")]])
    ax.set(ylim=(0,110), ylabel="AP50 (%)", title="Final detector performance by annotated defect class")
    ax.set_xticks(range(5), NAMES[:5])
    ax.legend(loc="upper center", ncols=2, frameon=False)
    save(fig,"10_per_class_ap")
    table(DATA/"per_class_detection_metrics.csv", per_rows)

    with (RUL/"test_comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    for ax, subset, title in zip(axes, ("all","true RUL <= 50"), ("All steps, 150-step cap", "Final 50 steps before failure")):
        selected = [row for row in rows if row["target"]=="capped" and row["subset"]==subset]
        bars=ax.barh([r["model"] for r in selected], [float(r["mae_steps"]) for r in selected],
                     color=["#8EA4B2"]*3+[BLUE,BLUE,GREEN])
        ax.invert_yaxis()
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=8)
        ax.set(xlabel="Mean absolute error (inspection steps)", title=title)
        ax.margins(x=.21)
    save(fig,"12_rul_model_errors")
    for original, name in (("five_test_units", "13_rul_test_trajectories"), ("error_by_life_stage","14_rul_error_by_life_stage"),
                           ("predicted_vs_true","15_rul_predicted_vs_true")):
        shutil.copy2(RUL/f"{original}.png", FIG/f"{name}.png")
    for name in ("test_comparison.csv","test_by_seed.csv"):
        shutil.copy2(RUL/name, DATA/f"rul_{name}")


if __name__ == "__main__":
    main()
