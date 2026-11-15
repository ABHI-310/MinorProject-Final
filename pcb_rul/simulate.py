"""Calibrate the checkpoint once, then generate split synthetic trajectories."""
import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from config import (
    BASE_RATE, CLASSES, GAMMA_SHAPE, GROWTH, LIFE_RANGE, N_UNITS, OUTPUTS, POWER,
    ROOT, RUL_CAP, SEED, SEVERITY, SPLIT, UNIT_NOISE,
)


def calibrate():
    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/runtime/ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/runtime/matplotlib"))
    import cv2
    import torch
    from ultralytics import YOLO
    from backend.mirror_merge import merge_mirror_boxes
    from backend.model_registry import PROJECT_MODEL
    from config import seed_everything

    seed_everything()
    torch.set_num_threads(4)
    folder = ROOT / "artifacts/models" / PROJECT_MODEL
    info = json.loads((folder / "model_info.json").read_text())
    weights = folder / "model.pt"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != info["checkpoint_sha256"]:
        raise ValueError("Detector checkpoint checksum mismatch")
    dataset = ROOT / "data/raw/pku/PCB_DATASET"
    with (ROOT / "reports/revamp/pcb100_defectdet_missing_hole_layout11_verification.json").open() as file:
        holdout = json.load(file)["splits"]["pku_layout_holdout"]
    groups = sorted(holdout["candidate_by_group"])
    paths = sorted(path for path in (dataset / "images").rglob("*.jpg") if path.stem[:2] in groups)
    if len(paths) != holdout["images"]:
        raise ValueError("Could not reconstruct the detector's published PKU holdout")

    canonical = {"open_circuit": "open", "mousebite": "mouse_bite"}
    model = YOLO(str(weights))
    detector_classes = [canonical.get(name, name) for name in model.names.values()]
    if set(detector_classes) != set(CLASSES):
        raise ValueError("Unexpected checkpoint classes")
    infer = info["inference"]
    options = dict(imgsz=info["image_size"], conf=0.001, iou=infer["iou"], rect=infer["rect"],
                   max_det=infer["max_det"], agnostic_nms=infer["agnostic_nms"],
                   augment=infer["augment"], device="cpu", verbose=False, save=False)
    confidence = 0.25
    totals = {name: {"ground_truth": 0, "tp": 0, "fp": 0} for name in CLASSES}
    validation_digest = hashlib.sha256()
    for path in paths:
        image_bytes = path.read_bytes()
        xml = (dataset / "Annotations" / path.parent.name / path.with_suffix(".xml").name).read_bytes()
        validation_digest.update(hashlib.sha256(image_bytes).digest())
        validation_digest.update(hashlib.sha256(xml).digest())
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        width = image.shape[1]
        original = model.predict(image, **options)[0].boxes.data.cpu()
        reflected = model.predict(cv2.flip(image, 1), **{**options, "imgsz": infer["mirror_image_size"]})[0]
        boxes = merge_mirror_boxes(original, reflected.boxes.data.cpu(), width, options["iou"], options["max_det"])
        truth = {name: [] for name in CLASSES}
        for item in ET.fromstring(xml).findall("object"):
            name = item.findtext("name")
            truth[canonical.get(name, name)].append(
                [float(item.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")],
            )
        for class_id, name in enumerate(detector_classes):
            predictions = boxes[(boxes[:, 5] == class_id) & (boxes[:, 4] >= confidence), :5].numpy()
            tp, fp = match(predictions, truth[name])
            totals[name]["ground_truth"] += len(truth[name])
            totals[name]["tp"] += tp
            totals[name]["fp"] += fp
    if any(value["ground_truth"] == 0 for value in totals.values()):
        raise ValueError("Every class must have ground-truth boxes")
    if totals["missing_hole"]["ground_truth"] != holdout["labels"]:
        raise ValueError("The holdout missing-hole box count changed")
    stats = {"validation_sha256": validation_digest.hexdigest(), "confidence": confidence, "matching_iou": 0.5, "images": len(paths),
             "validation": "PKU detector holdout with full six-class annotations",
             "layouts": groups, "checkpoint_sha256": digest,
             "inference": "Original and horizontal mirror merged before a global confidence cutoff",
             "per_class": {name: dict(value, recall=value["tp"] / value["ground_truth"],
                                      fp_per_image=value["fp"] / len(paths))
                           for name, value in totals.items()}}
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "detector_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def match(predictions, labels):
    remaining = np.asarray(labels, dtype=float).reshape(-1, 4)
    tp = 0
    for box in sorted(predictions, key=lambda p: -p[4]):
        if not len(remaining):
            break
        intersection = np.maximum(np.minimum(box[2:4], remaining[:, 2:]) -
                                  np.maximum(box[:2], remaining[:, :2]), 0).prod(axis=1)
        area = (box[2:4] - box[:2]).prod()
        union = area + (remaining[:, 2:] - remaining[:, :2]).prod(axis=1) - intersection
        ious = intersection / np.maximum(union, 1e-12)
        best = int(ious.argmax())
        if ious[best] >= 0.5:
            tp += 1
            remaining = np.delete(remaining, best, axis=0)
    return tp, len(predictions) - tp


def simulate_unit(rng, recall, false_positives):
    life = rng.uniform(*LIFE_RANGE)
    multiplier = rng.lognormal(*UNIT_NOISE)
    # Continue drawing increments if the initial block does not reach failure.
    increments = []
    total = 0.0
    while total < 1:
        block = rng.gamma(GAMMA_SHAPE, 1 / (GAMMA_SHAPE * life), size=int(LIFE_RANGE[1]))
        increments.append(block)
        total += block.sum()
    damage = np.cumsum(np.concatenate(increments))
    failure = int(np.searchsorted(damage, 1)) + 1
    damage = np.r_[0.0, damage[:failure]]
    latent_rate = multiplier * (BASE_RATE + GROWTH * damage[:, None] ** POWER)
    true = rng.binomial(rng.poisson(latent_rate), recall)
    false = rng.poisson(false_positives, size=true.shape)
    counts = true + false
    true_per_step = true.sum(axis=1)
    false_per_step = false.sum(axis=1)
    count = true_per_step + false_per_step
    index = np.repeat(np.arange(len(count)), count)
    area = np.exp(rng.normal(-6 + damage[index], 0.5))
    area_max = np.zeros(len(count))
    np.maximum.at(area_max, index, area)
    confidence_sum = np.bincount(np.repeat(np.arange(len(count)), true_per_step),
                                 rng.beta(8, 3, true_per_step.sum()), minlength=len(count))
    confidence_sum += np.bincount(np.repeat(np.arange(len(count)), false_per_step),
                                  rng.beta(3, 6, false_per_step.sum()), minlength=len(count))
    observed = np.column_stack((counts, np.bincount(index, area, minlength=len(count)), area_max,
                                confidence_sum / np.maximum(count, 1), counts @ SEVERITY))
    return observed, failure - np.arange(len(damage)), damage


def assignment():
    order = np.random.default_rng(SEED).permutation(N_UNITS)
    stop = np.cumsum(list(SPLIT.values()))[:-1]
    return dict(zip(SPLIT, np.split(order, stop), strict=True))


def generate(stats):
    recall = np.array([stats["per_class"][name]["recall"] for name in CLASSES])
    fp = np.array([stats["per_class"][name]["fp_per_image"] for name in CLASSES])
    rngs = np.random.SeedSequence(SEED).spawn(N_UNITS)
    for split, units in assignment().items():
        observations, rul, auxiliary, ids, steps = [], [], [], [], []
        for unit in sorted(units):
            x, remaining, damage = simulate_unit(np.random.default_rng(rngs[unit]), recall, fp)
            if np.any(np.diff(damage) < 0) or damage[-1] < 1 or np.any(damage[:-1] >= 1):
                raise ValueError(f"Invalid damage history for unit {unit}")
            observations.append(x.astype(np.float32))
            rul.append(remaining.astype(np.int16))
            auxiliary.append(np.minimum(damage, 1).astype(np.float32))
            ids.append(np.full(len(remaining), unit, dtype=np.int16))
            steps.append(np.arange(len(remaining), dtype=np.int16))
        np.savez_compressed(OUTPUTS / f"{split}.npz", observed=np.concatenate(observations),
                            unit=np.concatenate(ids), step=np.concatenate(steps),
                            true_rul=np.concatenate(rul),
                            y=np.minimum(np.concatenate(rul), RUL_CAP).astype(np.float32) / RUL_CAP,
                            auxiliary=np.concatenate(auxiliary))
        print(f"{split}: {len(units)} units, {sum(map(len, steps))} steps", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "generate"))
    stage = parser.parse_args().stage
    if stage == "calibrate":
        if (OUTPUTS / "detector_stats.json").exists():
            raise SystemExit("Calibration already exists.")
        print(json.dumps(calibrate(), indent=2), flush=True)
    else:
        generate(json.loads((OUTPUTS / "detector_stats.json").read_text()))
