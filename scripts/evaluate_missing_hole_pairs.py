"""Score the verified layout 11 checkpoint on labeled inspection and reference photos."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from _bootstrap import ROOT
from backend.mirror_merge import merge_mirror_boxes
from evaluate_k500_development import match

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CHECKPOINT_SHA256 = "931f4ceaeb6eb5d4b827eeff2dde51e7d14846554dc2eddb062607ede21df072"
ARCHIVE = ROOT / "artifacts/downloaded_runs/whole_board/pcb_defectdet_k100_r100_missing_hole_layout11_result.zip"
LAYOUT11_REPORT = ROOT / "reports/revamp/pcb100_defectdet_missing_hole_layout11_verification.json"
MISSING_HOLE_ID = 5
CONFIDENCE = 0.40


def read_photo(path):
    raw = path.read_bytes()
    photo = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if photo is None:
        raise ValueError(f"Cannot decode photo: {path}")
    return photo, sha256(raw).hexdigest()


def predict(model, photo, settings, mirror_size):
    original = model.predict(source=photo, **settings)[0].boxes.data.cpu()
    reflected = model.predict(
        source=cv2.flip(photo, 1), **{**settings, "imgsz": mirror_size}
    )[0].boxes.data.cpu()
    boxes = merge_mirror_boxes(original, reflected, photo.shape[1], settings["iou"], settings["max_det"])
    missing_holes = boxes[(boxes[:, 5] == MISSING_HOLE_ID) & (boxes[:, 4] >= CONFIDENCE)]
    return [
        {"class": MISSING_HOLE_ID, "box": box[:4], "confidence": box[4]}
        for box in missing_holes.tolist()
    ]


def evaluate(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("The manifest needs a nonempty pairs list.")
    if not isinstance(manifest["dataset"], str) or not manifest["dataset"].strip():
        raise ValueError("The manifest needs a dataset name.")

    settings = json.loads(LAYOUT11_REPORT.read_text())["settings"]
    mirror_size = settings.pop("mirror_image_size")
    settings.pop("confidence_cutoff")
    settings["classes"] = list(range(6))

    totals = {"inspection": Counter(), "reference": Counter()}
    results = []
    seen_inspections = set()
    torch.set_num_threads(4)
    with ZipFile(ARCHIVE) as archive, TemporaryDirectory(prefix="pcb-hole-pairs-") as temporary:
        weights = archive.read("last.pt")
        if sha256(weights).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Layout 11 checkpoint checksum mismatch.")
        weight_path = Path(temporary) / "layout11.pt"
        weight_path.write_bytes(weights)
        model = YOLO(str(weight_path))
        if list(model.names.values()) != ["short", "spur", "spurious_copper", "open", "mousebite", "missing_hole"]:
            raise ValueError("Unexpected checkpoint classes.")

        for pair in pairs:
            design = pair["board_design"]
            if not isinstance(design, str) or not design.strip():
                raise ValueError("Each pair needs a board_design name.")
            inspection_path = manifest_path.parent / pair["inspection"]
            reference_path = manifest_path.parent / pair["reference"]
            inspection, inspection_sha = read_photo(inspection_path)
            reference, reference_sha = read_photo(reference_path)
            if inspection_sha == reference_sha or inspection_sha in seen_inspections:
                raise ValueError(f"Inspection photo must be distinct: {inspection_path}")
            seen_inspections.add(inspection_sha)

            height, width = inspection.shape[:2]
            labels = []
            for box in pair["missing_hole_boxes"]:
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"Box outside inspection image {inspection_path}: {box}")
                labels.append({"class": MISSING_HOLE_ID, "box": box})

            entry = {"board_design": design}
            for role, photo, path, digest, truth in (
                ("inspection", inspection, pair["inspection"], inspection_sha, labels),
                ("reference", reference, pair["reference"], reference_sha, []),
            ):
                detections = predict(model, photo, settings, mirror_size)
                counts = match(detections, truth)[MISSING_HOLE_ID]
                totals[role].update(counts)
                entry[role] = {
                    "image": path,
                    "sha256": digest,
                    "missing_hole_boxes": [label["box"] for label in truth],
                    "detections": detections,
                    "counts": counts,
                }
            results.append(entry)

    return {
        "dataset": manifest["dataset"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "confidence_cutoff": CONFIDENCE,
        "matching_iou": 0.50,
        "matching": "Confidence-ordered one-to-one matching of missing-hole boxes",
        "reference_role": "Known-good control scored independently of the inspection image",
        "totals": totals,
        "pairs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest with inspection and reference photos")
    parser.add_argument("output", type=Path, help="Path for the evaluation report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    report = evaluate(args.manifest.resolve())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
