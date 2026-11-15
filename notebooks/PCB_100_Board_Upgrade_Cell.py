# Paste this entire file into one new cell in the existing Kaggle notebook.
# Uses a separate output folder and the verified 50-board checkpoint.
def run_100_board_upgrade():
    from collections import Counter, defaultdict
    from contextlib import ExitStack
    import hashlib
    import json
    from pathlib import Path
    import re
    from zipfile import ZipFile

    INPUT = Path("/kaggle/input")
    WORK = Path("/kaggle/working/pcb_100_board_upgrade")
    WORK.mkdir(parents=True, exist_ok=True)
    ANN_NAME = "_annotations.coco.json"
    ANN_SHA = "5d684e6d9f0a98de5015069228552d0f80ae4a539ddd288e60b7034e4f18f658"
    IMAGE_SET_SHA = "60eee441e374315bd13b5f4907161381b59c4cd2de6ee8801948c27dc63d1959"
    ZIP_SHA = "a5fb17e7009f96ffdb833cd9309b36da775545c8cc36a6dca3a8e9b4fbcb5f2a"
    BASE_GROUPS = set(['52', '55', '58', '62', '82'])
    ADDED_GROUPS = set(['57', '64', '65', '67', '80'])
    TRAIN_GROUPS = BASE_GROUPS | ADDED_GROUPS
    NAMES = ["short", "spur", "spurious_copper", "open", "mousebite",
             "hole_breakout", "conductor_scratch", "conductor_foreign_object",
             "base_material_foreign_object"]
    COCO_TO_MODEL = {2: 4, 3: 3, 4: 0, 5: 1, 6: 2}
    CLASSES = list(range(5))
    DISPLAY_NAMES = dict(enumerate(["short", "spur", "spurious_copper", "open_circuit", "mouse_bite"]))

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def sha(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    def save_verified(path, data):
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError(f"Existing file differs: {path}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def prepare_dataset():
        extracted = next((p for p in INPUT.rglob(ANN_NAME) if sha(p) == ANN_SHA), None)
        with ExitStack() as stack:
            if extracted:
                annotations = extracted.read_bytes()
                image_dir = extracted.parent.parent / "images"
                read_image = lambda name: (image_dir / name).read_bytes()
            else:
                archive = next((p for p in INPUT.rglob("*.zip")
                                if p.stat().st_size == 158048862 and sha(p) == ZIP_SHA), None)
                if archive is None:
                    raise FileNotFoundError("Attach the verified PCB_Defect ZIP as an input to this notebook.")
                zipped = stack.enter_context(ZipFile(archive))
                annotations = zipped.read("PCB_Defect/annotation/" + ANN_NAME)
                read_image = lambda name: zipped.read("PCB_Defect/images/" + name)

            assert digest(annotations) == ANN_SHA
            coco = json.loads(annotations)
            assert len(coco["images"]) == 230 and len(coco["annotations"]) == 1704
            assert {c["id"]: c["name"] for c in coco["categories"] if c["id"] > 0} == {
                1: "missing_pad", 2: "mouse_bite", 3: "open_circuit", 4: "short",
                5: "spur", 6: "spurious_copper"
            }
            by_image = defaultdict(list)
            for box in coco["annotations"]:
                if box["category_id"] in COCO_TO_MODEL:
                    by_image[box["image_id"]].append(box)

            manifest = []
            image_set = hashlib.sha256()
            counts = defaultdict(Counter)
            seen = set()
            for row in sorted(coco["images"], key=lambda r: r["file_name"]):
                name = row["file_name"]
                assert re.fullmatch(r"pcb_defect_\d{3}\.jpg", name), name
                group = row["extra"]["name"].split("-", 1)[0]
                split = "train" if group in TRAIN_GROUPS else "holdout"
                data = read_image(name)
                image_sha = digest(data)
                assert image_sha not in seen, f"Duplicate image: {name}"
                seen.add(image_sha)
                image_set.update(name.encode() + b"\0" + bytes.fromhex(image_sha))
                labels = []
                width, height = row["width"], row["height"]
                for annotation in by_image[row["id"]]:
                    cls = COCO_TO_MODEL[annotation["category_id"]]
                    x, y, w, h = annotation["bbox"]
                    assert 0 <= x < x+w <= width and 0 <= y < y+h <= height
                    labels.append(f"{cls} {(x+w/2)/width:.10f} {(y+h/2)/height:.10f} {w/width:.10f} {h/height:.10f}")
                    counts[split][DISPLAY_NAMES[cls]] += 1
                save_verified(WORK / "data/images" / split / name, data)
                label_path = WORK / "data/labels" / split / Path(name).with_suffix(".txt")
                label_data = "".join(line + "\n" for line in labels).encode()
                save_verified(label_path, label_data)
                manifest.append(dict(image=name, group=group, original_name=row["extra"]["name"],
                                     split=split, image_sha256=image_sha, label_sha256=digest(label_data),
                                     scored_boxes=len(labels)))
            assert image_set.hexdigest() == IMAGE_SET_SHA, "Image bytes differ from the audited dataset."
            assert Counter(row["split"] for row in manifest) == {"train": 100, "holdout": 130}
            assert sum(row["group"] in BASE_GROUPS for row in manifest) == 50
            assert sum(row["group"] in ADDED_GROUPS for row in manifest) == 50
            assert sum(counts["holdout"].values()) == 821
            for split in ("train", "holdout"):
                expected = {row["image"] for row in manifest if row["split"] == split}
                assert {p.name for p in (WORK / "data/images" / split).glob("*")} == expected
                assert {p.name for p in (WORK / "data/labels" / split).glob("*")} == {
                    Path(name).with_suffix(".txt").name for name in expected
                }
            assert {s: set(c) for s, c in counts.items()} == {
                "train": set(DISPLAY_NAMES.values()), "holdout": set(DISPLAY_NAMES.values())
            }
            write_json(WORK / "split.json", dict(dataset_doi="10.17632/vdj74sngvn.1",
                       annotation_sha256=ANN_SHA, image_set_sha256=IMAGE_SET_SHA,
                       train_groups=sorted(TRAIN_GROUPS), retained_50_groups=sorted(BASE_GROUPS),
                       added_50_groups=sorted(ADDED_GROUPS), group_rule="first prefix of COCO extra.name",
                       group_limit="Filename grouping and a local near-duplicate audit are proxies; layout independence is unknown.",
                       excluded_category="missing_pad", class_counts=counts, images=manifest))
            # JSON is valid YAML. The trainer can only validate against its 100 training boards.
            for name, validation in (("train_only", "train"), ("held_out", "holdout")):
                write_json(WORK / f"{name}.yaml", dict(path=str(WORK / "data"), train="images/train",
                           val=f"images/{validation}", names=NAMES))
            print("SPLIT READY | 100 training boards | 130 development boards")
            for split, values in counts.items():
                print(split, sum(values.values()), "boxes", dict(values))

    prepare_dataset()

    import importlib
    import importlib.metadata as metadata
    import os
    import shutil
    import subprocess
    import sys
    from urllib.request import urlopen

    os.environ["YOLO_CONFIG_DIR"] = str(WORK / "settings")
    os.environ["MPLCONFIGDIR"] = str(WORK / "matplotlib")
    os.environ["WANDB_MODE"] = "disabled"

    try:
        installed = metadata.version("ultralytics")
    except metadata.PackageNotFoundError:
        installed = None
    if installed != "8.4.142":
        pins = WORK / "dependency_pins.txt"
        pins.write_text("".join(f"{name}=={metadata.version(name)}\n" for name in ("torch", "torchvision", "numpy")))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(pins),
                        "ultralytics==8.4.142"], check=True)

    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    assert ultralytics.__version__ == "8.4.142"
    assert torch.cuda.is_available(), "Enable a GPU in Kaggle notebook settings."
    DEVICE = 0
    IMGSZ = 1024
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda:0")
    scores = torch.tensor([.9, .8], device="cuda:0")
    assert torchvision.ops.nms(boxes, scores, .5).cpu().tolist() == [0]
    del boxes, scores

    SOURCE = WORK / "yolo11s-dspcbsd.pt"
    WEIGHTS_URL = "https://github.com/Alpha-Ldz/lazarus/releases/download/weights-v1/yolo11s-dspcbsd.pt"
    WEIGHTS_SHA = "834b8dc02142e866d1a98881c202e87b86a24a3b0048708024e6eb7cd94b3af5"
    previous_source = WORK.parent / "pcb_50_board_transfer/yolo11s-dspcbsd.pt"
    if not SOURCE.exists() and previous_source.is_file() and sha(previous_source) == WEIGHTS_SHA:
        save_verified(SOURCE, previous_source.read_bytes())
    if not SOURCE.is_file():
        temporary = SOURCE.with_suffix(".download")
        with urlopen(WEIGHTS_URL, timeout=90) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        assert sha(temporary) == WEIGHTS_SHA
        temporary.replace(SOURCE)
    assert sha(SOURCE) == WEIGHTS_SHA

    def load_source():
        modules = {
            "torch.nn.modules.activation": "SiLU",
            "torch.nn.modules.batchnorm": "BatchNorm2d",
            "torch.nn.modules.container": "Sequential ModuleList",
            "torch.nn.modules.conv": "Conv2d",
            "torch.nn.modules.linear": "Identity",
            "torch.nn.modules.pooling": "MaxPool2d",
            "torch.nn.modules.upsampling": "Upsample",
            "ultralytics.nn.modules.block": "Attention Bottleneck C2PSA C3k C3k2 DFL PSABlock SPPF",
            "ultralytics.nn.modules.conv": "Concat Conv DWConv",
            "ultralytics.nn.modules.head": "Detect",
            "ultralytics.nn.tasks": "DetectionModel",
        }
        allowed = [getattr(importlib.import_module(m), n) for m, names in modules.items() for n in names.split()]
        previous = os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            with torch.serialization.safe_globals(allowed):
                model = YOLO(str(SOURCE))
        finally:
            if previous is None:
                os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = previous
        assert [model.names[i] for i in range(len(model.names))] == NAMES
        return model

    print("ENVIRONMENT READY", torch.cuda.get_device_name(0), "Ultralytics", ultralytics.__version__)


    REFERENCE_SHA = "2c62bf676e72f10ce59506f1df6ed1787722af04515df8aed2be236bfee0f70c"
    REFERENCE = WORK / "pcb50_epoch60.pt"

    def prepare_reference():
        if REFERENCE.is_file():
            assert sha(REFERENCE) == REFERENCE_SHA
            return
        previous = WORK.parent / "pcb_50_board_transfer/train_60/weights/last.pt"
        for candidate in [previous, *INPUT.rglob("last.pt"), *INPUT.rglob("model.pt")]:
            if candidate.is_file() and candidate.stat().st_size == 19217562 and sha(candidate) == REFERENCE_SHA:
                save_verified(REFERENCE, candidate.read_bytes())
                return
        for candidate in INPUT.rglob("*.zip"):
            if candidate.stat().st_size != 17653516:
                continue
            with ZipFile(candidate) as archive:
                member = "train_60/weights/last.pt"
                if member not in archive.namelist():
                    continue
                data = archive.read(member)
            if digest(data) == REFERENCE_SHA:
                save_verified(REFERENCE, data)
                return
        raise FileNotFoundError("Attach pcb_50_board_60_epoch_results.zip as a private input and run this cell again.")

    prepare_reference()
    print("50-board reference checkpoint verified.")

    import csv
    import numpy as np
    from ultralytics.utils.torch_utils import strip_optimizer
    import zipfile
    from IPython.display import FileLink, display

    EPOCHS = 60
    CHECKPOINT_EPOCH = 40
    TRAIN_BATCH = 2
    NMS_IOU = 0.4
    EVAL_CONF = 0.001
    training_dir = WORK / "train_100"
    final_weights = training_dir / "weights/last.pt"
    mid_weights = training_dir / f"weights/epoch_{CHECKPOINT_EPOCH}.pt"

    def save_intermediate(trainer):
        if trainer.epoch + 1 == CHECKPOINT_EPOCH:
            temp = mid_weights.with_suffix(".tmp")
            if not strip_optimizer(trainer.last, s=str(temp)):
                raise RuntimeError(f"Could not export epoch {CHECKPOINT_EPOCH}")
            temp.replace(mid_weights)

    def fixed_threshold_metrics(model):
        # Box metrics at the app cutoff. AP is computed separately over confidence scores.
        totals = {c: dict(tp=0, fp=0, fn=0) for c in CLASSES}
        images = 0
        for result in model.predict(
            source=str(WORK / "data/images/holdout"), stream=True, batch=1, imgsz=IMGSZ,
            device=DEVICE, conf=.20, iou=NMS_IOU, rect=False, max_det=300,
            classes=CLASSES, agnostic_nms=False, augment=False, verbose=False, save=False,
        ):
            images += 1
            height, width = result.orig_shape
            labels = WORK / "data/labels/holdout" / Path(result.path).with_suffix(".txt").name
            targets = {c: [] for c in CLASSES}
            for line in labels.read_text().splitlines():
                c, x, y, w, h = map(float, line.split())
                targets[int(c)].append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])

            boxes = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for c in CLASSES:
                actual = np.asarray(targets[c]).reshape(-1, 4)
                remaining = np.ones(len(actual), dtype=bool)
                order = np.flatnonzero(classes == c)
                order = order[np.argsort(-confidence[order], kind="stable")]
                for index in order:
                    candidates = np.flatnonzero(remaining)
                    if candidates.size:
                        prediction = boxes[index]
                        ground_truth = actual[candidates]
                        intersection = np.maximum(
                            np.minimum(prediction[2:], ground_truth[:, 2:])
                            - np.maximum(prediction[:2], ground_truth[:, :2]), 0
                        ).prod(axis=1)
                        union = ((prediction[2:] - prediction[:2]).prod()
                                 + (ground_truth[:, 2:] - ground_truth[:, :2]).prod(axis=1) - intersection)
                        iou = intersection / np.maximum(union, 1e-12)
                        best = int(np.argmax(iou))
                        if iou[best] >= .5:
                            totals[c]["tp"] += 1
                            remaining[candidates[best]] = False
                            continue
                    totals[c]["fp"] += 1
                totals[c]["fn"] += int(remaining.sum())

        def summarize(counts):
            tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
            return dict(**counts, precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                        f1=2*tp/max(2*tp+fp+fn, 1))

        pooled = {k: sum(row[k] for row in totals.values()) for k in ("tp", "fp", "fn")}
        assert images == 130 and pooled["tp"] + pooled["fn"] == 821
        return dict(confidence=.20, matching_iou=.5, overall=summarize(pooled),
                    per_class={DISPLAY_NAMES[c]: summarize(row) for c, row in totals.items()})

    def evaluate(model, name, checkpoint):
        assert [model.names[i] for i in range(len(model.names))] == NAMES
        metrics = model.val(data=str(WORK / "held_out.yaml"), split="val", imgsz=IMGSZ,
            batch=1, device=DEVICE, workers=0, rect=False, conf=EVAL_CONF, iou=NMS_IOU,
            max_det=300, classes=CLASSES, agnostic_nms=False, augment=False,
            plots=False, save_json=False, project=str(WORK), name=name, exist_ok=True, verbose=False)
        assert set(map(int, metrics.box.ap_class_index)) == set(CLASSES)
        assert int(metrics.nt_per_class.sum()) == 821, "Evaluation box count differs from the verified split"
        return dict(checkpoint_sha256=sha(checkpoint), mAP50=float(metrics.box.map50),
            mAP50_95=float(metrics.box.map), operating_point=fixed_threshold_metrics(model), per_class_AP50={
                DISPLAY_NAMES[int(c)]: float(ap)
                for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)
            })

    baseline = evaluate(YOLO(str(REFERENCE)), "baseline_50_eval", REFERENCE)
    write_json(WORK / "baseline.json", baseline)
    print(f"50 BOARDS | mAP50 {baseline['mAP50']:.2%} | mAP50–95 {baseline['mAP50_95']:.2%}")
    torch.cuda.empty_cache()

    # The recipe is fixed. The 130 comparison boards are absent from this training YAML.
    recipe = dict(data=str(WORK / "train_only.yaml"), epochs=EPOCHS, imgsz=IMGSZ,
        batch=TRAIN_BATCH, nbs=8, device=DEVICE, workers=0, amp=False, cache=False,
        val=False, patience=0, save=True, plots=False, optimizer="AdamW", lr0=0.0001,
        lrf=0.1, cos_lr=True, warmup_epochs=1.0, seed=42, deterministic=True,
        rect=False, freeze=0, classes=CLASSES, augmentations=[], mosaic=0.0, mixup=0.0,
        copy_paste=0.0, cutmix=0.0, degrees=0.0, translate=0.03, scale=0.1, shear=0.0,
        perspective=0.0, fliplr=0.5, flipud=0.5, hsv_h=0.0, hsv_s=0.1, hsv_v=0.1,
        project=str(WORK), name=training_dir.name, exist_ok=False)
    identity = dict(recipe=recipe, source_sha256=WEIGHTS_SHA, reference_sha256=REFERENCE_SHA, intermediate_epoch=CHECKPOINT_EPOCH, split_sha256=sha(WORK / "split.json"),
                    ultralytics=ultralytics.__version__, torch=str(torch.__version__))
    identity_file = WORK / "training_identity.json"

    def completed_epochs():
        results = training_dir / "results.csv"
        if not results.is_file():
            return []
        with results.open(newline="") as stream:
            return [int(row["epoch"]) for row in csv.DictReader(stream)]

    if training_dir.exists():
        if (not final_weights.is_file() or not mid_weights.is_file() or completed_epochs() != list(range(1, EPOCHS + 1))
                or not identity_file.is_file() or json.loads(identity_file.read_text()) != identity):
            raise RuntimeError(f"Existing run is incomplete or differs. Preserve {training_dir}, then move it aside and rerun.")
    else:
        write_json(identity_file, identity)
        candidate = load_source()
        candidate.add_callback("on_model_save", save_intermediate)
        candidate.train(**recipe)
        del candidate

    assert final_weights.is_file() and completed_epochs() == list(range(1, EPOCHS + 1))
    assert mid_weights.is_file()
    torch.cuda.empty_cache()
    intermediate = evaluate(YOLO(str(mid_weights)), "adapted_40_eval", mid_weights)
    adapted = evaluate(YOLO(str(final_weights)), "adapted_eval", final_weights)

    report = dict(reference_train_boards=50, train_boards=100, comparison_boards=130,
        comparison_role="Reused development set", scored_class_names=list(DISPLAY_NAMES.values()),
        imgsz=IMGSZ, nms_iou=NMS_IOU, ap_confidence_floor=EVAL_CONF,
        source_url=WEIGHTS_URL, checkpoint_selection="Prespecified 40- and 60-epoch snapshots",
        split_sha256=sha(WORK / "split.json"),
        model_50=baseline, model_100_epoch40=intermediate, model_100=adapted,
        delta_mAP50=adapted["mAP50"]-baseline["mAP50"],
        delta_mAP50_95=adapted["mAP50_95"]-baseline["mAP50_95"])
    write_json(WORK / "comparison.json", report)
    print("\n=== 100-BOARD UPGRADE CHECK ===")
    for name, row in (("Current 50-board", baseline), ("100 boards / epoch 40", intermediate), ("100 boards / epoch 60", adapted)):
        print(f"{name}: mAP50 {row['mAP50']:.2%} | mAP50–95 {row['mAP50_95']:.2%}")
        point = row["operating_point"]["overall"]
        print(f"  At 0.20 confidence | Precision {point['precision']:.2%} | Recall {point['recall']:.2%} | F1 {point['f1']:.2%}")
        for label, ap in row["per_class_AP50"].items():
            print(f"  {label}: AP50 {ap:.2%}")
    print(f"Change in mAP50: {100*report['delta_mAP50']:+.2f} percentage points")

    backup = WORK / "pcb_100_board_results.zip"
    temporary = backup.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in (final_weights, mid_weights, training_dir / "results.csv", training_dir / "args.yaml",
                     WORK / "split.json", WORK / "training_identity.json", WORK / "comparison.json"):
            z.write(path, arcname=path.relative_to(WORK).as_posix())
    temporary.replace(backup)
    os.chdir(WORK.parent)
    display(FileLink(str(backup.relative_to(WORK.parent))))


run_100_board_upgrade()
