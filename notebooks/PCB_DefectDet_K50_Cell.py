# Paste this file into one Kaggle cell with the matching input ZIP attached.
# Set all ADD_* switches to False to run the original DefectDet experiment.
from contextlib import ExitStack
from hashlib import file_digest, sha256
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import importlib.metadata
import json
import os
import random
import subprocess
import sys


TARGET_IMAGES = 100
REPLAY_IMAGES = 100
TRAIN_IMAGE_SIZE = 1280
ADD_PCB_IND_SHORTS = False
ADD_REVIEWED_SOURCE_SHORTS = False
ADD_MISSING_HOLE = True
MISSING_HOLE_LAYOUT_REPLAY = True
ADD_MISSING_HOLE_BACKGROUNDS = False
MISSING_HOLE_BACKGROUNDS_ZIP = "pcb_missing_hole_backgrounds_input.zip"
MISSING_HOLE_BACKGROUNDS_MANIFEST = "pcb_missing_hole_backgrounds_manifest.json"
LAYOUT11_SHA = "931f4ceaeb6eb5d4b827eeff2dde51e7d14846554dc2eddb062607ede21df072"
MISSING_HOLE_INPUT = "pcb_missing_hole_layout11_input" if MISSING_HOLE_LAYOUT_REPLAY else "pcb_missing_hole_input"
MISSING_HOLE_ZIP = f"{MISSING_HOLE_INPUT}.zip"
MISSING_HOLE_MANIFEST = (
    "pcb_missing_hole_layout11_manifest.json" if MISSING_HOLE_LAYOUT_REPLAY else "pcb_missing_hole_manifest.json"
)
PCB_IND_ZIP = "PCB-IND_v4.zip"
PCB_IND_MD5 = "1325f8dffeb73bef4db34fbd22b31ec5"
PCB_IND_BUDGET = 64
PCB_IND_CLASS_MAP = {0: 4, 3: 2, 6: 0, 7: 3}
PCB_IND_NAMES = {
    0: "mouse_bite", 1: "missing_copper", 2: "scratch", 3: "spurious_copper",
    4: "copper_burr", 5: "stain", 6: "short", 7: "open",
}
RUN_ID = f"k{TARGET_IMAGES}" if REPLAY_IMAGES == 50 else f"k{TARGET_IMAGES}_r{REPLAY_IMAGES}"
MANIFEST = f"defectdet_{RUN_ID}_manifest.json"
INPUT_ZIP = f"pcb_defectdet_{RUN_ID}_input.zip"
BASE_SHA = "d3451921ab46341c85c84214209da24ad21ebd6462e06b44c27a1eb884ee64b5"
BALANCED_SHA = "56d6ecf91e82e696a6ca2215ee8090b94576414bf459f2d1cd6fb10cdc3b29e1"
VERSION = "8.4.142"
EPOCHS = 30

# PCB-Defect v2 short boxes aligned to the original single-layer replay images.
# Coordinates are normalized center x, center y, width, height on the original scan.
REVIEWED_SOURCE_SHORTS = {
    "pcb_defect_073.jpg": (
        (0.5353173039, 0.8756415938, 0.0451772197, 0.0319069251),
    ),
    "pcb_defect_074.jpg": (
        (0.1422402875, 0.4527301173, 0.0212618423, 0.0686265576),
        (0.5472767246, 0.8758260665, 0.0456886031, 0.0264304992),
    ),
    "pcb_defect_075.jpg": (
        (0.5088029552, 0.1438042279, 0.0707228896, 0.0208238723),
        (0.3895287753, 0.6600207384, 0.0286732791, 0.0448435945),
    ),
    "pcb_defect_076.jpg": (
        (0.6358262784, 0.5979956006, 0.0478386495, 0.0251065319),
        (0.093963623, 0.4817311267, 0.0202796679, 0.0661725278),
    ),
    "pcb_defect_077.jpg": (
        (0.4281773649, 0.3772637025, 0.0763208529, 0.0231029106),
        (0.3527132942, 0.7132355446, 0.0730945772, 0.0270584461),
    ),
    "pcb_defect_078.jpg": (
        (0.3970507778, 0.5579116738, 0.0190189967, 0.070781209),
    ),
    "pcb_defect_079.jpg": (
        (0.2288941726, 0.6753983564, 0.0540647869, 0.0343242464),
    ),
    "pcb_defect_081.jpg": (
        (0.1463057881, 0.3942737828, 0.0556458479, 0.0421968178),
    ),
    "pcb_defect_082.jpg": (
        (0.6016150726, 0.138306307, 0.0372194255, 0.0497564047),
    ),
    "pcb_defect_083.jpg": (
        (0.1630274727, 0.3715378802, 0.0575220729, 0.0293034734),
    ),
    "pcb_defect_084.jpg": (
        (0.6631619984, 0.1874870042, 0.0319115808, 0.0484767274),
        (0.9065500678, 0.312576882, 0.1047350604, 0.0192649191),
    ),
    "pcb_defect_173.jpg": (
        (0.6929163023, 0.3932797041, 0.0272102322, 0.0329202916),
    ),
    "pcb_defect_180.jpg": (
        (0.4261433594, 0.1596869698, 0.0251283974, 0.0552558175),
        (0.4091974719, 0.3040330289, 0.0345853334, 0.0268127395),
    ),
    "pcb_defect_205.jpg": (
        (0.874839072, 0.7091306659, 0.0257993045, 0.0300131959),
        (0.5318285001, 0.9545210172, 0.0162467771, 0.0300729699),
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def prepare_input(inputs, workspace):
    assert (TARGET_IMAGES, REPLAY_IMAGES) in ((50, 50), (100, 50), (100, 100))
    manifests = sorted(inputs.rglob(MANIFEST))
    archives = sorted(inputs.rglob(INPUT_ZIP)) if not manifests else []
    if len(manifests) + len(archives) != 1:
        raise RuntimeError(f"Attach one Kaggle input containing {INPUT_ZIP} or its extracted contents.")

    with ExitStack() as stack:
        if manifests:
            manifest_path = manifests[0]
            manifest_bytes = manifest_path.read_bytes()
            def open_member(name):
                return (manifest_path.parent / name).open("rb")
        else:
            archive = stack.enter_context(ZipFile(archives[0]))
            manifest_bytes = archive.read(MANIFEST)
            open_member = archive.open

        manifest = json.loads(manifest_bytes)
        support, replay, test = (manifest[key] for key in ("target_support", "source_replay", "target_test"))
        assert len(support) == TARGET_IMAGES and len(replay) == REPLAY_IMAGES and len(test) == 148
        assert {row["group"] for row in support}.isdisjoint(row["group"] for row in test)
        assert {row["image_sha256"] for row in support}.isdisjoint(row["image_sha256"] for row in test)
        files = manifest["files"]
        expected = {"base.pt"}
        for row in support + replay:
            name = row["image"]
            assert PurePosixPath(name).name == name and name.endswith((".png", ".jpg"))
            expected.add(f"data/images/train/{name}")
            expected.add(f"data/labels/train/{Path(name).stem}.txt")
        assert len(expected) == 1 + 2 * (TARGET_IMAGES + REPLAY_IMAGES) and set(files) == expected
        assert files["base.pt"] == BASE_SHA
        if archives:
            assert len(archive.namelist()) == len(expected) + 1
            assert set(archive.namelist()) == expected | {MANIFEST}

        for name, expected_hash in files.items():
            destination = workspace / name
            if destination.is_file() and digest(destination) == expected_hash:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix(destination.suffix + ".partial")
            with open_member(name) as source, pending.open("wb") as output:
                copyfileobj(source, output)
            assert digest(pending) == expected_hash, name
            pending.replace(destination)

    for folder in ("data/images/train", "data/labels/train"):
        actual = {str(path.relative_to(workspace)) for path in (workspace / folder).iterdir()}
        assert actual == {name for name in expected if name.startswith(folder + "/")}

    manifest_hash = sha256(manifest_bytes).hexdigest()
    print(f"Verified {TARGET_IMAGES} target and {REPLAY_IMAGES} replay images. Held-out group list: {len(test)} images.")
    print(f"Input manifest SHA-256: {manifest_hash}", flush=True)
    return manifest, manifest_hash


def prepare_balanced_checkpoint(inputs, work, manifest_hash):
    result_name = "pcb_defectdet_k100_r100_result.zip"
    local_result = work.parent / result_name
    candidates = [local_result] if local_result.is_file() else sorted(inputs.rglob(result_name))
    if candidates:
        if len(candidates) != 1:
            raise RuntimeError(f"Attach one copy of {result_name}.")
        with ZipFile(candidates[0]) as archive:
            result = json.loads(archive.read("result.json"))
            identity = json.loads(archive.read("training_identity.json"))
            weights = archive.read("last.pt")
    else:
        for report in inputs.rglob("result.json"):
            result = json.loads(report.read_text())
            if result.get("checkpoint_sha256") == BALANCED_SHA:
                identity = json.loads(report.with_name("training_identity.json").read_text())
                weights = report.with_name("last.pt").read_bytes()
                break
        else:
            raise FileNotFoundError(f"Attach {result_name} as a Kaggle input or rerun in the session that created it.")
    assert result["training_complete"] and result["epochs"] == 30
    assert result["target_images"] == result["replay_images"] == 100
    assert result["manifest_sha256"] == identity["manifest_sha256"] == manifest_hash
    assert identity["start_sha256"] == BASE_SHA
    assert sha256(weights).hexdigest() == result["checkpoint_sha256"] == BALANCED_SHA
    destination = work / "balanced_start.pt"
    pending = destination.with_suffix(".partial")
    pending.write_bytes(weights)
    pending.replace(destination)
    return destination


def prepare_pcb_ind_shorts(inputs, work):
    archives = sorted(inputs.rglob(PCB_IND_ZIP))
    manifests = sorted(
        path for path in inputs.rglob("classes.json")
        if (path.parent / "COCO/annotations/train.json").is_file()
    ) if not archives else []
    if len(archives) + len(manifests) != 1:
        raise RuntimeError(f"Attach one copy of {PCB_IND_ZIP} or its extracted contents.")

    with ExitStack() as stack:
        if archives:
            with archives[0].open("rb") as stream:
                assert file_digest(stream, "md5").hexdigest() == PCB_IND_MD5, "Wrong PCB-IND release"
            archive = stack.enter_context(ZipFile(archives[0]))
            read_member = archive.read
        else:
            def read_member(name):
                return (manifests[0].parent / name).read_bytes()

        metadata = json.loads(read_member("classes.json"))
        assert metadata["dataset_name"] == "PCB-IND" and metadata["version"] == "v4"
        assert {row["id"]: row["name"] for row in metadata["classes"]} == PCB_IND_NAMES
        splits = {}
        annotation_hashes = {}
        for split, size in (("train", 3833), ("val", 478), ("test", 478)):
            raw = read_member(f"COCO/annotations/{split}.json")
            annotation_hashes[split] = sha256(raw).hexdigest()
            dataset = json.loads(raw)
            assert len(dataset["images"]) == size
            assert {row["id"]: row["name"] for row in dataset["categories"]} == PCB_IND_NAMES
            splits[split] = dataset

        def prefix(row):
            value = PurePosixPath(row["file_name"]).stem.split("_", 1)[0]
            assert value.isdigit(), row["file_name"]
            return value

        excluded = {prefix(row) for split in ("val", "test") for row in splits[split]["images"]}
        evaluation_hashes = {
            sha256(read_member("COCO/" + row["file_name"])).hexdigest()
            for split in ("val", "test") for row in splits[split]["images"]
        }
        labels_by_id = {}
        for ann in splits["train"]["annotations"]:
            labels_by_id.setdefault(ann["image_id"], []).append(ann)
        by_prefix = {}
        for row in sorted(splits["train"]["images"], key=lambda row: row["file_name"]):
            labels = labels_by_id.get(row["id"], [])
            classes = {ann["category_id"] for ann in labels}
            group = prefix(row)
            if group not in excluded and 6 in classes and classes <= PCB_IND_CLASS_MAP.keys():
                by_prefix.setdefault(group, []).append(row)

        eligible = sum(map(len, by_prefix.values()))
        assert eligible == 182 and len(by_prefix) == 102
        rng = random.Random(42)
        selected = [rng.choice(by_prefix[group]) for group in rng.sample(sorted(by_prefix), PCB_IND_BUDGET)]
        records = []
        for row in sorted(selected, key=lambda row: row["file_name"]):
            source = PurePosixPath(row["file_name"])
            assert source.parent == PurePosixPath("images/train") and source.suffix == ".jpg"
            raw_image = read_member(f"COCO/{source}")
            image_hash = sha256(raw_image).hexdigest()
            assert image_hash not in evaluation_hashes, source
            width, height = row["width"], row["height"]
            assert (width, height) == (300, 300)
            lines = []
            for ann in labels_by_id[row["id"]]:
                x, y, w, h = ann["bbox"]
                assert 0 <= x < x+w <= width and 0 <= y < y+h <= height, (source, ann["bbox"])
                box = ((x+w/2)/width, (y+h/2)/height, w/width, h/height)
                cid = PCB_IND_CLASS_MAP[ann["category_id"]]
                lines.append(f"{cid} " + " ".join(f"{v:.10f}" for v in box))
            raw_label = ("\n".join(lines) + "\n").encode()
            name = f"pcbind_{source.name}"
            image_path = work / "images/train" / name
            label_path = work / "labels/train" / f"{image_path.stem}.txt"
            for path, data in ((image_path, raw_image), (label_path, raw_label)):
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_suffix(".partial")
                pending.write_bytes(data)
                pending.replace(path)
            records.append(dict(image=name, source=str(source), prefix=prefix(row),
                                image_sha256=image_hash, label_sha256=sha256(raw_label).hexdigest(),
                                short_boxes=sum(line.startswith("0 ") for line in lines)))

    assert len({row["image_sha256"] for row in records}) == PCB_IND_BUDGET
    assert {p.name for p in (work / "images/train").iterdir()} == {row["image"] for row in records}
    assert {p.stem for p in (work / "labels/train").iterdir()} == {Path(row["image"]).stem for row in records}
    return dict(dataset="PCB-IND v4", annotation_sha256=annotation_hashes,
                eligible=eligible, seed=42, patches=records)


def add_reviewed_source_shorts(work, manifest):
    replay = {row["image"]: row for row in manifest["source_replay"]}
    assert REVIEWED_SOURCE_SHORTS.keys() <= replay.keys()
    assert manifest["checkpoint_classes"][0] == "short"
    reviewed = []
    for image_name, boxes in sorted(REVIEWED_SOURCE_SHORTS.items()):
        relative = f"data/labels/train/{Path(image_name).stem}.txt"
        label_path = work / relative
        original = label_path.read_bytes()
        original_hash = sha256(original).hexdigest()
        assert original_hash == manifest["files"][relative], relative
        lines = []
        for x, y, w, h in boxes:
            assert 0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1, image_name
            lines.append(f"0 {x:.10f} {y:.10f} {w:.10f} {h:.10f}")
        updated = original
        if updated and not updated.endswith(b"\n"):
            updated += b"\n"
        updated += ("\n".join(lines) + "\n").encode()
        pending = label_path.with_suffix(".partial")
        pending.write_bytes(updated)
        pending.replace(label_path)
        reviewed.append(dict(image=image_name, group=replay[image_name]["group"],
                             boxes=boxes, original_label_sha256=original_hash,
                             updated_label_sha256=sha256(updated).hexdigest()))
    return dict(source="PCB-Defect v2", reviewed=reviewed)


def prepare_missing_hole_input(inputs, work, replay_manifest):
    manifests = sorted(inputs.rglob(MISSING_HOLE_MANIFEST))
    archives = sorted(inputs.rglob(MISSING_HOLE_ZIP)) if not manifests else []
    if len(manifests) + len(archives) != 1:
        raise RuntimeError(f"Attach one Kaggle input containing {MISSING_HOLE_ZIP}.")
    with ExitStack() as stack:
        if manifests:
            path = manifests[0]
            raw_manifest = path.read_bytes()

            def open_member(name):
                return (path.parent / name).open("rb")
        else:
            archive = stack.enter_context(ZipFile(archives[0]))
            raw_manifest = archive.read(MISSING_HOLE_MANIFEST)
            open_member = archive.open
        addon = json.loads(raw_manifest)
        rows = addon["images"]
        assert addon["names"] == replay_manifest["checkpoint_classes"][:5] + ["missing_hole"]
        expected_counts = (180, 375, 90) if MISSING_HOLE_LAYOUT_REPLAY else (160, 325, 80)
        image_count, box_count, positive_count = expected_counts
        assert len(rows) == image_count and sum(row["missing_hole_boxes"] for row in rows) == box_count
        assert sum(row["missing_hole_boxes"] > 0 for row in rows) == positive_count
        if MISSING_HOLE_LAYOUT_REPLAY:
            added = [row for row in rows if row["group"] == "11"]
            assert len(added) == 20 and sum(row["missing_hole_boxes"] for row in added) == 50
            assert set(addon["validation_groups"]) == {"05", "06", "10"}
        assert {row["group"] for row in rows}.isdisjoint(addon["validation_groups"])
        assert {row["image_sha256"] for row in rows}.isdisjoint(
            row["image_sha256"]
            for key in ("source_replay", "target_support", "target_test")
            for row in replay_manifest[key]
        )
        files = {"balanced_start.pt": BALANCED_SHA}
        for row in rows:
            name = row["image"]
            assert name == PurePosixPath(name).name and name.endswith(".jpg")
            files[f"pku/images/train/{name}"] = row["image_sha256"]
            files[f"pku/labels/train/{Path(name).stem}.txt"] = row["label_sha256"]
        assert len(files) == 1 + 2 * image_count
        for name, expected_hash in files.items():
            destination = work / name
            if destination.is_file() and digest(destination) == expected_hash:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix(destination.suffix + ".partial")
            with open_member(name) as source, pending.open("wb") as output:
                copyfileobj(source, output)
            assert digest(pending) == expected_hash, name
            pending.replace(destination)
        for folder in ("pku/images/train", "pku/labels/train"):
            assert {str(p.relative_to(work)) for p in (work / folder).iterdir()} == {
                name for name in files if name.startswith(folder + "/")
            }
    return addon["names"], sha256(raw_manifest).hexdigest(), image_count, box_count


def prepare_missing_hole_backgrounds(inputs, work, replay_manifest, layout_hash):
    manifests = sorted(inputs.rglob(MISSING_HOLE_BACKGROUNDS_MANIFEST))
    archives = sorted(inputs.rglob(MISSING_HOLE_BACKGROUNDS_ZIP)) if not manifests else []
    if len(manifests) + len(archives) != 1:
        raise RuntimeError(f"Attach one Kaggle input containing {MISSING_HOLE_BACKGROUNDS_ZIP}.")
    with ExitStack() as stack:
        if manifests:
            path = manifests[0]
            raw_manifest = path.read_bytes()

            def open_member(name):
                return (path.parent / name).open("rb")
        else:
            archive = stack.enter_context(ZipFile(archives[0]))
            raw_manifest = archive.read(MISSING_HOLE_BACKGROUNDS_MANIFEST)
            open_member = archive.open
        manifest = json.loads(raw_manifest)
        assert manifest["layout_manifest_sha256"] == layout_hash
        assert manifest["checkpoint_sha256"] == LAYOUT11_SHA
        training = {row["image"]: row for row in replay_manifest["target_support"]}
        holdout_groups = {row["group"] for row in replay_manifest["target_test"]}
        files = {"layout11_start.pt": LAYOUT11_SHA}
        rows = manifest["backgrounds"]
        assert rows and len({row["image_sha256"] for row in rows}) == len(rows)
        for row in rows:
            parent = training[row["parent"]]
            assert row["group"] == parent["group"] and row["group"] not in holdout_groups
            assert row["parent_sha256"] == parent["image_sha256"]
            name = row["image"]
            assert name == PurePosixPath(name).name and name.endswith(".png")
            files[f"backgrounds/images/train/{name}"] = row["image_sha256"]
            files[f"backgrounds/labels/train/{Path(name).stem}.txt"] = sha256(b"").hexdigest()
        assert len(files) == 1 + 2 * len(rows)
        for name, expected_hash in files.items():
            destination = work / name
            if destination.is_file() and digest(destination) == expected_hash:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            pending = destination.with_suffix(destination.suffix + ".partial")
            with open_member(name) as source, pending.open("wb") as output:
                copyfileobj(source, output)
            assert digest(pending) == expected_hash, name
            pending.replace(destination)
        for folder in ("backgrounds/images/train", "backgrounds/labels/train"):
            assert {str(p.relative_to(work)) for p in (work / folder).iterdir()} == {
                name for name in files if name.startswith(folder + "/")
            }
    return work / "layout11_start.pt", sha256(raw_manifest).hexdigest(), len(rows)


def run():
    assert TRAIN_IMAGE_SIZE in (1280, 1536)
    assert sum((ADD_PCB_IND_SHORTS, ADD_REVIEWED_SOURCE_SHORTS, ADD_MISSING_HOLE)) <= 1
    assert not ADD_MISSING_HOLE_BACKGROUNDS or (ADD_MISSING_HOLE and MISSING_HOLE_LAYOUT_REPLAY)
    run_name = f"pcb_defectdet_{RUN_ID}"
    high_resolution = TRAIN_IMAGE_SIZE == 1536
    if high_resolution:
        assert (TARGET_IMAGES, REPLAY_IMAGES) == (100, 100)
        run_name += "_1536"
    if ADD_PCB_IND_SHORTS:
        assert (TARGET_IMAGES, REPLAY_IMAGES, TRAIN_IMAGE_SIZE) == (100, 100, 1280)
        run_name += "_pcbind64"
    if ADD_REVIEWED_SOURCE_SHORTS:
        assert (TARGET_IMAGES, REPLAY_IMAGES, TRAIN_IMAGE_SIZE) == (100, 100, 1280)
        run_name += "_reviewed_shorts"
    if ADD_MISSING_HOLE:
        assert (TARGET_IMAGES, REPLAY_IMAGES, TRAIN_IMAGE_SIZE) == (100, 100, 1280)
        run_name += "_missing_hole"
        if MISSING_HOLE_LAYOUT_REPLAY:
            run_name += "_layout11"
        if ADD_MISSING_HOLE_BACKGROUNDS:
            run_name += "_backgrounds"
    work = Path("/kaggle/working") / run_name
    work.mkdir(parents=True, exist_ok=True)
    inputs = Path("/kaggle/input")
    combined_dataset = ADD_PCB_IND_SHORTS or ADD_MISSING_HOLE
    replay_work = work / "replay" if combined_dataset else work
    manifest, manifest_hash = prepare_input(inputs, replay_work)
    starting_checkpoint = (prepare_balanced_checkpoint(inputs, work, manifest_hash)
                           if high_resolution or ADD_PCB_IND_SHORTS or ADD_REVIEWED_SOURCE_SHORTS
                           else replay_work / "base.pt")
    names = manifest["checkpoint_classes"]
    if ADD_MISSING_HOLE:
        names, missing_hole_hash, pku_images, missing_hole_boxes = prepare_missing_hole_input(inputs, work, manifest)
        starting_checkpoint = work / "balanced_start.pt"
        print(f"Training data: 200 original images and {pku_images} PKU images "
              f"with {missing_hole_boxes} missing-hole boxes.", flush=True)
    if ADD_MISSING_HOLE_BACKGROUNDS:
        starting_checkpoint, background_hash, background_count = prepare_missing_hole_backgrounds(
            inputs, work, manifest, missing_hole_hash
        )
        print(f"Adding {background_count} reviewed training crops with intact holes or markings.", flush=True)
    selection_path = work / "pcb_ind_selection.json"
    reviewed_path = work / "reviewed_source_shorts.json"
    if ADD_REVIEWED_SOURCE_SHORTS:
        reviewed = add_reviewed_source_shorts(work, manifest)
        reviewed_path.write_text(json.dumps(reviewed, indent=2) + "\n")
        reviewed_count = sum(len(row["boxes"]) for row in reviewed["reviewed"])
        print(f"Training data: 200 original images, {reviewed_count} added short labels on "
              f"{len(reviewed['reviewed'])} source images.", flush=True)
    if ADD_PCB_IND_SHORTS:
        selection = prepare_pcb_ind_shorts(inputs, work / "pcb_ind")
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        print(f"Training data: 200 whole-board images and {len(selection['patches'])} PCB-IND short patches.",
              flush=True)
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["MPLCONFIGDIR"] = str(work / "matplotlib")

    try:
        installed = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != VERSION:
        constraints = work / "runtime_constraints.txt"
        constraints.write_text("".join(
            f"{name}=={importlib.metadata.version(name)}\n"
            for name in ("torch", "torchvision", "numpy")
        ))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints),
                        f"ultralytics=={VERSION}"], check=True)

    import torch
    import ultralytics
    from torchvision.ops import nms
    from ultralytics import YOLO

    assert ultralytics.__version__ == VERSION
    assert torch.cuda.is_available(), "Enable the Kaggle GPU accelerator and rerun this cell."
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda:0")
    scores = torch.tensor([.9, .8], device="cuda:0")
    assert nms(boxes, scores, .5).tolist() == [0]
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    data = work / "train.yaml"
    # Ultralytics runs a final validator. Give it only the training images.
    train_images = "images/train"
    if combined_dataset:
        extra_images = "pku/images/train" if ADD_MISSING_HOLE else "pcb_ind/images/train"
        train_images = ["replay/data/images/train", extra_images]
        if ADD_MISSING_HOLE_BACKGROUNDS:
            train_images.append("backgrounds/images/train")
    dataset_root = work if combined_dataset else work / "data"
    data.write_text(json.dumps(dict(path=str(dataset_root), train=train_images,
                                    val=train_images, names=names)))
    train_dir = work / "training"
    recipe = dict(data=str(data), epochs=EPOCHS, imgsz=1280, batch=2, nbs=8, device=0,
        workers=0, amp=False, cache=False, val=False, patience=0, save=True, plots=False,
        optimizer="AdamW", lr0=.0002, lrf=.1, cos_lr=True, warmup_epochs=1.0,
        warmup_bias_lr=0.0, seed=42, deterministic=True, rect=False, freeze=0,
        classes=list(range(5)), augmentations=[], mosaic=0.0, mixup=0.0,
        copy_paste=0.0, cutmix=0.0, degrees=0.0, translate=.03, scale=.1,
        shear=0.0, perspective=0.0, fliplr=.5, flipud=.5, hsv_h=0.0, hsv_s=.1, hsv_v=.1,
        project=str(work), name=train_dir.name, exist_ok=False)
    if high_resolution:
        recipe.update(epochs=20, imgsz=1536, batch=1, lr0=.00005)
    if ADD_PCB_IND_SHORTS or ADD_REVIEWED_SOURCE_SHORTS:
        recipe.update(epochs=12, lr0=.00005)
    if ADD_MISSING_HOLE:
        recipe.update(epochs=20, lr0=.0001, classes=list(range(len(names))), cls_remap=True)
    if ADD_MISSING_HOLE_BACKGROUNDS:
        recipe.update(epochs=8, lr0=.00003)
    epochs = recipe["epochs"]
    expected_epochs = list(range(1, epochs + 1))
    identity = dict(start_sha256=digest(starting_checkpoint), manifest_sha256=manifest_hash,
                    ultralytics=ultralytics.__version__, torch=str(torch.__version__), recipe=recipe)
    if ADD_PCB_IND_SHORTS:
        identity["pcb_ind_selection_sha256"] = digest(selection_path)
    if ADD_REVIEWED_SOURCE_SHORTS:
        identity["reviewed_source_shorts_sha256"] = digest(reviewed_path)
    if ADD_MISSING_HOLE:
        identity["missing_hole_manifest_sha256"] = missing_hole_hash
    if ADD_MISSING_HOLE_BACKGROUNDS:
        identity["background_manifest_sha256"] = background_hash
    identity_path = work / "training_identity.json"
    results = train_dir / "results.csv"
    last = train_dir / "weights" / "last.pt"

    def completed_epochs():
        if not results.is_file():
            return []
        with results.open(newline="") as stream:
            return [int(row["epoch"]) for row in csv.DictReader(stream)]

    if train_dir.exists():
        assert identity_path.is_file() and json.loads(identity_path.read_text()) == identity
        if not last.is_file():
            raise RuntimeError(f"No checkpoint in {train_dir}. Inspect the earlier failure before retrying.")
        if completed_epochs() != expected_epochs:
            YOLO(str(last)).train(resume=True, device=0)
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        model = YOLO(str(starting_checkpoint))
        starting_names = names if ADD_MISSING_HOLE_BACKGROUNDS else manifest["checkpoint_classes"]
        assert list(model.names.values()) == starting_names
        model.train(**recipe)

    assert last.is_file() and completed_epochs() == expected_epochs
    report = dict(training_complete=True, epochs=epochs, target_images=TARGET_IMAGES, replay_images=REPLAY_IMAGES,
                  checkpoint_sha256=digest(last), manifest_sha256=manifest_hash)
    if ADD_PCB_IND_SHORTS:
        report.update(pcb_ind_images=PCB_IND_BUDGET,
                      pcb_ind_selection_sha256=identity["pcb_ind_selection_sha256"])
    if ADD_REVIEWED_SOURCE_SHORTS:
        report.update(added_source_short_boxes=reviewed_count,
                      reviewed_source_shorts_sha256=identity["reviewed_source_shorts_sha256"])
    if ADD_MISSING_HOLE:
        assert list(YOLO(str(last)).names.values()) == names
        report.update(classes=names, pku_images=pku_images, missing_hole_boxes=missing_hole_boxes,
                      missing_hole_manifest_sha256=missing_hole_hash)
    if ADD_MISSING_HOLE_BACKGROUNDS:
        report.update(background_count=background_count, background_manifest_sha256=background_hash)
    result_path = work / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    destination = work.parent / f"{run_name}_result.zip"
    pending = destination.with_suffix(".partial")
    with ZipFile(pending, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(last, "last.pt")
        archive.write(results, "results.csv")
        archive.write(identity_path, "training_identity.json")
        archive.write(result_path, "result.json")
        if ADD_PCB_IND_SHORTS:
            archive.write(selection_path, selection_path.name)
        if ADD_REVIEWED_SOURCE_SHORTS:
            archive.write(reviewed_path, reviewed_path.name)
    pending.replace(destination)
    print(f"Download {destination}. This is the fixed-epoch checkpoint for local evaluation.", flush=True)
    from IPython.display import FileLink, display
    display(FileLink(destination.name))


if __name__ == "__main__":
    run()
