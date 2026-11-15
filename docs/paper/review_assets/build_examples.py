"""Render the PKU demonstration and audited double-layer false alert."""
import hashlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryFile
from shutil import copyfileobj
from zipfile import ZipFile

os.environ.setdefault("YOLO_CONFIG_DIR", "/private/tmp/pcb-paper-yolo")
from build_figures import ROOT, REPORTS, Image, plt, Rectangle, save, np, NAMES, COLORS

sys.path.insert(0, str(ROOT / "src"))
from backend.model_registry import ModelRegistry, PROJECT_MODEL, PROJECT_CONFIDENCE
import cv2

pku = ROOT / "data/raw/pku/PCB_DATASET"
image_file = pku / "images/Missing_hole/10_missing_hole_02.jpg"
annotation = pku / "Annotations/Missing_hole/10_missing_hole_02.xml"
raw = image_file.read_bytes()
image = Image.open(io.BytesIO(raw)).convert("RGB")
result = ModelRegistry([PROJECT_MODEL]).infer(cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR),
                                              PROJECT_MODEL, PROJECT_CONFIDENCE, False)
truth = [[float(obj.findtext("bndbox/" + key)) for key in ("xmin","ymin","xmax","ymax")]
         for obj in ET.parse(annotation).findall("object") if obj.findtext("name") == "missing_hole"]
pred = [d["bbox"] for d in result["detections"] if d["class_name"] == "missing_hole"]
from build_figures import iou
overlap = iou(np.array(truth), np.array(pred))
assert len(truth) == len(pred) == 5 and (overlap >= .5).sum() == 5
fig, axes = plt.subplots(1,2,figsize=(10,4.8),layout="constrained")
for ax, label, boxes in zip(axes, ("PKU annotations", "Final app predictions"), (truth,pred)):
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(label)
    for x1,y1,x2,y2 in boxes:
        ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, ec="#FF6572", lw=2))
fig.suptitle("Synthetic PKU layout 10 | Five annotated missing holes",weight="bold")
save(fig,"06_missing_hole_example",False)

audit = json.loads((REPORTS/"pcb100_defectdet_missing_hole_double_layer_alert_audit.json").read_text())
rows = [audit["alerts"][0], next(x for x in audit["alerts"] if x["visual_observation"] == "visible hole")]
with ZipFile(ROOT/"artifacts/pcb_defectdet_k100_r100_input.zip") as archive:
    manifest = json.loads(archive.read("defectdet_k100_r100_manifest.json"))
hashes = {r["image"]:r["image_sha256"] for r in manifest["target_test"]}
fig, axes = plt.subplots(1,2,figsize=(8,3.5),layout="constrained")
if len(sys.argv) > 1:
    cache = Path(sys.argv[1])
    images = {row["image"]: (cache / row["image"]).read_bytes() for row in rows}
else:
    with ZipFile(ROOT / "data/raw/pcb_archives/DefectDet An Annotated Dataset of Printed Circuit.zip") as outer:
        with outer.open(outer.namelist()[0]) as source, TemporaryFile() as nested:
            copyfileobj(source, nested, length=8 * 1024 * 1024)
            nested.seek(0)
            with ZipFile(nested) as archive:
                images = {row["image"]: archive.read(f'DefectDet/images/{row["image"]}') for row in rows}
for ax, row in zip(axes, rows):
    raw = images[row["image"]]
    assert hashlib.sha256(raw).hexdigest() == hashes[row["image"]]
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    x1,y1,x2,y2 = row["xyxy"]
    margin = max(x2-x1,y2-y1)*.7
    ax.imshow(image)
    ax.set(xlim=(x1-margin,x2+margin),ylim=(y2+margin,y1-margin))
    ax.axis("off")
    ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,ec="#FF6572",lw=2))
    ax.set_title(f"{row['visual_observation'].capitalize()} | {row['image']}\nMissing-hole score {row['confidence']:.2f}")
fig.suptitle("Examples of false missing-hole alerts on DefectDet",weight="bold")
save(fig,"11_false_alert_examples",False)
