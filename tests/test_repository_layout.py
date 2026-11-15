import json
from hashlib import file_digest
from pathlib import Path

from backend.model_registry import PROJECT_MODEL
from config import ROOT
from rul_synthesis.ensemble_runtime import MODEL_DIR
from revamp.paths import historical_path


def test_historical_paths_use_longest_matching_prefix(tmp_path):
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/legacy_paths.json").write_text(json.dumps({"old":"archive/old","old/special":"data/new"}))
    assert historical_path(tmp_path,"old/special/image.jpg") == tmp_path/"data/new/image.jpg"
    assert historical_path(tmp_path,"old/other") == tmp_path/"archive/old/other"
    assert historical_path(tmp_path,"older/file") == tmp_path/"older/file"


def test_final_entrypoints_and_artifacts_are_present():
    files = [
        "scripts/manage.py",
        "src/backend/main.py",
        "notebooks/PCB_DefectDet_K50_Cell.py",
        "notebooks/PCB_RUL_Colab_Evaluation.ipynb",
        "output/pcb_ieee_latex/main.tex",
        "pcb_rul/outputs/detector_stats.json",
        "examples/pcb_defect_002.jpg",
        "examples/img62.png",
    ]
    for relative in files:
        assert (ROOT / relative).is_file(), relative


def test_final_model_files_match_their_manifests():
    detector = ROOT / "artifacts/models" / PROJECT_MODEL
    info = json.loads((detector / "model_info.json").read_text())
    with (detector / "model.pt").open("rb") as checkpoint:
        assert file_digest(checkpoint, "sha256").hexdigest() == info["checkpoint_sha256"]

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        with (MODEL_DIR / name).open("rb") as file:
            assert file_digest(file, "sha256").hexdigest() == expected, name
