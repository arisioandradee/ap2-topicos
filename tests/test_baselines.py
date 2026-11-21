import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import baselines
from tests.synthetic import make_synthetic


def test_baselines_smoke(tmp_path):
    img_dir, csv, df = make_synthetic(str(tmp_path), n_patients=60, per_patient=6, size=64)
    baselines.main(["--img_dir", img_dir, "--splits", csv, "--out", str(tmp_path / "runs"), "--seeds", "0", "1", "--n_jobs", "1", "--smoke"])
    runs = sorted(os.listdir(tmp_path / "runs"))
    assert any("baseline_trivial_s0" in r for r in runs) and any("baseline_hog_rf_s1" in r for r in runs)
    z = np.load(tmp_path / "runs" / [r for r in runs if "hog_rf_s0" in r][0] / "preds.npz")
    assert z["test_prob"].shape[1] == 6
    info = json.load(open(tmp_path / "runs" / [r for r in runs if "hog_rf_s1" in r][0] / "run.json"))
    assert info["args"]["cfg(n_est,max_depth,min_split)"] is not None
