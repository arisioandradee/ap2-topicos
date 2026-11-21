import json, os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sklearn.metrics import roc_auc_score
from src import stats, split
from src.train import main as train_main
from tests.synthetic import make_synthetic


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("syn")
    return (*make_synthetic(str(root), n_patients=80, per_patient=6), str(root))


def test_split_no_leak_and_ratio(syn):
    _, _, df, _ = syn
    split.check_no_leak(df)
    frac = df.split.value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8 and frac["val"] > 0.08 and frac["test"] > 0.08


def test_split_detects_leak(syn):
    _, _, df, _ = syn
    bad = df.copy(); bad.loc[bad.split == "test", "PatientID"] = bad.loc[bad.split == "train", "PatientID"].iloc[0]
    with pytest.raises(AssertionError):
        split.check_no_leak(bad)


def test_bootstrap_and_delong():
    rng = np.random.default_rng(0)
    n = 3000; pat = np.repeat(np.arange(n // 5), 5)
    y = (rng.random(n) < 0.3).astype(int)
    p1 = y * 0.8 + rng.normal(0, 0.6, n)
    p2 = y * 0.2 + rng.normal(0, 0.6, n)
    est, lo, hi = stats.bootstrap_ci(y, p1, pat, n_boot=200)
    assert lo < est < hi and abs(est - roc_auc_score(y, p1)) < 1e-9
    d, dlo, dhi, pv = stats.paired_bootstrap(y, p1, p2, pat, n_boot=200)
    assert d > 0 and dlo > 0 and pv < 0.05
    a1, a2, z, pd_ = stats.delong_test(y, p1, p2)
    assert abs(a1 - roc_auc_score(y, p1)) < 1e-9 and pd_ < 0.05
    # mesmo modelo contra ele mesmo -> sem diferenca
    _, _, _, pself = stats.delong_test(y, p1, p1 + 1e-9 * rng.normal(size=n))
    assert pself > 0.05


def test_kaggle_logloss():
    Y = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], float)
    assert stats.kaggle_weighted_logloss(Y, np.full((2, 6), 0.5)) == pytest.approx(np.log(2))


@pytest.mark.parametrize("model,regime,lr_mode,aug", [
    ("efficientnet_b0", "linear", "cosine", "main"),
    ("efficientnet_b0", "finetune", "const", "none"),
    ("efficientnet_b0", "finetune", "cosine", "main"),
    ("efficientnet_b0", "finetune", "llrd", "alt"),
    ("efficientnet_b0", "finetune", "gradual", "main"),
])
def test_smoke_train(syn, tmp_path, model, regime, lr_mode, aug):
    img_dir, csv, df, root = syn
    train_main(["--model", model, "--regime", regime, "--lr_mode", lr_mode, "--aug", aug, "--res", "64",
                "--bs", "16", "--img_dir", img_dir, "--splits", csv, "--out", str(tmp_path / "runs"),
                "--cache_dir", str(tmp_path / "cache"), "--smoke", "--seed", "1"])
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    z = np.load(runs[0] / "preds.npz")
    assert z["test_prob"].shape[1] == 6 and ((z["test_prob"] >= 0) & (z["test_prob"] <= 1)).all()
    info = json.load(open(runs[0] / "run.json"))
    assert info["smoke"] and info["params_total"] > 0


def test_make_report_end_to_end(syn, tmp_path):
    import json as js, pandas as pd
    from src import make_report
    from src.common import SUBTYPES
    _, csv, df, _ = syn
    te, va = df[df.split == "test"], df[df.split == "val"]
    rng = np.random.default_rng(0)
    comp = []
    for cfg, strength in [("resnet50_finetune_cosine_aug-main_r160", 1.5), ("resnet50_linear_aug-main_r160", 0.6)]:
        for s in range(3):
            d = tmp_path / "runs" / f"{cfg}_s{s}"; d.mkdir(parents=True)
            def prob(part):
                z = part["any"].values * strength + rng.normal(0, 1, len(part))
                P = np.tile(1 / (1 + np.exp(-z))[:, None], (1, 6)); return P.astype(np.float32)
            np.savez(d / "preds.npz", val_ids=va.image_id.values, val_prob=prob(va), test_ids=te.image_id.values, test_prob=prob(te))
            js.dump({"params_total": 1, "params_trainable": 1, "train_time_s": 1.0, "device": "cpu"}, open(d / "run.json", "w"))
    comp = [{"label": "x", "a": "resnet50_finetune_cosine_aug-main_r160", "b": "resnet50_linear_aug-main_r160"}]
    js.dump(comp, open(tmp_path / "c.json", "w"))
    make_report.main(["--splits", csv, "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "t"),
                      "--n_boot", "100", "--comparisons", str(tmp_path / "c.json")])
    t = pd.read_csv(tmp_path / "t" / "main_table.csv")
    assert t.iloc[0]["config"].startswith("resnet50_finetune") and t.iloc[0]["n_seeds"] == 3
    assert len(pd.read_csv(tmp_path / "t" / "paired_tests.csv")) == 1


def _fake_dicom(path, seed, bits12=False):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    rng = np.random.default_rng(seed)
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = f"P{seed % 3}"; ds.StudyInstanceUID = "1.2.3"; ds.SeriesInstanceUID = "1.2.4"
    ds.Rows = ds.Columns = 128; ds.BitsAllocated = 16; ds.BitsStored = 12 if bits12 else 16
    ds.HighBit = ds.BitsStored - 1; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0 if bits12 else 1
    ds.RescaleSlope = 1; ds.RescaleIntercept = 0 if bits12 else -1024
    base = 1040 if bits12 else 1064  # ambos = 40 HU
    px = (base + rng.normal(0, 10, (128, 128))).astype(np.uint16 if bits12 else np.int16)
    ds.PixelData = px.tobytes(); ds.ImagePositionPatient = [0, 0, float(seed)]; ds.SliceThickness = 5
    ds.save_as(str(path), enforce_file_format=True)


def test_prep_dicom_pack_and_read(tmp_path):
    from src import prep_dicom
    from src.data import ImageStore
    d = tmp_path / "dcm"; d.mkdir()
    for i in range(6):
        _fake_dicom(d / f"ID_{i:03d}.dcm", i, bits12=(i % 2 == 0))
    (d / "ID_bad.dcm").write_bytes(b"lixo")
    out = tmp_path / "out"
    import sys
    sys.argv = ["x", "--dicom_dir", str(d), "--out_dir", str(out), "--size", "64", "--workers", "1"]
    prep_dicom.main()
    assert "ID_bad" in (out / "corrupted_shard0.txt").read_text()
    st = ImageStore(str(out))
    img = st.get("ID_001")
    assert img.size == (64, 64) and len(st.loc) == 6
    a = np.asarray(img).astype(float)
    # janela do cerebro (40,80): HU~0 -> ~0.5 (apos correcao do offset de 12 bits tambem)
    assert abs(np.asarray(st.get("ID_000"))[..., 0].mean() - np.asarray(st.get("ID_001"))[..., 0].mean()) < 25


def test_build_dataset_merge_and_split(tmp_path):
    import pandas as pd
    from src import build_dataset
    from src.common import SUBTYPES
    rng = np.random.default_rng(0)
    rows, lab = [], []
    n = 0
    for p in range(60):
        for k in range(5):
            iid = f"ID_{n:06d}"; n += 1
            rows.append({"image_id": iid, "PatientID": f"P{p}", "StudyInstanceUID": f"S{p}", "SliceThickness": "5"})
            y = int(rng.random() < 0.3)
            for s in SUBTYPES:
                lab.append({"ID": f"{iid}_{s}", "Label": y if s == "any" else int(y and rng.random() < 0.3)})
    meta = pd.DataFrame(rows)
    pd.DataFrame(lab).to_csv(tmp_path / "labels.csv", index=False)
    for k in range(3):
        d = tmp_path / "in" / ("datasets" if k else "notebooks") / f"x{k}" / "img192"; d.mkdir(parents=True)
        part = meta.iloc[k::3]
        part.to_csv(d / f"meta_shard{k}.csv", index=False)
        pd.DataFrame({"image_id": part.image_id, "bin": f"images_shard{k}.bin", "offset": 0, "length": 1}).to_csv(d / f"index_shard{k}.csv", index=False)
        (d / f"images_shard{k}.bin").write_bytes(b"x")
        (d / f"corrupted_shard{k}.txt").write_text("ID_bad\terro" if k == 1 else "")
    (tmp_path / "in" / "competitions" / "c").mkdir(parents=True)
    df, rep, summ = build_dataset.build(str(tmp_path / "in"), str(tmp_path / "labels.csv"), str(tmp_path / "out"))
    assert rep["convertidas"] == 300 and rep["corrompidos"] == 1 and len(df) == 300
    assert (tmp_path / "out" / "splits.csv").exists() and (tmp_path / "out" / "images_shard2.bin").exists()
    build_dataset.split.check_no_leak(df) if hasattr(build_dataset, "split") else None
    # parte faltando -> erro
    import shutil
    shutil.rmtree(tmp_path / "in" / "notebooks")
    with pytest.raises(AssertionError):
        build_dataset.build(str(tmp_path / "in"), str(tmp_path / "labels.csv"), str(tmp_path / "out2"))
