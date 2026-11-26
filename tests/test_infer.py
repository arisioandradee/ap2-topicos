import os, sys
import numpy as np, pytest, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import infer
from src.train import main as train_main
from tests.synthetic import make_synthetic


@pytest.mark.parametrize("regime", ["linear", "finetune"])
def test_infer_reproduz_predicoes(tmp_path, regime, monkeypatch):
    """As predicoes refeitas a partir dos pesos salvos devem bater com as do treino.
    A reducao para meia precisao e desligada aqui, para testar so a reconstrucao do modelo."""
    monkeypatch.setattr(torch.Tensor, "half", lambda self, *a, **k: self)
    img_dir, csv, _ = make_synthetic(str(tmp_path), n_patients=60, per_patient=6)
    train_main(["--model", "efficientnet_b0", "--regime", regime, "--aug", "none", "--res", "64", "--bs", "16",
                "--img_dir", img_dir, "--splits", csv, "--out", str(tmp_path / "runs"),
                "--cache_dir", str(tmp_path / "cache"), "--smoke", "--seed", "0"])
    monkeypatch.undo()
    nome = os.listdir(tmp_path / "runs")[0]
    infer.main(["--runs", str(tmp_path / "runs"), "--splits", csv, "--img_dir", img_dir,
                "--out", str(tmp_path / "inferido"), "--limit", "64", "--workers", "0", "--bs", "32"])
    a = np.load(tmp_path / "runs" / nome / "preds.npz", allow_pickle=True)
    b = np.load(tmp_path / "inferido" / nome / "preds.npz", allow_pickle=True)
    for s in ("val", "test"):
        assert list(a[s + "_ids"]) == list(b[s + "_ids"])
        assert np.allclose(a[s + "_prob"], b[s + "_prob"], atol=1e-4)
