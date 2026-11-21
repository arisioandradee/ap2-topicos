"""Baselines do TP1 RETREINADOS no treino COMPLETO e avaliados no mesmo teste dos modelos novos.

  trivial : DummyClassifier (classe majoritaria / prior)
  hog_rf  : HOG (canal janela-cerebro, 128x128, parametros do TP1) + RandomForest
            (grade reduzida: 200 arvores x max_depth {None,20} x min_samples_split {2,5}, escolhida por
             validacao em particao FIXA por paciente; semente 0 escolhe, as demais retreinam a melhor)

Saida no mesmo formato dos modelos (preds.npz com 6 colunas; so a coluna 'any' e treinada,
as demais recebem a prevalencia do treino).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from .common import SUBTYPES, seed_everything
from .data import ImageStore


def hog_feat(img, size=128):
    img = img.convert("RGB").split()[0].resize((size, size), Image.BILINEAR)  # canal 0 = janela cerebro
    x = np.asarray(img, np.float32) / 255.0
    return hog(x, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys").astype(np.float32)


def features(df, img_dir, n_jobs):
    store = ImageStore(img_dir)
    ids = df["image_id"].tolist()
    def chunk(sub):
        return [hog_feat(store.get(i)) for i in sub]
    parts = [ids[i:i + 2000] for i in range(0, len(ids), 2000)]
    out = Parallel(n_jobs=n_jobs)(delayed(chunk)(p) for p in tqdm(parts))
    return np.stack([f for o in out for f in o])


def save(odir, name, df, p_val, p_test, prior, info):
    os.makedirs(odir, exist_ok=True)
    va, te = df[df.split == "val"], df[df.split == "test"]
    def six(p):
        m = np.tile(prior, (len(p), 1)).astype(np.float32); m[:, 0] = p; return m
    np.savez_compressed(os.path.join(odir, "preds.npz"), val_ids=va.image_id.values, val_prob=six(p_val),
                        test_ids=te.image_id.values, test_prob=six(p_test))
    json.dump(info, open(os.path.join(odir, "run.json"), "w"), indent=2, default=str)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results/runs"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_jobs", type=int, default=-1); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.splits)
    if a.smoke:
        df = pd.concat([df[df.split == s].head(80) for s in ("train", "val", "test")]).reset_index(drop=True)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    prior = tr[SUBTYPES].mean().values
    suf = "_SMOKE" if a.smoke else ""
    for seed in a.seeds:  # trivial e determinista; mantemos 3 execucoes por consistencia de formato
        d = DummyClassifier(strategy="prior").fit(np.zeros((len(tr), 1)), tr["any"])
        save(os.path.join(a.out, f"baseline_trivial_s{seed}{suf}"), "trivial", df,
             d.predict_proba(np.zeros((len(va), 1)))[:, 1], d.predict_proba(np.zeros((len(te), 1)))[:, 1], prior,
             {"args": {"model": "baseline_trivial", "seed": seed}, "family": "baseline", "smoke": a.smoke,
              "params_total": 0, "params_trainable": 0, "train_time_s": 0})
    t0 = time.time()
    Xtr, Xva, Xte = (features(p, a.img_dir, a.n_jobs) for p in (tr, va, te))
    t_feat = time.time() - t0
    # Orcamento: grade reduzida (4 configs x 200 arvores) escolhida na VALIDACAO com a semente 0 e treino COMPLETO;
    # as demais sementes retreinam so a melhor configuracao.
    grid = [(200, md, ms) for md in (None, 20) for ms in (2, 5)]
    best_cfg, best = None, -1
    for idx, seed in enumerate(a.seeds):
        seed_everything(seed)
        cfgs = grid if idx == 0 else [best_cfg]
        top_auc, top_cfg, top_rf = -1, None, None
        for n, md, ms in cfgs:
            rf = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_split=ms, random_state=seed, n_jobs=a.n_jobs)
            rf.fit(Xtr, tr["any"].values)
            auc = roc_auc_score(va["any"].values, rf.predict_proba(Xva)[:, 1])  # escolha na VALIDACAO
            print("seed", seed, (n, md, ms), "val AUC", round(auc, 4), flush=True)
            if auc > top_auc: top_auc, top_cfg, top_rf = auc, (n, md, ms), rf
        if idx == 0:
            best_cfg, best = top_cfg, top_auc
        save(os.path.join(a.out, f"baseline_hog_rf_s{seed}{suf}"), "hog_rf", df,
             top_rf.predict_proba(Xva)[:, 1], top_rf.predict_proba(Xte)[:, 1], prior,
             {"args": {"model": "baseline_hog_rf", "seed": seed, "cfg(n_est,max_depth,min_split)": top_cfg,
                       "grid": grid, "grid_escolhida_com_semente": a.seeds[0]},
              "family": "baseline", "smoke": a.smoke, "val_auc_any_full": top_auc, "params_total": 0,
              "params_trainable": 0, "train_time_s": time.time() - t0, "feature_time_s": t_feat})


if __name__ == "__main__":
    main()
