"""Regenera TODAS as tabelas do artigo a partir de results/runs/*/preds.npz (sem retreinar).

Saidas em results/tables/:
  main_table.csv        : por configuracao: AUC-ROC media +- dp entre sementes, IC95% (bootstrap por paciente,
                          da media das probabilidades das sementes? NAO: IC da semente mediana, ver abaixo),
                          AUC-PR, bal.acc, F1 (limiar de Youden na VALIDACAO), log-loss Kaggle, params, tempo
  paired_tests.csv      : comparacoes centrais (bootstrap pareado por paciente + DeLong)
  stratified.csv        : AUC por subtipo / subgrupo
Uso: python -m src.make_report --splits docs/splits.csv --runs results/runs --out results/tables
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from .common import SUBTYPES
from . import stats

# comparacoes centrais do enunciado: (rotulo, config A, config B)  [prefixos de nome sem _s<seed>]
DEFAULT_COMPARISONS = "configs/comparisons.json"


def load_runs(runs_dir):
    runs = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if d.endswith("_SMOKE") or not os.path.exists(os.path.join(d, "preds.npz")):
            continue
        cfg, seed = d.rsplit("_s", 1)
        info = json.load(open(os.path.join(d, "run.json")))
        runs.setdefault(os.path.basename(cfg), {})[int(seed)] = {"dir": d, "info": info}
    return runs


def load_preds(d):
    z = np.load(os.path.join(d, "preds.npz"), allow_pickle=True)
    return z


def aligned(df_test, ids, prob):
    s = pd.Series(range(len(ids)), index=ids)
    return prob[s.loc[df_test["image_id"]].values]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/tables"); ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--comparisons", default=DEFAULT_COMPARISONS)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.splits)
    te, va = df[df.split == "test"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    y, pat = te["any"].values, te["PatientID"].values
    runs = load_runs(a.runs)
    rows, cache = [], {}
    for cfg, seeds in runs.items():
        aucs, per_seed = [], []
        for s, r in sorted(seeds.items()):
            z = load_preds(r["dir"])
            pt = aligned(te, z["test_ids"], z["test_prob"]); pv = aligned(va, z["val_ids"], z["val_prob"])
            cache[(cfg, s)] = pt
            thr = stats.best_threshold_youden(va["any"].values, pv[:, 0])  # limiar so na validacao
            m = stats.metrics_any(y, pt[:, 0], thr)
            m["logloss_kaggle"] = stats.kaggle_weighted_logloss(te[SUBTYPES].values.astype(float), pt)
            per_seed.append(m); aucs.append(m["auc_roc"])
        # IC 95%: bootstrap por paciente da semente de AUC mediana (representativa)
        mid = sorted(seeds)[int(np.argsort(aucs)[len(aucs) // 2])]
        est, lo, hi = stats.bootstrap_ci(y, cache[(cfg, mid)][:, 0], pat, n_boot=a.n_boot)
        row = {"config": cfg, "n_seeds": len(seeds), "ic95_semente": mid, "auc_ic_lo": lo, "auc_ic_hi": hi}
        for k in per_seed[0]:
            mu, sd = stats.mean_std_over_seeds([m[k] for m in per_seed])
            row[k + "_mean"], row[k + "_std"] = mu, sd
        info = seeds[mid]["info"]
        row.update(params_total=info.get("params_total"), params_trainable=info.get("params_trainable"),
                   train_time_s=np.mean([r["info"].get("train_time_s", 0) for r in seeds.values()]),
                   hardware=info.get("gpu") or info.get("device"))
        rows.append(row)
    main_t = pd.DataFrame(rows).sort_values("auc_roc_mean", ascending=False)
    main_t.to_csv(os.path.join(a.out, "main_table.csv"), index=False)
    print(main_t[["config", "n_seeds", "auc_roc_mean", "auc_roc_std", "auc_ic_lo", "auc_ic_hi"]].to_string(index=False))

    # testes pareados (media das probabilidades entre sementes = predicao da configuracao)
    def cfg_pred(c):
        return np.mean([cache[(c, s)][:, 0] for s in runs[c]], axis=0)
    comp = json.load(open(a.comparisons)) if os.path.exists(a.comparisons) else []
    prow = []
    for c in comp:
        A, B = c["a"], c["b"]
        if A not in runs or B not in runs:
            continue
        d, lo, hi, pb = stats.paired_bootstrap(y, cfg_pred(A), cfg_pred(B), pat, n_boot=a.n_boot)
        _, _, z, pd_ = stats.delong_test(y, cfg_pred(A), cfg_pred(B))
        prow.append({"comparacao": c["label"], "A": A, "B": B, "dAUC": d, "ic_lo": lo, "ic_hi": hi,
                     "p_bootstrap_pareado": pb, "p_delong": pd_,
                     "conclusao": "diferenca significativa" if (lo > 0 or hi < 0) else "dentro do ruido"})
    pd.DataFrame(prow).to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)

    # estratificada: AUC por subtipo (positivos do subtipo vs. todos os negativos de 'any') para a melhor config
    best = main_t.iloc[0]["config"]
    P = np.mean([cache[(best, s)] for s in runs[best]], axis=0)
    Y = te[SUBTYPES].values.astype(float)
    rows = [{"estrato": s, "n_pos": int(Y[:, i].sum()), "auc": v} for i, (s, v) in
            enumerate(stats.subtype_auc(Y, P).items())]
    for col in ("SliceThickness", "KVP"):
        if col in te:
            sa = stats.stratified_auc(te, P[:, 0], col)
            rows += [{"estrato": f"{col}={r[col]}", "n_pos": r["positivos"], "auc": r["auc"]} for _, r in sa.iterrows()]
    pd.DataFrame(rows).assign(config=best).to_csv(os.path.join(a.out, "stratified.csv"), index=False)
    print("melhor config:", best)


if __name__ == "__main__":
    main()
