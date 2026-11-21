"""Estatistica: IC 95% por bootstrap em nivel de PACIENTE, bootstrap pareado, DeLong, subgrupos."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, log_loss
from .common import SUBTYPES, KAGGLE_WEIGHTS


def kaggle_weighted_logloss(Y, P, eps=1e-15):
    """Metrica do desafio RSNA: log-loss media ponderada (any=2, subtipos=1)."""
    P = np.clip(P, eps, 1 - eps)
    ll = -(Y * np.log(P) + (1 - Y) * np.log(1 - P))  # N x 6
    w = np.asarray(KAGGLE_WEIGHTS, float)
    return float((ll.mean(0) * w).sum() / w.sum())


def metrics_any(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p),
            "bal_acc": balanced_accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0)}


def best_threshold_youden(y, p):
    """Limiar escolhido por J de Youden (usar SOMENTE na validacao)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def _patient_index(patients):
    codes, uniq = pd.factorize(pd.Series(patients))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def bootstrap_indices(patients, n_boot=1000, seed=0):
    """Gera n_boot reamostragens de PACIENTES (com reposicao); devolve lista de indices de imagens."""
    groups = _patient_index(patients)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[i] for i in pick])


def bootstrap_ci(y, p, patients, metric=roc_auc_score, n_boot=1000, seed=0):
    """Retorna (estimativa, lo, hi) com IC 95% percentil, bootstrap por paciente."""
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for idx in bootstrap_indices(patients, n_boot, seed):
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    return float(metric(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y, p1, p2, patients, metric=roc_auc_score, n_boot=1000, seed=0):
    """Diferenca metric(p1)-metric(p2) pareada (mesmas reamostragens). Retorna diff, lo, hi, p-valor bilateral."""
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = []
    for idx in bootstrap_indices(patients, n_boot, seed):
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(metric(y[idx], p1[idx]) - metric(y[idx], p2[idx]))
    d = np.asarray(d)
    obs = metric(y, p1) - metric(y, p2)
    pval = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(max(pval, 1.0 / len(d)))


# ---------------- DeLong (Sun & Xu, 2014) ----------------
def _compute_midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx, ty, tz = (np.empty([k, s]) for s in (m, n, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r]); tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx, sy = np.atleast_2d(np.cov(v01)), np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Teste de DeLong pareado (nivel de imagem). Retorna (auc1, auc2, z, p). Complementar ao bootstrap por paciente."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-20))
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def stratified_auc(df, p, col, min_pos=20, min_neg=20):
    """AUC de 'any' por estrato de df[col] (ex.: subtipo, espessura de corte, faixa de posicao)."""
    rows = []
    for k, g in df.assign(_p=p).groupby(col):
        npos = int(g["any"].sum()); nneg = len(g) - npos
        if npos >= min_pos and nneg >= min_neg:
            rows.append({col: k, "n": len(g), "positivos": npos, "auc": roc_auc_score(g["any"], g["_p"])})
    return pd.DataFrame(rows)


def subtype_auc(Y, P):
    """AUC por subtipo (coluna) usando todas as imagens; Y,P: N x 6."""
    return {s: roc_auc_score(Y[:, i], P[:, i]) for i, s in enumerate(SUBTYPES) if Y[:, i].sum() > 0}


def mean_std_over_seeds(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
