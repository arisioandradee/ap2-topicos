"""Particao FIXA por paciente (70/15/15), estratificada por 'any'. Sem vazamento (assert).

Uso: python -m src.split --labels stage_2_train.csv --meta meta_all.csv --out docs/splits.csv
"""
import argparse
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .labels import load_labels

SEED = 2026


def make_split(df, seed=SEED, group="PatientID"):
    """df com image_id, any, PatientID -> coluna split in {train,val,test}."""
    df = df.reset_index(drop=True).copy()
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)  # 14/3/3 folds = 70/15/15
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df["any"], df[group])):
        fold[idx] = k
    df["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    check_no_leak(df, group)
    return df


def check_no_leak(df, group="PatientID"):
    s = {k: set(df.loc[df.split == k, group]) for k in ("train", "val", "test")}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) and not (s["val"] & s["test"]), \
        "VAZAMENTO: paciente em mais de uma particao"


def summary(df):
    agg = {"imagens": ("image_id", "size"), "pacientes": ("PatientID", "nunique"), "prevalencia_any": ("any", "mean")}
    if "StudyInstanceUID" in df:
        agg["exames"] = ("StudyInstanceUID", "nunique")
    return df.groupby("split").agg(**agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = load_labels(a.labels)
    meta = pd.read_csv(a.meta).drop_duplicates("image_id")
    df = lab.merge(meta, on="image_id", how="inner")
    print(f"rotulos={len(lab)} com_metadados={len(df)} (excluidos/corrompidos={len(lab) - len(df)})")
    df = make_split(df)
    df.to_csv(a.out, index=False)
    print(summary(df))


if __name__ == "__main__":
    main()
