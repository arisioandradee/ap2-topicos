"""Rotulos do Kaggle (formato longo) -> tabela larga, sem duplicatas."""
import pandas as pd
from .common import SUBTYPES


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["ID"])
    df[["image_id", "subtype"]] = df["ID"].str.rsplit("_", n=1, expand=True)
    wide = df.pivot(index="image_id", columns="subtype", values="Label").reset_index()
    return wide[["image_id"] + SUBTYPES]
