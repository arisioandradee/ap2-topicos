"""Dados SINTETICOS (so para testar o codigo; nada disto e reportado no artigo)."""
import os
import numpy as np, pandas as pd
from PIL import Image
from src.common import SUBTYPES
from src.split import make_split


def make_synthetic(root, n_patients=60, per_patient=8, size=64, seed=0):
    rng = np.random.default_rng(seed)
    img_dir = os.path.join(root, "img"); os.makedirs(img_dir, exist_ok=True)
    rows = []
    for p in range(n_patients):
        for k in range(per_patient):
            iid = f"ID_{p:03d}{k:02d}"
            y_any = int(rng.random() < 0.3)
            img = rng.normal(90, 20, (size, size, 3)).clip(0, 255)
            if y_any:  # "hemorragia": mancha clara
                cx, cy = rng.integers(16, size - 16, 2)
                img[cy - 6:cy + 6, cx - 6:cx + 6] += 90
            Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(os.path.join(img_dir, iid + ".jpg"), quality=95)
            r = {"image_id": iid, "PatientID": f"P{p:03d}", "StudyInstanceUID": f"S{p:03d}", "any": y_any}
            for s in SUBTYPES[1:]:
                r[s] = int(y_any and rng.random() < 0.3)
            rows.append(r)
    df = pd.DataFrame(rows)[["image_id"] + SUBTYPES + ["PatientID", "StudyInstanceUID"]]
    df = make_split(df)
    csv = os.path.join(root, "splits.csv"); df.to_csv(csv, index=False)
    return img_dir, csv, df
