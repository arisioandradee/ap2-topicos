"""DICOM -> JPEG de 3 canais (janelas cerebro/subdural/osso) + metadados.

Usa TODO o stage_2_train (permitido: reduzir resolucao, converter formato e excluir
corrompidos, desde que listados). Retomavel: pula o que ja existe.

Uso (Kaggle):
  python -m src.prep_dicom --dicom_dir <.../stage_2_train> --out_dir /kaggle/working/img224 \
      --size 224 --workers 4 [--shard 0 --nshards 4]
Saidas: out_dir/<ID>.jpg, out_dir/meta_shard{k}.csv, out_dir/corrupted_shard{k}.txt
"""
import argparse, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from .common import WINDOWS

META_TAGS = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SliceThickness", "KVP",
             "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "Rows", "Columns",
             "PhotometricInterpretation", "BitsStored", "PixelRepresentation", "RescaleSlope", "RescaleIntercept"]


def to_hu(ds):
    """Pixels -> HU, tratando exames do RSNA com BitsStored=12 sem sinal (offset de +1000)."""
    px = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = px * slope + inter
    if int(getattr(ds, "BitsStored", 16)) == 12 and int(getattr(ds, "PixelRepresentation", 0)) == 0 \
            and hu.min() >= 0 and inter > -100:
        hu = hu - 1000.0
    return hu


def window(hu, c, w):
    lo, hi = c - w / 2, c + w / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def convert_one(args):
    import pydicom
    path, out_path, size = args
    iid = os.path.splitext(os.path.basename(path))[0]
    try:
        ds = pydicom.dcmread(path)
        meta = {"image_id": iid}
        for t in META_TAGS:
            v = getattr(ds, t, None)
            if t in ("PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"):
                meta[t] = "|".join(str(float(x)) for x in v) if v is not None else ""
            else:
                meta[t] = str(v) if v is not None else ""
        if os.path.exists(out_path):
            return meta, None
        hu = to_hu(ds)
        chans = [window(hu, c, w) for c, w in WINDOWS]
        img = (np.stack(chans, -1) * 255).astype(np.uint8)
        im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
        im.save(out_path, quality=95, subsampling=0)
        return meta, None
    except Exception as e:  # arquivo corrompido -> listado
        return None, f"{iid}\t{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="SOMENTE para teste de codigo")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.dicom_dir) if f.endswith(".dcm"))[a.shard::a.nshards]
    if a.limit:
        files = files[:a.limit]
    jobs = [(os.path.join(a.dicom_dir, f), os.path.join(a.out_dir, f[:-4] + ".jpg"), a.size) for f in files]
    metas, bad = [], []
    with Pool(a.workers) as p:
        for m, e in tqdm(p.imap_unordered(convert_one, jobs, chunksize=64), total=len(jobs)):
            if m: metas.append(m)
            if e: bad.append(e)
    pd.DataFrame(metas).to_csv(os.path.join(a.out_dir, f"meta_shard{a.shard}.csv"), index=False)
    with open(os.path.join(a.out_dir, f"corrupted_shard{a.shard}.txt"), "w") as f:
        f.write("\n".join(bad))
    print(f"ok={len(metas)} corrompidos={len(bad)}")


if __name__ == "__main__":
    main()
