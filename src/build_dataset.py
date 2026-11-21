"""Junta as partes convertidas (images_shard*.bin + index + meta + corrompidos) num unico diretorio
e gera a particao FIXA por paciente (splits.csv) com o resumo de contagens.

Uso (Kaggle): python -m src.build_dataset --input_root /kaggle/input --labels <stage_2_train.csv> --out_dir /kaggle/working/ich192
Procura as partes em todo o --input_root (ignorando a pasta 'competitions', que tem milhares de arquivos).
"""
import argparse, json, os, re, shutil
import pandas as pd
from .labels import load_labels
from .split import make_split, summary

PATTERNS = ("images_shard{k}.bin", "index_shard{k}.csv", "meta_shard{k}.csv", "corrupted_shard{k}.txt")


def find_shards(root, skip=("competitions",)):
    """Retorna {k: diretorio} para cada images_shard{k}.bin encontrado (cada k deve aparecer uma vez)."""
    found = {}
    for dp, dn, fn in os.walk(root):
        if os.path.abspath(dp) == os.path.abspath(root):
            dn[:] = [d for d in dn if d not in skip]
        for f in fn:
            m = re.fullmatch(r"images_shard(\d+)\.bin", f)
            if m:
                k = int(m.group(1))
                if k in found:
                    raise RuntimeError(f"parte {k} encontrada duas vezes: {found[k]} e {dp}")
                found[k] = dp
    return found


def build(input_root, labels_csv, out_dir, expected_shards=(0, 1, 2)):
    shards = find_shards(input_root)
    assert sorted(shards) == sorted(expected_shards), f"partes encontradas {sorted(shards)} != esperadas {sorted(expected_shards)}"
    os.makedirs(out_dir, exist_ok=True)
    metas, bad = [], []
    for k, d in sorted(shards.items()):
        for p in PATTERNS:
            src = os.path.join(d, p.format(k=k))
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, p.format(k=k)))
        metas.append(pd.read_csv(os.path.join(d, f"meta_shard{k}.csv")))
        cp = os.path.join(d, f"corrupted_shard{k}.txt")
        if os.path.exists(cp):
            bad += [l for l in open(cp, encoding="utf8").read().splitlines() if l.strip()]
    meta = pd.concat(metas, ignore_index=True)
    assert meta.image_id.is_unique, "image_id repetido entre as partes"
    labels = load_labels(labels_csv)
    df = labels.merge(meta, on="image_id", how="inner")
    report = {"rotulos": len(labels), "convertidas": len(meta), "com_rotulo_e_imagem": len(df),
              "corrompidos": len(bad), "sem_imagem": len(labels) - len(df),
              "partes": {int(k): v for k, v in shards.items()}}
    df = make_split(df)
    df.to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    with open(os.path.join(out_dir, "corrupted.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(bad))
    summ = summary(df)
    summ.to_csv(os.path.join(out_dir, "split_summary.csv"))
    report["resumo"] = json.loads(summ.to_json(orient="index"))
    with open(os.path.join(out_dir, "build_report.json"), "w", encoding="utf8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return df, report, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", default="/kaggle/input")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    df, rep, summ = build(a.input_root, a.labels, a.out_dir)
    print(json.dumps({k: v for k, v in rep.items() if k != "resumo"}, indent=2, ensure_ascii=False))
    print(summ.to_string())
    print("prevalencia por subtipo (todas as imagens):")
    print(df[["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
