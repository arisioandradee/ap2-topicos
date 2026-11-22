"""Executa uma fase do plano (src.plan) em sequencia, modelo a modelo, apagando o cache de features
de cada modelo depois de treinar as cabecas de todas as sementes (economiza disco no Kaggle).

  python -m src.run_phase --phase A --img_dir <dir> --splits <splits.csv> --out runs --cache_dir /tmp/cache \
         --models resnet50 convnext_tiny vit_small_sup dinov2_small dino_vits16
"""
import argparse, os, shutil, subprocess, sys, time
from types import SimpleNamespace
from . import plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A"); ap.add_argument("--img_dir", required=True)
    ap.add_argument("--splits", required=True); ap.add_argument("--out", default="runs")
    ap.add_argument("--cache_dir", default="/tmp/cache"); ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--best_model", default="resnet50")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--epoch_fraction", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    p = SimpleNamespace(img_dir=a.img_dir, splits=a.splits, out=a.out, seeds=a.seeds, best_model=a.best_model,
                        epochs=a.epochs, epoch_fraction=a.epoch_fraction, bs=a.bs,
                        radimagenet_weights="weights/radimagenet_resnet50.pt")
    cmds = plan.cmds(p)[a.phase]
    models = a.models or sorted({c.split("--model ")[1].split()[0] for c in cmds})
    for m in models:
        mine = [c for c in cmds if f"--model {m} " in c]
        for c in mine:
            c += f" --cache_dir {a.cache_dir}"
            print(">>", c, flush=True)
            if a.dry:
                continue
            t0 = time.time()
            r = subprocess.run(c, shell=True)
            print(f"<< codigo {r.returncode} em {time.time() - t0:.0f}s", flush=True)
            if r.returncode != 0:
                print("FALHOU:", c, file=sys.stderr, flush=True)
        if not a.dry and os.path.isdir(a.cache_dir):
            shutil.rmtree(a.cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
