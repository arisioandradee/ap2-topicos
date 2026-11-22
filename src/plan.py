"""Gera a lista ordenada de comandos de TODOS os experimentos (mesmo orcamento de treino).

  python -m src.plan --img_dir /kaggle/input/ich224 --splits docs/splits.csv --best_cnn resnet50 [--phase A]

Fases: A congelados | B ajuste fino (todos os modelos) | C ablacao de ajuste fino (melhor modelo)
       D ablacao de aumento (melhor modelo) | E baselines TP1.  Rode A->E; troque --best_cnn/--best_model
depois das fases A/B (a escolha usa so a VALIDACAO).
"""
import argparse

RES = {"resnet50": 160, "convnext_tiny": 160, "efficientnet_b0": 160, "vit_small_sup": 160,
       "dinov2_small": 154, "dino_vits16": 160, "radimagenet_resnet50": 160}
FROZEN = ["resnet50", "convnext_tiny", "vit_small_sup", "dinov2_small", "dino_vits16", "radimagenet_resnet50"]
FT = ["resnet50", "vit_small_sup", "dinov2_small", "dino_vits16", "radimagenet_resnet50"]


def cmds(a):
    base = f"python -m src.train --img_dir {a.img_dir} --splits {a.splits} --out {a.out} " \
           f"--epochs {a.epochs} --epoch_fraction {a.epoch_fraction} --bs {a.bs}"
    w = " --weights {w}"
    def one(m, regime, lr_mode, aug, s):
        c = f"{base} --model {m} --regime {regime} --lr_mode {lr_mode} --aug {aug} --res {RES[m]} --seed {s}"
        if m == "radimagenet_resnet50": c += w.format(w=a.radimagenet_weights)
        return c
    P = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for s in a.seeds:
        for m in FROZEN: P["A"].append(one(m, "linear", "cosine", "none", s))
        for m in FT: P["B"].append(one(m, "finetune", "cosine", "main", s))
        for lm in ("const", "llrd", "gradual"): P["C"].append(one(a.best_model, "finetune", lm, "main", s))
        for au in ("none", "alt"): P["D"].append(one(a.best_model, "finetune", "cosine", au, s))
    P["E"].append(f"python -m src.baselines --img_dir {a.img_dir} --splits {a.splits} --out {a.out} --seeds {' '.join(map(str, a.seeds))}")
    return P


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True); ap.add_argument("--splits", default="docs/splits.csv")
    ap.add_argument("--out", default="results/runs"); ap.add_argument("--phase", default="ABCDE")
    ap.add_argument("--best_model", default="resnet50"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--epoch_fraction", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--radimagenet_weights", default="weights/radimagenet_resnet50.pt")
    a = ap.parse_args()
    P = cmds(a)
    for ph in a.phase:
        print(f"# ---- fase {ph}")
        print("\n".join(P[ph]))
