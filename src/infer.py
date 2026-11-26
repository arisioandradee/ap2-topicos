"""Regenera as predicoes (preds.npz) de validacao e teste a partir dos PESOS treinados, sem retreinar.

  python -m src.infer --runs <pasta_com_os_pesos> --splits <splits.csv> --img_dir <imagens> --out <runs_inferidos>

Para cada subpasta de --runs:
  weights.pt (ajuste fino)         -> carrega o modelo completo e infere
  head.pt    (extrator congelado)  -> backbone pre-treinado + padronizacao do treino + cabeca linear
  sem pesos  (baselines do TP1)    -> copia o preds.npz existente (o Random Forest nao e salvo)

Os pesos do ajuste fino sao guardados em meia precisao, entao as probabilidades podem diferir
das originais nas ultimas casas decimais.
"""
import argparse, glob, json, os, shutil
import numpy as np, pandas as pd


def carregar(run_dir, args, device):
    """Reconstroi o modelo de uma execucao. Retorna (modelo, mean, std, cabeca_ou_None)."""
    import torch, torch.nn as nn, timm
    from . import models as M
    nome, res = args["model"], int(args["res"])
    medico = M.REGISTRY[nome][1] == "medico"
    wpath = os.path.join(run_dir, "weights.pt")
    if os.path.exists(wpath):
        if medico:
            net = timm.create_model("resnet50", pretrained=False, num_classes=6)
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            net, mean, std = M.build(nome, res, 6, pretrained=False)
        sd = torch.load(wpath, map_location="cpu", weights_only=True)
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        return net.to(device).eval(), mean, std, None
    # extrator congelado: backbone pre-treinado + cabeca salva
    pesos_medico = args.get("weights") if medico else None
    net, mean, std = M.build(nome, res, 6, True, pesos_medico)
    # head.pt guarda tambem mu/sd (arrays numpy) -> use apenas arquivos publicados pelo grupo
    hp = torch.load(os.path.join(run_dir, "head.pt"), map_location="cpu", weights_only=False)
    head = nn.Linear(hp["head"]["weight"].shape[1], 6)
    head.load_state_dict(hp["head"])
    return net.to(device).eval(), mean, std, (head.to(device).eval(), np.asarray(hp["mu"]), np.asarray(hp["sd"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--splits", required=True)
    ap.add_argument("--img_dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="nomes de execucoes (padrao: todas)")
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--radimagenet_weights", default=None, help="ResNet50.pt do RadImageNet (regime congelado)")
    ap.add_argument("--limit", type=int, default=0, help="teste de codigo com poucas imagens; NUNCA reportar")
    a = ap.parse_args(argv)

    import torch
    from .data import build_transform
    from .train import make_loader, predict, sigmoid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.splits)
    feats = {}  # os embeddings do backbone congelado sao iguais entre sementes
    for run_dir in sorted(glob.glob(os.path.join(a.runs, "*"))):
        nome = os.path.basename(run_dir)
        if not os.path.isdir(run_dir) or (a.only and nome not in a.only):
            continue
        odir = os.path.join(a.out, nome); os.makedirs(odir, exist_ok=True)
        shutil.copy(os.path.join(run_dir, "run.json"), os.path.join(odir, "run.json"))
        tem_ft = os.path.exists(os.path.join(run_dir, "weights.pt"))
        tem_head = os.path.exists(os.path.join(run_dir, "head.pt"))
        if not (tem_ft or tem_head):
            shutil.copy(os.path.join(run_dir, "preds.npz"), os.path.join(odir, "preds.npz"))
            print("copiado (sem pesos):", nome, flush=True)
            continue
        args = json.load(open(os.path.join(run_dir, "run.json")))["args"]
        if a.radimagenet_weights:
            args["weights"] = a.radimagenet_weights
        net, mean, std, cab = carregar(run_dir, args, device)
        tf = build_transform("none", False, int(args["res"]), mean, std)
        saida = {}
        for s in ("val", "test"):
            part = df[df.split == s]
            if a.limit:
                part = part.head(a.limit)
            loader = make_loader(part, a.img_dir, tf, a.bs, a.workers)
            if cab is None:
                logits = predict(net, loader, device, bool(args.get("tta")))
            else:
                chave = (args["model"], args["res"], s)
                if chave not in feats:
                    feats[chave] = predict(net, loader, device, feats=True).astype(np.float16)
                head, mu, sd = cab
                z = torch.from_numpy((feats[chave].astype(np.float32) - mu) / sd).float().to(device)
                with torch.no_grad():
                    logits = head(z).cpu().numpy()
            saida[s + "_ids"] = part["image_id"].values
            saida[s + "_prob"] = sigmoid(logits).astype(np.float32)
        np.savez_compressed(os.path.join(odir, "preds.npz"), **saida)
        print("inferido:", nome, flush=True)


if __name__ == "__main__":
    main()
