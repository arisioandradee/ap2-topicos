"""Executa a analise de explicabilidade (ver docs/xai_selection_criterion.md).

Dois passos:
  1) select  : le SOMENTE as predicoes (preds.npz) e fixa os casos (configs/xai_cases.json). Roda no PC, sem GPU.
  2) explain : carrega os pesos treinados, gera Grad-CAM e Oclusao para esses casos, calcula delecao/insercao,
               fracao em moldura/osso, teste de sanidade (randomizacao de pesos) e salva figuras + tabelas.

  python -m src.run_xai select  --runs <dir_runs> --splits <splits.csv> --out configs/xai_cases.json
  python -m src.run_xai explain --cases configs/xai_cases.json --search <raiz_com_os_runs> --img_dir <dir_imagens> --out xai_out
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

# modelo analisado -> (nome no REGISTRY, regime, pasta da execucao com a semente 0)
MODELOS = {
    "resnet50_ft": ("resnet50", "finetune", "resnet50_finetune_cosine_aug-main_r160_s0"),
    "vit_small_sup_ft": ("vit_small_sup", "finetune", "vit_small_sup_finetune_cosine_aug-main_r160_s0"),
    "radimagenet_resnet50_ft": ("radimagenet_resnet50", "finetune", "radimagenet_resnet50_finetune_cosine_aug-main_r160_s0"),
    "dino_vits16_congelado": ("dino_vits16", "linear", "dino_vits16_linear_aug-none_r160_s0"),
}
CATS = ["TP", "FP", "FN", "TN", "limiar"]


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


# ------------------------------------------------------------------------------ 1) selecao
def cmd_select(a):
    from .xai import select_cases
    sp = pd.read_csv(a.splits, usecols=["image_id", "any", "split"]).set_index("image_id")
    out = {}
    for key, (_, _, run) in MODELOS.items():
        d = np.load(os.path.join(a.runs, run, "preds.npz"), allow_pickle=True)
        yv = sp.loc[d["val_ids"], "any"].values
        thr = youden_threshold(yv, d["val_prob"][:, 0])
        ids, p = d["test_ids"], d["test_prob"][:, 0]
        y = sp.loc[ids, "any"].values
        cases = select_cases(y, p, k=a.k, thr=thr)
        out[key] = {"run": run, "limiar_youden_val": thr,
                    "casos": {c: [{"image_id": str(ids[i]), "y": int(y[i]), "p": float(p[i])} for i in idx]
                              for c, idx in cases.items()}}
        print(key, "limiar=%.3f" % thr, {c: len(v) for c, v in cases.items()})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("salvo:", a.out)


# ------------------------------------------------------------------------------ 2) explicacao
def _achar(search, run, arq):
    hits = glob.glob(os.path.join(search, "**", run, arq), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{arq} de {run} nao encontrado em {search}")
    return hits[0]


def construir(key, search, device, res=160):
    """Retorna (modelo, rede_para_camadas, nome_registry, mean, std). Pesos treinados, modo eval."""
    import torch, torch.nn as nn, timm
    from . import models as M
    name, regime, run = MODELOS[key]
    if regime == "finetune":
        if M.REGISTRY[name][1] == "medico":
            net = timm.create_model("resnet50", pretrained=False, num_classes=6)
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            net, mean, std = M.build(name, res, 6, pretrained=False)
        sd = torch.load(_achar(search, run, "weights.pt"), map_location="cpu", weights_only=True)
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        model = net
    else:
        net, mean, std = M.build(name, res, 6, pretrained=True)
        hp = torch.load(_achar(search, run, "head.pt"), map_location="cpu", weights_only=False)

        class Congelado(nn.Module):  # extrator congelado + padronizacao + cabeca linear
            def __init__(s):
                super().__init__()
                s.backbone = net
                s.head = nn.Linear(hp["head"]["weight"].shape[1], 6)
                s.head.load_state_dict(hp["head"])
                s.register_buffer("mu", torch.as_tensor(hp["mu"], dtype=torch.float32))
                s.register_buffer("sd", torch.as_tensor(hp["sd"], dtype=torch.float32))

            def forward(s, x):
                f = M.extract_features(s.backbone, x)
                return s.head((f - s.mu) / s.sd)
        model = Congelado()
    return model.to(device).eval(), name, mean, std


def camada(model, name):
    from .xai import target_layer
    net = model.backbone if hasattr(model, "backbone") else model
    return target_layer(net, name)


def cmd_explain(a):
    import torch, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import xai
    from .data import get_store, build_transform

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = json.load(open(a.cases))
    store = get_store(a.img_dir)
    os.makedirs(a.out, exist_ok=True)
    rows, srows = [], []
    rng = np.random.RandomState(0)
    for key in (a.models or list(MODELOS)):
        if key not in cases:
            continue
        model, name, mean, std = construir(key, a.search, device, a.res)
        tf = build_transform("none", False, a.res, mean, std)
        layer, reshape = camada(model, name)
        odir = os.path.join(a.out, key); os.makedirs(odir, exist_ok=True)

        def gradcam(m, x):
            lay, rs = camada(m, name)
            return xai.saliency_gradcam(m, x, lay, rs, idx=0)

        def occl(m, x):
            return xai.saliency_occlusion(m, x, idx=0, window=a.window, stride=a.stride)

        for cat in CATS:
            fig, ax = plt.subplots(3, len(cases[key]["casos"][cat]), figsize=(3 * len(cases[key]["casos"][cat]), 9), squeeze=False)
            for j, c in enumerate(cases[key]["casos"][cat]):
                iid = c["image_id"]
                img = store.get(iid)
                x = tf(img).unsqueeze(0).to(device)
                disp = np.asarray(img.resize((a.res, a.res)))  # uint8 (H,W,3): canais = janelas cerebro/subdural/osso
                maps = {"gradcam": gradcam(model, x)[0], "oclusao": occl(model, x)[0]}
                maps["aleatorio"] = rng.rand(a.res, a.res).astype(np.float32)  # controle p/ delecao/insercao
                for mname, sal in maps.items():
                    dl, ins, _ = xai.deletion_insertion(model, x, sal, idx=0, steps=a.steps)
                    rows.append(dict(modelo=key, categoria=cat, image_id=iid, y=c["y"], p=c["p"], metodo=mname,
                                     delecao_auc=dl, insercao_auc=ins,
                                     moldura=xai.border_fraction(sal), osso=xai.bone_fraction(sal, disp[..., 0])))
                # teste de sanidade: pesos randomizados (completo e so o topo) -> correlacao deve ser BAIXA
                for mname, fn in (("gradcam", gradcam),) + ((("oclusao", occl),) if (j < a.sanity_occ) else ()):
                    for mode in ("full", "top"):
                        srows.append(dict(modelo=key, categoria=cat, image_id=iid, metodo=mname, modo=mode,
                                          spearman=xai.sanity_correlation(fn, model, x, mode=mode, seed=0)))
                # figura: imagem, Grad-CAM, oclusao
                ax[0, j].imshow(disp[..., 0], cmap="gray", vmin=0, vmax=255)
                ax[0, j].set_title(f"{cat} y={c['y']} p={c['p']:.2f}", fontsize=9)
                for r_, mname in ((1, "gradcam"), (2, "oclusao")):
                    ax[r_, j].imshow(disp[..., 0], cmap="gray", vmin=0, vmax=255)
                    ax[r_, j].imshow(maps[mname], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                for r_ in range(3):
                    ax[r_, j].axis("off")
            ax[0, 0].text(-0.05, 0.5, "imagem (janela cerebro)", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right")
            ax[1, 0].text(-0.05, 0.5, "Grad-CAM", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right")
            ax[2, 0].text(-0.05, 0.5, "Oclusao", transform=ax[2, 0].transAxes, rotation=90, va="center", ha="right")
            fig.suptitle(f"{key} | caso: {cat}", fontsize=11)
            fig.tight_layout(); fig.savefig(os.path.join(odir, f"casos_{cat}.png"), dpi=130); plt.close(fig)
        print("ok:", key, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "xai_metricas_por_caso.csv"), index=False)
        pd.DataFrame(srows).to_csv(os.path.join(a.out, "xai_sanidade_por_caso.csv"), index=False)
    m = pd.DataFrame(rows)
    s = pd.DataFrame(srows)
    if len(m):
        m.groupby(["modelo", "metodo"])[["delecao_auc", "insercao_auc", "moldura", "osso"]].mean().round(4).to_csv(
            os.path.join(a.out, "xai_resumo_metricas.csv"))
        s.groupby(["modelo", "metodo", "modo"])["spearman"].agg(["mean", "std", "count"]).round(4).to_csv(
            os.path.join(a.out, "xai_resumo_sanidade.csv"))
    print("pronto")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--runs", required=True); s.add_argument("--splits", required=True)
    s.add_argument("--out", default="configs/xai_cases.json"); s.add_argument("--k", type=int, default=4)
    e = sub.add_parser("explain")
    e.add_argument("--cases", default="configs/xai_cases.json"); e.add_argument("--search", required=True)
    e.add_argument("--img_dir", required=True); e.add_argument("--out", default="xai_out")
    e.add_argument("--models", nargs="*", default=None); e.add_argument("--res", type=int, default=160)
    e.add_argument("--window", type=int, default=24); e.add_argument("--stride", type=int, default=12)
    e.add_argument("--steps", type=int, default=20); e.add_argument("--sanity_occ", type=int, default=2,
                   help="quantos casos por categoria entram no teste de sanidade da oclusao (mais caro)")
    a = ap.parse_args(argv)
    {"select": cmd_select, "explain": cmd_explain}[a.cmd](a)


if __name__ == "__main__":
    main()
