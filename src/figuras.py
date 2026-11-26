"""Monta as tabelas resumidas e a figura de explicabilidade a partir das saidas do make_report e do run_xai.

  python -m src.figuras --tables results/tables --xai_tables results/tables_xai --xai_figs figures/xai --out figures

Saidas em --out: fig_xai.png (ViT-S/16: 2 primeiros casos VP, FP e FN, na ordem do criterio pre-declarado)
e tabelas.md (tabela principal, ablacoes e medidas de explicabilidade).
"""
import argparse, json, os
import pandas as pd

BASE = "vit_small_sup_finetune_cosine_aug-main_r160"


def tabela_md(df):
    linhas = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    linhas += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(linhas)


def tabelas(tables, xai_tables):
    mt = pd.read_csv(os.path.join(tables, "main_table.csv"))
    principal = pd.DataFrame({
        "configuracao": mt.config,
        "AUC-ROC": [f"{m:.3f} ± {s:.3f}" for m, s in zip(mt.auc_roc_mean, mt.auc_roc_std)],
        "IC 95%": [f"[{a:.3f}; {b:.3f}]" for a, b in zip(mt.auc_ic_lo, mt.auc_ic_hi)],
        "AUC-PR": [f"{m:.3f} ± {s:.3f}" for m, s in zip(mt.auc_pr_mean, mt.auc_pr_std)]})
    pt = pd.read_csv(os.path.join(tables, "paired_tests.csv"))
    pares = pt[["comparacao", "dAUC", "ic_lo", "ic_hi", "p_bootstrap_pareado", "p_delong", "conclusao"]].round(4)
    est = pd.read_csv(os.path.join(tables, "stratified.csv")).drop(columns="config").round(4)
    for c in ("n", "n_pos", "pacientes"):
        if c in est:
            est[c] = est[c].astype("Int64")
    est = est.astype(object).where(est.notna(), "")
    partes = ["## Tabela principal (teste)", tabela_md(principal), "## Testes pareados", tabela_md(pares),
              "## Analise estratificada", tabela_md(est)]
    for arq, titulo in (("xai_resumo_metricas.csv", "Explicabilidade: metricas"),
                        ("xai_resumo_sanidade.csv", "Explicabilidade: teste de sanidade")):
        caminho = os.path.join(xai_tables, arq)
        if os.path.exists(caminho):
            partes += ["## " + titulo, tabela_md(pd.read_csv(caminho).round(4))]
    return "\n\n".join(partes) + "\n"


def figura(xai_figs, casos_json, destino):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    casos = json.load(open(casos_json))["vit_small_sup_ft"]["casos"]
    paineis = []
    for cat in ("TP", "FP", "FN"):
        im = Image.open(os.path.join(xai_figs, "vit_small_sup_ft", f"casos_{cat}.png")).convert("RGB")
        e = im.size[0] / 1560.0  # as figuras do run_xai tem layout fixo (3 linhas x 4 colunas)
        for j, (x0, x1) in enumerate(((58, 403), (435, 780))):
            caixa = lambda y0, y1: im.crop((int(x0 * e), int(y0 * e), int(x1 * e), int(y1 * e)))
            paineis.append((cat, casos[cat][j], caixa(445, 790), caixa(806, 1150)))
    fig, ax = plt.subplots(2, 6, figsize=(7.2, 2.55))
    for j, (cat, c, cam, ocl) in enumerate(paineis):
        ax[0, j].imshow(cam); ax[1, j].imshow(ocl)
        ax[0, j].set_title(f"{cat}  $y$={c['y']}  $p$={c['p']:.2f}", fontsize=7, pad=2)
        ax[0, j].axis("off"); ax[1, j].axis("off")
    for i, rotulo in enumerate(("Grad-CAM", "Oclusão")):
        ax[i, 0].text(-0.06, 0.5, rotulo, transform=ax[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    plt.subplots_adjust(left=0.03, right=0.995, top=0.9, bottom=0.01, wspace=0.03, hspace=0.04)
    fig.savefig(destino, dpi=300); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables"); ap.add_argument("--xai_tables", default="results/tables_xai")
    ap.add_argument("--xai_figs", default="figures/xai"); ap.add_argument("--cases", default="configs/xai_cases.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tabelas.md"), "w", encoding="utf-8") as f:
        f.write(tabelas(a.tables, a.xai_tables))
    figura(a.xai_figs, a.cases, os.path.join(a.out, "fig_xai.png"))
    print("salvo em", a.out)


if __name__ == "__main__":
    main()
