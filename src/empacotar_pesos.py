"""Monta o pacote de pesos a partir das saidas dos notebooks de treino (uma pasta por execucao).

  python -m src.empacotar_pesos --search /kaggle/input --out /kaggle/working/pesos --llrd_de <nome_do_notebook>

Leva run.json + weights.pt (ajuste fino) ou head.pt (extrator congelado). Nos baselines do TP1, que nao
tem pesos salvos, leva o preds.npz. As execucoes de LLRD so sao aceitas se vierem de --llrd_de e tiverem
as escalas por camada registradas (a primeira versao do LLRD foi descartada).
"""
import argparse, glob, json, os, shutil


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--llrd_de", required=True, help="trecho do caminho de onde vem o LLRD valido")
    ap.add_argument("--esperado", type=int, default=54)
    a = ap.parse_args(argv)
    escolhidas = {}
    for rj in sorted(glob.glob(os.path.join(a.search, "**", "run.json"), recursive=True)):
        d = os.path.dirname(rj)
        nome = os.path.basename(d)
        info = json.load(open(rj))
        if nome.endswith("_SMOKE") or info.get("smoke"):
            continue
        if "_llrd_" in nome and (a.llrd_de not in d or not info.get("llrd_scales")):
            print("ignorado (LLRD descartado):", d)
            continue
        if nome in escolhidas:
            raise SystemExit(f"execucao repetida: {nome}\n  {escolhidas[nome]}\n  {d}")
        escolhidas[nome] = d
    for nome, d in sorted(escolhidas.items()):
        destino = os.path.join(a.out, nome); os.makedirs(destino, exist_ok=True)
        if nome.startswith("baseline_"):
            arquivos = ["run.json", "preds.npz"]
        else:
            arquivos = ["run.json", "weights.pt" if "_finetune_" in nome else "head.pt"]
        for f in arquivos:
            shutil.copy(os.path.join(d, f), os.path.join(destino, f))
        print(nome, "<-", d)
    print("total de execucoes:", len(escolhidas))
    if len(escolhidas) != a.esperado:
        raise SystemExit(f"esperadas {a.esperado} execucoes, encontradas {len(escolhidas)}")


if __name__ == "__main__":
    main()
