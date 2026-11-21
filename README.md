# TP2 — Transfer Learning, Data Augmentation e Interpretabilidade
**Equipe G1 · Desafio RSNA 2019 Intracranial Hemorrhage Detection** (continuação do TP1)

Pergunta central: modelos pré-treinados (CNN, ViT, DINOv2, modelo médico) superam o baseline clássico do TP1 (HOG + Random Forest)? Quais escolhas de pré-treino, ajuste fino e aumento de dados explicam o ganho, e as regiões usadas são clinicamente plausíveis?

## Protocolo
- **Dados:** todo o `stage_2_train` (~752 mil imagens). Só há conversão (DICOM → JPEG 224×224, 3 canais = janelas cérebro/subdural/osso) e exclusão de arquivos corrompidos (listados em `docs/corrupted.txt`).
- **Partição fixa por paciente** (70/15/15, `docs/splits.csv`, semente 2026, sem interseção de pacientes — verificada por `assert`). Época, hiperparâmetros e limiar são escolhidos **somente na validação**; o teste é inferido uma única vez.
- **Submissão tardia no Kaggle** (`stage_2_test`) com comprovante em `docs/`; métricas complementares na partição de teste interna.
- **Baselines do TP1** (trivial e HOG+RF) retreinados no treino completo, avaliados no mesmo teste.
- ≥ 3 sementes por configuração; IC 95% por bootstrap em nível de paciente; testes pareados (bootstrap pareado + DeLong).

## Estrutura
| Caminho | Função |
|---|---|
| `src/prep_dicom.py` | DICOM → JPEG 3 janelas + metadados + lista de corrompidos |
| `src/split.py` | partição por paciente sem vazamento |
| `src/data.py` | dataset e políticas de aumento (`none`, `main`, `alt`) |
| `src/models.py` | backbones pré-treinados (timm), descongelamento gradual, LLRD |
| `src/train.py` | regimes `linear` (congelado) e `finetune` (`const`, `cosine`, `llrd`, `gradual`) |
| `src/baselines.py` | baselines do TP1 no treino completo |
| `src/stats.py` | bootstrap por paciente, bootstrap pareado, DeLong, subgrupos |
| `src/xai.py` | Grad-CAM, oclusão, IG, sanidade por randomização, deleção/inserção, atalhos |
| `src/plan.py` | lista de comandos de todos os experimentos |
| `src/make_report.py` | **regenera as tabelas do artigo a partir de `preds.npz`, sem retreinar** |
| `docs/xai_selection_criterion.md` | critério de seleção de casos (fixado antes dos mapas) |

## Reprodução (Kaggle Notebook com GPU)
```bash
pip install -r requirements.txt
# 1) converter (em partes, se necessário) e montar splits
python -m src.prep_dicom --dicom_dir <stage_2_train> --out_dir img224 --size 224 --workers 4
python -m src.split --labels <stage_2_train.csv> --meta meta_all.csv --out docs/splits.csv
# 2) todos os experimentos (3 sementes)
python -m src.plan --img_dir img224 --splits docs/splits.csv --best_model <escolhido na validação>
# 3) tabelas
python -m src.make_report --splits docs/splits.csv --runs results/runs --out results/tables
```
Testes do código (dados sintéticos): `python -m pytest tests`.

## Pesos, hardware e custo
Preencher após os experimentos: link dos pesos, GPU, tempo por configuração (`results/tables/main_table.csv`).
