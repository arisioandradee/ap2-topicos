## Tabela principal (teste)

| configuracao | AUC-ROC | IC 95% | AUC-PR |
|---|---|---|---|
| vit_small_sup_finetune_cosine_aug-main_r160 | 0.964 ± 0.001 | [0.961; 0.967] | 0.879 ± 0.001 |
| dino_vits16_finetune_cosine_aug-main_r160 | 0.963 ± 0.000 | [0.960; 0.967] | 0.877 ± 0.000 |
| resnet50_finetune_cosine_aug-main_r160 | 0.963 ± 0.001 | [0.959; 0.967] | 0.877 ± 0.002 |
| vit_small_sup_finetune_cosine_aug-alt_r160 | 0.963 ± 0.000 | [0.959; 0.966] | 0.875 ± 0.001 |
| vit_small_sup_finetune_gradual_aug-main_r160 | 0.962 ± 0.000 | [0.958; 0.965] | 0.871 ± 0.000 |
| dinov2_small_finetune_cosine_aug-main_r154 | 0.961 ± 0.002 | [0.959; 0.965] | 0.871 ± 0.003 |
| vit_small_sup_finetune_cosine_aug-none_r160 | 0.960 ± 0.001 | [0.956; 0.964] | 0.869 ± 0.002 |
| vit_small_sup_finetune_const_aug-main_r160 | 0.960 ± 0.002 | [0.957; 0.964] | 0.868 ± 0.004 |
| vit_small_sup_finetune_llrd_aug-main_r160 | 0.952 ± 0.000 | [0.948; 0.956] | 0.847 ± 0.001 |
| radimagenet_resnet50_finetune_cosine_aug-main_r160 | 0.951 ± 0.000 | [0.947; 0.954] | 0.841 ± 0.001 |
| dino_vits16_linear_aug-none_r160 | 0.908 ± 0.001 | [0.902; 0.913] | 0.714 ± 0.000 |
| dinov2_small_linear_aug-none_r154 | 0.893 ± 0.000 | [0.886; 0.899] | 0.662 ± 0.001 |
| convnext_tiny_linear_aug-none_r160 | 0.889 ± 0.000 | [0.883; 0.895] | 0.640 ± 0.001 |
| vit_small_sup_linear_aug-none_r160 | 0.886 ± 0.000 | [0.878; 0.892] | 0.665 ± 0.001 |
| resnet50_linear_aug-none_r160 | 0.878 ± 0.000 | [0.870; 0.884] | 0.608 ± 0.002 |
| baseline_hog_rf | 0.856 ± 0.000 | [0.848; 0.864] | 0.533 ± 0.001 |
| radimagenet_resnet50_linear_aug-none_r160 | 0.809 ± 0.001 | [0.802; 0.817] | 0.412 ± 0.003 |
| baseline_trivial | 0.500 ± 0.000 | [0.500; 0.500] | 0.144 ± 0.000 |

## Testes pareados

| comparacao | dAUC | ic_lo | ic_hi | p_bootstrap_pareado | p_delong | conclusao |
|---|---|---|---|---|---|---|
| ViT supervisionado: congelado vs ajuste fino | 0.0805 | 0.0753 | 0.0863 | 0.001 | 0.0 | diferenca significativa |
| CNN: congelado vs ajuste fino | 0.0875 | 0.0818 | 0.0933 | 0.001 | 0.0 | diferenca significativa |
| DINO ViT-S/16: congelado vs ajuste fino | 0.0585 | 0.0544 | 0.0628 | 0.001 | 0.0 | diferenca significativa |
| DINOv2: congelado vs ajuste fino | 0.0712 | 0.0663 | 0.0761 | 0.001 | 0.0 | diferenca significativa |
| Congelado: auto-supervisionado (DINO) vs supervisionado (ViT) | 0.0219 | 0.0183 | 0.0259 | 0.001 | 0.0 | diferenca significativa |
| Ajuste fino: ViT supervisionado vs ResNet50 (CNN vs ViT) | 0.0003 | -0.001 | 0.0015 | 0.66 | 0.5157 | dentro do ruido |
| Ajuste fino: ViT supervisionado vs DINO | 0.0001 | -0.0007 | 0.0009 | 0.83 | 0.8445 | dentro do ruido |
| Ajuste fino: ViT supervisionado vs DINOv2 | 0.0021 | 0.0009 | 0.0034 | 0.001 | 0.0 | diferenca significativa |
| Pre-treino geral vs medico (ajuste fino, ResNet50) | -0.014 | -0.0157 | -0.0122 | 0.001 | 0.0 | diferenca significativa |
| Pre-treino geral vs medico (congelado, ResNet50) | -0.0678 | -0.0743 | -0.0611 | 0.001 | 0.0 | diferenca significativa |
| Ajuste fino: cosseno vs LR constante | 0.0022 | 0.0014 | 0.003 | 0.001 | 0.0 | diferenca significativa |
| Ajuste fino: LLRD vs cosseno | -0.0129 | -0.0146 | -0.0112 | 0.001 | 0.0 | diferenca significativa |
| Ajuste fino: descong. gradual vs cosseno | -0.0029 | -0.0035 | -0.0023 | 0.001 | 0.0 | diferenca significativa |
| Aumento: politica principal vs nenhum | 0.0029 | 0.002 | 0.0039 | 0.001 | 0.0 | diferenca significativa |
| Aumento: politica principal vs alternativa | 0.0014 | 0.0007 | 0.0021 | 0.001 | 0.0 | diferenca significativa |
| Melhor modelo vs baseline classico TP1 | 0.1074 | 0.1005 | 0.1147 | 0.001 | 0.0 | diferenca significativa |
| Baseline classico vs trivial | 0.3589 | 0.3506 | 0.3667 | 0.001 | 0.0 | diferenca significativa |

## Analise estratificada

| estrato | n_pos | auc | n | prevalencia | pacientes | auc_ic_lo | auc_ic_hi |
|---|---|---|---|---|---|---|---|
| any | 16245 | 0.9663 |  |  |  |  |  |
| epidural | 409 | 0.9485 |  |  |  |  |  |
| intraparenchymal | 5709 | 0.9727 |  |  |  |  |  |
| intraventricular | 4080 | 0.9859 |  |  |  |  |  |
| subarachnoid | 5229 | 0.9472 |  |  |  |  |  |
| subdural | 7091 | 0.9586 |  |  |  |  |  |
| posicao: terco inferior (base) | 2368 | 0.9521 | 36402.0 | 0.0651 | 2856.0 | 0.9456 | 0.9584 |
| posicao: terco medio | 9764 | 0.9618 | 37726.0 | 0.2588 | 2856.0 | 0.9567 | 0.9658 |
| posicao: terco superior | 4113 | 0.9684 | 38865.0 | 0.1058 | 2856.0 | 0.9647 | 0.9716 |
| RescaleIntercept=-1024 | 15467 | 0.9665 | 111007.0 | 0.1393 | 2797.0 | 0.9633 | 0.9694 |
| RescaleIntercept=-1000 | 427 | 0.9228 | 1238.0 | 0.3449 | 41.0 | 0.8783 | 0.9583 |
| RescaleIntercept=0 | 331 | 0.9373 | 716.0 | 0.4623 | 23.0 | 0.9054 | 0.9664 |

## Explicabilidade: metricas

| modelo | metodo | delecao_auc | insercao_auc | moldura | osso |
|---|---|---|---|---|---|
| dino_vits16_congelado | aleatorio | 0.3579 | 0.373 | 0.4183 | 0.0453 |
| dino_vits16_congelado | gradcam | 0.1444 | 0.3176 | 0.331 | 0.0489 |
| dino_vits16_congelado | oclusao | 0.1184 | 0.2469 | 0.4542 | 0.0394 |
| radimagenet_resnet50_ft | aleatorio | 0.2937 | 0.2923 | 0.4191 | 0.0573 |
| radimagenet_resnet50_ft | gradcam | 0.0964 | 0.3682 | 0.2421 | 0.037 |
| radimagenet_resnet50_ft | oclusao | 0.1261 | 0.3527 | 0.5345 | 0.0261 |
| resnet50_ft | aleatorio | 0.3644 | 0.362 | 0.419 | 0.0665 |
| resnet50_ft | gradcam | 0.2582 | 0.5211 | 0.3751 | 0.0423 |
| resnet50_ft | oclusao | 0.2356 | 0.5743 | 0.3319 | 0.0581 |
| vit_small_sup_ft | aleatorio | 0.3142 | 0.3108 | 0.418 | 0.0468 |
| vit_small_sup_ft | gradcam | 0.2313 | 0.3396 | 0.2742 | 0.0554 |
| vit_small_sup_ft | oclusao | 0.1424 | 0.4382 | 0.3798 | 0.0494 |

## Explicabilidade: teste de sanidade

| modelo | metodo | modo | mean | std | count |
|---|---|---|---|---|---|
| dino_vits16_congelado | gradcam | full | 0.0476 | 0.3445 | 20 |
| dino_vits16_congelado | gradcam | top | -0.0477 | 0.235 | 20 |
| dino_vits16_congelado | oclusao | full | 0.0056 | 0.1106 | 10 |
| dino_vits16_congelado | oclusao | top | 0.1895 | 0.3526 | 10 |
| radimagenet_resnet50_ft | gradcam | full | 0.0402 | 0.2769 | 20 |
| radimagenet_resnet50_ft | gradcam | top | 0.0928 | 0.2631 | 20 |
| radimagenet_resnet50_ft | oclusao | full | -0.0465 | 0.1726 | 10 |
| radimagenet_resnet50_ft | oclusao | top | 0.1524 | 0.2173 | 10 |
| resnet50_ft | gradcam | full | 0.1162 | 0.3729 | 20 |
| resnet50_ft | gradcam | top | 0.0928 | 0.287 | 20 |
| resnet50_ft | oclusao | full | 0.041 | 0.1196 | 10 |
| resnet50_ft | oclusao | top | 0.0646 | 0.1758 | 10 |
| vit_small_sup_ft | gradcam | full | 0.0443 | 0.1115 | 20 |
| vit_small_sup_ft | gradcam | top | -0.2474 | 0.1996 | 20 |
| vit_small_sup_ft | oclusao | full | -0.0601 | 0.26 | 10 |
| vit_small_sup_ft | oclusao | top | 0.0173 | 0.178 | 10 |
