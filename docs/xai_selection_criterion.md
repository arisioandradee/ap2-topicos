# Criterio de selecao dos casos de explicabilidade (fixado ANTES de gerar/ver qualquer mapa)

Data de fixacao: 2026-11-20 (ver data do commit deste arquivo).

Para cada modelo analisado (melhor CNN, melhor ViT, extrator congelado, modelo medico):
1. Limiar = ponto de Youden da curva ROC **na validacao**.
2. No conjunto de teste, usando **apenas as predicoes** (nunca os mapas), selecionar k = 4 casos de cada tipo:
   - verdadeiros positivos de maior confianca (maior |p - limiar|);
   - falsos positivos de maior confianca;
   - falsos negativos de maior confianca;
   - verdadeiros negativos de maior confianca;
   - os 4 casos mais proximos do limiar (independente do rotulo).
3. Empates resolvidos pela ordem do indice no arquivo de teste (determinista).
4. Os mesmos criterios (e, quando possivel, as mesmas imagens) sao aplicados aos modelos comparados.
5. Nenhum caso e trocado, removido ou adicionado depois de ver os mapas. Figuras do artigo = recorte
   destes casos; o restante fica em `figures/xai/`.

Metodos: Grad-CAM (gradiente/ativacao; ViT com reorganizacao de tokens em grade) e Oclusao
(perturbacao). Testes: sanidade por randomizacao de pesos; curvas de delecao/insercao; fracao de
saliencia na moldura e em pixels saturados na janela de cerebro (osso) para atalhos.
