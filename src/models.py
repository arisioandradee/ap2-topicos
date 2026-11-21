"""Backbones pre-treinados (nunca aleatorios) via timm + utilitarios de ajuste fino.

Familias:
  cnn_sup   : resnet50 / convnext_tiny / efficientnet_b0 (ImageNet supervisionado)
  vit_sup   : vit_small (ImageNet-21k->1k supervisionado)
  vit_dino  : dinov2_small / dino_vits16 (auto-supervisionado)
  medico    : radimagenet_resnet50 (CNN pre-treinada em TC/RM/US; pesos locais via --weights)
"""
import torch, timm
from timm.models import group_parameters

REGISTRY = {
    "resnet50": ("resnet50.a1_in1k", "cnn_sup"),
    "convnext_tiny": ("convnext_tiny.fb_in1k", "cnn_sup"),
    "efficientnet_b0": ("efficientnet_b0.ra_in1k", "cnn_sup"),
    "vit_small_sup": ("vit_small_patch16_224.augreg_in21k_ft_in1k", "vit_sup"),
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", "vit_dino"),
    "dino_vits16": ("vit_small_patch16_224.dino", "vit_dino"),
    "radimagenet_resnet50": ("resnet50", "medico"),
}


def is_vit(name):
    return REGISTRY[name][1] in ("vit_sup", "vit_dino")


def build(name, res=160, num_outputs=6, pretrained=True, weights_path=None):
    """Retorna (modelo, mean, std). Cabeca de num_outputs saidas (any + 5 subtipos)."""
    tname, fam = REGISTRY[name]
    kw = {}
    if is_vit(name):
        kw = dict(img_size=res)  # timm interpola os embeddings de posicao
    if fam == "medico":
        model = timm.create_model(tname, pretrained=False, num_classes=num_outputs)
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier."))}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[radimagenet] carregado; missing={len(missing)} unexpected={len(unexpected)}")
            assert len(missing) <= 2, "pesos do RadImageNet nao casaram com o resnet50 do timm"
        else:
            raise ValueError("modelo medico exige --weights (nunca pesos aleatorios)")
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return model, mean, std
    model = timm.create_model(tname, pretrained=pretrained, num_classes=num_outputs, **kw)
    cfg = model.pretrained_cfg
    return model, tuple(cfg["mean"]), tuple(cfg["std"])


def n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def layer_groups(model):
    """Grupos de parametros ordenados da base (0) ao topo/cabeca (ultimo)."""
    g = group_parameters(model, model.group_matcher(coarse=False))
    named = dict(model.named_parameters())
    groups = [[named[n] for n in g[k]] for k in sorted(g)]
    assert sum(len(x) for x in groups) == len(named), "grupos nao cobrem todos os parametros"
    return groups


def set_gradual_stage(model, groups, stage, n_stages):
    """Descongelamento gradual: stage 0 = so cabeca; stage n_stages-1 = tudo.
    Libera blocos do topo para a base ao longo dos estagios."""
    n = len(groups)
    n_free = 1 + round((n - 1) * stage / max(1, n_stages - 1))  # >=1 (cabeca)
    for i, params in enumerate(groups):
        trainable = i >= n - n_free
        for p in params:
            p.requires_grad = trainable
    return n_free


def freeze_backbone(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def extract_features(model, x):
    """Embedding antes da cabeca (para o regime de extrator congelado)."""
    f = model.forward_features(x)
    return model.forward_head(f, pre_logits=True)
