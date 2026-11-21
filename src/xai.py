"""Explicabilidade: Grad-CAM (gradiente/ativacao) + Oclusao (perturbacao) [+ Integrated Gradients],
teste de sanidade por randomizacao de pesos, curvas de delecao/insercao e metricas de atalho.

Criterio de selecao dos casos (definido ANTES de ver mapas; ver docs/xai_selection_criterion.md):
  para cada modelo analisado, k verdadeiros positivos, k falsos positivos, k falsos negativos e
  k verdadeiros negativos de MAIOR CONFIANCA, e os k casos MAIS PROXIMOS do limiar (limiar = Youden na validacao).
  Sempre calculado so a partir das predicoes (nunca dos mapas).
"""
import copy
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr

# ---------------------------------------------------------------- selecao de casos
def select_cases(y, p, k=4, thr=0.5):
    """Indices (posicoes) dos casos por criterio fixo. y,p: arrays da coluna 'any' no teste."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    pred = (p >= thr).astype(int)
    conf = np.abs(p - thr)
    out = {}
    for name, mask in {"TP": (y == 1) & (pred == 1), "FP": (y == 0) & (pred == 1),
                       "FN": (y == 1) & (pred == 0), "TN": (y == 0) & (pred == 0)}.items():
        idx = np.where(mask)[0]
        out[name] = idx[np.argsort(-conf[idx], kind="stable")][:k].tolist()  # maior confianca
    out["limiar"] = np.argsort(conf, kind="stable")[:k].tolist()              # mais proximos do limiar
    return out


# ---------------------------------------------------------------- alvos / reshape
def vit_reshape_factory(model):
    npref = getattr(model, "num_prefix_tokens", 1)
    def f(t):  # (B, tokens, C) -> (B, C, H, W)
        t = t[:, npref:, :]
        h = w = int(round(t.shape[1] ** 0.5))
        return t.reshape(t.shape[0], h, w, t.shape[2]).permute(0, 3, 1, 2)
    return f


def target_layer(model, name):
    """Ultima camada espacial de cada familia (nome do REGISTRY)."""
    if name in ("resnet50", "radimagenet_resnet50"):
        return model.layer4[-1], None
    if name == "convnext_tiny":
        return model.stages[-1].blocks[-1], None
    if name == "efficientnet_b0":
        return model.blocks[-1][-1], None
    if "vit" in name or "dino" in name:
        return model.blocks[-1].norm1, vit_reshape_factory(model)
    raise ValueError(name)


# ---------------------------------------------------------------- mapas
def saliency_gradcam(model, x, layer, reshape=None, idx=0, variant="gradcam"):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cls = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigencam": EigenCAM}[variant]
    for p in model.parameters():
        p.requires_grad_(True)
    with cls(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)] * x.shape[0])  # (B,H,W) em [0,1]


def saliency_occlusion(model, x, idx=0, window=24, stride=12):
    """Oclusao deslizante: queda do logit 'any' ao cobrir a regiao com a media (0 apos normalizacao)."""
    from captum.attr import Occlusion
    occ = Occlusion(lambda z: model(z)[:, idx:idx + 1])
    a = occ.attribute(x, sliding_window_shapes=(3, window, window), strides=(3, stride, stride), baselines=0.0)
    s = a.sum(1).detach().cpu().numpy()  # (B,H,W); positivo = regiao sustenta a predicao
    return _norm01(np.clip(s, 0, None))


def saliency_ig(model, x, idx=0, steps=32):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(lambda z: model(z)[:, idx:idx + 1])
    a = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=steps)
    return _norm01(np.clip(a.abs().sum(1).detach().cpu().numpy(), 0, None))


def _norm01(s):
    s = np.asarray(s, np.float32)
    mn = s.min(axis=(-2, -1), keepdims=True); mx = s.max(axis=(-2, -1), keepdims=True)
    return (s - mn) / (mx - mn + 1e-8)


# ---------------------------------------------------------------- avaliacao quantitativa
@torch.no_grad()
def deletion_insertion(model, x, sal, idx=0, steps=20):
    """x: (1,3,H,W) normalizado; sal: (H,W). Retorna (auc_delecao, auc_insercao, curvas).
    Delecao: remove (substitui por 0=media) os pixels mais salientes primeiro; esperado QUEDA rapida (AUC baixa).
    Insercao: parte de imagem zerada e revela os mais salientes; esperado SUBIDA rapida (AUC alta)."""
    model.eval()
    H, W = sal.shape
    order = np.argsort(-sal.reshape(-1), kind="stable")
    fr = np.linspace(0, 1, steps + 1)
    dels, inss = [], []
    flat = x.reshape(1, 3, -1)
    for f in fr:
        n = int(f * H * W)
        m = torch.zeros(H * W, device=x.device); m[order[:n]] = 1
        m = m.reshape(1, 1, H, W)
        dels.append(torch.sigmoid(model(x * (1 - m)))[0, idx].item())
        inss.append(torch.sigmoid(model(x * m))[0, idx].item())
    auc = lambda v: float(np.sum((np.asarray(v)[1:] + np.asarray(v)[:-1]) / 2 * np.diff(fr)))  # trapezio
    return auc(dels), auc(inss), (fr, np.array(dels), np.array(inss))


def border_fraction(sal, margin=0.12):
    """Fracao da massa de saliencia na moldura externa (crania/bordas/rotulos de texto)."""
    H, W = sal.shape; mh, mw = int(H * margin), int(W * margin)
    m = np.ones_like(sal, bool); m[mh:H - mh, mw:W - mw] = False
    return float(sal[m].sum() / (sal.sum() + 1e-8))


def bone_fraction(sal, brain_channel_uint8):
    """Fracao da saliencia sobre pixels saturados na janela de cerebro (HU>=80: osso/calcificacao/contraste)."""
    m = brain_channel_uint8 >= 250
    return float(sal[m].sum() / (sal.sum() + 1e-8))


def energy_in_mask(sal, mask):
    """Fracao de energia dentro de uma regiao anotada (se houver mascara/caixa)."""
    return float(sal[mask > 0].sum() / (sal.sum() + 1e-8))


def pointing_game(sal, mask):
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    return bool(mask[y, x] > 0)


# ---------------------------------------------------------------- teste de sanidade (Adebayo et al.)
def randomize_weights(model, mode="full", seed=0):
    """full: reinicializa TUDO; top: reinicializa somente o ultimo estagio + cabeca (cascata parcial)."""
    g = torch.Generator().manual_seed(seed)
    m = copy.deepcopy(model)
    mods = [mm for mm in m.modules() if hasattr(mm, "reset_parameters")]
    if mode == "top":
        mods = mods[-max(1, len(mods) // 6):]
    for mm in mods:
        torch.manual_seed(seed); mm.reset_parameters()
    return m


def sanity_correlation(make_map, model, x, mode="full", seed=0):
    """Correlacao de Spearman entre mapa do modelo treinado e do randomizado.
    Mapa confiavel => correlacao BAIXA (o mapa depende dos pesos aprendidos)."""
    s0 = make_map(model, x).reshape(-1)
    s1 = make_map(randomize_weights(model, mode, seed), x).reshape(-1)
    r = spearmanr(s0, s1).correlation
    return float(0.0 if np.isnan(r) else r)
