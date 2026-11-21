"""Dataset (JPEG 3 canais) + politicas de aumento de dados (somente no treino).

Politicas (justificativa clinica no artigo):
  none : nenhum aumento
  main : flip horizontal (hemorragia/subtipos nao sao lateralizados nos rotulos; o encefalo e ~simetrico),
         rotacao +-10 graus e translacao/escala leves (variacao de posicionamento do paciente),
         ruido gaussiano leve (variacao de dose/ruido do tomografo).
         NAO usa: flip vertical (inverte anatomia), deformacao elastica forte e variacao grande de
         brilho/contraste (a janela de HU ja fixa a intensidade; alterar apaga hemorragias sutis).
  alt  : TrivialAugmentWide (politica automatica, agnostica ao dominio) + flip horizontal.
         Operacoes de cor alteram o significado das janelas de HU -> serve de contraste.
"""
import os
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from .common import SUBTYPES


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.02, p=0.3):
        super().__init__(); self.std, self.p = std, p

    def forward(self, x):  # x float [0,1]
        if torch.rand(1).item() < self.p:
            x = (x + torch.randn_like(x) * self.std).clamp(0, 1)
        return x


def build_transform(policy, train, res, mean, std):
    norm = [T.ToDtype(torch.float32, scale=True)]
    tail = [T.Normalize(mean, std)]
    base = [T.ToImage(), T.Resize((res, res), antialias=True)]
    if not train or policy == "none":
        return T.Compose(base + norm + tail)
    if policy == "main":
        aug = [T.RandomHorizontalFlip(0.5),
               T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))]
        return T.Compose(base + aug + norm + [AddGaussianNoise()] + tail)
    if policy == "alt":
        aug = [T.RandomHorizontalFlip(0.5), T.TrivialAugmentWide()]
        return T.Compose(base + aug + norm + tail)
    raise ValueError(policy)


class CTDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_labels=True):
        self.ids = df["image_id"].tolist()
        self.y = df[SUBTYPES].values.astype(np.float32) if with_labels else None
        self.dir, self.tf = img_dir, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.dir, self.ids[i] + ".jpg")).convert("RGB")
        x = self.tf(img)
        y = torch.from_numpy(self.y[i]) if self.y is not None else torch.zeros(len(SUBTYPES))
        return x, y, i
