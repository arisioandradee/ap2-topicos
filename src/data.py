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
import glob, io, os
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


class ImageStore:
    """Leitura de imagens: arquivos .bin empacotados (index_shard*.csv, buscados recursivamente em `root`)
    ou, se nao houver indice, arquivos <root>/<image_id>.jpg."""

    def __init__(self, root):
        self.root, self.loc, self._mm = root, {}, {}
        for ip in sorted(glob.glob(os.path.join(root, "**", "index_shard*.csv"), recursive=True)):
            d = os.path.dirname(ip)
            idx = pd.read_csv(ip)
            for iid, b, o, n in zip(idx.image_id, idx.bin, idx.offset, idx.length):
                self.loc[iid] = (os.path.join(d, b), int(o), int(n))

    def _map(self, path):
        import mmap
        if path not in self._mm:
            f = open(path, "rb"); self._mm[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mm[path]

    def __getstate__(self):  # mmap nao e serializavel (DataLoader com workers)
        d = dict(self.__dict__); d["_mm"] = {}; return d

    def get(self, iid):
        if iid in self.loc:
            path, o, n = self.loc[iid]
            return Image.open(io.BytesIO(self._map(path)[o:o + n])).convert("RGB")
        return Image.open(os.path.join(self.root, iid + ".jpg")).convert("RGB")


_STORES = {}


def get_store(root):
    if root not in _STORES:
        _STORES[root] = ImageStore(root)
    return _STORES[root]


class CTDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_labels=True, store=None):
        self.ids = df["image_id"].tolist()
        self.y = df[SUBTYPES].values.astype(np.float32) if with_labels else None
        self.store, self.tf = store or get_store(img_dir), transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.store.get(self.ids[i])
        x = self.tf(img)
        y = torch.from_numpy(self.y[i]) if self.y is not None else torch.zeros(len(SUBTYPES))
        return x, y, i
