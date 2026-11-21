"""Constantes e utilitarios compartilhados."""
import os, random
import numpy as np

SUBTYPES = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
# pesos da metrica oficial do Kaggle (RSNA 2019): any pesa 2x
KAGGLE_WEIGHTS = [2, 1, 1, 1, 1, 1]
# janelas (centro, largura) -> canais R,G,B: cerebro, subdural, osso
WINDOWS = [(40, 80), (80, 200), (600, 2800)]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
