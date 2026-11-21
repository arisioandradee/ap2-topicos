"""Treino/avaliacao de uma configuracao (um modelo, um regime, uma semente).

Regimes:
  --regime linear    : extrator CONGELADO + classificador linear (embeddings em cache)
  --regime finetune  : ajuste fino completo; --lr_mode const | cosine | llrd | gradual

Orcamento identico entre configuracoes: --epochs e --epoch_fraction (fracao do treino sorteada por epoca;
todos os dados continuam sendo usados ao longo das epocas). Selecao de epoca SOMENTE na validacao.
O conjunto de teste so e inferido ao final, uma unica vez.

Exemplo:
  python -m src.train --model resnet50 --regime finetune --lr_mode cosine --aug main --seed 0 \
     --img_dir /kaggle/input/<dataset>/img224 --splits docs/splits.csv --out results/runs
"""
import argparse, json, math, os, platform, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import timm.optim

from .common import SUBTYPES, KAGGLE_WEIGHTS, seed_everything
from .data import CTDataset, build_transform
from . import models as M

FAMILY_LR = {"cnn_sup": 2e-4, "vit_sup": 5e-5, "vit_dino": 5e-5, "medico": 2e-4}


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(M.REGISTRY))
    ap.add_argument("--regime", choices=["linear", "finetune"], required=True)
    ap.add_argument("--lr_mode", choices=["const", "cosine", "llrd", "gradual"], default="cosine")
    ap.add_argument("--aug", choices=["none", "main", "alt"], default="main")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epoch_fraction", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--layer_decay", type=float, default=0.75)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--tta", action="store_true", help="TTA (flip) -> declarar como fator separado")
    ap.add_argument("--val_eval_n", type=int, default=20000, help="imagens da val usadas para escolher a epoca")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weights", default=None, help="pesos locais (RadImageNet)")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="teste de codigo em poucas imagens; NUNCA reportar")
    return ap.parse_args(argv)


def run_name(a):
    base = f"{a.model}_{a.regime}"
    if a.regime == "finetune":
        base += f"_{a.lr_mode}"
    base += f"_aug-{a.aug}_r{a.res}"
    if a.tag: base += f"_{a.tag}"
    return f"{base}_s{a.seed}" + ("_SMOKE" if a.smoke else "")


def weighted_bce(logits, y, label_smooth=0.0):
    w = torch.tensor(KAGGLE_WEIGHTS, device=logits.device, dtype=logits.dtype)
    if label_smooth > 0:
        y = y * (1 - label_smooth) + 0.5 * label_smooth
    bce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (bce * w).sum(1).div(w.sum()).mean()


@torch.no_grad()
def predict(model, loader, device, tta=False, feats=False):
    model.eval(); out = []
    amp = device.type == "cuda"
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            if feats:
                o = M.extract_features(model, x)
            else:
                o = model(x)
                if tta:
                    o = (o + model(torch.flip(x, dims=[3]))) / 2
        out.append(o.float().cpu())
    return torch.cat(out).numpy()


def make_loader(df, img_dir, tf, bs, workers, shuffle=False, sampler=None):
    ds = CTDataset(df, img_dir, tf)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle if sampler is None else False, sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=workers > 0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def eval_auc(df_part, logits):
    return roc_auc_score(df_part["any"].values, logits[:, 0])


def build_scheduler(opt, total_steps, warmup_frac):
    warm = max(1, int(total_steps * warmup_frac))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def finetune(a, model, df, mean, std, device, info):
    tr, va = df[df.split == "train"], df[df.split == "val"]
    rng = np.random.RandomState(0)  # subconjunto FIXO da val (independe da semente)
    va_sel = va.iloc[rng.permutation(len(va))[:a.val_eval_n]]
    tf_tr = build_transform(a.aug, True, a.res, mean, std)
    tf_ev = build_transform("none", False, a.res, mean, std)
    n_per_epoch = max(a.bs, int(len(tr) * a.epoch_fraction))
    g = torch.Generator(); g.manual_seed(a.seed)
    ds_tr = CTDataset(tr, a.img_dir, tf_tr)
    sampler = RandomSampler(ds_tr, replacement=False, num_samples=n_per_epoch, generator=g)
    ld_tr = DataLoader(ds_tr, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
                       pin_memory=device.type == "cuda", drop_last=True, persistent_workers=a.workers > 0)
    ld_va = make_loader(va_sel, a.img_dir, tf_ev, a.bs * 2, a.workers)

    lr = a.lr or FAMILY_LR[M.REGISTRY[a.model][1]]
    ld = a.layer_decay if a.lr_mode == "llrd" else None
    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=lr, weight_decay=a.wd, layer_decay=ld)
    steps_per_epoch = len(ld_tr)
    total = steps_per_epoch * a.epochs
    sched = build_scheduler(opt, total, a.warmup_frac) if a.lr_mode != "const" else None
    groups, n_stages = None, 4
    if a.lr_mode == "gradual":
        groups = M.layer_groups(model)
        M.set_gradual_stage(model, groups, 0, n_stages)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    amp = device.type == "cuda"
    best, best_state, hist, step = -1, None, [], 0
    info.update(lr=lr, steps_per_epoch=steps_per_epoch, total_steps=total, imgs_per_epoch=n_per_epoch)
    for ep in range(a.epochs):
        model.train(); run, nb = 0.0, 0
        for x, y, _ in tqdm(ld_tr, desc=f"ep{ep}", leave=False):
            if groups is not None:  # estagios nos primeiros 50% dos passos
                st = min(n_stages - 1, int(step / max(1, total * 0.5) * n_stages))
                M.set_gradual_stage(model, groups, st, n_stages)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = weighted_bce(model(x).float(), y, a.label_smooth)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            if sched: sched.step()
            run += loss.item(); nb += 1; step += 1
        auc = eval_auc(va_sel, predict(model, ld_va, device))
        hist.append({"epoch": ep, "train_loss": run / max(1, nb), "val_auc_any": auc})
        print(hist[-1])
        if auc > best:
            best = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            info["best_epoch"] = ep
    model.load_state_dict(best_state)
    info.update(history=hist, best_val_auc_any_subset=best)
    return model, best_state


def run_linear(a, model, df, mean, std, device, info):
    """Extrator congelado: features em cache (uma passada) + cabeca linear treinada sobre elas."""
    os.makedirs(a.cache_dir, exist_ok=True)
    cpath = os.path.join(a.cache_dir, f"feats_{a.model}_r{a.res}{'_SMOKE' if a.smoke else ''}.npy")
    model.to(device).eval()
    if os.path.exists(cpath):
        F = np.load(cpath)
    else:
        tf_ev = build_transform("none", False, a.res, mean, std)
        ld = make_loader(df, a.img_dir, tf_ev, a.bs * 2, a.workers)
        F = predict(model, ld, device, feats=True).astype(np.float16)
        np.save(cpath, F)
    F = F.astype(np.float32)
    tr = (df.split == "train").values; va = (df.split == "val").values
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = torch.from_numpy((F - mu) / sd)
    Y = torch.from_numpy(df[SUBTYPES].values.astype(np.float32))
    head = nn.Linear(F.shape[1], 6).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    idx_tr = np.where(tr)[0]; rng = np.random.RandomState(a.seed)
    best, best_state, hist = -1, None, []
    n_ep = 20
    for ep in range(n_ep):
        rng.shuffle(idx_tr)
        for i in range(0, len(idx_tr), 4096):
            b = idx_tr[i:i + 4096]
            loss = weighted_bce(head(Z[b].to(device)), Y[b].to(device), a.label_smooth)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            lv = head(Z[va].to(device)).cpu().numpy()
        auc = roc_auc_score(df.loc[va, "any"].values, lv[:, 0])
        hist.append({"epoch": ep, "val_auc_any": auc})
        if auc > best:
            best = auc; best_state = {k: v.clone() for k, v in head.state_dict().items()}; info["best_epoch"] = ep
    head.load_state_dict(best_state)
    with torch.no_grad():
        logits = head(Z.to(device)).cpu().numpy()
    info.update(history=hist, best_val_auc_any_subset=best, lr=1e-3, head_params=sum(p.numel() for p in head.parameters()))
    head.mu, head.sd = mu, sd  # padronizacao do treino (necessaria para explicabilidade do modelo congelado)
    return logits, head


def main(argv=None):
    a = get_args(argv)
    name = run_name(a)
    odir = os.path.join(a.out, name)
    if os.path.exists(os.path.join(odir, "preds.npz")) and not a.force:
        print("ja existe:", odir); return
    os.makedirs(odir, exist_ok=True)
    seed_everything(a.seed)
    df = pd.read_csv(a.splits)
    if a.smoke:
        df = pd.concat([df[df.split == s].head(64) for s in ("train", "val", "test")]).reset_index(drop=True)
        a.epochs, a.val_eval_n, a.workers = 1, 64, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = M.build(a.model, a.res, 6, True, a.weights)
    fam = M.REGISTRY[a.model][1]
    info = {"args": vars(a), "family": fam, "device": str(device), "host": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__, "params_total": M.n_params(model), "smoke": a.smoke}
    t0 = time.time()
    model.to(device)
    if a.regime == "linear":
        logits, head = run_linear(a, model, df, mean, std, device, info)
        info["params_trainable"] = info["head_params"]
        te = (df.split == "test").values; va = (df.split == "val").values
        out_logits = {"val": logits[va], "test": logits[te]}
        ids = {"val": df.loc[va, "image_id"].values, "test": df.loc[te, "image_id"].values}
        torch.save({"head": head.state_dict(), "mu": head.mu, "sd": head.sd}, os.path.join(odir, "head.pt"))
    else:
        model, state = finetune(a, model, df, mean, std, device, info)
        info["params_trainable"] = M.n_params(model, trainable_only=False)  # fine-tuning completo
        tf_ev = build_transform("none", False, a.res, mean, std)
        out_logits, ids = {}, {}
        for s in ("val", "test"):
            part = df[df.split == s]
            out_logits[s] = predict(model, make_loader(part, a.img_dir, tf_ev, a.bs * 2, a.workers), device, a.tta)
            ids[s] = part["image_id"].values
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in state.items()}, os.path.join(odir, "weights.pt"))
    info["train_time_s"] = time.time() - t0
    np.savez_compressed(os.path.join(odir, "preds.npz"),
                        val_ids=ids["val"], val_prob=sigmoid(out_logits["val"]).astype(np.float32),
                        test_ids=ids["test"], test_prob=sigmoid(out_logits["test"]).astype(np.float32))
    info["val_auc_any_full"] = float(roc_auc_score(df.loc[df.split == "val", "any"].values, out_logits["val"][:, 0])) \
        if len(set(df.loc[df.split == "val", "any"])) > 1 else None
    with open(os.path.join(odir, "run.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("OK", name, "val AUC any (completa):", info["val_auc_any_full"])


if __name__ == "__main__":
    main()
