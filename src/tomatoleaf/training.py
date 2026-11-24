"""PyTorch training loop with mixed precision, class-weighted loss, LR scheduling and early stopping."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from .config import NUM_CLASSES


class LeafDataset(Dataset):
    """Holds decoded uint8 images in memory; applies the given transform per sample.

    With device='cuda' the uint8 images (~300 MB for the training split) live on the GPU and the same torchvision
    transforms run there, so augmentation does not bottleneck on the CPU.
    """

    def __init__(self, images: np.ndarray, labels, transform, device=None):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        if device is not None:
            self.images, self.labels = self.images.to(device), self.labels.to(device)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.transform(tv_tensors.Image(self.images[i]))
        return x.as_subclass(torch.Tensor), self.labels[i]


def make_loader(images, labels, transform, batch_size: int, shuffle: bool, seed: int = 42, device=None) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    on_gpu = device is not None and torch.device(device).type == "cuda"
    return DataLoader(LeafDataset(images, labels, transform, device=device if on_gpu else None), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available() and not on_gpu,
                      generator=generator)


class WeightedSmoothedCE(torch.nn.Module):
    """Class-weighted cross-entropy with label smoothing, weighting each sample by its TRUE class only.

    torch's CrossEntropyLoss(weight=w, label_smoothing=e) also multiplies the smoothing term of every class by w_c,
    which with a 31:0.26 weight ratio drags all predictions towards the rarest class. Here:
    loss_n = w[y_n] * [(1 - e) * (-log p_{y_n}) + e/K * sum_c (-log p_c)],  averaged with weights sum(w[y_n]).
    """

    def __init__(self, weight: torch.Tensor, smoothing: float = 0.1):
        super().__init__()
        self.register_buffer("weight", weight)
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_sample = torch.nn.functional.cross_entropy(logits.float(), target, reduction="none",
                                                       label_smoothing=self.smoothing)
        w = self.weight[target]
        return (w * per_sample).sum() / w.sum()


def balanced_class_weights(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """w_c = N / (K * n_c) — the same formula as scikit-learn's class_weight='balanced'."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Returns (logits N x K, labels N, mean loss)."""
    model.eval()
    logits_all, y_all, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, y).item() * len(y)
        logits_all.append(logits.cpu())
        y_all.append(y.cpu())
        n += len(y)
    return torch.cat(logits_all).numpy(), torch.cat(y_all).numpy(), (total_loss / n if criterion else float("nan"))


def train(model, train_loader, val_loader, *, device, criterion, epochs: int, lr: float, weight_decay: float,
          patience: int, stage: str, freeze_backbone: bool = False, history=None, epoch_offset: int = 0, log=print,
          schedule: str = "plateau", warmup_epochs: int = 2):
    """Train until early stopping on validation macro-F1; restores and returns the best weights' summary.

    schedule="plateau": halve the LR when validation macro-F1 stalls; "cosine": linear warm-up then cosine decay.
    """
    history = history if history is not None else []
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if schedule == "cosine":
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * 0.01)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    bad_epochs, start = 0, time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        if freeze_backbone:
            model.features.eval()  # frozen backbone: keep its BatchNorm statistics fixed too
        run_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_logits, val_y, val_loss = evaluate(model, val_loader, device, criterion)
        val_pred = val_logits.argmax(1)
        val_f1 = f1_score(val_y, val_pred, labels=list(range(val_logits.shape[1])), average="macro", zero_division=0)
        current_lr = optimizer.param_groups[0]["lr"]
        if schedule == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_f1)
        row = {"epoch": epoch_offset + epoch, "stage": stage, "train_loss": run_loss / seen, "train_acc": correct / seen,
               "val_loss": val_loss, "val_acc": float((val_pred == val_y).mean()), "val_f1_macro": float(val_f1),
               "lr": current_lr, "seconds": time.perf_counter() - t0}
        history.append(row)
        log(f"[{stage}] epoch {row['epoch']:3d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
            f"val loss {val_loss:.4f} acc {row['val_acc']:.3f} macro-F1 {val_f1:.3f} | lr {current_lr:.1e} | {row['seconds']:.1f}s")

        improved = val_f1 > best["f1"] + 1e-4 or (abs(val_f1 - best["f1"]) <= 1e-4 and val_loss < best["loss"])
        if improved:
            best = {"f1": float(val_f1), "loss": float(val_loss), "epoch": epoch_offset + epoch,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"[{stage}] early stopping after epoch {epoch_offset + epoch} (best epoch {best['epoch']})")
                break

    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_val_f1_macro": best["f1"], "best_val_loss": best["loss"],
            "epochs_run": epoch, "train_seconds": time.perf_counter() - start, "history": history}


# =====================================================================================================================
# Upgrade round: fixed-epoch fine-tuning (no early stopping) with layer-wise LR decay, EMA and bf16 autocast.
# =====================================================================================================================
def class_weights(labels, num_classes: int, mode: str = "balanced") -> torch.Tensor:
    if mode == "none":
        return torch.ones(num_classes, dtype=torch.float32)
    w = balanced_class_weights(labels, num_classes)
    return w.sqrt() if mode == "sqrt" else w


def _gpu_loader(images, labels, transform, batch_size, seed, device, drop_last=True, sampling: str = "uniform"):
    """Shuffled loader; sampling="class_balanced" draws every class equally often (with replacement, N per epoch)."""
    generator = torch.Generator().manual_seed(seed)
    on_gpu = torch.device(device).type == "cuda"
    dataset = LeafDataset(images, labels, transform, device=device if on_gpu else None)
    if sampling == "class_balanced":
        labels = np.asarray(labels)
        per_sample = 1.0 / np.bincount(labels)[labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(per_sample, dtype=torch.double),
                                                         num_samples=len(labels), replacement=True,
                                                         generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=drop_last)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator,
                      drop_last=drop_last)


def uniform_target_loss(logits: torch.Tensor) -> torch.Tensor:
    """Outlier exposure (Hendrycks et al. 2019): cross-entropy between the uniform distribution and the softmax,
    -1/K * sum_c log p_c, averaged over the outlier photos. Minimal when the model is maximally unsure."""
    return -torch.log_softmax(logits.float(), dim=1).mean(dim=1).mean()


def _cycle(loader):
    while True:
        yield from loader


def _optimizer(model, recipe, lr: float):
    """AdamW; timm models with a layer decay get one parameter group per depth (lr_scale = decay^(depth gap))."""
    if recipe.layer_decay and recipe.arch.startswith("timm:"):
        from timm.optim import create_optimizer_v2

        opt = create_optimizer_v2(model, opt="adamw", lr=lr, weight_decay=recipe.weight_decay,
                                  layer_decay=recipe.layer_decay)
        for g in opt.param_groups:  # bake the per-depth scale into the lr so a plain LambdaLR schedule keeps it
            g["lr"] = lr * g.get("lr_scale", 1.0)
        return opt
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                             weight_decay=recipe.weight_decay)


def final_out_features(model) -> int:
    from .models import final_linear

    return final_linear(model).out_features


def fit_recipe(model, recipe, images, labels, *, device, seed: int = 42, monitor=None, log=print, outliers=None,
               monitor_every: int = 5):
    """Train `model` with `recipe` on uint8 images (N x 256 x 256 x 3). Returns (model with final weights, info).

    Stage 1 trains only the new classification head (backbone frozen and in eval mode); stage 2 fine-tunes every
    layer for exactly `recipe.epochs` epochs (warm-up then cosine decay, per step). With `ema_decay`, the returned
    weights are the exponential moving average of stage 2. `monitor(model) -> dict` (optional) is only logged,
    never used to pick an epoch (every `monitor_every` epochs and at the end, to save time).
    `outliers` (uint8 images): with recipe.outlier_weight > 0, every step adds recipe.outlier_batch of them, trained
    towards a uniform output (outlier exposure).
    """
    from .models import normalisation, set_backbone_trainable
    from .preprocessing import train_transform, train_transform_robust

    labels = np.asarray(labels)
    mean, std = normalisation(recipe.arch)
    if recipe.aug == "base":
        tf = train_transform(recipe.input_size, mean, std)
    else:  # "robust" (mild) or "heavy"
        tf = train_transform_robust(recipe.input_size, mean, std, strength="heavy" if recipe.aug == "heavy" else "mild")
    loader = _gpu_loader(images, labels, tf, recipe.batch, seed, device, sampling=recipe.sampling)
    criterion = WeightedSmoothedCE(class_weights(labels, final_out_features(model), recipe.class_weighting).to(device),
                                   smoothing=recipe.label_smoothing)
    oe_stream = None
    if recipe.outlier_weight > 0:
        if outliers is None or not len(outliers):
            raise ValueError(f"recipe {recipe.name} needs outlier photos (outlier exposure)")
        oe_stream = _cycle(_gpu_loader(outliers, np.zeros(len(outliers), dtype=int), tf, recipe.outlier_batch,
                                       seed + 1, device))
    autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    clip = None if recipe.aug == "base" else 1.0
    history, start = [], time.perf_counter()
    last_epoch = recipe.head_epochs + recipe.epochs

    def run_epoch(opt, sched, stage, epoch, ema=None, step0=0, head_only=False):
        model.eval() if head_only else model.train()
        tot, correct, seen, step, oe_tot = 0.0, 0, 0, step0, 0.0
        t0 = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            n_in = len(x)
            if oe_stream is not None:
                x = torch.cat([x, next(oe_stream)[0].to(device, non_blocking=True)])
            with torch.autocast(**autocast):
                logits = model(x)
            oe_loss = None
            if oe_stream is not None:
                oe_loss = uniform_target_loss(logits[n_in:])
                logits = logits[:n_in]
            loss = criterion(logits.float(), y)
            if oe_loss is not None:
                loss = loss + recipe.outlier_weight * oe_loss
                oe_tot += oe_loss.item() * n_in
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if ema is not None:
                ema.update(model, step=step)
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        row = {"epoch": epoch, "stage": stage, "train_loss": tot / seen, "train_acc": correct / seen,
               "lr": opt.param_groups[-1]["lr"], "seconds": time.perf_counter() - t0}
        if oe_stream is not None:
            row["outlier_loss"] = oe_tot / seen
        if monitor is not None and (epoch % monitor_every == 0 or epoch == last_epoch):
            row.update(monitor(ema.module if ema is not None else model))
        history.append(row)
        extra = " | ".join(f"{k} {v:.3f}" for k, v in row.items() if k.startswith("val_"))
        log(f"[{stage}] epoch {epoch:3d} | loss {row['train_loss']:.4f} acc {row['train_acc']:.3f}"
            + (f" | outlier loss {row['outlier_loss']:.4f}" if "outlier_loss" in row else "")
            + (f" | {extra}" if extra else "") + f" | lr {row['lr']:.1e} | {row['seconds']:.1f}s")
        return step

    # ---- stage 1: head only
    set_backbone_trainable(model, False)
    head_opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=recipe.head_lr,
                                 weight_decay=recipe.weight_decay)
    for e in range(1, recipe.head_epochs + 1):
        run_epoch(head_opt, None, "head", e, head_only=True)

    # ---- stage 2: all layers, warm-up + cosine per step
    set_backbone_trainable(model, True)
    opt = _optimizer(model, recipe, recipe.lr)
    total, warm = recipe.epochs * len(loader), recipe.warmup_epochs * len(loader)

    def lr_lambda(step):
        if step < warm:
            return 0.1 + 0.9 * step / max(1, warm)
        progress = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = None
    if recipe.ema_decay:
        from timm.utils import ModelEmaV3

        ema = ModelEmaV3(model, decay=recipe.ema_decay, use_warmup=True)
    step = 0
    for e in range(1, recipe.epochs + 1):
        step = run_epoch(opt, sched, "finetune", recipe.head_epochs + e, ema=ema, step0=step)
    if ema is not None:
        model.load_state_dict(ema.module.state_dict())
    model.eval()
    return model, {"history": history, "train_seconds": time.perf_counter() - start}


@torch.no_grad()
def infer(model, images, device, *, size: int, mean, std, resize: str = "resize", views: int = 1,
          batch_size: int = 32, features: bool = False):
    """float32 logits (N x K) — and penultimate features (N x D) if asked — for uint8 images (N x 256 x 256 x 3).

    With views > 1, logits and features are averaged over the dihedral views (test-time augmentation).
    No autocast: the same float32 arithmetic as the CPU deployment path."""
    from .models import FeatureTap
    from .preprocessing import dihedral_views, eval_transform

    tf = eval_transform(size, mean, std, resize)
    tap = FeatureTap(model) if features else None
    model.eval()
    logits_out, feats_out = [], []
    try:
        for i in range(0, len(images), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size])).permute(0, 3, 1, 2).to(device)
            x = tf(tv_tensors.Image(chunk)).as_subclass(torch.Tensor)
            b = len(x)
            if views > 1:
                x = dihedral_views(x, views)
            lg = model(x).float()
            logits_out.append(lg.view(views, b, -1).mean(0).cpu() if views > 1 else lg.cpu())
            if tap is not None:
                f = tap.features.float()
                feats_out.append(f.view(views, b, -1).mean(0).cpu() if views > 1 else f.cpu())
    finally:
        if tap is not None:
            tap.close()
    logits = torch.cat(logits_out).numpy()
    return (logits, torch.cat(feats_out).numpy()) if features else logits
