"""Phase 4b — improvement round with stronger pretrained backbones (selection still on VALIDATION only).

Changes versus 07_train_dl.py, each a standard fine-tuning practice:
- stronger ImageNet backbones: EfficientNet-B0 (5.3M params) and ConvNeXt-Tiny (28M params)
- native 256x256 input (no down-sampling of the 256x256 photos; thin leaf-miner trails survive)
- stage 2 uses linear warm-up + cosine learning-rate decay and label smoothing 0.1 (softer targets, less over-confidence)
Stage 1 (frozen backbone, 5 epochs) warms up the new head first, as before.

--protocol official retrains the same recipe on the official (leaky) folders for the leakage benchmark only.
Outputs: experiments/models/dl/<protocol>/<arch>_v2.pt, experiments/results/dl_<protocol>_<arch>_v2.json
"""
import argparse

import torch
from torch import nn

from tomatoleaf.config import CLASS_KEYS, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index, load_splits
from tomatoleaf.models import build_model, count_parameters, set_backbone_trainable
from tomatoleaf.preprocessing import eval_transform, train_transform
from tomatoleaf.training import WeightedSmoothedCE, balanced_class_weights, make_loader, train
from tomatoleaf.utils import save_json, set_seed

RECIPES = {
    "efficientnet_b0": {"batch": 32, "lr": 3e-4, "weight_decay": 1e-4},
    "convnext_tiny": {"batch": 16, "lr": 1e-4, "weight_decay": 0.05},
    "mobilenetv2": {"batch": 32, "lr": 3e-4, "weight_decay": 1e-4},
}
SIZE, FROZEN_EPOCHS, FINETUNE_EPOCHS, PATIENCE, LABEL_SMOOTHING = 256, 5, 30, 10, 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(RECIPES), required=True)
    parser.add_argument("--protocol", choices=["clean", "official"], default="clean")
    parser.add_argument("--classes", default="", help="comma-separated class keys to keep (default: all 8)")
    args = parser.parse_args()
    keep = [k for k in CLASS_KEYS if k in args.classes.split(",")] if args.classes else list(CLASS_KEYS)
    recipe = RECIPES[args.arch]

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.protocol == "clean":
        s = load_splits()
        tr, va = s[s.split == "train"], s[s.split == "val"]
    else:
        idx = load_index()
        tr, va = idx[idx.official_split == "train"], idx[idx.official_split == "val"]
    if len(keep) < len(CLASS_KEYS):
        tr, va = tr[tr.class_key.isin(keep)].copy(), va[va.class_key.isin(keep)].copy()
        remap = {CLASS_KEYS.index(k): i for i, k in enumerate(keep)}
        tr["label"], va["label"] = tr.label.map(remap), va.label.map(remap)
    X_tr, X_va = load_images(tr.path), load_images(va.path)
    print(f"{args.arch} | protocol {args.protocol}: {len(tr)} train / {len(va)} val images | input {SIZE}px")

    train_loader = make_loader(X_tr, tr.label.to_numpy(), train_transform(SIZE), recipe["batch"], shuffle=True,
                               seed=SEED, device=device)
    val_loader = make_loader(X_va, va.label.to_numpy(), eval_transform(SIZE), 64, shuffle=False, device=device)
    weights = balanced_class_weights(tr.label.to_numpy(), num_classes=len(keep))
    criterion = WeightedSmoothedCE(weights.to(device), smoothing=LABEL_SMOOTHING)
    model = build_model(args.arch, num_classes=len(keep)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    set_backbone_trainable(model, False)
    res1 = train(model, train_loader, val_loader, device=device, criterion=criterion, epochs=FROZEN_EPOCHS, lr=1e-3,
                 weight_decay=recipe["weight_decay"], patience=FROZEN_EPOCHS, stage="frozen", freeze_backbone=True,
                 history=history)
    set_backbone_trainable(model, True)
    res2 = train(model, train_loader, val_loader, device=device, criterion=criterion, epochs=FINETUNE_EPOCHS,
                 lr=recipe["lr"], weight_decay=recipe["weight_decay"], patience=PATIENCE, stage="finetune",
                 history=history, epoch_offset=len(history), schedule="cosine")

    key = f"{args.arch}_v2" if len(keep) == len(CLASS_KEYS) else f"{args.arch}_{len(keep)}class"
    path = MODELS_DIR / "dl" / args.protocol / f"{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "model_key": key, "state_dict": model.state_dict(), "class_keys": keep,
                "input_size": SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "protocol": args.protocol,
                "best_epoch": res2["best_epoch"]}, path)
    summary = {"protocol": args.protocol, "arch": args.arch, "model_key": key, "input_size": SIZE, "recipe": recipe,
               "label_smoothing": LABEL_SMOOTHING, "train_images": len(tr), "val_images": len(va),
               "total_params": count_parameters(model),
               "models": {key: {"train_time_s": res1["train_seconds"] + res2["train_seconds"],
                                "best_epoch": res2["best_epoch"], "epochs_run": res1["epochs_run"] + res2["epochs_run"],
                                "best_val_f1_macro": res2["best_val_f1_macro"],
                                "trainable_params": count_parameters(model, trainable_only=True)}},
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
               "history": history}
    summary["class_keys"] = keep
    save_json(summary, RESULTS_DIR / f"dl_{args.protocol}_{key}.json")
    print({k: v for k, v in summary.items() if k != "history"})


if __name__ == "__main__":
    main()
