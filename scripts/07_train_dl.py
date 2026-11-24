"""Phase 4 — deep learning on the GPU.

custom_cnn : CNN trained from scratch (AdamW lr 1e-3, up to 60 epochs, early stopping patience 12).
mobilenetv2: ImageNet-pretrained. Stage 1 "frozen" = feature extraction (only the new head trains, lr 1e-3);
             stage 2 "finetune" = all layers train with a 10x smaller lr (1e-4).
Both: class-weighted cross-entropy, ReduceLROnPlateau on validation macro-F1, best-checkpoint restore,
mixed precision (float16 autocast), batch size 32, 224x224 input, train-only augmentation.

--protocol clean    : our group-aware split (original photos only)  [default]
--protocol official : the dataset's official folders (leaky) — used only for the leakage benchmark
Outputs: experiments/models/dl/<protocol>/<model>.pt, experiments/results/dl_<protocol>_<arch>.json
"""
import argparse

import torch
from torch import nn

from tomatoleaf.config import CLASS_KEYS, IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE, MODELS_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index, load_splits
from tomatoleaf.models import build_model, count_parameters, set_backbone_trainable
from tomatoleaf.preprocessing import eval_transform, train_transform
from tomatoleaf.training import balanced_class_weights, make_loader, train
from tomatoleaf.utils import save_json, set_seed

BATCH_SIZE = 32
CONFIG = {
    "custom_cnn": {"epochs": 60, "lr": 1e-3, "weight_decay": 1e-4, "patience": 12},
    "frozen": {"epochs": 15, "lr": 1e-3, "weight_decay": 1e-4, "patience": 5},
    "finetune": {"epochs": 30, "lr": 1e-4, "weight_decay": 1e-4, "patience": 7},
}


def load_protocol(protocol: str):
    if protocol == "clean":
        splits = load_splits()
        return splits[splits.split == "train"], splits[splits.split == "val"]
    index = load_index()
    return index[index.official_split == "train"], index[index.official_split == "val"]


def save_checkpoint(model, arch: str, key: str, protocol: str, extra: dict) -> None:
    path = MODELS_DIR / "dl" / protocol / f"{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": arch, "model_key": key, "state_dict": model.state_dict(), "class_keys": CLASS_KEYS,
                "input_size": MODEL_INPUT_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "protocol": protocol, **extra}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["custom_cnn", "mobilenetv2"], required=True)
    parser.add_argument("--protocol", choices=["clean", "official"], default="clean")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    tr, va = load_protocol(args.protocol)
    X_tr, X_va = load_images(tr.path), load_images(va.path)
    print(f"Protocol {args.protocol}: {len(tr)} training images, {len(va)} validation images")

    train_loader = make_loader(X_tr, tr.label.to_numpy(), train_transform(), BATCH_SIZE, shuffle=True, seed=SEED,
                               device=device)
    val_loader = make_loader(X_va, va.label.to_numpy(), eval_transform(), 64, shuffle=False, device=device)
    weights = balanced_class_weights(tr.label.to_numpy())
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    model = build_model(args.arch).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, summary = [], {"protocol": args.protocol, "arch": args.arch, "batch_size": BATCH_SIZE,
                            "train_images": len(tr), "val_images": len(va), "class_weights": weights.tolist(),
                            "total_params": count_parameters(model), "config": CONFIG}
    if args.arch == "custom_cnn":
        res = train(model, train_loader, val_loader, device=device, criterion=criterion, stage="scratch",
                    history=history, **CONFIG["custom_cnn"])
        save_checkpoint(model, "custom_cnn", "custom_cnn", args.protocol, {"best_epoch": res["best_epoch"]})
        summary["models"] = {"custom_cnn": {"train_time_s": res["train_seconds"], "best_epoch": res["best_epoch"],
                                            "epochs_run": res["epochs_run"], "best_val_f1_macro": res["best_val_f1_macro"],
                                            "trainable_params": count_parameters(model, trainable_only=True)}}
    else:
        set_backbone_trainable(model, False)
        head_params = count_parameters(model, trainable_only=True)
        res1 = train(model, train_loader, val_loader, device=device, criterion=criterion, stage="frozen",
                     freeze_backbone=True, history=history, **CONFIG["frozen"])
        save_checkpoint(model, "mobilenetv2", "mobilenetv2_fe", args.protocol, {"best_epoch": res1["best_epoch"]})
        set_backbone_trainable(model, True)
        res2 = train(model, train_loader, val_loader, device=device, criterion=criterion, stage="finetune",
                     history=history, epoch_offset=len(history), **CONFIG["finetune"])
        save_checkpoint(model, "mobilenetv2", "mobilenetv2_ft", args.protocol, {"best_epoch": res2["best_epoch"]})
        summary["models"] = {
            "mobilenetv2_fe": {"train_time_s": res1["train_seconds"], "best_epoch": res1["best_epoch"],
                               "epochs_run": res1["epochs_run"], "best_val_f1_macro": res1["best_val_f1_macro"],
                               "trainable_params": head_params},
            "mobilenetv2_ft": {"train_time_s": res1["train_seconds"] + res2["train_seconds"],
                               "best_epoch": res2["best_epoch"], "epochs_run": res1["epochs_run"] + res2["epochs_run"],
                               "best_val_f1_macro": res2["best_val_f1_macro"],
                               "trainable_params": count_parameters(model, trainable_only=True)},
        }
    if device.type == "cuda":
        summary["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
    summary["history"] = history
    save_json(summary, RESULTS_DIR / f"dl_{args.protocol}_{args.arch}.json")
    print({k: v for k, v in summary.items() if k not in ("history", "config")})


if __name__ == "__main__":
    main()
