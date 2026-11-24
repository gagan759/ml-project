"""Training recipes compared in the upgrade round (grouped 5-fold cross-validation on train+val photos).

Each recipe trains for a FIXED number of epochs (no early stopping), so the out-of-fold predictions are an unbiased
estimate: no epoch is chosen by looking at the held-out fold.

- `convnext_t_in1k_base` re-creates the original study's recipe (torchvision ConvNeXt-Tiny, ImageNet-1k weights,
  'base' augmentation, all layers at one learning rate) as the reference point.
- `convnext_t_in22k_base` changes only the pretrained weights (ImageNet-22k) — the backbone ablation.
- `*_robust` recipes add, to the original recipe: 'robust' augmentation (8 dihedral views so every test-time view is
  in-distribution, aspect jitter, mild JPEG and blur) and an exponential moving average (EMA) of the weights.
- `convnext_t_in22k_llrd` was the first robust version (heavier augmentation, layer-wise LR decay 0.8, stochastic
  depth). It under-fitted on fold 0 (held-out macro-F1 0.582 vs 0.626 for the reference; train accuracy 0.92 vs 0.99)
  and was stopped; it is kept here so the record is complete.
- `dinov2_b_linear` is not fine-tuned at all: frozen DINOv2 ViT-B/14 features + class-balanced logistic regression
  (scripts/21b_frozen_features.py). Out-of-fold macro-F1 0.574 / accuracy 0.771 — well below fine-tuning.

Final round (docs/AUDIT.md §6, rules fixed before training): ONE imbalance mechanism per arm, same backbone and
augmentation as `convnext_t_in1k_robust` (which is arm A, class-weighted loss):
- `convnext_t_in1k_sampler` (arm B): plain cross-entropy, class-balanced sampling (every class drawn equally often);
- `convnext_t_in1k_robust_oe` (focused task only): arm A plus outlier exposure (Hendrycks et al., ICLR 2019). The
  training folds' photos of the three unsupported conditions are pushed towards a uniform output over the five classes
  (weight 0.5, 8 outlier photos per step, both fixed in advance), so the model learns to be unsure on them while
  keeping exactly five outputs. (Sampling / plain variants with outlier exposure were dropped for time.)
"""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Recipe:
    name: str
    arch: str
    label: str
    input_size: int = 256
    eval_resize: str = "resize"  # "resize" or "crop" (see preprocessing.eval_transform)
    aug: str = "robust"  # "base", "robust" (mild) or "heavy"
    batch: int = 16
    head_epochs: int = 5
    head_lr: float = 1e-3
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 0.05
    layer_decay: float | None = None
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    class_weighting: str = "balanced"  # "balanced" (N / (K n_c)), "sqrt" (its square root) or "none"
    sampling: str = "uniform"  # "uniform" (shuffle) or "class_balanced" (each class drawn equally often)
    outlier_weight: float = 0.0  # > 0: outlier exposure with out-of-scope photos (focused task only)
    outlier_batch: int = 8
    crop: str = "none"  # "leaf": leaf-centred crop before everything else (tomatoleaf.crop)
    ema_decay: float | None = None
    model_kwargs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


RECIPES = {r.name: r for r in [
    Recipe("convnext_t_in1k_base", "convnext_tiny", "ConvNeXt-Tiny (ImageNet-1k), original recipe", aug="base"),
    Recipe("convnext_t_in22k_base", "timm:convnext_tiny.fb_in22k_ft_in1k",
           "ConvNeXt-Tiny (ImageNet-22k), original recipe", aug="base"),
    Recipe("convnext_t_in22k_robust", "timm:convnext_tiny.fb_in22k_ft_in1k",
           "ConvNeXt-Tiny (ImageNet-22k), robust recipe", ema_decay=0.998),
    Recipe("convnext_t_in1k_robust", "convnext_tiny", "ConvNeXt-Tiny (ImageNet-1k), robust recipe", ema_decay=0.998),
    Recipe("dinov2_s_robust", "timm:vit_small_patch14_dinov2.lvd142m",
           "ViT-S/14 (DINOv2 self-supervised), robust recipe",
           input_size=252, eval_resize="crop", batch=32, layer_decay=0.8, ema_decay=0.998,
           model_kwargs={"img_size": 252, "drop_path_rate": 0.05}),
    Recipe("effnetv2_s_robust", "timm:tf_efficientnetv2_s.in21k_ft_in1k",
           "EfficientNetV2-S (ImageNet-21k), robust recipe",
           batch=24, lr=2e-4, weight_decay=0.01, ema_decay=0.998, model_kwargs={"drop_path_rate": 0.05}),
    Recipe("dinov2_b_linear", "timm:vit_base_patch14_dinov2.lvd142m",
           "ViT-B/14 (DINOv2, frozen) + logistic regression (scripts/21b)", input_size=224, aug="none",
           head_epochs=0, epochs=0),
    Recipe("convnext_t_in22k_llrd", "timm:convnext_tiny.fb_in22k_ft_in1k",
           "ConvNeXt-Tiny (ImageNet-22k), heavy aug + layer decay (stopped after fold 0)",
           aug="heavy", head_epochs=2, lr=1.5e-4, layer_decay=0.8, ema_decay=0.998, model_kwargs={"drop_path_rate": 0.1}),
    # ---- final round: imbalance arms (B, C) and outlier exposure (focused task) ----
    Recipe("convnext_t_in1k_sampler", "convnext_tiny", "ConvNeXt-Tiny, robust recipe, class-balanced sampling",
           class_weighting="none", sampling="class_balanced", ema_decay=0.998),
    # last round: leaf-centred crop (tomatoleaf.crop); focused with stronger outlier exposure (weight 1.0)
    Recipe("convnext_t_in1k_robust_oe_crop", "convnext_tiny",
           "ConvNeXt-Tiny, leaf crop, class-weighted loss + outlier exposure (weight 1.0)", outlier_weight=1.0,
           crop="leaf", ema_decay=0.998),
    Recipe("convnext_t_in1k_sampler_crop", "convnext_tiny", "ConvNeXt-Tiny, leaf crop, class-balanced sampling",
           class_weighting="none", sampling="class_balanced", crop="leaf", ema_decay=0.998),
    Recipe("convnext_t_in1k_robust_oe", "convnext_tiny",
           "ConvNeXt-Tiny, robust recipe, class-weighted loss + outlier exposure", outlier_weight=0.5, ema_decay=0.998),
]}
