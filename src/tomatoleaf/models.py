"""Deep learning architectures.

- torchvision: custom CNN, MobileNetV2, EfficientNet-B0, ConvNeXt-Tiny (the original study).
- timm (this upgrade): any timm model, written as arch="timm:<name>", e.g. "timm:convnext_tiny.fb_in22k_ft_in1k".
Checkpoints store the arch string, class list, input size, normalisation and model kwargs, so `load_checkpoint`
rebuilds any of them without extra information.
"""
import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v2,
)

from .config import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CustomCNN(nn.Module):
    """4 blocks of [conv-BN-ReLU x2 -> max-pool], global average pooling, dropout, linear classifier.

    224x224 input -> 112 -> 56 -> 28 -> 14 spatial resolution; ~1.2M parameters.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, channels=(32, 64, 128, 256), dropout: float = 0.4):
        super().__init__()
        blocks, in_ch = [], 3
        for out_ch in channels:
            blocks.append(nn.Sequential(conv_bn_relu(in_ch, out_ch), conv_bn_relu(out_ch, out_ch), nn.MaxPool2d(2)))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def build_mobilenetv2(num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained MobileNetV2 with a new 8-class head (1280 -> 8)."""
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes))
    return model


def build_efficientnet_b0(num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ImageNet-pretrained EfficientNet-B0 (compound-scaled MBConv network, 5.3M parameters) with an 8-class head."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes))
    return model


def build_convnext_tiny(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ImageNet-pretrained ConvNeXt-Tiny (modernised ResNet: 7x7 depthwise convs, LayerNorm, GELU; 28M parameters)."""
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def build_model(arch: str, pretrained: bool = True, num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    if arch.startswith("timm:"):
        import timm

        return timm.create_model(arch[5:], pretrained=pretrained, num_classes=num_classes, **kwargs)
    if arch == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    if arch == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, pretrained=pretrained)
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    if arch == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {arch}")


def normalisation(arch: str) -> tuple:
    """(mean, std) the backbone was pretrained with."""
    if arch.startswith("timm:"):
        import timm

        cfg = timm.get_pretrained_cfg(arch[5:])
        return tuple(cfg.mean), tuple(cfg.std)
    return IMAGENET_MEAN, IMAGENET_STD


def load_checkpoint(path, device="cpu"):
    """Rebuild the architecture with as many outputs as the checkpoint's class list, load weights, eval mode.

    Weights saved in float16 (to halve download size) are loaded into a float32 model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["class_keys"]),
                        **ckpt.get("model_kwargs", {}))
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    return model.to(device).eval(), ckpt


def is_timm(arch: str) -> bool:
    return arch.startswith("timm:")


def classifier_params(model: nn.Module):
    """Parameters of the final classification layer (trained alone in the warm-up stage)."""
    if hasattr(model, "get_classifier"):
        return list(model.get_classifier().parameters())
    return list(final_linear(model).parameters())


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head.

    torchvision models keep the original study's behaviour (only `model.features` is frozen)."""
    if not hasattr(model, "get_classifier") and hasattr(model, "features"):
        for p in model.features.parameters():
            p.requires_grad = trainable
        return
    head = {id(p) for p in classifier_params(model)}
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


def final_linear(model: nn.Module) -> nn.Linear:
    """The last nn.Linear layer: its INPUT is the penultimate feature vector used for OOD detection."""
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Linear):
            return head
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("model has no linear layer")
    return last


class FeatureTap:
    """Captures the penultimate features (input of the final linear layer) during a normal forward pass."""

    def __init__(self, model: nn.Module):
        self.features = None
        self._handle = final_linear(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features = inputs[0].detach()

    def close(self):
        self._handle.remove()


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    """Module whose output feature maps Grad-CAM uses (last convolutional stage / last transformer block)."""
    if is_timm(arch):
        name = arch[5:]
        if name.startswith(("convnext", "convnextv2")):
            return model.stages[-1]
        if name.startswith("vit"):
            # the head reads only the CLS token, so the last block's OUTPUT patch tokens get no gradient; its first
            # LayerNorm's output still reaches the CLS token through the block's attention
            return model.blocks[-1].norm1
        if hasattr(model, "conv_head"):  # EfficientNet family
            return model.conv_head
        raise ValueError(f"no Grad-CAM layer defined for {arch}")
    return model.features[-1]


def gradcam_reshape(model: nn.Module, arch: str):
    """For ViTs: token sequence (B, prefix+N, C) -> feature map (B, C, h, w); identity for CNNs."""
    if is_timm(arch) and arch[5:].startswith("vit"):
        prefix = getattr(model, "num_prefix_tokens", 1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            tokens = t[:, prefix:, :]
            side = int(round(tokens.shape[1] ** 0.5))
            return tokens.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)

        return reshape
    return None


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
