"""Architectures, checkpoints and Grad-CAM targets (no pretrained download needed: pretrained=False)."""
import numpy as np
import pytest
import torch

from tomatoleaf.gradcam import GradCAM
from tomatoleaf.models import (
    FeatureTap,
    build_model,
    final_linear,
    gradcam_layer,
    gradcam_reshape,
    load_checkpoint,
    set_backbone_trainable,
)
from tomatoleaf.preprocessing import dihedral_views

ARCHS = [
    ("convnext_tiny", {}, 64),
    ("timm:convnext_tiny.fb_in22k_ft_in1k", {}, 64),
    ("timm:vit_small_patch14_dinov2.lvd142m", {"img_size": 56}, 56),
    ("timm:tf_efficientnetv2_s.in21k_ft_in1k", {}, 64),
]


@pytest.mark.parametrize("arch,kwargs,size", ARCHS)
def test_build_forward_features_and_gradcam(arch, kwargs, size):
    pytest.importorskip("timm")
    model = build_model(arch, pretrained=False, num_classes=5, **kwargs).eval()
    x = torch.randn(2, 3, size, size)
    tap = FeatureTap(model)
    logits = model(x)
    tap.close()
    assert logits.shape == (2, 5)
    assert tap.features.shape[0] == 2 and tap.features.shape[1] == final_linear(model).in_features
    cam = GradCAM(model, gradcam_layer(model, arch), gradcam_reshape(model, arch))
    heat, idx, probs = cam(x[:1])
    cam.close()
    assert heat.shape == (size, size) and 0 <= heat.min() and heat.max() <= 1 + 1e-6
    assert heat.max() > 0, "Grad-CAM map is all zero (wrong target layer?)"
    assert abs(probs.sum() - 1) < 1e-5


def test_head_only_stage_freezes_backbone():
    pytest.importorskip("timm")
    model = build_model("timm:convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=3)
    set_backbone_trainable(model, False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("head.fc") for n in trainable)
    set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_checkpoint_roundtrip_in_float16(tmp_path):
    model = build_model("convnext_tiny", pretrained=False, num_classes=4).eval()
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    path = tmp_path / "m.pt"
    torch.save({"arch": "convnext_tiny", "state_dict": state, "class_keys": list("abcd"), "input_size": 64}, path)
    loaded, ckpt = load_checkpoint(path)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        ref = model.half().float()(x)
        out = loaded(x)
    assert torch.allclose(ref, out, atol=1e-4)
    assert next(loaded.parameters()).dtype == torch.float32


def test_dihedral_views_are_the_eight_symmetries():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    views = dihedral_views(x, 8)
    assert views.shape[0] == 8
    flat = {tuple(v.flatten().tolist()) for v in views}
    assert len(flat) == 8
    assert np.isclose(views.mean().item(), x.mean().item())
