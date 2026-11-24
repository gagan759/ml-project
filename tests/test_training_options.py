"""Final-round training options: no class weights, class-balanced sampling, outlier exposure (scripts/21, 36)."""
import dataclasses

import numpy as np
import pytest
import torch

from tomatoleaf.recipes import RECIPES
from tomatoleaf.training import _gpu_loader, class_weights, fit_recipe, uniform_target_loss


def test_class_weight_modes():
    labels = np.array([0] * 90 + [1] * 10)
    assert torch.equal(class_weights(labels, 2, "none"), torch.ones(2))
    w = class_weights(labels, 2, "balanced")
    assert w[1] / w[0] == pytest.approx(9.0)  # N / (K n_c)
    assert class_weights(labels, 2, "sqrt")[1] / class_weights(labels, 2, "sqrt")[0] == pytest.approx(3.0)


def test_class_balanced_sampling_draws_classes_equally():
    labels = np.array([0] * 180 + [1] * 15 + [2] * 5)
    images = np.zeros((len(labels), 8, 8, 3), dtype=np.uint8)
    loader = _gpu_loader(images, labels, lambda x: x, batch_size=50, seed=0, device="cpu", drop_last=False,
                         sampling="class_balanced")
    drawn = torch.cat([y for _, y in loader] * 1)
    for _ in range(19):  # 20 epochs of N draws each
        drawn = torch.cat([drawn] + [y for _, y in loader])
    share = np.bincount(drawn.numpy(), minlength=3) / len(drawn)
    assert len(drawn) == 20 * len(labels)
    assert np.allclose(share, 1 / 3, atol=0.03), share
    uniform = _gpu_loader(images, labels, lambda x: x, batch_size=50, seed=0, device="cpu", drop_last=False)
    assert np.bincount(torch.cat([y for _, y in uniform]).numpy()).tolist() == [180, 15, 5]


def test_uniform_target_loss_is_minimal_for_an_unsure_model():
    k = 5
    flat = uniform_target_loss(torch.zeros(4, k))
    peaked = uniform_target_loss(torch.tensor([[8.0, 0, 0, 0, 0]] * 4))
    assert flat.item() == pytest.approx(np.log(k), abs=1e-6)
    assert peaked.item() > flat.item() + 1.0


def test_outlier_recipes_need_outlier_photos():
    recipe = RECIPES["convnext_t_in1k_robust_oe"]
    assert recipe.outlier_weight > 0 and recipe.class_weighting == "balanced" and recipe.sampling == "uniform"
    assert RECIPES["convnext_t_in1k_sampler"].sampling == "class_balanced"
    assert RECIPES["convnext_t_in1k_sampler"].class_weighting == "none"  # one compensation mechanism per arm
    tiny = dataclasses.replace(recipe, epochs=0, head_epochs=0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 5))
    with pytest.raises(ValueError, match="outlier"):
        fit_recipe(model, tiny, np.zeros((4, 8, 8, 3), np.uint8), np.arange(4), device=torch.device("cpu"))
