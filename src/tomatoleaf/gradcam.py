"""Grad-CAM: weight the last feature maps by the average gradient of the class score, sum, ReLU, upsample.

Works for CNNs (last convolutional stage) and, through `reshape`, for ViTs (last block's patch tokens laid out on
their grid, as in Chefer et al. / the pytorch-grad-cam reshape_transform convention)."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, reshape=None):
        self.model = model
        self.reshape = reshape
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        if not output.requires_grad:  # e.g. a concurrent torch.no_grad() prediction on the same model
            return
        self.activations = output
        output.register_hook(lambda grad: setattr(self, "gradients", grad))

    def __call__(self, x: torch.Tensor, class_index: int | None = None):
        """x: 1x3xHxW normalised tensor. Returns (heatmap HxW in [0,1], class index, softmax probabilities)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = int(logits.argmax(1).item()) if class_index is None else class_index
            self.model.zero_grad(set_to_none=True)
            logits[0, idx].backward()
        acts, grads = self.activations, self.gradients
        if self.reshape is not None:
            acts, grads = self.reshape(acts), self.reshape(grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        return cam.detach().cpu().numpy(), idx, probs

    def close(self):
        self._handle.remove()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap (resized to the image) onto an RGB uint8 image."""
    import cv2
    import matplotlib

    heat = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = (matplotlib.colormaps["inferno"](heat)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * rgb.astype(np.float32) + alpha * colored, 0, 255).astype(np.uint8)


def heatmap_png(cam: np.ndarray, size: int = 256) -> np.ndarray:
    """[0,1] heatmap -> RGBA uint8 (inferno colours, alpha = intensity) for client-side blending with an opacity slider."""
    import cv2
    import matplotlib

    heat = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    rgba = (matplotlib.colormaps["inferno"](heat) * 255).astype(np.uint8)
    rgba[..., 3] = np.clip(heat * 255 * 1.15, 0, 255).astype(np.uint8)
    return rgba
