"""Optional ONNX Runtime execution for the PyTorch models (CPU inference optimisation).

Measured on the deployment CPU (scripts/24_cpu_inference.py): for one image, ONNX Runtime runs ConvNeXt-Tiny and
ViT-S/14 about 1.6-1.9x faster than eager PyTorch with logits equal to ~1e-5. Each model is exported once with two
outputs (logits and the penultimate features used by the out-of-distribution check) and cached on disk next to the
weights; the PyTorch model is still used for Grad-CAM. Any export or runtime problem falls back to PyTorch.
"""
import hashlib
import os
from pathlib import Path

import torch

from tomatoleaf.models import final_linear


def enabled() -> bool:
    if os.getenv("TOMATO_RUNTIME", "onnx").lower() != "onnx":
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


class _WithFeatures(torch.nn.Module):
    """Wraps a classifier so a single forward pass returns (logits, penultimate features)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self._feats = None
        self.handle = final_linear(model).register_forward_hook(lambda m, i, o: setattr(self, "_feats", i[0]))

    def forward(self, x):
        logits = self.model(x)
        return logits, self._feats


def _cache_path(ckpt_path: Path) -> Path:
    stat = ckpt_path.stat()
    key = hashlib.sha1(f"{ckpt_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{torch.__version__}".encode()).hexdigest()[:16]
    return ckpt_path.parent / ".onnx_cache" / f"{ckpt_path.stem}-{key}.onnx"


class OnnxModel:
    """ONNX Runtime session for one checkpoint: __call__(x: float32 NCHW tensor) -> (logits, features) numpy."""

    def __init__(self, model: torch.nn.Module, ckpt_path: Path, input_size: int, threads: int | None = None):
        import onnxruntime as ort

        path = _cache_path(Path(ckpt_path))
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            wrapper = _WithFeatures(model).eval()
            dummy = torch.zeros(1, 3, input_size, input_size)
            with torch.no_grad():
                torch.onnx.export(wrapper, dummy, tmp, input_names=["x"], output_names=["logits", "features"],
                                  dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}, "features": {0: "n"}},
                                  opset_version=17, dynamo=False)
            wrapper.handle.remove()  # the torch model is used again (Grad-CAM, parity check)
            tmp.replace(path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(threads or os.getenv("TOMATO_THREADS", min(8, os.cpu_count() or 4)))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # several sessions run one after another (ensemble members): idle worker threads must not spin-wait
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.path = path

    def __call__(self, x: torch.Tensor):
        logits, feats = self.session.run(["logits", "features"], {"x": x.detach().cpu().numpy()})
        return logits, feats
