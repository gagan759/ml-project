"""Upgrade round, step 5 — CPU inference: measure the options and keep the fastest one that gives the same answers.

For one fold model of each deployed recipe, with the deployed number of test-time views (a batch of 1 or 8):
  - PyTorch eager, float32 (the original serving path)
  - PyTorch eager with channels-last memory format
  - ONNX Runtime (CPU execution provider, all graph optimisations), exported with logits AND penultimate features
at 1, 4 and 8 threads. Equivalence: max |probability difference| vs PyTorch eager on 32 real photos must be < 1e-4.
Run it on an otherwise idle machine. Output: experiments/results/cpu_inference.json
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tomatoleaf.config import EXPERIMENTS_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.data import load_images, load_splits  # noqa: E402
from tomatoleaf.models import final_linear, load_checkpoint  # noqa: E402
from tomatoleaf.preprocessing import dihedral_views, preprocess_for_cnn  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402


class WithFeatures(torch.nn.Module):
    """Model wrapper returning (logits, penultimate features) so ONNX serves the OOD check too."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feats = None
        final_linear(model).register_forward_hook(lambda m, i, o: setattr(self, "feats", i[0]))

    def forward(self, x):
        logits = self.model(x)
        return logits, self.feats


def timeit(fn, x, n=15, warm=3):
    for _ in range(warm):
        fn(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return 1000 * float(np.median(t))


def main():
    import onnxruntime as ort

    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    defaults = deployment["defaults_by_task"]
    splits = load_splits()
    imgs = load_images(splits[splits.split == "test"].path.iloc[:32])
    results = []
    seen = set()
    for key in defaults.values():
        entry = next(e for e in deployment["models"] if e["key"] == key)
        for member in entry["members"]:
            recipe = member["path"].split("/")[2]
            if (recipe, member["tta_views"]) in seen:
                continue
            seen.add((recipe, member["tta_views"]))
            model, ckpt = load_checkpoint(MODELS_DIR / member["path"], "cpu")
            views = int(member["tta_views"])

            def batch(rgb, ckpt=ckpt, views=views):
                x = preprocess_for_cnn(rgb, ckpt["input_size"], ckpt["mean"], ckpt["std"], ckpt.get("eval_resize", "resize"))
                return dihedral_views(x, views) if views > 1 else x

            xs = [batch(r) for r in imgs]
            wrapped = WithFeatures(model).eval()
            onnx_path = Path(tempfile.gettempdir()) / f"{recipe}_{views}.onnx"
            with torch.no_grad():
                torch.onnx.export(wrapped, xs[0], onnx_path, input_names=["x"], output_names=["logits", "features"],
                                  dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}, "features": {0: "n"}}, opset_version=17,
                                  dynamo=False)
            row = {"recipe": recipe, "views": views, "input_size": ckpt["input_size"], "onnx_mb": onnx_path.stat().st_size / 2**20,
                   "timings_ms": {}}
            with torch.no_grad():
                ref = np.stack([torch.softmax(model(x).float().mean(0), 0).numpy() for x in xs])
            for threads in (1, 4, 8):
                torch.set_num_threads(threads)
                so = ort.SessionOptions()
                so.intra_op_num_threads = threads
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                with torch.inference_mode():
                    eager = timeit(lambda x, m=model: m(x), xs[0])
                    cl_model = load_checkpoint(MODELS_DIR / member["path"], "cpu")[0].to(memory_format=torch.channels_last)
                    cl = timeit(lambda x, m=cl_model: m(x.contiguous(memory_format=torch.channels_last)), xs[0])
                onnx = timeit(lambda x, s=sess: s.run(None, {"x": x.numpy()}), xs[0])
                row["timings_ms"][threads] = {"torch_eager": eager, "torch_channels_last": cl, "onnxruntime": onnx}
                print(f"{recipe} x{views} @ {threads} threads: eager {eager:.0f} ms | channels-last {cl:.0f} ms | "
                      f"onnxruntime {onnx:.0f} ms", flush=True)
            torch.set_num_threads(8)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            out = np.stack([np.exp(lg.mean(0) - lg.mean(0).max()) / np.exp(lg.mean(0) - lg.mean(0).max()).sum()
                            for lg in (sess.run(["logits"], {"x": x.numpy()})[0] for x in xs)])
            row["max_abs_prob_diff_onnx_vs_torch"] = float(np.abs(out - ref).max())
            results.append(row)
            os.remove(onnx_path)
    best = {}
    for r in results:
        t8 = r["timings_ms"][8]
        fastest = min(t8, key=t8.get)
        best[f"{r['recipe']}@{r['views']}"] = {"fastest": fastest, "speedup_vs_eager": t8["torch_eager"] / t8[fastest],
                                                "equivalent": r["max_abs_prob_diff_onnx_vs_torch"] < 1e-4}
    save_json({"machine": {"cpu_count": os.cpu_count(), "torch": torch.__version__, "onnxruntime": ort.__version__},
               "results": results, "summary": best}, RESULTS_DIR / "cpu_inference.json")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
