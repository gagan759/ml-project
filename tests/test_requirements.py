"""Streamlit Community Cloud installs only requirements.txt. Every third-party module the served app can import at
runtime — including imports inside functions, such as matplotlib for the Grad-CAM colours — must be installed by it
(a missing one breaks the live app while every local test still passes)."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = [ROOT / "streamlit_app.py", *sorted((ROOT / "streamlit_ui").rglob("*.py")),
           *sorted((ROOT / "backend" / "services").glob("*.py")),
           *(ROOT / "src" / "tomatoleaf" / f"{m}.py"
             for m in ("config", "features", "gradcam", "models", "ood", "preprocessing", "utils", "weights"))]
PIP_NAME = {"PIL": "pillow", "sklearn": "scikit-learn", "skimage": "scikit-image", "cv2": "opencv-python-headless"}
NOT_NEEDED = {"timm": "only for timm architectures; the served models are torchvision ConvNeXt-Tiny",
              "onnxruntime": "optional runtime, imported inside try/except; the app runs PyTorch"}
LOCAL = {"tomatoleaf", "backend", "streamlit_ui", "__future__"}


def imported_modules(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def requirement_names() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[ @=<>;\[]", line, maxsplit=1)[0].lower())
    return names


def test_requirements_cover_every_runtime_import():
    required, missing = requirement_names(), {}
    for path in RUNTIME:
        for mod in imported_modules(path):
            if mod in sys.stdlib_module_names or mod in LOCAL or mod in NOT_NEEDED:
                continue
            if PIP_NAME.get(mod, mod).lower() not in required:
                missing.setdefault(mod, []).append(path.relative_to(ROOT).as_posix())
    assert not missing, f"imported at runtime but not installed by requirements.txt: {missing}"
