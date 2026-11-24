"""Model weights: which files a deployment needs, whether they are present, and a verified download.

The weights are GitHub release assets (files over 100 MB cannot live in a repository). The manifest
(experiments/weights_manifest.json, written by scripts/26_package_release.py) lists every file with its size,
SHA-256 and download URL. Standard library only, so the downloader also runs before anything heavy is installed.
"""
import hashlib
import json
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "experiments" / "weights_manifest.json"
DEPLOYMENT = ROOT / "experiments" / "deployment.json"
MODELS_DIR = ROOT / "experiments" / "models"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_paths(deployment: dict, keys: Iterable[str] | None = None) -> set[str]:
    """Weight paths (relative to experiments/models) used by the given deployment models (default: all)."""
    wanted = set(keys) if keys is not None else None
    paths = set()
    for entry in deployment["models"]:
        if wanted is None or entry["key"] in wanted:
            paths |= {m["path"] for m in entry.get("members", [])}
            if "path" in entry:
                paths.add(entry["path"])
    return paths


def files_for(keys: Iterable[str] | None = None, manifest: dict | None = None, deployment: dict | None = None) -> list[dict]:
    manifest = manifest or load_manifest()
    deployment = deployment or json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    needed = model_paths(deployment, keys)
    return [f for f in manifest["files"] if f["path"] in needed]


def is_present(entry: dict, models_dir: Path = MODELS_DIR, verify: bool = False) -> bool:
    """File exists with the right size (and, with verify=True, the right SHA-256)."""
    path = models_dir / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    return not verify or sha256(path) == entry["sha256"]


def fetch(entry: dict, models_dir: Path = MODELS_DIR, on_progress: Callable[[int, int], None] | None = None,
          user_agent: str = "tomato-care-weights-downloader", timeout: int = 60) -> Path:
    """Download one file to a temporary name, check its SHA-256, then move it into place. Raises on any failure."""
    dest = (models_dir / entry["path"]).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(entry["url"], headers={"User-Agent": user_agent})
    h, done = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, entry["bytes"])
        if h.hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {entry['path']}")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest
