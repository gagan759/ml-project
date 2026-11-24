"""Weights manifest and the verified downloader (no network: file:// URLs)."""
import hashlib
import json

import pytest

from tomatoleaf import weights

needs_manifest = pytest.mark.skipif(not weights.MANIFEST.exists(), reason="weights manifest not built")


@needs_manifest
def test_manifest_lists_every_deployed_file_with_size_hash_and_url():
    manifest = weights.load_manifest()
    deployment = json.loads(weights.DEPLOYMENT.read_text(encoding="utf-8"))
    listed = {f["path"] for f in manifest["files"]}
    assert weights.model_paths(deployment) <= listed
    for f in manifest["files"]:
        assert f["bytes"] > 0 and len(f["sha256"]) == 64
        # each file lives in the release it was first published in (unchanged files keep their earlier URL)
        assert f["url"].endswith(f"/{f['asset']}") and "/releases/download/weights-v" in f["url"]
        assert manifest["repo"] in f["url"]


@needs_manifest
def test_files_for_selects_only_the_requested_model():
    deployment = json.loads(weights.DEPLOYMENT.read_text(encoding="utf-8"))
    default = next(e for e in deployment["models"] if e["key"] == deployment["default_model"])
    files = weights.files_for([default["key"]])
    assert {f["path"] for f in files} == {m["path"] for m in default["members"]}


def _entry(tmp_path, payload: bytes, sha: str | None = None) -> dict:
    src = tmp_path / "release" / "asset.bin"
    src.parent.mkdir()
    src.write_bytes(payload)
    return {"path": "sub/model.pt", "bytes": len(payload), "url": src.as_uri(),
            "sha256": sha or hashlib.sha256(payload).hexdigest()}


def test_fetch_downloads_verifies_and_reports_progress(tmp_path):
    payload = b"weights" * 300_000  # > 1 MB: several chunks
    entry, seen = _entry(tmp_path, payload), []
    dest = weights.fetch(entry, tmp_path / "models", on_progress=lambda done, total: seen.append((done, total)))
    assert dest.read_bytes() == payload
    assert weights.is_present(entry, tmp_path / "models", verify=True)
    assert seen[-1] == (len(payload), len(payload)) and len(seen) >= 2


def test_fetch_rejects_a_corrupted_download_and_leaves_nothing_behind(tmp_path):
    entry = _entry(tmp_path, b"weights", sha="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        weights.fetch(entry, tmp_path / "models")
    assert not [p for p in (tmp_path / "models").rglob("*") if p.is_file()]


def test_is_present_needs_the_exact_size(tmp_path):
    entry = _entry(tmp_path, b"abc")
    target = tmp_path / "models" / entry["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"abcd")
    assert not weights.is_present(entry, tmp_path / "models")
    target.write_bytes(b"abc")
    assert weights.is_present(entry, tmp_path / "models", verify=True)
