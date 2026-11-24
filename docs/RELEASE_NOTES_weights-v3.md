## Model weights v3 (Tomato Care — final round)

Weights of the two served pipelines (`experiments/deployment.json`): one validated model per mode, chosen on
cross-validation data only (`scripts/36_final_pipeline.py`, rules in `docs/AUDIT.md` §6 and `docs/METHODOLOGY.md` §6c).

```bash
python scripts/download_weights.py        # the served models (SHA-256 checked)
python scripts/download_weights.py --all  # + the earlier models used as the before/after baseline
```

* **This release** holds only the new or changed files: the 5 fold networks of each final pipeline, float16 weights
  plus each network's training-feature statistics (used by the "unusual image" check).
  Files: `deploy__<task>__<recipe>__fold<k>.pt`.
* Files that did not change keep their download URL in the earlier releases (`weights-v2`); the manifest
  (`experiments/weights_manifest.json`) lists every file with its release, size and SHA-256.

Asset names are flat (`/` → `__`). The Tomato-Village images are not included (no redistribution licence).
