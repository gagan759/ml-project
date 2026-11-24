## Model weights v2 (Tomato Care)

Weights for the models served by the API (`experiments/deployment.json`). Download and verify them all with

```bash
python scripts/download_weights.py        # the served version-2 models (SHA-256 checked)
python scripts/download_weights.py --all  # + the original study's models (to re-run the final benchmark)
```

* **Upgrade defaults** — fold ensembles chosen by cross-validation (`scripts/22_select_models.py`): the 5 fold models
  of each member recipe, float16 weights plus each fold model's training-feature statistics (used by the
  "Unusual image" check). Files: `deploy__<task>__<recipe>__fold<k>.pt`.
* **Original study** — the previous models (float32, unchanged); not served by the apps, kept so the comparison
  in the final benchmark can be reproduced: `dl__clean__*.pt`, `ml__*.joblib`.

Asset names are flat (`/` → `__`); the manifest maps each asset to its path under `experiments/models/`.
The Tomato-Village images are not included (no redistribution licence).
