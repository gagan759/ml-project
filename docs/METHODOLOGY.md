# Methodology

How the data were checked, how the models were trained and chosen without touching the test set, and how every
reported number was computed. All numbers quoted from experiments are in [RESULTS.md](RESULTS.md), which is
generated from the result files; this document explains the procedure.

## 1. Dataset

**Tomato-Village, Variant-a** (Gehlot, Saxena & Gandhi, *Multimedia Systems*, 2023, doi:10.1007/s00530-023-01158-y):
single tomato leaves photographed on paper in Jodhpur and Jaipur (Rajasthan, India), eight classes — Early Blight,
Healthy, Late Blight, Leaf Miner, Magnesium / Nitrogen / Potassium Deficiency, Spotted Wilt Virus.

| Fact | Value | How it was established |
|---|---|---|
| Files | 4,525 JPEGs, all 256×256, none corrupt or byte-identical | `scripts/01_build_index.py` (full decode, MD5) |
| Distinct photographs | 2,103 | file names `IMGyyyymmddhhmmss[_n]`: 346 photos have 7 augmented copies |
| Re-shots of the same leaf | 340 near-duplicate pairs merged | ImageNet embeddings (cosine ≥ 0.871) and capture times ≤ 120 s, `scripts/02` |
| Leaf groups | 1,786 | union–find over copies and re-shots |
| Imbalance | 1,024 Leaf Miner vs 9 Potassium photographs (114:1) | |
| Licence | none declared | images are not redistributed; cite the paper |

## 2. A leakage-free split

The official split puts augmented copies of one photograph into train, validation and test: 61.2% of the official
test images have a sibling in the training folder (100% for the five augmented classes). A model can score highly
by recognising photographs it has already seen. We therefore split **whole leaf groups** with `StratifiedGroupKFold`
(the best-balanced of 200 seeded draws, chosen by label counts only): 1,502 / 311 / 290 original photographs for
train / validation / test, with no leaf group on two sides (`scripts/03_make_splits.py`, asserted in tests).
Trained and tested on the official folders, the original study's MobileNetV2 and ConvNeXt-Tiny reach 93.9% and 97.6%
accuracy; with the leakage-free split the same recipes reach 74.8% and 82.1% — the difference measures the inflation.

## 3. Preprocessing audit

The original web app gave different answers for photos that looked alike. Before training anything new we checked
whether a real phone photo reaches the model in the same form as a training photo. It did not.

**Shape.** Every dataset photo was taken with a *realme GT Master Edition* at 3468×4624 (72%, portrait) or 4624×3468
(EXIF of all 2,103 originals) and stored at 256×256. The authors resized the **whole** 3:4 frame into the square —
a non-uniform squash. Evidence: registering each original against the authors' rotated copies (SIFT + RANSAC affine)
gives singular-value ratios up to 1.79 ≈ (4/3)² at 90° rotations (and ≈ 1 for flips), exactly the signature of
`squash · rotate · squash⁻¹`. The original upload path instead **centre-cropped a square**: on a phone photo the
leaf appeared 25% narrower than in training and the top and bottom eighths were cut away.

**Colour.** The dataset files carry a **Display P3** ICC profile and P3-encoded pixel values. The original upload
path ignored colour profiles, so an sRGB photo (most web images and many phones) reached the model with different
values — and colour is how nutrient deficiencies show.

**Compression.** Dataset files are JPEG quality 90 with 4:2:0 chroma (estimated from the quantisation tables of
200 files); a downscaled phone photo is much cleaner, and an earlier version flipped borderline predictions after a
q95 re-encode.

**Fix** (`tomatoleaf.preprocessing.prepare_upload`): EXIF orientation → convert into Display P3 (sRGB, untagged and
other ICC profiles via LittleCMS) → centre-crop the aspect ratio into [3:4, 4:3] → resize the whole frame to 256×256 →
JPEG quality 90. Dataset images pass through bit-for-bit unchanged (tested on all 4,525 files), so training data and
earlier results are unaffected.

**Measured effect** (`scripts/19_baseline_audit.py`, validation photos only): each validation photo is turned back
into a full-frame phone photo (3:4 or 4:3 as its EXIF says) saved either with a P3 profile or as untagged sRGB, then
sent through the old and the new upload path. For the original deployed model, the share of predictions that change
falls from 2.1% to 0.0% (P3 photo) and from 3.5% to 1.4% (sRGB photo). Tables for all models: RESULTS.md §5.

## 4. Model selection protocol

The original study selected models on one validation split of 311 photos containing a single Potassium and six
Nitrogen photos: one photo moved validation macro-F1 by up to 0.125, so the choice was largely noise. The upgrade
uses **grouped, stratified 5-fold cross-validation over the 1,813 train+validation photos**
(`scripts/20_make_cv_folds.py`; every class in every fold; leaf groups never split):

1. Each recipe trains for a **fixed number of epochs** — no early stopping — so no epoch is chosen by looking at the
   held-out fold; held-out scores are logged for the training curves only.
2. Every photo receives an **out-of-fold (OOF) prediction** from a model that never saw its leaf group. Recipes,
   test-time augmentation and ensembles are ranked on these 1,813 predictions.
3. The **deployed model of a recipe is its 5 fold models** (a fold ensemble). Each fold model has genuinely unseen
   data — its own fold — so its temperature and out-of-distribution threshold are fitted on photos it never saw.
4. The selection rule was fixed in code before the test set was scored (`scripts/22_select_models.py`):
   *highest pooled OOF macro-F1 among candidates whose estimated CPU time for the whole fold ensemble is ≤ 1 s per
   photo; ties within 0.001 → higher OOF accuracy → lower latency.*
5. The 290 test photos were then scored **once** (`scripts/25_final_benchmark.py`).

Outcome worth knowing: for the full task, out-of-fold data ranked the two-recipe ensemble first (macro-F1 0.634) and
the robust recipe alone lower; on the test photos the robust recipe alone scored higher (0.598 vs 0.559) and the
original study's model too (0.586). Recipes this close do not keep their order between data sets of this size. The
pre-registered choice was kept — choosing again after seeing test results would make the test score optimistic —
and every model's test result is published (RESULTS.md §3).

Caveat: the same 290 test photos were used to report the original study's results, so they are not perfectly
fresh; no decision in the upgrade used them.

## 5. Models and training recipes

| Recipe | Backbone (pretraining) | Changes vs original | Run |
|---|---|---|---|
| `convnext_t_in1k_base` | ConvNeXt-Tiny (ImageNet-1k, torchvision) | none — the original recipe, as the reference | 5 folds |
| `convnext_t_in1k_robust` | ConvNeXt-Tiny (ImageNet-1k) | robust recipe | 5 folds |
| `dinov2_s_robust` | ViT-S/14 (DINOv2 self-supervised, LVD-142M) | backbone + robust recipe, 252 px, layer decay 0.8 | 5 folds |
| `convnext_t_in22k_base` | ConvNeXt-Tiny (ImageNet-22k → 1k, timm) | backbone only (ablation) | fold 0 screen |
| `convnext_t_in22k_robust` | ConvNeXt-Tiny (ImageNet-22k → 1k) | backbone + robust recipe | fold 0 screen |
| `convnext_t_in22k_llrd` | ConvNeXt-Tiny (ImageNet-22k → 1k) | first robust version: heavy augmentation, layer decay 0.8, stochastic depth — under-fitted (train accuracy 0.92 vs 0.99) and was stopped | fold 0 |
| `dinov2_b_linear` | ViT-B/14 (DINOv2), **frozen** | no fine-tuning: CLS + mean-patch features, class-balanced logistic regression, C by nested grouped CV | 5 folds |

The ImageNet-22k variants were screened on fold 0 first (to save GPU time) and only recipes that looked competitive
were run on all five folds; all screens are reported in RESULTS.md §4. The frozen-feature baseline reached
out-of-fold macro-F1 0.574 and accuracy 0.771, well below fine-tuning (≈ 0.61–0.62 and ≈ 0.85), so it was not
deployed.

Common to all: 256×256 input (252 for the ViT), class-weighted cross-entropy with label smoothing 0.1 (weights
N / (K·n_c), applied to the true class only), AdamW, a head-only warm-up stage, then all layers with linear warm-up
and cosine decay, bf16 autocast, fixed seeds.

The **robust recipe** adds: (a) all eight flips/90° rotations as training augmentation, so every test-time
augmentation view is in-distribution (the original recipe never rotated by 90°, yet averaged 90° views at test time);
(b) aspect-ratio jitter 3:4…4:3 (the dataset mixes photos squashed along either axis); (c) JPEG re-compression
(quality 50–95) and mild Gaussian blur, so compression and resampling differences between uploads and dataset files
matter less; (d) layer-wise learning-rate decay (lower layers move less, keeping generic pretrained features);
(e) an exponential moving average of the weights, evaluated at the end; (f) gradient clipping and stochastic depth.
Hue is never jittered: colour is the diagnostic signal for deficiencies.

**Focused task.** The five-class scope (Healthy, Leaf Miner, Magnesium, Nitrogen, Spotted Wilt) was fixed in the
original study on validation F1 (the four conditions recognised best, plus Healthy); it is kept for comparability.

## 6. Calibration and uncertainty

* **Temperature scaling**: one temperature per recipe and view setting, fitted on its pooled OOF logits (minimum
  negative log-likelihood); ensembles average the members' calibrated probabilities.
* **“Uncertain”**: the lowest calibrated-confidence threshold whose OOF accuracy on answered photos reaches the
  target (95% for the focused task, 90% for the full task) with at least 50% coverage.
* **“Unusual image”** (out-of-distribution): each fold model scores a photo by the cosine distance to the k-th
  nearest (k = 5) of its training photos in penultimate-feature space (kNN, Sun et al. 2022), or by the Mahalanobis
  distance to class means (Lee et al. 2018, Ledoit–Wolf covariance) — whichever separates OOD inputs better on OOF
  data. Each fold model's threshold keeps 95% of its own held-out photos; the ensemble flags a photo when the mean
  ratio score/threshold, rescaled on OOF data, exceeds 1. OOD inputs for choosing the score: training photos of the
  three out-of-scope conditions (focused task) and half of a fixed set of 30 non-leaf probe images (scikit-image
  photographs and synthetic images); the other half is used only for reporting.

## 6b. Class bias and per-class confidence

Training already weights each class by its rarity. The served models add two post-hoc corrections, fitted on
cross-fitted out-of-fold predictions (`src/tomatoleaf/balance.py`, `scripts/31_balance_models.py`):

* **Per-class calibration** (vector scaling, penalised NLL): each condition's confidence matches its own accuracy.
* **Class-prior correction** (logit adjustment): the answered class is the argmax of p / prior^τ, so frequent
  conditions cannot out-vote rare ones. The confidence shown stays the calibrated probability of that answer. τ is the
  value with the highest out-of-fold balanced accuracy whose accuracy is within 2 points of the uncorrected model
  (focused 1.2, full 0.65).

The variants were scored once on the test set (`scripts/32_evaluate_variants.py`); RESULTS.md and the README list
them next to the uncorrected networks and the original study. A GAN was not used for minority-class augmentation: with
9 Potassium and 45 Nitrogen photographs it cannot learn new variation, only reproduce the few training leaves.

**Superseded by §6c.** The v2.1 variants combined a class-weighted training loss with a class-prior correction, i.e.
they compensated for imbalance twice. Their answer (the argmax of p / prior^τ) could also differ from the tallest bar of
the probabilities they displayed.

## 6c. Final round: one validated pipeline per model

An audit (`docs/AUDIT.md`) traced wrong answers on the app's example images to the models, not the pipeline: the
focused model gave confident wrong diagnoses to Early/Late Blight and Potassium photos, the full model rarely
recognised Early Blight or Potassium, the refusal rules were biased against rare in-scope conditions, and the
imbalance was compensated twice. The rules below were written down in `docs/AUDIT.md` §6 **before** any training of
this round; every choice uses cross-fitted out-of-fold (OOF) predictions of the grouped 5-fold CV, never test photos.

1. **Imbalance — one mechanism per arm** (same backbone, augmentation, epochs and folds; `src/tomatoleaf/recipes.py`):
   A class-weighted cross-entropy (inverse frequency); B class-balanced sampling with plain cross-entropy; C plain
   cross-entropy plus a post-hoc logit adjustment (τ · log prior subtracted from the logits; τ chosen on 4 folds by
   macro-F1 and applied to the 5th). No arm combines a training-time and a post-hoc correction. For time, the
   final round trained arms A and B for the full model and arm A (with outlier exposure) for the focused model;
   the other arms were dropped before any of their results existed.
2. **Focused model: outlier exposure** (Hendrycks et al., ICLR 2019) on arm A (arms B/C were dropped for time
   before any result was seen; the weighting-vs-sampling comparison is made on the full model). The training folds'
   Early Blight, Late Blight and Potassium photos are trained towards a uniform output over the five classes (weight
   0.5, 8 outlier photos per step, fixed in advance). The model keeps exactly five outputs; it learns to be unsure on
   look-alike conditions it does not cover. The held-out folds' photos of these conditions measure refusal OOF.
3. **Selection**: highest OOF macro-F1 (argmax of the final calibrated probabilities); within 0.01, the lower OOF
   NLL. Focused model: eligible only if the operating threshold refuses ≥ 80% of the OOF out-of-scope photos.
4. **Calibration** (probability reliability, not accuracy): temperature scaling vs per-class vector scaling, both
   cross-fitted; the lower OOF NLL wins (within 0.005: temperature scaling). The answer is the argmax of the final
   calibrated probabilities, so it is always the tallest bar. Reported: Brier, ECE (15 bins), class-wise ECE, NLL,
   reliability diagrams (OOF and test).
5. **One threshold per model**: the lowest calibrated confidence whose OOF accuracy on answered photos is ≥ 95%
   (focused) / ≥ 90% (full) with ≥ 50% of in-scope photos answered; for the focused model an answered out-of-scope
   photo counts as an error.
6. **Unusual-image gate**: kNN vs Mahalanobis feature distance on the selected networks, chosen by OOF AUROC against
   synthetic non-leaf probes, leaf-free paper and (focused) OOF out-of-scope photos. Limits keep 95% of OOF genuine
   photos **per answered condition** (shrunk towards the global limit for conditions with few photos), so rare
   conditions are not flagged more often than common ones.
7. **Evaluation** (`scripts/38_final_test.py`): the test photos are scored once with the final pipelines and, for
   paired comparisons, with the models they replace; stress tests built from test photos (blur, dark, over-exposed,
   heavy JPEG, four leaves in one frame, unusual backgrounds); out-of-dataset photos (PlantVillage leaves of other
   crops and of tomato diseases outside the dataset, tomato photos from another lab, everyday photos; downloaded by
   `scripts/37`, evaluation only); synthetic non-leaf probes and empty paper.

Scripts: `21_train_cv.py` (arms), `36_final_pipeline.py` (selection, calibration, threshold, gate),
`37_fetch_external_ood.py`, `38_final_test.py`; audits `34_audit_examples.py`, `35_audit_behaviour.py`.

## 6d. After the diagnosis: photo quality, reasons, candidates (no change to the models' answers)

- **Photo quality check** (`tomatoleaf.quality`). The quality check runs on the 256×256 image the model sees. It
  measures:
  - sharpness (variance of the Laplacian);
  - brightness and highlights (99th-percentile grey level);
  - the leaf, found as the region far from the paper colour in CIELAB. The paper colour is the median colour of the
    image border, so paper of any colour works. From this it measures the contrast between leaf and paper, the leaf's
    share of the frame, a second large region (several leaves) and the texture of the paper.

  Each limit is the 0.5th or 99.5th percentile of the measure on the train + validation photos (`scripts/41`).
  Evaluation, nothing fitted:
  - **Genuine photos:** 2% of the held-out test photos are flagged.
  - **Deliberately damaged test photos:** blur, darkening, over-exposure, other backgrounds and four leaves in one
    frame are flagged in 100%–100% of cases. Heavy JPEG
    compression is flagged in only 3% of cases.
  - **Photos from outside the dataset** are flagged in 99%–100% of cases.

  A failed check becomes a specific tip, and it never changes the diagnosis.
- **Why no diagnosis.** Each refusal states its cause:
  - low confidence, against the mode's threshold;
  - or an unusual image, against the feature-distance limit.

  The quality tips follow. In Focused mode, a pointer to Full mode lists the conditions the mode does not cover; the
  list is read from the model's class list.
- **Top-3 candidates.** "Not sure" results show the three most probable conditions.
- **Several photos of one leaf** were evaluated and not shipped (`scripts/42`, out-of-fold). The test used the dataset's
  leaves photographed 2–5 times, averaging their calibrated probabilities at the same threshold:

  | Mode | Answered | Correct when answered |
  |---|---|---|
  | Focused | 88% → 84% | 96.9% → 97.8% |
  | Full | 84% → 80% | 93.3% → 94.1% |

  These photos were taken seconds apart and add little new information.
- **int8 quantisation** was measured and not shipped:
  - it is about 1.5× faster on 2 CPU threads (0.28 s vs 0.43 s per photo);
  - it shifts confidences by up to 2 points, so the validated thresholds would no longer hold exactly;
  - peak memory rose to about 2.2 GB, close to the Streamlit Cloud limit.

## 6e. Five conditions, two models: deep learning and classical ML

**Scope.** The app diagnoses the five conditions that the eight-condition study could recognise reliably. The
eight-condition model (`full_8class`) is no longer served; its results stay in `experiments/` as the reason for the
scope. Photos of Early Blight, Late Blight and Potassium Deficiency are used only to check refusals.

**Classical model** (`scripts/43_classical_model.py`, rules in its docstring, fixed before training):
- Features of the same 256×256 image the network sees (`tomatoleaf.features`):
  - colour: HSV histograms and L\*a\*b\* moments over leaf pixels (73 values);
  - texture: uniform LBP histograms at radius 1 and 2, and GLCM contrast, dissimilarity, homogeneity, energy and
    correlation at distances 1, 2 and 4 over leaf pixels (43);
  - HOG: 64 principal components, fitted inside each training fold.
- Arms: four class-balanced classifiers (logistic regression, RBF SVM, random forest, histogram gradient boosting) on
  three feature sets. Hyperparameters were tuned by nested grouped cross-validation inside the network's 5 folds.
- Calibration, threshold and unusual-image gate follow the network's rules (§6c). The gate uses the distance to the
  training photos in the standardised colour + texture space.

Best arms (out-of-fold):

| Arm | Macro-F1 | Accuracy |
|---|---:|---:|
| Gradient boosting · Color + Texture | 0.697 | 78.7% |
| Gradient boosting · Color | 0.672 | 77.5% |
| Random forest · Color + Texture | 0.671 | 76.8% |
| Random forest · HOG + Color + Texture | 0.666 | 76.5% |

Chosen: **Gradient boosting on colour + texture features** (highest out-of-fold macro-F1 (0.6969)). Texture features added about
2.5 macro-F1 points over colour alone; HOG did not help.

**Test, once:**

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Deep learning | 90.9% | 0.838 |
| Classical ML | 78.1% | 0.634 |

Paired difference (classical − deep learning): -12.8 accuracy points (95% CI -18.9 to -7.2), macro-F1 -0.204 (-0.318 to -0.095).

## 7. Final evaluation

All models — classical ML, the original networks and the upgrade — are scored by the **same predictor code that the
API runs** (`backend/services/predictor.py`), in float32, on identical test photos (CPU/GPU parity checked). Reported:
accuracy, macro precision/recall/F1, weighted F1; 95% confidence intervals from 2,000 bootstrap resamples of whole
leaf groups; differences between models from a **paired** bootstrap on the same resamples; per-class metrics;
confusion matrices; calibration (ECE with 15 bins, NLL, Brier); selective accuracy at each model's threshold;
robustness to harmless changes and to simulated phone uploads; CPU latency (median, 8 threads).

## 8. Limitations

* One phone, two districts, single leaves on paper: performance on field photos, other cameras or regions is unknown.
* Potassium Deficiency has 9 photographs in total (2 in the test set) and Nitrogen Deficiency 45: their per-class
  scores are very uncertain, and macro-F1 on the full task inherits that uncertainty.
* 290 test photos: differences of a few points are within noise — read the confidence intervals and paired tests.
* Thresholds come from OOF predictions of single fold models but are applied to the 5-model average, which is
  typically less confident; the realised coverage on new photos can differ (measured on test in RESULTS.md).
* The OOD check was evaluated on out-of-scope leaf conditions and generic images, not on other crops or field scenes.
* Grad-CAM shows where a network looked, not why a leaf is diseased.

## 9. Reproducing

```bash
python scripts/01_build_index.py && python scripts/02_near_duplicates.py && python scripts/03_make_splits.py
python scripts/19_baseline_audit.py                     # audit of the original models (validation only)
python scripts/20_make_cv_folds.py
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in1k_base      # all 5 folds; likewise for
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in1k_robust    # 8class dinov2_s_robust and 5class
python scripts/21_train_cv.py --task 8class --recipe convnext_t_in22k_base --folds 0   # screening runs: fold 0 only
python scripts/21b_frozen_features.py --task 8class                            # frozen DINOv2 ViT-B baseline
python scripts/22_select_models.py --task 8class && python scripts/22_select_models.py --task 5class
python scripts/23_build_deployment.py
python scripts/24_cpu_inference.py                      # optional: CPU inference options
python scripts/25_final_benchmark.py                    # the test set, once
python scripts/26_package_release.py && python scripts/27_export_platform_data.py && python scripts/29_results_report.py
python scripts/28_e2e_check.py                          # React platform in a browser (after npm run build)
streamlit run streamlit_app.py --server.port 8510 & python scripts/30_streamlit_check.py   # Streamlit app in a browser
```

## 10. Serving

The REST API exports each network once to ONNX Runtime (about 1.6–2× faster on a CPU, outputs within 1e-6 of
PyTorch). The Streamlit app runs on small shared servers in **low-memory mode** (`TOMATO_LOWMEM=1`): all fold
networks of one architecture share a single float32 network and copy their memory-mapped float16 weights in just
before running — bit-identical predictions, 0.9 GB instead of 2.4 GB of private memory for the two version-2 models
(`tests/test_lowmem.py`).

Hardware used: one RTX 4050 Laptop GPU (6 GB); software versions in `requirements*.txt` and
`experiments/platform.json → environment`.
