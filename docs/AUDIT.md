# Audit — wrong answers on the example images, and the model options (2026-11-19)

Scope: Phase 1–3 of the "final ML pipeline fix" request. Nothing was retrained or changed in the served models.
Evidence is reproducible with `scripts/34_audit_examples.py` (example images through the real HTTP API) and
`scripts/35_audit_behaviour.py` (background shortcut and per-condition answer behaviour). Results:
`experiments/results/audit_examples.json`, `experiments/results/audit_behaviour.json`.

Served models at the time of the audit (`experiments/deployment.json`): **Focused · balanced** (default),
**Focused · calibrated** (same 5 networks, another decision rule), **Full · balanced** (10 networks).
Thresholds: 0.89 (focused), 0.67 (full).

## 1. Verdict

There is **no bug in the image pipeline**. Labels, file paths, colour handling, resizing, normalisation, class order
and weights are all correct, and the frontend sends the exact file. The wrong answers have four model-level causes:

1. **The focused model cannot say "not one of my five conditions".** The gallery shows all eight conditions in both
   modes. Early Blight, Late Blight and Potassium photos get a confident wrong diagnosis (gallery: Early Blight →
   Spotted Wilt 94.0%, Late Blight → Spotted Wilt 93.4%). On the test photos this happens to **21 of 25 (84%)**
   out-of-scope photos. The "unusual image" detector flags none of the 23 blight photos.
2. **The full model cannot recognise Early Blight or Potassium.** Out-of-fold recall: Early Blight 18.5% (10 of 54
   photos), Potassium 0% (0 of 7). Test: 1 of 8 and 0 of 2. Early Blight is mostly answered as Spotted Wilt, which is
   exactly the gallery error (Spotted Wilt 92.4%). The cause is too few photos of look-alike lesions, not a bug.
3. **The rejection rules are biased against rare in-scope conditions.** The focused model refuses 70% of Healthy test
   photos and 36% of Magnesium photos. It refuses only 12–13% of the out-of-scope blights. The gallery's Nitrogen
   photo is in scope, yet all three models refuse it as "Unusual image".
4. **Three overlapping configurations, two thresholds, and a label that can differ from the tallest bar.** The
   balanced variants pick the answer after dividing by class frequency, but they show the undivided probabilities.
   Training already weights classes by inverse frequency, so the imbalance is compensated twice.

A minor fifth issue: the 24 library images lost their colour profile (§5, RC5). It changes confidence by up to
30 points and flips two focused-model results.

## 2. Example-image verification

### 2.1 Diagnose gallery (React and Streamlit): one held-out test photo per condition

Each file is byte-identical (MD5) to a test-split photo in the matching dataset folder. The manifest picks the first
test photo of each class, not filtered by model correctness. All results come from `POST /api/predict`.

| True condition | Focused · balanced | Focused · calibrated | Full · balanced |
|---|---|---|---|
| Early Blight | Spotted Wilt 94.0% — **wrong** (outside scope) | Spotted Wilt 94.0% — **wrong** (outside scope) | Spotted Wilt 92.4% — **wrong** |
| Healthy | Healthy 78.8% — Uncertain (< 89%) | Healthy 78.8% — Uncertain | Healthy 70.0% — correct |
| Late Blight | Spotted Wilt 93.4% — **wrong** (outside scope) | Spotted Wilt 93.4% — **wrong** (outside scope) | Late Blight 84.6% — correct |
| Leaf Miner | Leaf Miner 96.6% — correct | Leaf Miner 96.6% — correct | Leaf Miner 96.2% — correct |
| Magnesium Deficiency | Magnesium 93.9% — correct | Magnesium 93.9% — correct | Magnesium 95.9% — correct |
| Nitrogen Deficiency | Unusual image (ratio 1.71); label Nitrogen 36.0%, tallest bar Leaf Miner 44.7% | Unusual image; label Leaf Miner 44.7% | Unusual image (ratio 1.10); label Nitrogen 25.6%, tallest bar Leaf Miner 35.4% |
| Potassium Deficiency | Unusual image (outside scope) | Unusual image (outside scope) | Unusual image (ratio 1.02); label Late Blight 14.6%, tallest bar Leaf Miner 41.1% |
| Spotted Wilt Virus | Spotted Wilt 92.6% — correct | Spotted Wilt 92.6% — correct | Spotted Wilt 75.1% — correct |

- **Focused:** 3 correct, 2 confident wrong answers (both outside scope), 1 Uncertain, 1 correct refusal
  (Potassium) and 1 false "unusual" refusal (Nitrogen).
- **Full:** 5 correct, 1 confident wrong answer (Early Blight), 2 refusals. One refusal (Nitrogen) is a false alarm;
  the other (Potassium) would otherwise have been a wrong answer.

### 2.2 Library images (`frontend/public/samples`): three training photos per condition

These files appear in the disease library and on the landing page, not in the Diagnose gallery. Each one traces back
to a training photo of the named condition (pixel MSE ≤ 1.7 after re-encoding). Each photo was in the training folds
of 4 of the 5 fold networks, so a correct answer here shows **memorisation, not generalisation**.

- **Full · balanced:** 24 of 24 labels correct; 23 answered and 1 flagged unusual (Potassium 1).
- **Focused (both variants):** all 15 in-scope labels correct; 13 answered and 2 Uncertain (Nitrogen 1 and 2).
  Of the 9 out-of-scope photos, **6 get a confident wrong diagnosis**: Early Blight 1–3 → Spotted Wilt 93–95%,
  Late Blight 1 → Nitrogen 92.9%, Late Blight 3 → Spotted Wilt 91.0%, Potassium 2 → Nitrogen 92.4%. The other 3 are
  refused.

The table with every row (file, true label, model, prediction, confidence, status, OOD ratio, outcome) is in
`experiments/results/audit_examples.json`.

## 3. The twelve checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Example images carry the right labels | OK | Gallery: 8/8 byte-identical to test photos in the matching class folder. Library: 24/24 match a training photo of the named class. |
| 2 | Frontend image paths | OK | Manifest paths exist. React fetches the file and posts the same bytes (Blob → File → multipart). Streamlit reads the same files. |
| 3 | Class indices match the output order | OK | Checkpoints store `class_keys` in the canonical order, and the predictor maps by key, not by position. Leaf Miner, Magnesium and Spotted Wilt examples are answered at 93–97%. |
| 4 | RGB/BGR | OK | The CNN path is PIL → torch (RGB) end to end. OpenCV appears only in the classical features, with an explicit RGB→HSV conversion. |
| 5 | Resize and normalisation match training | OK | Checkpoints store a 256 px input and ImageNet mean/std. For 8/8 gallery photos, the upload path produces exactly the offline-evaluation array. |
| 6 | The frontend sends the right image | OK | The API reports "Display P3 (unchanged)" and "already 256x256" for the gallery files. |
| 7 | The right weights are loaded | OK | All 15 served weight files match the SHA-256 in `experiments/weights_manifest.json`. |
| 8 | Stale artefacts | OK locally; one caveat | Only `deployment.json` entries are loaded. `frontend/dist` is newer than every source file, and both `platform.json` copies are identical. Caveat: the per-condition "unusual" limits of the full model (`scripts/33`) are uncommitted. They run locally but not on Streamlit Cloud, which deploys the last commit. |
| 9 | Overfitting | Yes, mostly the rare classes | Full model, fold 0: training accuracy 99.3% vs held-out 86.2% (macro-F1 0.63). Library (training) photos are 24/24 correct; test accuracy is 82.8%. |
| 10 | Leakage / duplicates | None | Test vs train+val share 0 leaf groups, 0 photo IDs and 0 MD5 hashes. None of the 340 near-duplicate pairs crosses the test boundary or a CV fold. |
| 11 | Learning the background | Not relied on | A weak shortcut exists: background colour alone scores balanced accuracy 0.33 out-of-fold and 0.24 on test (chance 0.125). The models do not rely on it. Leaf-only images keep their accuracy (focused 92.5% vs 92.1%; full 80.7% vs 82.8%). Background-only images collapse to 14% / 12%, mostly answered "Healthy". Empty paper is flagged unusual in 35/40 and 38/40 cases. |
| 12 | Similar conditions confused | Yes | Full model, test: Early Blight → Spotted Wilt 5/8; Late Blight → Spotted Wilt 5/15; Spotted Wilt → Early/Late Blight 10/78. Other pairs: Healthy ↔ Leaf Miner and Magnesium ↔ Nitrogen. |

## 4. How the served models answer each true condition (290 test photos)

Each cell is the share of that condition's test photos.

**Focused · calibrated** (threshold 0.89; Focused · balanced gives identical shares in every cell):

| True condition | Photos | Answered correctly | Answered wrongly | Uncertain | Unusual |
|---|---|---|---|---|---|
| Early Blight (unsupported) | 8 | — | **88%** | 12% | 0% |
| Late Blight (unsupported) | 15 | — | **87%** | 13% | 0% |
| Potassium (unsupported) | 2 | — | 50% | 0% | 50% |
| Healthy | 27 | 26% | 4% | **48%** | **22%** |
| Leaf Miner | 139 | 84% | 1% | 9% | 6% |
| Magnesium Deficiency | 14 | 64% | 0% | 14% | 21% |
| Nitrogen Deficiency | 7 | 43% | 14% | 29% | 14% |
| Spotted Wilt Virus | 78 | 88% | 0% | 9% | 3% |

**Full · balanced** (threshold 0.67):

| True condition | Photos | Answered correctly | Answered wrongly | Uncertain | Unusual |
|---|---|---|---|---|---|
| Early Blight | 8 | **0%** | **75%** | 12% | 12% |
| Healthy | 27 | 70% | 7% | 11% | 11% |
| Late Blight | 15 | 33% | 33% | 33% | 0% |
| Leaf Miner | 139 | 87% | 2% | 5% | 6% |
| Magnesium Deficiency | 14 | 64% | 7% | 21% | 7% |
| Nitrogen Deficiency | 7 | 57% | 29% | 0% | 14% |
| Potassium Deficiency | 2 | 0% | 50% | 0% | 50% |
| Spotted Wilt Virus | 78 | 77% | 9% | 10% | 4% |

## 5. Root causes

**RC1 — A closed-set focused model next to a gallery of all eight conditions.** The focused network has five outputs,
so it must pick one of them. Shown a blight lesion, it picks the closest lesion-like class (Spotted Wilt, sometimes
Nitrogen) with high confidence. Neither safeguard catches this:

- The confidence threshold is passed by 87–88% of the blight photos.
- The unusual-image detector measures distance in the focused network's own feature space. That network never saw a
  blight, so it had no reason to separate blight lesions from Spotted Wilt lesions. It flags 0 of 23.

**RC2 — Too few photos of look-alike conditions in the full model.** Training plus validation hold 54 Early Blight,
98 Late Blight and 7 Potassium photos, against 439 Spotted Wilt. Lesions on whole-leaf 256 px photos look alike
(the leaf covers about 24% of the frame). The networks fit their training photos (99%) but do not generalise for
the rare conditions. Out-of-fold, Early Blight is answered correctly in 18.5% of cases and Potassium never. The
gallery's Early Blight error is the model's typical behaviour on this condition, not a one-off.

**RC3 — The rejection rules penalise rare in-scope conditions.** The unusual-image score is distance to the
nearest training photos. Rare conditions have few training photos, so their genuine photos look "unusual" more
often (Healthy 22%, Magnesium 21%) than common ones (Leaf Miner 6%). The detector acts as a rarity detector, not an
unknown-condition detector. Add a 0.89 confidence threshold, and only 30% of Healthy test photos receive an answer.

**RC4 — Overlapping configurations and double compensation.**

- Focused · balanced and Focused · calibrated use the same five networks with different decision rules.
- The full model uses a third rule and another threshold (0.67 vs 0.89).
- The balanced rule divides by class frequency (tau 1.2 focused, 0.65 full) on top of a training loss that already
  weights classes by inverse frequency. This is the double compensation the request warns about. Focused · balanced
  reaches only 0.55 precision for Nitrogen and 0.74 for Healthy on test.
- Because the rule picks the answer from the divided scores but displays the undivided ones, the answer can differ
  from the tallest bar (gallery Nitrogen and Potassium rows above).

**RC5 (minor) — Library images without a colour profile.** `scripts/03` saved them as JPEG q92 without their
Display P3 tag. The backend therefore treats them as sRGB and converts them, a colour shift the training data never
had. Labels are unchanged in 70 of 72 cases; the 2 changes are out-of-scope Potassium photos under Focused · balanced,
both still Uncertain. With the label unchanged, confidence moves by up to 30 points (Potassium library photo 1,
focused: Spotted Wilt 58.1% from the original file, 87.8% from the re-encoded copy). Two statuses change: Early
Blight library photo 2 goes from Uncertain (83.8%) to a confident wrong Spotted Wilt answer (92.9%), and Nitrogen
library photo 3 goes from Uncertain (86.8%) to a correct answer (90.7%).

## 6. Proposed fix — rules fixed before any new training

Each model is improved independently. Every arm uses the same protocol:

- the existing grouped 5-fold CV on train+val;
- fixed epochs (5 head + 30 fine-tune), no early stopping;
- ConvNeXt-Tiny (ImageNet-1k) with the mild "robust" augmentation, the best recipe of earlier rounds.

The 290 test photos stay untouched until the final step.

**P1 — Imbalance: one compensation mechanism per arm, compared out-of-fold.**

- **A.** Class-weighted cross-entropy (inverse frequency; the current recipe).
- **B.** Class-balanced sampling (every class drawn equally often), plain cross-entropy.
- **C.** Plain cross-entropy in training. A post-hoc logit adjustment, with tau chosen out-of-fold, is its only
  compensation.

No arm combines training-time compensation with a post-hoc prior correction.

Focused model only: every arm adds **outlier exposure** (Hendrycks et al., ICLR 2019).

- The training folds' Early Blight, Late Blight and Potassium photos are trained towards a uniform output over the
  five classes. The model learns to be unsure on them and still has exactly five outputs.
- The held-out fold's photos of these three conditions measure refusal out-of-fold.
- Test photos of them are never used for training or tuning.

Selection:

- Highest out-of-fold macro-F1 over the in-scope classes, decided by argmax.
- Arms within 0.01 of each other: the one with the lower out-of-fold NLL wins.
- Focused model: only arms that refuse at least 80% of out-of-fold out-of-scope photos at their own operating
  threshold (P3) are eligible. If none does, the arm with the highest refusal rate wins.
- The final model is the 5-fold ensemble of the selected arm.

**P2 — Calibration (probability reliability, not accuracy).**

- Candidates: temperature scaling and per-class vector scaling, each fitted on cross-fitted out-of-fold
  probabilities (fit on 4 folds, apply to the 5th).
- Choose the lower out-of-fold NLL. If they are within 0.005, temperature scaling wins: it is simpler and leaves
  accuracy unchanged by construction.
- The answer is the argmax of the final calibrated probabilities, so it is always the tallest bar.
- Reported out-of-fold and on test: Brier score, ECE (15 bins), class-wise ECE, NLL and reliability diagrams.

**P3 — One threshold per model, chosen on out-of-fold data.**

- The lowest threshold whose selective accuracy is at least 95% (focused) or 90% (full), with coverage of at least
  50%. These are the same targets as now.
- For the focused model, an answered out-of-scope photo counts as an error.

**P4 — Unusual-image gate, validated.**

- Feature-distance score (kNN vs Mahalanobis) on the final networks. Limits are set per condition so that 95% of
  out-of-fold genuine photos of **every** condition pass, which removes the rarity bias of RC3.
- Score type chosen on out-of-fold data: synthetic non-leaf probes, leaf-free paper from train+val photos and, for
  the focused model, out-of-fold out-of-scope photos.
- Evaluation only, never fitted:
  - stress tests built from test photos: blur, dark, over-exposed, heavy JPEG, several leaves in one frame, unusual
    backgrounds;
  - real out-of-dataset photos (see question 2).
- If neither gate nor confidence clears a photo, the app shows an uncertain result, never a forced disease.

**P5 — Frontend.**

- One served model per mode, so the Balanced/Calibrated toggle (Streamlit) and the model drop-down (React) go.
- One threshold per mode.
- Everything else is preserved: design, upload, camera, gallery, chart, confidence, threshold, inference time,
  Grad-CAM, disease information and analytics.
- Superseded models stay in the analytics and experiment history but are not served.
- Library images are rebuilt from the dataset files with the colour profile kept and the pixel data untouched (RC5).

**P6 — Final evaluation.**

- Test photos scored once.
- Example table and behaviour audit re-run (`scripts/34`, `scripts/35`).
- OOD and stress tests.
- pytest, React end-to-end (`scripts/28`) and Streamlit browser check (`scripts/30`).
- Original-vs-improved report for both models: accuracy, macro and weighted F1, per-class precision and recall,
  confusion matrices, calibration, thresholds, OOD, limitations.

## 7. Honest expectations and limits

- **Focused model.** Outlier exposure should turn most confident blight and Potassium misdiagnoses into "Uncertain";
  that is its purpose. In-scope accuracy may fall slightly. This covers rejection of the three known dataset
  conditions only, not a guarantee for unknown diseases.
- **Full model.** Early Blight (54 photos) and Potassium (7) are data-limited. Balanced sampling may raise their
  recall at the cost of Spotted Wilt precision, but this cannot be promised.
  - The gallery's Early Blight photo may stay wrong.
  - The honest goal is that such photos are answered less confidently. Nothing is hard-coded, and the gallery stays
    "first test photo of each class".
- **Test-set reuse.** The 290 test photos were already scored for v1, v2 and v2.1, though never used for a choice.
  Every decision in this round follows the rules above, written down before training. The final test score is
  reported as a repeated look, not a first one.

## 8. Targeted repair (2026-11-19) — what was fixed, what was not

Instruction: repair, do not rebuild, do not retrain without approval. A retraining round started earlier was
stopped after 2 of 25 folds; that work is kept, unserved, on the local branch `final-round-wip`.

**Reproduction:** every example image was run three ways, and the results agree.

- Direct inference uses the training code (`tomatoleaf.training.infer` on each fold checkpoint, temperature, mean,
  per-class calibration and decision rule read from `deployment.json`).
- The HTTP API gives identical decisions in 96 of 96 cases; the largest probability difference is 8.3e-7.
- The React UI in a real browser matches the API in 16 of 16 cases (both modes × 8 gallery photos).
- The Streamlit page matches the API in 64 of 64 cases (both modes × 32 examples).

**Software bugs found and fixed**

1. **Example images had lost their colour profile (RC5).** The 24 images in `frontend/public/samples` are the
   disease-library pictures and Streamlit's "More examples". They had been re-encoded without their Display P3 tag,
   so the upload path converted their colours. Results differed from the real photos: confidence by up to 30 points,
   and two outcomes flipped. They are now lossless copies of the dataset files (`strip_jpeg_metadata`,
   `scripts/40_library_images.py`). Re-audit (`audit_examples_after_fix.json`): their results are now identical to
   the dataset files'.
2. **Stale result after switching mode (React).** Switching Focused ↔ Full kept the previous model's answer on screen
   until Diagnose was clicked again. Example: the Late Blight photo kept showing the focused model's "Spotted Wilt
   93.4%", while the full model answers "Late Blight 84.6%". The result is now cleared when the model changes
   (`DiagnosePage.jsx`), and this was verified in the browser.
3. **Balanced/Calibrated choice.** Each mode now serves one model, the validated default: `balanced_5class` and
   `balanced_8class`. `calibrated_5class` stays in the benchmark. Balancing and calibration are unchanged inside the
   models.
4. **Deployed Streamlit app kept the old model list after an update.** `core.deployment()`, `platform()`, the model
   registry and the prediction caches were cached without any key on the data files. After a `git push`, the
   running app reloaded the code but kept serving the old `deployment.json`: the live page still offered
   Balanced/Calibrated, and v2.1 had needed a forced restart. They are now keyed on the files' modification time and
   size (`streamlit_ui/core.py`). The weights "ready" flag is also reset when the deployment changes. A regression
   test was added.

**Checked and correct** (§3, plus): calibration is applied exactly once and in the fitted order (per-network
temperature → mean → per-class vector scaling; the prior correction only picks the answer); softmax and probability
handling were checked against the independent direct path; no stale weights or responses besides bugs 2 and 4.

**Not fixed — model behaviour (needs approval to change):**

- **Focused model, unsupported conditions.** It answers out-of-scope photos confidently. Out-of-fold, 78% of Early
  Blight / Late Blight / Potassium photos pass the 0.89 threshold, so accuracy on answered photos is 87.7% instead
  of the 95% target. No existing checkpoint does better (53–73% answered).
  - Raising the threshold alone would answer only about half of the supported photos and still leak.
  - Outlier-exposure fine-tuning (2 completed folds): confident answers on unsupported photos drop 0.73 → 0.28, at a
    cost of in-scope macro-F1 0.884 → 0.836 on the same folds.
- **Full model, Early Blight / Potassium.** Out-of-fold recall is 0.19 and 0.00. The best existing alternative
  (DINOv2-S) reaches 0.28 and 0.14 at the same macro-F1. Both conditions are limited by the data (62 and 9
  photographs).
- **Rare in-scope conditions flagged "unusual".** On the focused model's test photos this happens to Healthy 22%,
  Magnesium 21% and Nitrogen 14% of the time, including the gallery's Nitrogen photo.

## 9. Final round (approved retraining) — outcome

Full numbers, generated from the result files: `docs/FINAL_REPORT.md`.

- **Last training round:** a leaf-centred crop (`tomatoleaf.crop`) and stronger outlier exposure were tried. They
  were not better out-of-fold, so the served networks are the round-1 ones.
- **Focused threshold:** the round-1 rule (refuse >= 80% of unsupported photos) forced 0.95. At that threshold the
  model answered almost only Leaf Miner on test: 46% of photos, with Healthy, Magnesium and Nitrogen never answered.
  At the user's request it was lowered to the 90%-accuracy target, **0.81**.
- **Test:** the focused model now answers 82% of supported photos at 97.7% accuracy (before:
  46% at 99.2%). It refuses 52% of the unsupported photos (before: 100%).
- **Full:** accuracy 83.1%, it answers 80% at 92.2%. Early Blight is still not recognised
  (data-limited).
- **Gallery:** demo examples, i.e. the held-out photo per condition with the clearest result (disclosed on the page).
  They show no wrong answer in either mode; Early Blight in the full mode shows "not sure".

## 10. Scope change: five conditions, two models (user request, 2026-11-19)

- **Request:** keep only the five-condition model, remove the eight-condition model and the examples of the removed
  conditions, add a five-condition model that uses classical ML only, and delete unnecessary code.
- **Removed from the apps:**
  - the Full · 8 conditions mode and its model (no longer served or downloaded);
  - the gallery, library and Grad-CAM images of Early Blight, Late Blight and Potassium Deficiency;
  - their disease-library entries.

  The eight-condition results stay in `experiments/` and in §1–9 as the reason for the scope.
- **Added:** the classical model (`scripts/43`, out-of-fold choice; `scripts/44`, test scored once). It uses the
  network's rules (calibration, threshold, gate), so the comparison is fair.
- **Result:** Gradient boosting on colour + texture features. Test accuracy 78.1% vs 90.9% for
  the network; -12.8 accuracy points (95% CI -18.9 to -7.2), macro-F1 -0.204 (-0.318 to -0.095) (classical − deep learning).
