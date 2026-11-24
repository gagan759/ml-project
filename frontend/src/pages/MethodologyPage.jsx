import { Link } from "react-router-dom";
import { AlertTriangle, BookMarked, Brain, Camera, CheckCircle2, Crop, Database, FlaskConical, Gauge, Layers3, Microscope, Palette, ScanEye, Split, Target } from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, Callout, Card, CardHeader, PageHeader, Skeleton } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, classicalModel, comparisonsFor, defaultModel, modelName, recipesFor, servedName, taskClasses, taskPhotos } from "../lib/analytics.js";
import { pct, signedPct } from "../lib/format.js";

const TOC = [
  ["dataset", "Dataset"],
  ["leakage", "Leakage-free split"],
  ["preprocessing", "Preprocessing audit"],
  ["protocol", "Model selection protocol"],
  ["models", "Models and training"],
  ["classical-ml", "Classical ML model"],
  ["uncertainty", "Calibration & uncertainty"],
  ["evaluation", "Final evaluation"],
  ["limitations", "Limitations"],
  ["reproduce", "Reproducibility"],
];

function Section({ id, icon: Icon, title, children }) {
  return (
    <section id={id} className="scroll-mt-24">
      <Card>
        <CardHeader icon={Icon} title={title} />
        <div className="space-y-4 p-5 text-[15px] leading-relaxed text-ink-2 sm:p-6">{children}</div>
      </Card>
    </section>
  );
}

const andList = (xs) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);
const GATE_NAME = { knn: "k-nearest-neighbour", maha: "Mahalanobis" };

function Table({ head, rows }) {
  return (
    <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
      <table className="w-full min-w-[560px] text-sm">
        <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
          <tr>
            {head.map((h, i) => (
              <th key={h} className={`px-4 py-2.5 font-medium ${i ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((r, i) => (
            <tr key={i} className="bg-surface">
              {r.map((c, j) => (
                <td key={j} className={`px-4 py-2.5 ${j ? "text-right font-mono tabular" : "text-ink"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MethodologyPage() {
  useDocumentTitle("Methodology");
  const { analytics } = useData();
  const ds = analytics?.dataset;
  const audit = analytics?.audit;
  const recipes = recipesFor(analytics, TASK);
  const dl = defaultModel(analytics, TASK);
  const ml = classicalModel(analytics, TASK);
  const ood = analytics?.ood;
  const keys = taskClasses(analytics, TASK);
  const diagnosed = (ds?.classes || []).filter((c) => keys.includes(c.key)).map((c) => c.name);
  const notDiagnosed = (ds?.classes || []).filter((c) => keys.length && !keys.includes(c.key)).map((c) => c.name);
  const cvPhotos = taskPhotos(analytics, TASK, "cv");
  const cl = analytics?.classical;
  const chosenArm = cl?.arms?.find((a) => a.chosen) || cl?.arms?.find((a) => a.arm === cl?.chosen);
  const mlGate = ood?.final?.[ml?.key]?.method;
  const dlGate = ood?.final?.[TASK];
  const headline = comparisonsFor(analytics, TASK).filter((c) => (c.headline || c.ml_vs_dl) && c.f1_ci);

  return (
    <div>
      <PageHeader
        eyebrow="Scientific protocol"
        title="Methodology"
        description="How the data were checked, how models were trained and chosen without touching the test set, and how every reported number was computed. Written so that each step can be questioned and reproduced."
      />
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[220px_minmax(0,1fr)]">
        <nav className="hidden lg:block" aria-label="On this page">
          <div className="sticky top-8">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-muted">On this page</p>
            <ul className="space-y-1 border-l border-line">
              {TOC.map(([id, label]) => (
                <li key={id}>
                  <a href={`#${id}`} className="-ml-px block border-l border-transparent py-1 pl-4 text-sm text-muted hover:border-leaf-500 hover:text-ink">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        <div className="flex min-w-0 flex-col gap-6">
          {!analytics && <Skeleton className="h-96 rounded-2xl" />}

          <Section id="dataset" icon={Database} title="1 · Dataset">
            <p>
              Tomato-Village, Variant-a (Gehlot et al., <em>Multimedia Systems</em>, 2023): {ds ? ds.files.toLocaleString() : "4,525"} JPEG files of single tomato leaves photographed on paper in Jodhpur and Jaipur, Rajasthan, in eight classes. Only{" "}
              {ds ? ds.photos.toLocaleString() : "2,103"} are distinct photographs: in five classes the authors added seven augmented copies of each photo.
            </p>
            <p>
              Classes are very unbalanced — from {ds?.classes ? Math.max(...ds.classes.map((c) => c.photos)) : "1,020"} Leaf Miner photos down to {ds?.classes ? Math.min(...ds.classes.map((c) => c.photos)) : 9} Potassium Deficiency photos — so we report <strong>macro-F1</strong> (every class weighted equally) next to accuracy, and use class-weighted losses.
            </p>
            {diagnosed.length > 0 && (
              <p>
                Tomato Care diagnoses <strong>{diagnosed.length} of these classes</strong>: {andList(diagnosed)}.
                {notDiagnosed.length > 0 && ` ${andList(notDiagnosed)} were too hard to recognise reliably and are not diagnosed: the models should refuse such photos, and their test photos check that they do.`}
              </p>
            )}
          </Section>

          <Section id="leakage" icon={Split} title="2 · A leakage-free split">
            <p>
              The official split places augmented copies of the same photograph in train, validation and test: {ds?.leakage ? pct(ds.leakage.test_with_train_sibling, 1) : "61%"} of official test images have a sibling in training. Re-shots of the same leaf a few seconds apart (found with image embeddings and capture times) make this worse.
            </p>
            <p>
              We merged copies and re-shots into <strong>{ds?.groups?.toLocaleString() ?? "1,786"} leaf groups</strong> (union–find) and split whole groups with StratifiedGroupKFold: {ds ? `${ds.split.train} / ${ds.split.val} / ${ds.split.test}` : "1,502 / 311 / 290"} original photos for train / validation / test. No leaf appears on both sides of any split.
            </p>
            {ds?.leakage?.benchmark && (
              <Table
                head={["Model", "Official (leaky) accuracy", "Leakage-free accuracy"]}
                rows={ds.leakage.benchmark.map((b) => [b.model, pct(b.official_accuracy), pct(b.clean_accuracy)])}
              />
            )}
            <p className="text-sm text-muted">The leaky split inflates results because the model has effectively seen the test photos; all numbers on this site use the leakage-free split.</p>
          </Section>

          <Section id="preprocessing" icon={Microscope} title="3 · Preprocessing audit: why predictions on real photos were inconsistent">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              {[
                { icon: Crop, title: "Shape", text: "Every dataset photo was taken at 3468×4624 (or 4624×3468) and squashed to 256×256. Registering the authors' rotated copies shows the matching (4/3)² anisotropy." },
                { icon: Palette, title: "Colour", text: "The files are Display P3 encoded (embedded ICC profile). sRGB uploads therefore reached the model with different colour values — and colour is how deficiencies show." },
                { icon: Camera, title: "Compression", text: "Dataset files are JPEG quality 90 with 4:2:0 chroma; downscaled phone photos are much cleaner, another shift." },
              ].map(({ icon: Icon, title, text }) => (
                <div key={title} className="rounded-xl bg-surface-2 p-4">
                  <Icon className="size-5 text-leaf-600" aria-hidden />
                  <p className="mt-2 font-semibold text-ink">{title}</p>
                  <p className="mt-1 text-sm text-muted">{text}</p>
                </div>
              ))}
            </div>
            <p>
              The original app <strong>centre-cropped a square</strong> (leaves 25% narrower than in training, top and bottom cut off) and ignored colour profiles. The new upload path reproduces the dataset's own procedure — whole frame, squash to 256×256, convert into Display P3, JPEG q90 — and leaves dataset images bit-for-bit unchanged.
            </p>
            {audit?.preprocessing?.length > 0 && (
              <>
                <p>Effect, measured on the validation photos turned back into full-size phone photos (original deployed model):</p>
                <Table head={["Simulated upload", "Accuracy", "Macro-F1", "Predictions changed"]} rows={audit.preprocessing.map((r) => [r.label, pct(r.accuracy), pct(r.f1_macro), pct(r.flip_rate)])} />
              </>
            )}
          </Section>

          <Section id="protocol" icon={Target} title="4 · Model selection without touching the test set">
            <p>
              The original study chose models on one validation split of 311 photos — with a single Potassium and six Nitrogen photos, one photo moves validation macro-F1 by up to 0.125, so the choice was largely noise. The upgrade uses <strong>grouped, stratified 5-fold cross-validation</strong> on all {analytics?.dataset?.cv?.photos?.toLocaleString() ?? "1,813"} train+validation photos
              {cvPhotos != null ? ` (${cvPhotos.toLocaleString()} of them show the five diagnosed conditions)` : ""}.
            </p>
            <ul className="space-y-2">
              {[
                "Every recipe trains for a fixed number of epochs — no early stopping — so no epoch is chosen by looking at the held-out fold.",
                `Each photo gets an out-of-fold prediction from a model that never saw its leaf group; recipes, ensembles and classical classifiers are ranked on the ${cvPhotos != null ? `${cvPhotos.toLocaleString()} ` : ""}out-of-fold predictions for the five conditions.`,
                "Temperatures, the selective-prediction threshold and the out-of-distribution threshold are fitted on out-of-fold outputs as well.",
                "The chosen models are retrained on all train+validation photos and the 290 test photos are scored once, at the end.",
              ].map((t) => (
                <li key={t} className="flex gap-2.5">
                  <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-leaf-600" aria-hidden />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
            {analytics?.cv?.selection_rule && <Callout tone="slate" title="Pre-registered selection rule">{analytics.cv.selection_rule}</Callout>}
            <Callout tone="sun" icon={AlertTriangle} title="Honest caveat">
              The same 290 test photos were already used to report the original study's results, so they are not perfectly “fresh”. No decision in the upgrade used them; they were scored once after every choice had been fixed and logged.
            </Callout>
          </Section>

          <Section id="models" icon={Brain} title="5 · Models and training recipes">
            <p>
              The upgrade compares stronger pretrained backbones — ConvNeXt-Tiny pretrained on ImageNet-22k, EfficientNetV2-S (ImageNet-21k) and a DINOv2 self-supervised ViT-S/14 — against the original ConvNeXt-Tiny (ImageNet-1k) recipe, all on 256×256 inputs.
            </p>
            <p>
              The <strong>robust recipe</strong> adds: all eight flips/rotations during training (so every test-time-augmentation view is in-distribution), aspect-ratio jitter, JPEG re-compression and mild blur (uploads differ from the dataset files), layer-wise learning-rate decay, an exponential moving average of the weights, and gradient clipping. Hue is never jittered: colour is the diagnostic signal for nutrient deficiencies.
            </p>
            {recipes.length > 0 && (
              <Table
                head={["Recipe (5 conditions)", "OOF macro-F1", "+ TTA", "OOF accuracy + TTA"]}
                rows={recipes.map((r) => [r.label, pct(r.f1_1), pct(r.f1_8), pct(r.acc_8)])}
              />
            )}
            <div id="classical-ml" className="scroll-mt-24 rounded-xl bg-surface-2 p-4">
              <p className="flex items-center gap-2 font-semibold text-ink">
                <Layers3 className="size-4 text-sun-600" aria-hidden /> The classical ML model — no neural network
              </p>
              <p className="mt-2 text-sm leading-relaxed">
                The second served model diagnoses the same five conditions from handcrafted features of the same 256×256 image: <strong>colour</strong> (HSV histograms and L*a*b* mean, spread and skewness over the leaf pixels), <strong>texture</strong> (LBP histograms and GLCM statistics such as contrast and homogeneity) and <strong>shape</strong> (HOG, reduced by PCA inside each training fold). Logistic regression, an RBF-kernel SVM, a random forest and gradient boosting were each tried on three feature sets (colour; colour + texture; HOG + colour + texture), tuned by nested grouped cross-validation, then selected, calibrated and given a threshold on the same five folds with the same pre-registered rules as the network. The test photos were scored once, at the end.
              </p>
              {chosenArm && (
                <p className="mt-2 text-sm leading-relaxed">
                  Chosen{cl.arms.length > 1 ? ` from ${cl.arms.length} classifier × feature-set combinations` : ""}: <strong>{chosenArm.classifier} · {chosenArm.feature_set}</strong>
                  {cl.why ? ` — ${cl.why}; out-of-fold accuracy ${pct(chosenArm.oof?.accuracy)}.` : ` — out-of-fold macro-F1 ${pct(chosenArm.oof?.f1_macro)}, accuracy ${pct(chosenArm.oof?.accuracy)}.`}{" "}
                  <Link to="/lab" className="font-medium text-leaf-700 hover:underline dark:text-leaf-300">
                    Every combination is in the Model lab.
                  </Link>
                </p>
              )}
            </div>
          </Section>

          <Section id="uncertainty" icon={ScanEye} title="6 · Calibration, “Uncertain” and “Unusual image”">
            <p>
              <strong>Calibration</strong> rescales each model's log-probabilities — one temperature T, or a scale and bias per class (vector scaling), whichever has the lower out-of-fold log-loss — so a stated 90% confidence is right about 90% of the time. Below a <strong>confidence threshold</strong> — the lowest threshold whose out-of-fold accuracy on answered photos reaches the target — the app answers “Uncertain”. Both models follow these rules, each on its own out-of-fold predictions.
            </p>
            <p>
              Photos far from every training photo in the network's feature space ({dlGate?.method ? `${GATE_NAME[dlGate.method] || dlGate.method} distance` : ood?.method || "k-nearest-neighbour distance"}) are reported as <strong>“Unusual image”</strong>. The threshold keeps {pct(dlGate?.keep ?? ood?.keep ?? 0.95, 0)} of in-distribution out-of-fold photos; it is tested on photos of the dataset's conditions that the app does not diagnose and on 30 non-leaf images.
              {mlGate && ` The classical model uses the same rule with ${GATE_NAME[mlGate] || mlGate} distance in its own colour + texture feature space.`}
            </p>
            {ood?.test && (
              <Table
                head={["Out-of-distribution check (test)", "Flagged"]}
                rows={[
                  ["In-scope test photos (false alarms)", pct(ood.test.flagged_in)],
                  ["Out-of-scope conditions (near-OOD)", pct(ood.test.flagged_near)],
                  ["Non-leaf images, held-out half (far-OOD)", pct(ood.test.flagged_far)],
                ]}
              />
            )}
          </Section>

          <Section id="evaluation" icon={Gauge} title="7 · Final evaluation">
            <p>
              All models — classical ML, the original networks and the upgrade — are scored by the same code that serves the API (CPU, float32) on identical test photos. Confidence intervals resample whole leaf groups (2,000 bootstrap resamples); differences between models use a paired bootstrap on the same resamples.
            </p>
            {dl && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {[
                  [dl, `${servedName(dl)} · deep learning`, "leaf"],
                  [ml, `${servedName(ml)} · classical ML`, "sun"],
                ]
                  .filter(([m]) => m?.test)
                  .map(([m, label, tone]) => (
                    <div key={m.key} className="rounded-xl bg-surface-2 p-4">
                      <Badge tone={tone}>{label}</Badge>
                      <p className="mt-2 font-semibold text-ink">{m.name}</p>
                      <p className="mt-1 text-sm text-muted">
                        accuracy {pct(m.test.accuracy)} · macro-F1 {pct(m.test.f1_macro)} · ECE {m.test.ece?.toFixed(3) ?? "—"}
                      </p>
                    </div>
                  ))}
              </div>
            )}
            {headline.map((c) => (
              <p key={`${c.a}-${c.b}`} className="text-sm">
                <strong>{c.a_name || modelName(analytics, c.a)}</strong> vs <strong>{c.b_name || modelName(analytics, c.b)}</strong>: macro-F1 {signedPct(c.f1_diff)} (95% CI {signedPct(c.f1_ci[0])} to {signedPct(c.f1_ci[1])}), accuracy {signedPct(c.accuracy_diff)}.
              </p>
            ))}
          </Section>

          <Section id="limitations" icon={AlertTriangle} title="8 · Limitations">
            <ul className="space-y-2">
              {(analytics?.limitations || [
                "One phone, two districts, leaves on paper: performance on field photos, other cameras or other regions is unknown.",
                "Five conditions only: Early Blight, Late Blight and Potassium Deficiency are in the dataset but are not diagnosed.",
                "The test set is small (265 photos of the five conditions), so differences of a few points are within noise — see the confidence intervals.",
              ]).map((t) => (
                <li key={t} className="flex gap-2.5">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0 text-sun-600" aria-hidden />
                  <span>{t}</span>
                </li>
              ))}
            </ul>
          </Section>

          <Section id="reproduce" icon={BookMarked} title="9 · Reproducibility">
            <p>
              Every number comes from versioned scripts with fixed seeds (<code className="font-mono text-sm">scripts/01…44</code>); results are written to <code className="font-mono text-sm">experiments/</code> and this page reads them directly. See the documentation for the exact command sequence, hardware and software versions.
            </p>
            <div className="flex flex-wrap gap-2">
              <Badge tone="slate" icon={FlaskConical}>PyTorch {analytics?.environment?.torch || "2.7"}</Badge>
              <Badge tone="slate">timm {analytics?.environment?.timm || "1.0"}</Badge>
              <Badge tone="slate">GPU: {analytics?.environment?.gpu || "RTX 4050 Laptop"}</Badge>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}
