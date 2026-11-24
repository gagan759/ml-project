import { motion } from "framer-motion";
import {
  ArrowRight,
  BadgeCheck,
  BrainCircuit,
  FlaskConical,
  Gauge,
  Layers3,
  Microscope,
  ScanEye,
  ScanSearch,
  ShieldQuestion,
  Sparkles,
  Split,
  Upload,
} from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { ButtonLink, Card } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, classicalModel, defaultModel, modelsFor, taskClasses } from "../lib/analytics.js";
import { CLASSES } from "../lib/classes.js";
import { pct } from "../lib/format.js";

// The five conditions the apps diagnose (until the results file has loaded; afterwards tasks["5class"].classes is used).
const DIAGNOSED = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"];

const FEATURES = [
  { icon: ScanSearch, title: "Instant diagnosis", text: "Upload or snap a leaf photo; get the condition, a calibrated confidence and the full probability distribution in about a second on a CPU." },
  { icon: ShieldQuestion, title: "Knows when it doesn't know", text: "Below a validated confidence threshold it answers “Uncertain”; photos unlike the training data are flagged as “Unusual” instead of guessed." },
  { icon: BrainCircuit, title: "Explainable", text: "Grad-CAM heat maps show where the deep-learning network looked, blended live over exactly the image the model received." },
  { icon: FlaskConical, title: "Model laboratory", text: "Classical ML, the original networks and the upgraded ones on identical test photos — leaderboards, confusion matrices, significance tests." },
  { icon: Split, title: "Leakage-free by design", text: "Near-duplicate shots are grouped so no leaf appears in both training and test; models are chosen by cross-validation and the test set is scored once." },
  { icon: Microscope, title: "Dataset-faithful preprocessing", text: "Uploads are resized, colour-converted and compressed exactly like the dataset photos, removing a hidden source of inconsistent predictions." },
];

const STEPS = [
  { icon: Upload, title: "Photograph", text: "One detached leaf on plain paper." },
  { icon: Layers3, title: "Normalise", text: "Whole frame → 256×256, Display P3, JPEG q90 — like the training photos." },
  { icon: Gauge, title: "Predict", text: "Fine-tuned networks with test-time augmentation — or classical ML on colour, texture and shape features — then calibration." },
  { icon: ScanEye, title: "Check & explain", text: "Confidence threshold, similarity to training photos, Grad-CAM." },
];

function HeroVisual({ hero }) {
  // A real, precomputed result for a held-out test photo (experiments/platform.json -> hero); nothing is invented.
  const bars = (hero?.probabilities || []).slice(0, 3);
  const meta = hero ? CLASSES.find((c) => c.key === hero.pred_key) : null;
  return (
    <div className="relative mx-auto w-full max-w-md">
      <div className="absolute -inset-6 -z-10 rounded-[2.5rem] bg-gradient-to-br from-leaf-400/25 via-transparent to-tomato-400/20 blur-2xl" />
      <Card className="overflow-hidden shadow-pop">
        <div className="relative aspect-square overflow-hidden bg-black">
          <img src={hero?.input || "/samples/leaf_miner-1.jpg"} alt="Held-out test photo of a tomato leaf" className="size-full object-cover" />
          {hero?.heatmap && <img src={hero.heatmap} alt="Grad-CAM heat map" className="absolute inset-0 size-full object-cover opacity-70" />}
          <span className="absolute inset-x-0 top-0 h-10 animate-scan bg-gradient-to-b from-transparent via-leaf-300/40 to-transparent" />
          {hero?.heatmap && <span className="absolute left-3 top-3 rounded-full bg-black/55 px-2.5 py-1 text-[11px] font-medium text-white backdrop-blur">Grad-CAM</span>}
        </div>
        {hero && meta ? (
          <div className="p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-muted">Diagnosis</p>
                <p className="font-display text-xl font-semibold text-ink">{meta.name}</p>
              </div>
              <span className="rounded-full bg-leaf-500/10 px-2.5 py-1 text-xs font-semibold text-leaf-700 ring-1 ring-leaf-500/20 dark:text-leaf-300">
                {pct(hero.confidence, 0)} confident
              </span>
            </div>
            <div className="mt-4 space-y-2.5">
              {bars.map((b, i) => {
                const m = CLASSES.find((c) => c.key === b.key);
                return (
                  <div key={b.key}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="text-ink-2">{m.name}</span>
                      <span className="font-mono text-muted">{pct(b.p, 1)}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-surface-3">
                      <motion.div className="h-full rounded-full" style={{ background: m.color }} initial={{ width: 0 }} animate={{ width: `${Math.max(1, b.p * 100)}%` }} transition={{ duration: 1, delay: 0.4 + i * 0.15 }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="mt-4 text-[11px] text-muted">
              Real output of {hero.model_name} for a held-out test photo (true label: {CLASSES.find((c) => c.key === hero.true_key)?.name}).
            </p>
          </div>
        ) : (
          <div className="p-5 text-sm text-muted">Upload a leaf photo to see a calibrated diagnosis and Grad-CAM.</div>
        )}
      </Card>
    </div>
  );
}

export default function LandingPage() {
  useDocumentTitle(null);
  const { analytics } = useData();
  const d5 = defaultModel(analytics, TASK);
  const cm = classicalModel(analytics, TASK);
  const nModels = modelsFor(analytics, TASK).length;
  const keys = taskClasses(analytics, TASK);
  const diagnosed = CLASSES.filter((c) => (keys.length ? keys : DIAGNOSED).includes(c.key));

  const stats = [
    { label: "Test accuracy · deep learning", value: pct(d5?.test?.accuracy), hint: `${analytics?.tasks?.[TASK]?.test_photos ?? "—"} unseen photos` },
    { label: "Accuracy when it answers", value: pct(d5?.selective?.test?.accuracy_kept ?? d5?.selective?.accuracy_kept), hint: "selective prediction" },
    { label: "Test accuracy · classical ML", value: pct(cm?.test?.accuracy), hint: "no neural network, same test photos" },
    { label: "Models benchmarked", value: nModels || "—", hint: "on identical test photos" },
  ];

  return (
    <div>
      <section className="relative overflow-hidden">
        <div className="bg-grid absolute inset-0 -z-10 [mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_70%)]" />
        <div className="mx-auto grid grid-cols-1 max-w-7xl items-center gap-14 px-4 pb-20 pt-14 sm:px-6 lg:grid-cols-[1.1fr_0.9fr] lg:px-8 lg:pb-28 lg:pt-20">
          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <span className="inline-flex items-center gap-2 rounded-full border border-leaf-500/25 bg-leaf-500/[0.07] px-3 py-1 text-xs font-semibold text-leaf-700 dark:text-leaf-300">
              <Sparkles className="size-3.5" aria-hidden /> Agricultural AI · research-grade evaluation
            </span>
            <h1 className="mt-6 font-display text-4xl font-semibold leading-[1.08] tracking-tight text-ink sm:text-5xl lg:text-[3.6rem]">
              Spot tomato leaf disease
              <span className="block bg-gradient-to-r from-leaf-600 via-leaf-500 to-sun-500 bg-clip-text text-transparent">before it spreads.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted">
              Tomato Care diagnoses five tomato leaf conditions from a single photo — with a deep-learning or a classical machine-learning model, calibrated confidence, an honest “I'm not sure”, out-of-distribution warnings and visual explanations.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <ButtonLink to="/diagnose" size="lg" iconRight={ArrowRight}>
                Diagnose a leaf
              </ButtonLink>
              <ButtonLink to="/lab" size="lg" variant="secondary" icon={FlaskConical}>
                Explore the model lab
              </ButtonLink>
            </div>
            <div className="mt-10 flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted">
              {["Leakage-free evaluation", "Cross-validated model choice", "Runs on a CPU"].map((t) => (
                <span key={t} className="flex items-center gap-1.5">
                  <BadgeCheck className="size-4 text-leaf-600" aria-hidden /> {t}
                </span>
              ))}
            </div>
          </motion.div>
          <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.7, delay: 0.15 }}>
            <HeroVisual hero={analytics?.hero} />
          </motion.div>
        </div>
      </section>

      <section className="border-y border-line bg-surface/60">
        <div className="mx-auto grid max-w-7xl grid-cols-2 divide-line px-4 sm:px-6 lg:grid-cols-4 lg:divide-x lg:px-8">
          {stats.map((s) => (
            <div key={s.label} className="px-2 py-8 lg:px-8">
              <p className="font-display text-3xl font-semibold tracking-tight text-ink tabular sm:text-4xl">{s.value}</p>
              <p className="mt-2 text-sm font-medium text-ink-2">{s.label}</p>
              <p className="text-xs text-muted">{s.hint}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-leaf-600 dark:text-leaf-400">Platform</p>
          <h2 className="mt-2 font-display text-3xl font-semibold tracking-tight text-ink">Everything from a photo to a defensible answer</h2>
        </div>
        <div className="mt-10 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ icon: Icon, title, text }) => (
            <div key={title}>
              <Card className="h-full p-6" interactive>
                <span className="grid size-10 place-items-center rounded-xl bg-leaf-500/10 text-leaf-600 dark:text-leaf-400">
                  <Icon className="size-5" aria-hidden />
                </span>
                <h3 className="mt-4 font-display text-lg font-semibold text-ink">{title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{text}</p>
              </Card>
            </div>
          ))}
        </div>
      </section>

      <section className="bg-leaf-950 text-white">
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-leaf-300">How it works</p>
          <h2 className="mt-2 max-w-2xl font-display text-3xl font-semibold tracking-tight">From phone photo to calibrated diagnosis</h2>
          <ol className="mt-12 grid grid-cols-1 gap-8 md:grid-cols-4">
            {STEPS.map(({ icon: Icon, title, text }, i) => (
              <li key={title} className="relative">
                <span className="grid size-12 place-items-center rounded-2xl bg-white/10 ring-1 ring-white/15">
                  <Icon className="size-5 text-leaf-300" aria-hidden />
                </span>
                <p className="mt-4 font-mono text-xs text-leaf-300">0{i + 1}</p>
                <p className="mt-1 font-display text-lg font-semibold">{title}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-white/70">{text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-10 lg:grid-cols-2 lg:items-center">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-leaf-600 dark:text-leaf-400">Five conditions</p>
            <h2 className="mt-2 font-display text-3xl font-semibold tracking-tight text-ink">Viral, pest and nutrient problems — and healthy leaves</h2>
            <p className="mt-4 text-[15px] leading-relaxed text-muted">
              Built on {analytics?.dataset?.photos ? analytics.dataset.photos.toLocaleString() : "2,103"} photographs of tomato leaves from Rajasthan, India (Tomato-Village dataset). Tomato Care diagnoses the five conditions with enough photos to be recognised reliably, with two models for the same five: a deep-learning network (the default) and a classical machine-learning model that uses no neural network.
            </p>
            <ButtonLink to="/library" variant="secondary" className="mt-6" iconRight={ArrowRight}>
              Open the disease library
            </ButtonLink>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {diagnosed.map(({ key, name, color, icon: Icon }) => (
              <a key={key} href={`/library/${key}`} className="group overflow-hidden rounded-2xl border border-line bg-surface transition hover:-translate-y-0.5 hover:shadow-pop">
                <div className="aspect-square overflow-hidden">
                  <img src={`/samples/${key}-2.jpg`} alt={name} className="size-full object-cover transition duration-500 group-hover:scale-105" loading="lazy" />
                </div>
                <p className="flex items-center gap-1.5 px-2.5 py-2 text-xs font-medium text-ink">
                  <Icon className="size-3.5 shrink-0" style={{ color }} aria-hidden />
                  <span className="truncate">{name}</span>
                </p>
              </a>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-24 sm:px-6 lg:px-8">
        <Card className="relative overflow-hidden p-8 sm:p-12">
          <div className="absolute -right-24 -top-24 size-72 rounded-full bg-leaf-500/15 blur-3xl" />
          <div className="relative flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div className="max-w-xl">
              <h2 className="font-display text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Numbers you can defend</h2>
              <p className="mt-3 text-[15px] leading-relaxed text-muted">
                Every figure on this site is computed from held-out photos the models never saw during training or selection, with confidence intervals and paired significance tests. Read how in the methodology.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <ButtonLink to="/methodology" variant="secondary" icon={Microscope}>
                Methodology
              </ButtonLink>
              <ButtonLink to="/dashboard" iconRight={ArrowRight}>
                Open dashboard
              </ButtonLink>
            </div>
          </div>
        </Card>
      </section>
    </div>
  );
}
