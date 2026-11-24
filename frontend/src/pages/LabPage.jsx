import { useEffect, useMemo, useState } from "react";
import { ArrowDownUp, Beaker, Check, FlaskConical, Gauge, Layers, Play, Scale, ShieldCheck, Sparkles, Timer } from "lucide-react";
import { FoldStrip, LatencyScatter, PerClassChart, RobustnessChart } from "../components/charts/charts.jsx";
import ConfusionMatrix from "../components/charts/ConfusionMatrix.jsx";
import CompareTable from "../components/diagnose/CompareTable.jsx";
import SampleGallery from "../components/diagnose/SampleGallery.jsx";
import FinalPipeline from "../components/lab/FinalPipeline.jsx";
import UploadPanel from "../components/diagnose/UploadPanel.jsx";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, Button, Callout, Card, CardHeader, ErrorState, InfoTip, PageHeader, Segmented, Skeleton } from "../components/ui/primitives.jsx";
import { SERIES } from "../components/charts/theme.js";
import { useData } from "../context/DataContext.jsx";
import { usePrediction } from "../context/PredictionContext.jsx";
import { TASK, comparisonsFor, defaultModel, isClassicalModel, modelColor, modelName, modelsFor, recipesFor, taskClasses, taskInfo, taskPhotos } from "../lib/analytics.js";
import { ci, compactParams, cx, ms, num, pct, signedPct } from "../lib/format.js";

const COLUMNS = [
  { key: "f1_macro", label: "Macro-F1", get: (m) => m.test?.f1_macro, fmt: (v) => pct(v), tip: "Mean F1 over classes: every class counts equally, so rare diseases matter as much as common ones." },
  { key: "accuracy", label: "Accuracy", get: (m) => m.test?.accuracy, fmt: (v) => pct(v) },
  { key: "f1_weighted", label: "Weighted F1", get: (m) => m.test?.f1_weighted, fmt: (v) => pct(v) },
  { key: "ece", label: "ECE", get: (m) => m.test?.ece, fmt: (v) => num(v, 3), lowerBetter: true, tip: "Expected calibration error: gap between confidence and accuracy (lower is better)." },
  { key: "cv", label: "CV / val F1", get: (m) => m.cv?.f1_macro ?? m.val?.f1_macro, fmt: (v) => pct(v), tip: "Upgrade models: pooled out-of-fold macro-F1 from 5-fold cross-validation on train+val. Original models: single validation split." },
  { key: "latency", label: "CPU", get: (m) => m.latency_ms, fmt: (v) => ms(v), lowerBetter: true },
];

function Leaderboard({ models, sort, setSort, selected, toggle, highlight, served = [] }) {
  const col = COLUMNS.find((c) => c.key === sort.key);
  const rows = [...models].sort((a, b) => {
    const va = col.get(a);
    const vb = col.get(b);
    if (va == null) return 1;
    if (vb == null) return -1;
    return sort.dir * (vb - va);
  });
  return (
    <div className="overflow-x-auto scrollbar-thin">
      <table className="w-full min-w-[880px] text-sm">
        <thead className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
          <tr>
            <th className="w-10 px-4 py-3" />
            <th className="px-2 py-3 font-medium">Model</th>
            {COLUMNS.map((c) => (
              <th key={c.key} className="px-3 py-3 text-right font-medium">
                <button className={cx("inline-flex items-center gap-1 hover:text-ink", sort.key === c.key && "text-ink")} onClick={() => setSort({ key: c.key, dir: sort.key === c.key ? -sort.dir : c.lowerBetter ? -1 : 1 })}>
                  {c.label}
                  <ArrowDownUp className="size-3" aria-hidden />
                </button>
                {c.tip && <InfoTip content={c.tip} className="ml-1 align-middle" />}
              </th>
            ))}
            <th className="px-4 py-3 text-right font-medium">Params</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((m) => {
            const on = selected.includes(m.key);
            return (
              <tr key={m.key} className={cx("transition", on ? "bg-leaf-500/[0.05]" : "hover:bg-surface-2/60")}>
                <td className="px-4 py-3">
                  <button
                    onClick={() => toggle(m.key)}
                    className={cx("grid size-5 place-items-center rounded-md border transition", on ? "border-leaf-600 bg-leaf-600 text-white" : "border-line-strong")}
                    aria-pressed={on}
                    aria-label={`Compare ${m.name}`}
                  >
                    {on && <Check className="size-3.5" aria-hidden />}
                  </button>
                </td>
                <td className="px-2 py-3">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="size-2 rounded-full" style={{ background: modelColor(m) }} />
                    <span className="font-medium text-ink">{m.name}</span>
                    {m.key === highlight && <Badge tone="leaf">default</Badge>}
                    {m.key !== highlight && served.includes(m.key) && <Badge tone="sun">served</Badge>}
                    {m.generation === "upgrade" && <Sparkles className="size-3.5 text-leaf-600" aria-label="upgraded" />}
                  </div>
                  <p className="mt-0.5 text-xs text-muted">
                    {m.family} · {m.generation === "upgrade" ? "upgrade" : "original study"}
                    {m.tta_views > 1 ? " · TTA" : ""}
                  </p>
                </td>
                {COLUMNS.map((c) => (
                  <td key={c.key} className="whitespace-nowrap px-3 py-3 text-right font-mono text-[13px] tabular text-ink-2">
                    {c.fmt(c.get(m))}
                    {c.key === "f1_macro" && m.test?.f1_macro_ci && <div className="text-[10.5px] text-muted">{ci(m.test.f1_macro_ci, 0)}</div>}
                    {c.key === "accuracy" && m.test?.accuracy_ci && <div className="text-[10.5px] text-muted">{ci(m.test.accuracy_ci, 0)}</div>}
                  </td>
                ))}
                <td className="px-4 py-3 text-right font-mono text-xs tabular text-muted">{compactParams(m.params)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function LiveBench({ online, classes }) {
  const { file, choose, comparison, compareStatus, compareError, runCompare } = usePrediction();
  const { models } = useData();
  const [sample, setSample] = useState(null);
  return (
    <Card>
      <CardHeader icon={Play} title="Live bench" subtitle="Run both served models — deep learning and classical ML — on one photo, right now" />
      <div className="grid grid-cols-1 gap-6 p-5 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
        <div className="flex flex-col gap-4">
          <UploadPanel
            file={file}
            onFile={(f) => {
              setSample(null);
              choose(f);
            }}
            disabled={compareStatus === "running"}
          />
          <SampleGallery
            onPick={(f, s) => {
              setSample(s);
              choose(f);
            }}
            activeKey={sample?.key}
            disabled={compareStatus === "running"}
            scope={classes}
            columns="grid-cols-4"
          />
          <Button icon={FlaskConical} onClick={() => runCompare()} disabled={!file || !online} loading={compareStatus === "running"}>
            Run both models
          </Button>
        </div>
        <div className="min-w-0">
          {compareStatus === "error" && <ErrorState error={compareError} />}
          {comparison && compareStatus === "done" ? (
            <CompareTable results={comparison.results} trueKey={sample?.key} models={models} />
          ) : (
            <div className="grid h-full min-h-48 place-items-center rounded-xl border border-dashed border-line-strong p-6 text-center text-sm text-muted">
              {online ? "Pick a photo and run the bench. Both models see exactly the same preprocessed image." : "The live bench needs the diagnosis server (offline now)."}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

export default function LabPage() {
  useDocumentTitle("Model lab");
  const { analytics, analyticsLoading, analyticsError, reloadAnalytics, online } = useData();
  const task = TASK;
  const [family, setFamily] = useState("all");
  const [sort, setSort] = useState({ key: "f1_macro", dir: 1 });
  const [selected, setSelected] = useState([]);

  const def = defaultModel(analytics, task);
  const info = taskInfo(analytics, task);
  const models = useMemo(() => modelsFor(analytics, task).filter((m) => family === "all" || (family === "dl" ? !isClassicalModel(m) : isClassicalModel(m))), [analytics, task, family]);

  // Start with the two served models (deep learning, classical ML) and the original study's default.
  useEffect(() => {
    if (!analytics) return;
    const i = taskInfo(analytics, task);
    setSelected([...new Set([i?.default_model, i?.classical_model, i?.previous_default].filter(Boolean))]);
  }, [analytics, task]);

  const toggle = (key) => setSelected((s) => (s.includes(key) ? s.filter((k) => k !== key) : [...s, key].slice(-3)));
  const chosen = selected.map((k) => modelsFor(analytics, task).find((m) => m.key === k)).filter(Boolean);
  const classes = taskClasses(analytics, task);
  // The machine-learning vs deep-learning test comes first; names fall back to the model list when a row has none.
  const comps = comparisonsFor(analytics, task)
    .filter((c) => c.f1_ci)
    .sort((x, y) => Boolean(y.ml_vs_dl) - Boolean(x.ml_vs_dl));
  const recipes = recipesFor(analytics, task);
  const robustness = analytics?.audit?.robustness?.[task];
  const screening = (analytics?.cv?.screening || []).filter((s) => s.task === task);
  const cvPhotos = taskPhotos(analytics, task, "cv") ?? analytics?.dataset?.cv?.photos;

  if (analyticsError && !analytics) return <ErrorState title="Could not load results" error={analyticsError} onRetry={reloadAnalytics} />;

  return (
    <div>
      <PageHeader
        eyebrow="Model comparison laboratory"
        title="Model lab"
        description="Every model for the five conditions — classical machine learning, the original study's networks and the upgraded ones — scored on exactly the same held-out test photos. Select up to three models to compare them class by class."
      />

      {analyticsLoading && !analytics && <Skeleton className="h-96 rounded-2xl" />}
      {analytics && (
        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader
              icon={Gauge}
              title={`Leaderboard · ${info?.test_photos ?? "—"} test photos`}
              subtitle="95% confidence intervals from 2,000 bootstrap resamples of whole leaf groups"
              action={
                <Segmented
                  size="sm"
                  label="Family"
                  value={family}
                  onChange={setFamily}
                  options={[
                    { value: "all", label: "All" },
                    { value: "dl", label: "Deep learning" },
                    { value: "ml", label: "Classical" },
                  ]}
                />
              }
            />
            <Leaderboard models={models} sort={sort} setSort={setSort} selected={selected} toggle={toggle} highlight={def?.key} served={info?.served_models || []} />
          </Card>

          <FinalPipeline final={analytics.final_round} classical={analytics.classical} mlGate={analytics.ood?.final?.[info?.classical_model]} />

          {chosen.length > 0 && (
            <>
              <Card>
                <CardHeader icon={Layers} title="Per-class F1 of the selected models" subtitle="Where each model succeeds or fails — the test support of every class is shown in the Performance page" />
                <div className="p-4">
                  <PerClassChart classes={classes} series={chosen.map((m, i) => ({ key: m.key, label: m.name, perClass: m.per_class, color: SERIES[i] }))} />
                </div>
              </Card>
              <div className={cx("grid grid-cols-1 gap-6", chosen.length > 1 ? "xl:grid-cols-2" : "")}>
                {chosen.slice(0, 2).map((m) => (
                  <Card key={m.key} className="p-5">
                    <ConfusionMatrix matrix={m.confusion} classes={classes} title={m.name} compact />
                  </Card>
                ))}
              </div>
            </>
          )}

          {comps.length > 0 && (
            <Card>
              <CardHeader icon={Scale} title="Is the difference real?" subtitle="Paired bootstrap over the same test leaf groups: a difference is reliable only if its 95% interval excludes zero" />
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full min-w-[720px] text-sm">
                  <thead className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <th className="px-5 py-3 font-medium">Comparison</th>
                      <th className="px-3 py-3 text-right font-medium">Δ accuracy</th>
                      <th className="px-3 py-3 text-right font-medium">Δ macro-F1</th>
                      <th className="px-3 py-3 text-right font-medium">95% CI (macro-F1)</th>
                      <th className="px-5 py-3 font-medium">Verdict</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {comps.map((c) => {
                      const reliable = c.f1_ci[0] > 0 || c.f1_ci[1] < 0;
                      return (
                        <tr key={`${c.a}-${c.b}`} className={c.ml_vs_dl ? "bg-sun-500/[0.04]" : undefined}>
                          <td className="px-5 py-3">
                            <span className="font-medium text-ink">{c.a_name || modelName(analytics, c.a)}</span> <span className="text-muted">vs</span> {c.b_name || modelName(analytics, c.b)}
                            {c.ml_vs_dl && (
                              <Badge tone="sun" className="ml-2">
                                classical ML vs deep learning
                              </Badge>
                            )}
                          </td>
                          <td className="px-3 py-3 text-right font-mono tabular">{signedPct(c.accuracy_diff)}</td>
                          <td className="px-3 py-3 text-right font-mono tabular">{signedPct(c.f1_diff)}</td>
                          <td className="px-3 py-3 text-right font-mono text-xs tabular text-muted">
                            {signedPct(c.f1_ci[0])} … {signedPct(c.f1_ci[1])}
                          </td>
                          <td className="px-5 py-3">
                            <Badge tone={reliable ? (c.f1_diff > 0 ? "leaf" : "tomato") : "slate"}>{reliable ? (c.f1_diff > 0 ? "reliably better" : "reliably worse") : "not significant"}</Badge>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader icon={Timer} title="Accuracy vs CPU cost" subtitle="Median CPU time per photo (log scale) against test macro-F1" />
              <div className="p-4">
                <LatencyScatter
                  points={modelsFor(analytics, task)
                    .filter((m) => m.latency_ms && m.test)
                    .map((m) => ({ key: m.key, label: m.name, latency: m.latency_ms, f1: m.test.f1_macro, acc: m.test.accuracy, color: modelColor(m), highlight: m.key === def?.key }))}
                />
                <div className="mt-2 flex flex-wrap justify-center gap-4 text-xs text-muted">
                  <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-[#15834e]" />upgrade</span>
                  <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-[#0891b2]" />original deep learning</span>
                  <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-[#d6760b]" />classical ML</span>
                </div>
              </div>
            </Card>
            {robustness ? (
              <Card>
                <CardHeader icon={ShieldCheck} title="Consistency under harmless changes" subtitle="Share of validation photos whose predicted class flips (lower is better)" />
                <div className="p-4">
                  <RobustnessChart rows={robustness.rows} series={robustness.series} />
                </div>
              </Card>
            ) : (
              <Card className="p-5 text-sm text-muted">Robustness audit not available in this results file.</Card>
            )}
          </div>

          {recipes.length > 0 && (
            <Card>
              <CardHeader
                icon={Beaker}
                title="Training recipes (5-fold cross-validation)"
                subtitle={`Out-of-fold macro-F1 on ${cvPhotos != null ? `the ${cvPhotos.toLocaleString()} ` : "the "}train+validation photos of the five conditions — how the upgrade was chosen without touching the test set`}
              />
              <div className="grid grid-cols-1 gap-6 p-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
                <div className="overflow-x-auto scrollbar-thin">
                  <table className="w-full min-w-[460px] text-sm">
                    <thead className="border-b border-line text-left text-xs uppercase tracking-wide text-muted">
                      <tr>
                        <th className="py-2.5 pr-3 font-medium">Recipe</th>
                        <th className="px-2 py-2.5 text-right font-medium">F1</th>
                        <th className="px-2 py-2.5 text-right font-medium">F1 + TTA</th>
                        <th className="py-2.5 pl-2 text-right font-medium">Acc + TTA</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line">
                      {recipes.map((r) => (
                        <tr key={r.recipe}>
                          <td className="py-2.5 pr-3">
                            <p className="font-medium text-ink">{r.label}</p>
                            <p className="text-xs text-muted">{r.train_minutes ? `${Math.round(r.train_minutes)} GPU-min for 5 folds` : ""}</p>
                          </td>
                          <td className="px-2 py-2.5 text-right font-mono tabular">{pct(r.f1_1)}</td>
                          <td className="px-2 py-2.5 text-right font-mono tabular">{pct(r.f1_8)}</td>
                          <td className="py-2.5 pl-2 text-right font-mono tabular">{pct(r.acc_8)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <FoldStrip recipes={recipes.map((r) => ({ label: r.label, short: r.short, pooled: r.f1_8, per_fold: r.per_fold.map((f) => ({ fold: f.fold, value: f.f1_8 })) }))} />
              </div>
              {screening.length > 0 && (
                <div className="px-5 pb-5">
                  <p className="mb-2 text-sm font-semibold text-ink">Every recipe tried — screening on fold 0</p>
                  <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
                    <table className="w-full min-w-[560px] text-sm">
                      <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                        <tr>
                          <th className="px-4 py-2 font-medium">Recipe</th>
                          <th className="px-3 py-2 text-right font-medium">F1</th>
                          <th className="px-3 py-2 text-right font-medium">F1 + TTA</th>
                          <th className="px-3 py-2 text-right font-medium">Acc + TTA</th>
                          <th className="px-4 py-2 font-medium">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-line">
                        {screening.map((s) => (
                          <tr key={s.recipe} className="bg-surface">
                            <td className="px-4 py-2 text-ink">{s.label}</td>
                            <td className="px-3 py-2 text-right font-mono tabular">{pct(s.f1_1)}</td>
                            <td className="px-3 py-2 text-right font-mono tabular">{pct(s.f1_8)}</td>
                            <td className="px-3 py-2 text-right font-mono tabular">{pct(s.acc_8)}</td>
                            <td className="px-4 py-2">
                              <Badge tone={s.full_cv ? "leaf" : "slate"}>{s.full_cv ? "full 5-fold CV" : "fold 0 only"}</Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-2 text-xs text-muted">
                    {screening.some((s) => !s.full_cv)
                      ? "A single fold holds out only a few photos of the rarest condition, so its macro-F1 is noisy; only recipes that looked competitive were run on all five folds."
                      : "Every recipe tried was then run on all five folds."}
                  </p>
                </div>
              )}
              {analytics.cv?.selection_rule && (
                <div className="px-5 pb-5">
                  <Callout tone="slate" title="Selection rule (fixed before the test set was scored)">
                    {analytics.cv.selection_rule}
                  </Callout>
                </div>
              )}
            </Card>
          )}

          <LiveBench online={online} classes={classes} />
        </div>
      )}
    </div>
  );
}
