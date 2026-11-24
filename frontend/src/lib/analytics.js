// Selectors over the platform analytics document (experiments/platform.json, served at /api/analytics and bundled at
// /data/platform.json). Pages use these helpers instead of reaching into the JSON directly.

/** The apps diagnose five conditions: every page reads this one task (eight-condition results are no longer exported). */
export const TASK = "5class";

export const taskInfo = (a, task) => a?.tasks?.[task] ?? null;
export const taskClasses = (a, task) => taskInfo(a, task)?.classes ?? [];
export const modelsFor = (a, task) => (a?.models || []).filter((m) => !task || m.task === task);
export const modelByKey = (a, key) => (a?.models || []).find((m) => m.key === key) ?? null;
export const modelName = (a, key) => modelByKey(a, key)?.name ?? key;
export const defaultModel = (a, task) => modelByKey(a, taskInfo(a, task)?.default_model);

/** Previous default (the original study's deployed model) for a task, if the document marks one. */
export const previousDefault = (a, task) => modelByKey(a, taskInfo(a, task)?.previous_default);

/** The served classical machine-learning model (handcrafted features, no neural network), if the document has one. */
export const classicalModel = (a, task) => modelByKey(a, taskInfo(a, task)?.classical_model);

/** Classical ML (no neural network). Analytics entries say so in `family`, API entries in `kind`. */
export const isClassicalModel = (m) => m?.kind === "classical" || m?.family === "Classical ML" || m?.family === "Machine learning";

/** The model's name in the apps (ConvNeXt-Tiny ensemble, Gradient Boosting), else its family. */
export const servedName = (m) => m?.model_name || (isClassicalModel(m) ? "Classical ML" : "Deep learning");

/** What model `m` does better than `other`, from the measured numbers (test photos, CPU time, download size). */
export function modelBenefit(m, other) {
  const family = isClassicalModel(m) ? "Classical ML" : "Deep learning";
  if (!m || !other) return family;
  const p = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
  const acc = (x) => x?.test_metrics?.accuracy ?? x?.test?.accuracy;
  const cov = (x) => x?.operating_point?.test?.coverage ?? x?.selective?.test?.coverage;
  const parts = [];
  if (acc(m) != null && acc(other) != null && acc(m) > acc(other)) parts.push(`more accurate (${p(acc(m))} vs ${p(acc(other))} on test photos)`);
  if (cov(m) != null && cov(other) != null && cov(m) > cov(other)) parts.push(`answers more photos (${p(cov(m), 0)} vs ${p(cov(other), 0)})`);
  if (!isClassicalModel(m)) parts.push("shows where it looked (Grad-CAM)");
  if (m.latency_ms && other.latency_ms && m.latency_ms < other.latency_ms)
    parts.push(`${(other.latency_ms / m.latency_ms).toFixed(1)}× faster (${Math.round(m.latency_ms)} vs ${Math.round(other.latency_ms)} ms per photo on CPU)`);
  if (m.weights_mb && other.weights_mb && m.weights_mb < other.weights_mb)
    parts.push(`${Math.round(other.weights_mb / m.weights_mb)}× smaller (${m.weights_mb.toFixed(1)} MB vs ${Math.round(other.weights_mb)} MB)`);
  if (isClassicalModel(m)) parts.push("no neural network");
  return parts.length ? `${family}: ${parts.join(", ")}.` : family;
}

/** Dataset class rows (photos, train/val/test/cv counts) of the conditions a task diagnoses. */
export const taskDatasetClasses = (a, task) => {
  const keys = taskClasses(a, task);
  return (a?.dataset?.classes || []).filter((c) => keys.includes(c.key));
};

/** Photo count of a split (train | val | test | cv) over the conditions a task diagnoses; null if unknown. */
export function taskPhotos(a, task, split) {
  const rows = taskDatasetClasses(a, task);
  return rows.length ? rows.reduce((s, c) => s + (c[split] || 0), 0) : null;
}

export function ranked(a, task, metric = "f1_macro") {
  return modelsFor(a, task)
    .filter((m) => m.test?.[metric] != null)
    .sort((x, y) => y.test[metric] - x.test[metric] || y.test.accuracy - x.test.accuracy);
}

export const comparisonsFor = (a, task) => (a?.comparisons || []).filter((c) => c.task === task);
export const recipesFor = (a, task) => (a?.cv?.recipes || []).filter((r) => r.task === task);


export function modelColor(m) {
  if (!m) return "#64748b";
  if (isClassicalModel(m)) return "#d6760b";
  return m.generation === "upgrade" ? "#15834e" : "#0891b2";
}
