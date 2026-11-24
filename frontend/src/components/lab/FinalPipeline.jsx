import { CheckCircle2, ShieldAlert, ShieldCheck, Sparkles } from "lucide-react";
import { Badge, Callout, Card, CardHeader } from "../ui/primitives.jsx";
import { TASK } from "../../lib/analytics.js";
import { ci, cx, num, pct, signedPct } from "../../lib/format.js";

const STRESS_LABEL = {
  blur_sigma2: "Blur (σ 2)",
  blur_sigma4: "Strong blur (σ 4)",
  "dark_x0.35": "Very dark",
  "overexposed_x1.7": "Over-exposed",
  jpeg_q15: "Heavy JPEG (q15)",
  background_black: "Black background",
  background_wood: "Wood background",
  background_grass: "Grass background",
  background_fabric: "Fabric background",
  four_leaves_same: "4 leaves, same condition",
  four_leaves_mixed: "4 leaves, mixed",
};
const GATE_NAME = { knn: "k-nearest-neighbour", maha: "Mahalanobis" };
const OOD_LABEL = {
  other_crop: "Other crops (PlantVillage)",
  tomato_unsupported: "Tomato diseases not in the dataset",
  everyday: "Everyday photos",
  probes: "Synthetic non-leaf images",
  empty_paper: "Empty paper (leaf removed)",
};

function Th({ children, right = false, className = "" }) {
  return <th className={cx("px-3 py-2.5 font-medium", right && "text-right", className)}>{children}</th>;
}
function Td({ children, right = true, className = "" }) {
  return <td className={cx("px-3 py-2.5", right && "text-right font-mono tabular", className)}>{children}</td>;
}

/** Share bar: answered correctly (green), answered wrongly (red), refused (grey). */
function OutcomeBar({ o }) {
  if (!o) return <span className="text-muted">—</span>;
  const right = o.answered_correct ?? 0;
  const wrong = o.answered_wrong ?? (o.answered_correct == null ? o.answered : 0);
  const refused = o.refused ?? 0;
  return (
    <div className="flex min-w-[140px] items-center gap-2">
      <div className="flex h-2 flex-1 overflow-hidden rounded-full bg-surface-2" aria-hidden>
        <span className="bg-leaf-500" style={{ width: `${100 * right}%` }} />
        <span className="bg-tomato-500" style={{ width: `${100 * wrong}%` }} />
        <span className="bg-line-strong" style={{ width: `${100 * refused}%` }} />
      </div>
      <span className="w-24 text-right font-mono text-xs tabular text-muted">
        {o.answered_correct != null ? `${pct(right, 0)} · ` : ""}
        <span className="text-tomato-600">{pct(wrong, 0)}</span>
      </span>
    </div>
  );
}

function PairedLine({ c, label }) {
  if (!c?.f1_ci) return null;
  return (
    <p className="mt-2 text-xs text-muted">
      Paired bootstrap, {label}: Δ accuracy {signedPct(c.accuracy_diff)}
      {c.accuracy_ci ? ` (95% CI ${signedPct(c.accuracy_ci[0])} … ${signedPct(c.accuracy_ci[1])})` : ""}, Δ macro-F1 {signedPct(c.f1_diff)} ({signedPct(c.f1_ci[0])} …{" "}
      {signedPct(c.f1_ci[1])}).
    </p>
  );
}

/**
 * The final round of the five-condition app: the deep-learning arms and the classical ML arms compared out-of-fold, the
 * chosen pipelines, and before vs after on the test photos with the served classical model alongside.
 */
export default function FinalPipeline({ final, classical, mlGate }) {
  const t = final?.tasks?.[TASK];
  if (!t) return null;
  const finalRow = t.test?.find((r) => r.final);
  const mlRow = t.test?.find((r) => r.classical);
  const before = t.test?.find((r) => !r.final && !r.classical);
  const mlArms = [...(classical?.arms || [])].sort((a, b) => (b.oof?.f1_macro ?? 0) - (a.oof?.f1_macro ?? 0));
  const mlChosen = mlArms.find((a) => a.chosen);
  const mlChosenLabel = mlChosen ? `${mlChosen.classifier} · ${mlChosen.feature_set}` : classical?.chosen;
  const stressNames = Object.keys(t.stress?.[finalRow?.key] || {});
  const oodNames = Object.keys(OOD_LABEL).filter((k) => t.out_of_dataset?.[finalRow?.key]?.[k]);
  const columns = [
    before && { key: before.key, label: "Before" },
    finalRow && { key: finalRow.key, label: finalRow.model_name || "Final" },
    mlRow && { key: mlRow.key, label: mlRow.model_name || "Classical ML" },
  ].filter(Boolean);
  const stressCols = columns.filter((c) => t.stress?.[c.key]);
  const oodCols = columns.filter((c) => t.out_of_dataset?.[c.key]);
  const showChecks = Boolean(finalRow) && stressNames.length > 0 && stressCols.length > 1;
  const steps = ["dl", mlArms.length > 0 && "ml", t.test && "test", showChecks && "stress", showChecks && "ood"].filter(Boolean);
  const step = (s) => steps.indexOf(s) + 1;
  const comparisons = t.comparisons || [];
  const vsBefore = comparisons.find((c) => before && c.a === finalRow?.key && c.b === before.key) || comparisons.find((c) => !c.ml_vs_dl && c.a !== mlRow?.key && c.b !== mlRow?.key);
  const vsMl = mlRow ? comparisons.find((c) => c.ml_vs_dl || c.a === mlRow.key || c.b === mlRow.key) : null;
  const short = (key, name) => (key === mlRow?.key ? "classical ML" : key === finalRow?.key ? "deep learning" : name || key);

  return (
    <Card>
      <CardHeader
        icon={Sparkles}
        title="Final pipeline — the two served models"
        subtitle="Deep learning (imbalance handling) and classical ML (classifier and features): calibration, threshold and unusual-image limits chosen on out-of-fold data only; the test photos were scored once at the end"
      />
      <div className="flex flex-col gap-6 p-5">
        <div>
          <p className="mb-2 text-sm font-semibold text-ink">{step("dl")} · Deep learning: imbalance arms (out-of-fold, cross-fitted)</p>
          <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
            <table className="w-full min-w-[860px] text-sm">
              <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                <tr>
                  <Th>Arm · training</Th>
                  <Th right>Macro-F1</Th>
                  <Th right>Accuracy</Th>
                  <Th right>NLL</Th>
                  <Th right>ECE</Th>
                  <Th>Calibration</Th>
                  <Th right>Threshold</Th>
                  <Th right>Answered</Th>
                  <Th right>Acc. answered</Th>
                  <Th right>Unsupported refused</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {t.arms.map((a) => (
                  <tr key={a.arm} className={a.chosen ? "bg-leaf-500/5" : "bg-surface"}>
                    <td className="px-3 py-2.5">
                      <span className="font-medium text-ink">{a.arm}</span> <span className="text-muted">· {a.label}</span>
                      {a.tau ? <span className="ml-1 text-xs text-muted">(τ {num(a.tau, 2)})</span> : null}
                      {a.chosen && <Badge tone="leaf" icon={CheckCircle2} className="ml-2">chosen</Badge>}
                    </td>
                    <Td>{pct(a.oof.f1_macro)}</Td>
                    <Td>{pct(a.oof.accuracy)}</Td>
                    <Td>{num(a.oof.nll, 3)}</Td>
                    <Td>{num(a.oof.ece, 3)}</Td>
                    <td className="px-3 py-2.5 text-xs text-muted">
                      {a.calibration} <span className="font-mono">(NLL {num(a.nll_temperature, 3)} vs {num(a.nll_vector, 3)})</span>
                    </td>
                    <Td>{num(a.threshold, 2)}</Td>
                    <Td>{pct(a.coverage)}</Td>
                    <Td>{pct(a.accuracy_kept)}</Td>
                    <Td>{pct(a.out_of_scope_refusal)}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted">
            Chosen: arm {t.chosen} — {t.why}. Unusual-image gate: {GATE_NAME[t.gate?.method] || t.gate?.method} feature distance (out-of-fold AUROC {num(t.gate?.auroc_select, 3)}).
          </p>
        </div>

        {mlArms.length > 0 && (
          <div>
            <p className="mb-2 text-sm font-semibold text-ink">{step("ml")} · Classical ML: classifier × handcrafted features (out-of-fold, same rules)</p>
            <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
              <table className="w-full min-w-[860px] text-sm">
                <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                  <tr>
                    <Th>Classifier · features</Th>
                    <Th right>Macro-F1</Th>
                    <Th right>Accuracy</Th>
                    <Th right>NLL</Th>
                    <Th right>ECE</Th>
                    <Th>Calibration</Th>
                    <Th right>Threshold</Th>
                    <Th right>Answered</Th>
                    <Th right>Acc. answered</Th>
                    <Th right>Unsupported refused</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {mlArms.map((a) => (
                    <tr key={a.arm} className={a.chosen ? "bg-leaf-500/5" : "bg-surface"}>
                      <td className="px-3 py-2.5">
                        <span className="font-medium text-ink">{a.classifier}</span> <span className="text-muted">· {a.feature_set}</span>
                        {a.chosen && <Badge tone="leaf" icon={CheckCircle2} className="ml-2">chosen</Badge>}
                      </td>
                      <Td>{pct(a.oof?.f1_macro)}</Td>
                      <Td>{pct(a.oof?.accuracy)}</Td>
                      <Td>{num(a.oof?.nll, 3)}</Td>
                      <Td>{num(a.oof?.ece, 3)}</Td>
                      <td className="px-3 py-2.5 text-xs text-muted">{a.calibration ?? "—"}</td>
                      <Td>{num(a.threshold, 2)}</Td>
                      <Td>{pct(a.coverage)}</Td>
                      <Td>{pct(a.accuracy_kept)}</Td>
                      <Td>{pct(a.out_of_scope_refusal)}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted">
              {mlChosenLabel ? `Chosen: ${mlChosenLabel}${classical.why ? ` — ${classical.why}` : ""}.` : ""}
              {mlGate?.method ? ` Unusual-image gate: ${GATE_NAME[mlGate.method] || mlGate.method} distance in the colour + texture feature space.` : ""} No neural network: features are computed from the same 256×256 image the network sees.
            </p>
          </div>
        )}

        {t.test && (
          <div>
            <p className="mb-2 text-sm font-semibold text-ink">{step("test")} · Before vs after on the held-out test photos</p>
            <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
              <table className="w-full min-w-[900px] text-sm">
                <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                  <tr>
                    <Th>Model</Th>
                    <Th right>Accuracy (95% CI)</Th>
                    <Th right>Macro-F1</Th>
                    <Th right>Weighted F1</Th>
                    <Th right>ECE</Th>
                    <Th right>Brier</Th>
                    <Th right>Threshold</Th>
                    <Th right>Answered → acc.</Th>
                    <Th right>Unsupported: wrong answer</Th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {t.test.map((r) => (
                    <tr key={r.key} className={r.final ? "bg-leaf-500/5" : r.classical ? "bg-sun-500/[0.05]" : "bg-surface"}>
                      <td className="px-3 py-2.5">
                        <span className={cx("font-medium", r.final || r.classical ? "text-ink" : "text-muted")}>{r.name}</span>
                        {r.final && <Badge tone="leaf" className="ml-2">served · deep learning</Badge>}
                        {r.classical && <Badge tone="sun" className="ml-2">served · classical ML</Badge>}
                      </td>
                      <Td>
                        {pct(r.accuracy)} <span className="text-xs text-muted">({ci(r.accuracy_ci, 0)})</span>
                      </Td>
                      <Td>{pct(r.f1_macro)}</Td>
                      <Td>{pct(r.f1_weighted)}</Td>
                      <Td>{num(r.ece, 3)}</Td>
                      <Td>{num(r.brier, 3)}</Td>
                      <Td>{num(r.threshold, 2)}</Td>
                      <Td>
                        {pct(r.coverage, 0)} → {pct(r.accuracy_kept)}
                      </Td>
                      <Td>{pct(r.out_of_scope?.answered_wrong, 0)}</Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <PairedLine c={vsBefore} label={`final vs ${vsBefore?.b_name || vsBefore?.b}`} />
            {vsMl && vsMl !== vsBefore && <PairedLine c={vsMl} label={`${short(vsMl.a, vsMl.a_name)} vs ${short(vsMl.b, vsMl.b_name)}`} />}
          </div>
        )}

        {showChecks && (
          <div className={cx("grid grid-cols-1 gap-6", stressCols.length < 3 && "xl:grid-cols-2")}>
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
                <ShieldCheck className="size-4 text-muted" aria-hidden /> {step("stress")} · Stress tests (test photos)
              </p>
              <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
                <table className={cx("w-full text-sm", stressCols.length > 2 ? "min-w-[640px]" : "min-w-[460px]")}>
                  <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <Th>Condition</Th>
                      {stressCols.map((c) => (
                        <Th key={c.key}>{c.label}</Th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {stressNames.map((n) => (
                      <tr key={n} className="bg-surface">
                        <td className="px-3 py-2 text-ink">{STRESS_LABEL[n] || n}</td>
                        {stressCols.map((c) => (
                          <td key={c.key} className="px-3 py-2">
                            <OutcomeBar o={t.stress?.[c.key]?.[n]} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div>
              <p className="mb-2 flex items-center gap-1.5 text-sm font-semibold text-ink">
                <ShieldAlert className="size-4 text-muted" aria-hidden /> {step("ood")} · Out-of-dataset images: share refused
              </p>
              <div className="overflow-x-auto scrollbar-thin rounded-xl border border-line">
                <table className={cx("w-full text-sm", oodCols.length > 2 ? "min-w-[520px]" : "min-w-[420px]")}>
                  <thead className="bg-surface-2 text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <Th>Images</Th>
                      {oodCols.map((c) => (
                        <Th key={c.key} right>
                          {c.label}
                        </Th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {oodNames.map((n) => (
                      <tr key={n} className="bg-surface">
                        <td className="px-3 py-2 text-ink">
                          {OOD_LABEL[n]} <span className="text-xs text-muted">({t.out_of_dataset[finalRow.key][n].photos})</span>
                        </td>
                        {oodCols.map((c) => (
                          <Td key={c.key}>{pct(t.out_of_dataset?.[c.key]?.[n]?.refused, 0)}</Td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-2 text-xs text-muted">Bars: answered correctly · answered wrongly (red %) · refused. Refused = “Uncertain” or “Unusual image”.</p>
            </div>
          </div>
        )}
        {final.note && (
          <Callout tone="slate" title="How the test set was used">
            {final.note}
          </Callout>
        )}
      </div>
    </Card>
  );
}
