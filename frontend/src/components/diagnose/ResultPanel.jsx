import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { AlertTriangle, ArrowRight, CheckCircle2, CircleHelp, Download, Info, Lightbulb, ScanEye, ShieldAlert, XCircle } from "lucide-react";
import { STATUS_STYLE, classMeta } from "../../lib/classes.js";
import { ms, pct } from "../../lib/format.js";
import ConfidenceRing from "../charts/ConfidenceRing.jsx";
import ProbabilityBars from "../charts/ProbabilityBars.jsx";
import { Badge, Button, Card, CardHeader, InfoTip } from "../ui/primitives.jsx";
import GradCamViewer from "./GradCamViewer.jsx";

const TONE_RING = { Healthy: "#15834e", Diseased: "#db3a25", Uncertain: "#d6760b", "Unusual image": "#64748b" };
const STATUS_ICON = { Healthy: CheckCircle2, Diseased: ShieldAlert, Uncertain: CircleHelp, "Unusual image": AlertTriangle };

function downloadReport(result) {
  const { explanation, input_image: _img, ...rest } = result;
  const report = { ...rest, explanation: explanation ? { method: explanation.method, class_index: explanation.class_index, note: explanation.note } : null, exported_at: new Date().toISOString() };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `tomatoleaf-diagnosis-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export default function ResultPanel({ result }) {
  const { prediction, probabilities, confidence_threshold: thr, ood, trueKey, operating_point: op } = result;
  const status = prediction.status;
  const style = STATUS_STYLE[status] || STATUS_STYLE.Uncertain;
  const Icon = STATUS_ICON[status] || Info;
  const meta = classMeta(prediction.key);
  const answered = status === "Healthy" || status === "Diseased";
  // Verdict against the known true label (gallery photos): a refusal ("Uncertain" / "Unusual image") is not a wrong
  // answer — and for a condition outside the model's classes it is the right behaviour.
  const inScope = trueKey ? (result.classes || []).includes(trueKey) : null;
  const verdict = !trueKey ? null : answered ? (trueKey === prediction.key ? "correct" : "wrong") : inScope ? "refused" : "refused_outside";
  const VERDICT = {
    correct: { tone: "leaf", icon: CheckCircle2, text: "matches true label" },
    wrong: { tone: "tomato", icon: XCircle, text: `wrong · true label: ${trueKey ? classMeta(trueKey).name : ""}` },
    refused: {
      tone: "sun",
      icon: CircleHelp,
      text: trueKey === prediction.key ? "below the threshold · best guess is right" : `not answered · true label: ${trueKey ? classMeta(trueKey).name : ""}`,
    },
    refused_outside: { tone: "leaf", icon: CheckCircle2, text: `correctly not diagnosed: ${trueKey ? classMeta(trueKey).name : ""} is not one of this model's conditions` },
  };

  return (
    <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="flex flex-col gap-5">
      <Card className="overflow-hidden">
        <div className="relative p-5 sm:p-6">
          <div className="absolute inset-x-0 top-0 h-1" style={{ background: TONE_RING[status] }} />
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center">
            <ConfidenceRing value={prediction.confidence} threshold={thr} color={TONE_RING[status]} />
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={style.tone} icon={Icon}>
                  {style.label}
                </Badge>
                {verdict && (
                  <Badge tone={VERDICT[verdict].tone} icon={VERDICT[verdict].icon} className="whitespace-normal! rounded-2xl! py-1!">
                    {VERDICT[verdict].text}
                  </Badge>
                )}
              </div>
              <h2 className="mt-3 font-display text-2xl font-semibold tracking-tight text-ink sm:text-[28px]">
                {answered ? meta.name : status === "Uncertain" ? "Not sure" : "No diagnosis"}
              </h2>
              {!answered && result.candidates?.length > 0 && (
                <p className="mt-0.5 text-sm text-ink-2">
                  Most likely: {result.candidates.map((c) => `${c.name} ${pct(c.probability, 0)}`).join(" · ")}
                </p>
              )}
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{style.description}</p>
              <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-muted">
                <span>
                  Model <strong className="font-medium text-ink-2">{result.model?.name}</strong>
                </span>
                <span>
                  Threshold <strong className="font-medium text-ink-2">{pct(thr, 0)}</strong>
                </span>
                <span>
                  Inference <strong className="font-medium text-ink-2">{ms(result.inference_ms)}</strong>
                </span>
              </div>
            </div>
          </div>

          {result.refusal && (
            <div className="mt-5 rounded-xl border border-sun-500/30 bg-sun-500/5 px-3.5 py-3 text-sm">
              <p className="flex items-center gap-2 font-semibold text-ink">
                <Lightbulb className="size-4 shrink-0 text-sun-600" aria-hidden /> Why no diagnosis
              </p>
              <p className="mt-1 text-ink-2">{result.refusal.message}</p>
              {result.refusal.hints?.length > 0 && (
                <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-2">
                  {result.refusal.hints.map((h) => (
                    <li key={h}>{h}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
          {answered && result.quality && !result.quality.ok && (
            <div className="mt-5 rounded-xl border border-sun-500/30 bg-sun-500/5 px-3.5 py-3 text-sm">
              <p className="flex items-center gap-2 font-semibold text-ink">
                <AlertTriangle className="size-4 shrink-0 text-sun-600" aria-hidden /> Photo quality: the result may be less reliable
              </p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-ink-2">
                {result.quality.issues.map((i) => (
                  <li key={i.code}>{i.tip}</li>
                ))}
              </ul>
            </div>
          )}
          {op?.test && (
            <p className="mt-5 rounded-xl bg-surface-2 px-3.5 py-2.5 text-xs leading-relaxed text-ink-2">
              <strong className="font-semibold">Operating point:</strong> on the held-out test photos this model answered {pct(op.test.coverage, 0)} of photos and was right on {pct(op.test.accuracy_kept, 1)} of those (threshold chosen on {op.chosen_on || "validation data"}).
            </p>
          )}
          {ood && (
            <div className="mt-3 flex items-center gap-3 rounded-xl border border-line px-3.5 py-2.5 text-xs">
              <ScanEye className="size-4 shrink-0 text-muted" aria-hidden />
              <span className="flex-1 text-ink-2">
                Similarity check: distance to training photos is <strong>{(ood.ratio * 100).toFixed(0)}%</strong> of the "unusual" limit
                {ood.flag ? " — flagged as unusual." : "."}
              </span>
              <InfoTip content={ood.note} />
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line bg-surface-2/60 px-5 py-3">
          {answered && !prediction.is_healthy ? (
            <Link to={`/library/${prediction.key}`} className="inline-flex items-center gap-1.5 text-sm font-medium text-leaf-700 hover:underline dark:text-leaf-300">
              About {meta.name}: symptoms & management <ArrowRight className="size-4" aria-hidden />
            </Link>
          ) : (
            <Link to="/library" className="inline-flex items-center gap-1.5 text-sm font-medium text-leaf-700 hover:underline dark:text-leaf-300">
              Browse the disease library <ArrowRight className="size-4" aria-hidden />
            </Link>
          )}
          <Button variant="ghost" size="sm" icon={Download} onClick={() => downloadReport(result)}>
            Report (JSON)
          </Button>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
        <Card>
          <CardHeader title="Class probabilities" subtitle="Calibrated on out-of-fold data — the answer is the tallest bar; the vertical tick marks the threshold" />
          <div className="p-5">
            <ProbabilityBars probabilities={probabilities} threshold={thr} trueKey={trueKey} />
          </div>
        </Card>
        <Card>
          <div className="p-5">
            <GradCamViewer input={result.input_image} heatmap={result.explanation?.heatmap} overlay={result.explanation?.overlay} note={result.explanation?.note} />
          </div>
        </Card>
      </div>

      {result.image && (
        <Card className="p-5">
          <p className="mb-2 text-sm font-semibold text-ink">Preprocessing applied</p>
          <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            <div>
              <dt className="text-xs text-muted">Original</dt>
              <dd className="text-ink-2">
                {result.image.width}×{result.image.height} {result.image.format}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Colour</dt>
              <dd className="text-ink-2">{result.image.colour || "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted">Geometry</dt>
              <dd className="text-ink-2">{result.image.geometry || "—"}</dd>
            </div>
          </dl>
        </Card>
      )}
    </motion.div>
  );
}
