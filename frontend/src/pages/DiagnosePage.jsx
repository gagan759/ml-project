import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Camera, FlaskConical, Lightbulb, ScanSearch, ServerOff, Sun, Wand2 } from "lucide-react";
import CompareTable from "../components/diagnose/CompareTable.jsx";
import ModelPicker from "../components/diagnose/ModelPicker.jsx";
import ResultPanel from "../components/diagnose/ResultPanel.jsx";
import SampleGallery from "../components/diagnose/SampleGallery.jsx";
import UploadPanel from "../components/diagnose/UploadPanel.jsx";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Button, Callout, Card, CardHeader, EmptyState, ErrorState, PageHeader } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { usePrediction } from "../context/PredictionContext.jsx";

const TIPS = [
  { icon: Camera, text: "One detached leaf on plain white paper, filling most of the frame (this is how the training photos were taken)." },
  { icon: Sun, text: "Even daylight, no harsh shadows or flash glare; keep the leaf in focus." },
  { icon: Lightbulb, text: "Photos of whole plants, fields or other crops are outside what the models learned — expect “Unusual image”." },
];

function Scanning() {
  return (
    <Card className="relative overflow-hidden p-6">
      <div className="flex items-center gap-4">
        <div className="relative grid size-28 shrink-0 place-items-center overflow-hidden rounded-2xl bg-leaf-500/10">
          <ScanSearch className="size-10 text-leaf-600" aria-hidden />
          <span className="absolute inset-x-0 top-0 h-6 animate-scan bg-gradient-to-b from-transparent via-leaf-400/50 to-transparent" />
        </div>
        <div className="flex-1 space-y-3">
          <p className="font-display font-semibold text-ink">Analysing the leaf…</p>
          <p className="text-sm text-muted">Preprocessing like the dataset, running the model, calibrating its confidence and checking the photo.</p>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
            <div className="skeleton h-full w-full" />
          </div>
        </div>
      </div>
    </Card>
  );
}

export default function DiagnosePage() {
  useDocumentTitle("Diagnose");
  const { online, models, health } = useData();
  const { file, choose, result, status, error, run, comparison, compareStatus, compareError, runCompare } = usePrediction();
  const [model, setModel] = useState(null);
  const [sample, setSample] = useState(null);

  useEffect(() => {
    const list = models?.models || [];
    if (!list.length) return;
    if (!list.some((m) => m.key === model)) setModel(models.default_model || list[0].key);
  }, [models, model]);

  // A result belongs to the model that produced it. Switching model (deep learning <-> classical ML) must not leave the other
  // model's answer on screen: clear it (same photo kept) so the next "Diagnose" shows this model's own answer.
  useEffect(() => {
    if (result?.model?.key && model && result.model.key !== model) choose(file);
  }, [model]); // eslint-disable-line react-hooks/exhaustive-deps

  const scope = useMemo(() => (models?.models || []).find((m) => m.key === model)?.classes, [models, model]);
  const running = status === "running";

  const onFile = (f) => {
    setSample(null);
    choose(f);
  };
  const onSample = (f, s) => {
    setSample(s);
    choose(f);
  };
  const diagnose = () => run(model, { source: sample ? "test-sample" : "upload", trueKey: sample?.key || null });

  return (
    <div>
      <PageHeader
        eyebrow="AI diagnosis"
        title="Diagnose a tomato leaf"
        description="Upload a photo or pick a held-out test photo. The model returns a calibrated diagnosis, says “Uncertain” when it is not confident enough, flags photos unlike its training data, and shows where it looked."
      />

      {!online && health.status !== "checking" && (
        <Callout tone="sun" icon={ServerOff} title="The diagnosis server is offline" className="mb-6">
          Start it from the project folder with <code className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-xs">uvicorn backend.main:app --port 8000</code> (or run the Docker image). The rest of the site still works from the bundled results snapshot.
        </Callout>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
        <div className="flex flex-col gap-6">
          <Card className="p-5">
            <UploadPanel file={file} onFile={onFile} disabled={running} />
            <div className="mt-5">
              <SampleGallery onPick={onSample} activeKey={sample?.key} disabled={running} scope={scope} columns="grid-cols-4" />
            </div>
          </Card>
          <Card className="p-5">
            <ModelPicker model={model} onModel={setModel} disabled={running} />
            <Button className="mt-5 w-full" size="lg" icon={Wand2} onClick={diagnose} disabled={!file || !online || !model} loading={running}>
              {running ? "Diagnosing…" : "Diagnose"}
            </Button>
            {!file && <p className="mt-2 text-center text-xs text-muted">Choose a photo first.</p>}
          </Card>
          <Card className="p-5">
            <p className="mb-3 text-sm font-semibold text-ink">For a reliable result</p>
            <ul className="space-y-3">
              {TIPS.map(({ icon: Icon, text }) => (
                <li key={text} className="flex gap-3 text-sm text-ink-2">
                  <Icon className="mt-0.5 size-4 shrink-0 text-leaf-600" aria-hidden />
                  {text}
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <div className="min-w-0">
          {status === "running" && <Scanning />}
          {status === "error" && <ErrorState title="Diagnosis failed" error={error} onRetry={diagnose} />}
          {status === "done" && result && <ResultPanel result={result} />}
          {status === "idle" && (
            <EmptyState
              icon={ScanSearch}
              title="Your diagnosis will appear here"
              description="Choose a photo on the left, then press Diagnose. Try a held-out test photo to see a case with a known answer."
              className="min-h-[420px] bg-surface/50"
            />
          )}

          {status === "done" && result && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="mt-6">
              <Card>
                <CardHeader
                  icon={FlaskConical}
                  title="Second opinion"
                  subtitle="Run both Tomato Care models — deep learning and classical ML — on this photo"
                  action={
                    <Button variant="secondary" size="sm" onClick={() => runCompare()} loading={compareStatus === "running"} disabled={!online}>
                      {comparison ? "Run again" : "Ask both models"}
                    </Button>
                  }
                />
                <div className="p-5">
                  {compareStatus === "error" && <ErrorState error={compareError} onRetry={() => runCompare()} />}
                  {compareStatus === "running" && <p className="text-sm text-muted">Running both models on the same photo…</p>}
                  {comparison && compareStatus === "done" && <CompareTable results={comparison.results} trueKey={result.trueKey} models={models} />}
                  {compareStatus === "idle" && !comparison && <p className="text-sm text-muted">See whether the other Tomato Care model agrees on exactly this photo.</p>}
                </div>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}
