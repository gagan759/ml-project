import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, CheckCircle2, Eye, ScanSearch, TriangleAlert, XCircle } from "lucide-react";
import GradCamViewer from "../components/diagnose/GradCamViewer.jsx";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, ButtonLink, Callout, Card, CardHeader, EmptyState, PageHeader, Segmented, Skeleton } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, modelByKey, taskClasses } from "../lib/analytics.js";
import { classMeta } from "../lib/classes.js";
import { cx, pct } from "../lib/format.js";

const STEPS = [
  { title: "Forward pass", text: "The photo goes through the network; the last feature layer holds a coarse grid of activation maps (8×8 for ConvNeXt at 256 px, 18×18 tokens for the ViT)." },
  { title: "Gradient of the class score", text: "We back-propagate the score of the predicted class to that layer: the gradient says how much each map raises the score." },
  { title: "Weight and sum", text: "Each map is weighted by its average gradient, the weighted maps are summed and negative values are cut (ReLU)." },
  { title: "Upsample and overlay", text: "The coarse map is upsampled to 256×256 and blended over the photo: bright = most influential region." },
];

export default function ExplainabilityPage() {
  useDocumentTitle("Explainability");
  const { analytics, analyticsLoading } = useData();
  // Only the five diagnosed conditions and five-condition networks (an older results file may hold more).
  const gallery = useMemo(() => {
    const g = analytics?.gradcam;
    if (!g) return g;
    const keys = taskClasses(analytics, TASK);
    const inTask = (key) => {
      const m = modelByKey(analytics, key);
      return !m || m.task === TASK;
    };
    const items = (g.items || [])
      .filter((it) => !keys.length || keys.includes(it.true_key))
      .map((it) => ({ ...it, overlays: (it.overlays || []).filter((o) => inTask(o.model)) }));
    return { ...g, items };
  }, [analytics]);
  const modelsInGallery = useMemo(() => {
    const seen = new Map();
    gallery?.items?.forEach((it) => it.overlays.forEach((o) => seen.set(o.model, o.model_name)));
    return [...seen.entries()].map(([value, label]) => ({ value, label }));
  }, [gallery]);
  const [model, setModel] = useState(null);
  const [active, setActive] = useState(null);

  useEffect(() => {
    if (!model && modelsInGallery.length) setModel(modelsInGallery[0].value);
  }, [modelsInGallery, model]);
  useEffect(() => {
    if (!active && gallery?.items?.length) setActive(gallery.items[0].sample);
  }, [gallery, active]);

  const item = gallery?.items?.find((i) => i.sample === active);
  const ov = item?.overlays.find((o) => o.model === model);

  return (
    <div>
      <PageHeader
        eyebrow="Explainable AI"
        title="Explainability"
        description="Grad-CAM shows which parts of the photo pushed the network towards its answer. It is a sanity check — does the model look at the leaf and its lesions, or at the paper and shadows? — not proof of why a leaf is diseased. It needs a neural network, so it is shown for the deep-learning model; the classical ML model has none."
        actions={<ButtonLink to="/diagnose" icon={ScanSearch}>Explain your own photo</ButtonLink>}
      />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {STEPS.map((s, i) => (
          <Card key={s.title} className="p-5">
            <span className="grid size-8 place-items-center rounded-lg bg-leaf-500/10 font-mono text-sm font-semibold text-leaf-700 dark:text-leaf-300">{i + 1}</span>
            <p className="mt-3 font-display font-semibold text-ink">{s.title}</p>
            <p className="mt-1.5 text-sm leading-relaxed text-muted">{s.text}</p>
          </Card>
        ))}
      </div>

      <Card className="mt-6">
        <CardHeader
          icon={Eye}
          title="Held-out test photos"
          subtitle={gallery ? `${gallery.items.length} photos, one per condition, never used for training or model choice` : "Precomputed for the deep-learning model"}
          action={modelsInGallery.length > 1 && <Segmented size="sm" options={modelsInGallery} value={model} onChange={setModel} label="Model" />}
        />
        {analyticsLoading && !analytics && <div className="p-5"><Skeleton className="h-80" /></div>}
        {analytics && !gallery?.items?.length && <EmptyState icon={BrainCircuit} title="No Grad-CAM gallery in this results file" className="m-5" />}
        {item && (
          <div className="grid grid-cols-1 gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)]">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {gallery.items.map((it) => {
                const o = it.overlays.find((x) => x.model === model);
                const meta = classMeta(it.true_key);
                return (
                  <button
                    key={it.sample}
                    onClick={() => setActive(it.sample)}
                    className={cx("group overflow-hidden rounded-xl border-2 text-left transition", active === it.sample ? "border-leaf-500" : "border-transparent hover:border-line-strong")}
                  >
                    <div className="relative aspect-square bg-black">
                      <img src={o?.overlay || it.input} alt={`Grad-CAM for ${meta.name}`} className="size-full object-cover" loading="lazy" />
                      {o && (
                        <span className="absolute right-1.5 top-1.5">
                          {o.correct ? <CheckCircle2 className="size-4 text-leaf-400 drop-shadow" aria-label="correct" /> : <XCircle className="size-4 text-tomato-400 drop-shadow" aria-label="wrong" />}
                        </span>
                      )}
                    </div>
                    <div className="bg-surface px-2 py-1.5">
                      <p className="truncate text-xs font-medium text-ink">{meta.name}</p>
                      {o && <p className="truncate text-[11px] text-muted">→ {classMeta(o.pred_key).short} {pct(o.confidence, 0)}</p>}
                    </div>
                  </button>
                );
              })}
            </div>
            <div>
              {ov ? (
                <>
                  <GradCamViewer input={item.input} heatmap={ov.heatmap} overlay={ov.overlay} />
                  <div className="mt-4 flex flex-wrap items-center gap-2 text-sm">
                    <Badge tone="slate">true: {classMeta(item.true_key).name}</Badge>
                    <Badge tone={ov.correct ? "leaf" : "tomato"}>
                      predicted: {classMeta(ov.pred_key).name} ({pct(ov.confidence, 0)})
                    </Badge>
                    {ov.out_of_scope && <Badge tone="sun" icon={TriangleAlert}>class outside this model</Badge>}
                  </div>
                  {ov.on_leaf_share != null && (
                    <p className="mt-3 text-xs text-muted">
                      {pct(ov.on_leaf_share, 0)} of the heat-map mass falls on the leaf (leaf mask from saturation thresholding).
                    </p>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted">No overlay for this model.</p>
              )}
            </div>
          </div>
        )}
      </Card>

      {gallery?.summary && (
        <Callout tone="leaf" title="What the gallery shows" className="mt-6">
          {gallery.summary}
        </Callout>
      )}
      <Callout tone="sun" title="Limits of Grad-CAM" className="mt-4">
        The map is coarse (one value per 32×32-pixel patch for the convolutional networks), depends on the chosen layer, and highlights what increased the score — including spurious cues. Two models can be right for different reasons; a highlighted lesion does not by itself confirm a diagnosis.
      </Callout>
    </div>
  );
}
