import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ExternalLink, Eye, GitCompareArrows, ListChecks, ScanSearch, ShieldPlus, Stethoscope } from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, ButtonLink, Callout, Card, CardHeader, EmptyState, Meter, Skeleton } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, classicalModel, defaultModel, servedName, taskClasses } from "../lib/analytics.js";
import { classMeta } from "../lib/classes.js";
import { cx, pct } from "../lib/format.js";

function List({ items }) {
  return (
    <ul className="space-y-2.5">
      {items.map((s) => (
        <li key={s} className="flex gap-2.5 text-sm leading-relaxed text-ink-2">
          <span className="mt-2 size-1.5 shrink-0 rounded-full bg-leaf-500" />
          {s}
        </li>
      ))}
    </ul>
  );
}

function lookAlikes(model, classes, key) {
  const i = classes.indexOf(key);
  if (!model || i < 0) return [];
  const row = model.confusion[i];
  const col = model.confusion.map((r) => r[i]);
  const total = row.reduce((a, b) => a + b, 0);
  return classes
    .map((k, j) => ({ key: k, missedAs: j === i ? 0 : row[j], mistakenFor: j === i ? 0 : col[j], total }))
    .filter((x) => x.missedAs + x.mistakenFor > 0)
    .sort((a, b) => b.missedAs + b.mistakenFor - (a.missedAs + a.mistakenFor))
    .slice(0, 3);
}

export default function DiseasePage() {
  const { key } = useParams();
  const { library, libraryLoading, analytics } = useData();
  const item = library?.classes?.find((c) => c.key === key);
  const meta = classMeta(key);
  useDocumentTitle(item?.name || "Disease");
  const [img, setImg] = useState(0);

  const dl = defaultModel(analytics, TASK);
  const ml = classicalModel(analytics, TASK);
  const classes = taskClasses(analytics, TASK);
  const alikes = useMemo(() => lookAlikes(dl, classes, key), [dl, classes, key]);
  const perfDl = dl?.per_class?.find((p) => p.key === key);
  const perfMl = ml?.per_class?.find((p) => p.key === key);

  if (libraryLoading && !library) return <Skeleton className="h-[520px] rounded-2xl" />;
  if (!item)
    return <EmptyState title="Condition not found" description={`No library entry for “${key}”.`} action={<ButtonLink to="/library" variant="secondary" icon={ArrowLeft}>Back to the library</ButtonLink>} />;
  const Icon = meta.icon;

  return (
    <div>
      <Link to="/library" className="mb-6 inline-flex items-center gap-1.5 text-sm font-medium text-muted hover:text-ink">
        <ArrowLeft className="size-4" aria-hidden /> Disease library
      </Link>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,440px)]">
        <div>
          <div className="flex items-center gap-3">
            <span className="grid size-11 place-items-center rounded-xl" style={{ background: `${meta.color}1a`, color: meta.color }}>
              <Icon className="size-5" aria-hidden />
            </span>
            <div>
              <h1 className="font-display text-3xl font-semibold tracking-tight text-ink">{item.name}</h1>
              <p className="text-sm font-medium" style={{ color: meta.color }}>
                {item.category} · {item.cause}
              </p>
            </div>
          </div>
          <p className="mt-5 text-[15px] leading-relaxed text-ink-2">{item.summary}</p>

          <div className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
            <Card>
              <CardHeader icon={Stethoscope} title="Symptoms" />
              <div className="p-5">
                <List items={item.symptoms} />
              </div>
            </Card>
            <Card>
              <CardHeader icon={Eye} title="What it looks like in a photo" />
              <div className="p-5">
                <List items={item.visual_characteristics} />
              </div>
            </Card>
          </div>
          <Card className="mt-6">
            <CardHeader icon={ShieldPlus} title={item.is_healthy ? "Keeping plants healthy" : "Management"} />
            <div className="p-5">
              <List items={item.management} />
            </div>
          </Card>
          <Callout tone="sun" className="mt-6" title="Decision support only">
            Confirm any diagnosis with a local agricultural extension service (in India, a Krishi Vigyan Kendra) before treating a crop, and follow product labels and local regulations.
          </Callout>
          <Card className="mt-6">
            <CardHeader icon={ListChecks} title="References" />
            <ul className="divide-y divide-line">
              {item.references.map((r) => (
                <li key={r.url}>
                  <a href={r.url} target="_blank" rel="noreferrer" className="flex items-start justify-between gap-3 px-5 py-3 text-sm hover:bg-surface-2">
                    <span>
                      <span className="font-medium text-ink">{r.title}</span>
                      <span className="block text-xs text-muted">{r.publisher}</span>
                    </span>
                    <ExternalLink className="mt-0.5 size-4 shrink-0 text-muted" aria-hidden />
                  </a>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <Card className="overflow-hidden">
            <div className="aspect-square bg-surface-3">
              <img src={item.images[img]} alt={`${item.name} example ${img + 1}`} className="size-full object-cover" />
            </div>
            <div className="flex gap-2 p-3">
              {item.images.map((src, i) => (
                <button key={src} onClick={() => setImg(i)} className={cx("size-16 overflow-hidden rounded-lg ring-2 transition", i === img ? "ring-leaf-500" : "ring-transparent hover:ring-line-strong")} aria-label={`Show example ${i + 1}`}>
                  <img src={src} alt="" className="size-full object-cover" />
                </button>
              ))}
            </div>
            <p className="px-4 pb-3 text-[11px] text-muted">Dataset photos (Tomato-Village, Gehlot et al. 2023), shown as the models see them: 256×256.</p>
          </Card>

          <Card>
            <CardHeader icon={ScanSearch} title="How well the models recognise it" subtitle="Held-out test photos" />
            <div className="space-y-5 p-5">
              {[
                [servedName(dl), dl, perfDl],
                [servedName(ml), ml, perfMl],
              ].map(([label, m, p]) =>
                p ? (
                  <div key={label}>
                    <div className="mb-2 flex items-center justify-between text-sm">
                      <span className="font-medium text-ink">{label}</span>
                      <Badge tone={p.support < 10 ? "sun" : "slate"}>{p.support} photos</Badge>
                    </div>
                    {["recall", "precision", "f1"].map((k) => (
                      <div key={k} className="mb-1.5 flex items-center gap-3 text-xs">
                        <span className="w-16 text-muted capitalize">{k === "f1" ? "F1" : k}</span>
                        <Meter value={p[k]} color={meta.color} className="flex-1" label={`${label} ${k}`} />
                        <span className="w-10 text-right font-mono tabular text-ink-2">{pct(p[k], 0)}</span>
                      </div>
                    ))}
                    <p className="mt-1 text-[11px] text-muted">{m.name}</p>
                  </div>
                ) : m ? (
                  <p key={label} className="text-sm text-muted">
                    {label}: this condition is outside its classes.
                  </p>
                ) : null,
              )}
            </div>
          </Card>

          {alikes.length > 0 && (
            <Card>
              <CardHeader icon={GitCompareArrows} title="Most often confused with" subtitle="From the deep-learning model's test confusion matrix" />
              <ul className="divide-y divide-line">
                {alikes.map((a) => (
                  <li key={a.key} className="flex items-center justify-between gap-3 px-5 py-3 text-sm">
                    <Link to={`/library/${a.key}`} className="flex items-center gap-2 font-medium text-ink hover:underline">
                      <span className="size-2.5 rounded-full" style={{ background: classMeta(a.key).color }} />
                      {classMeta(a.key).name}
                    </Link>
                    <span className="text-right text-xs text-muted">
                      {a.missedAs} missed as · {a.mistakenFor} mistaken for
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
