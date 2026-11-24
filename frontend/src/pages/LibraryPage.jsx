import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, BookOpenText, Search } from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, Card, EmptyState, ErrorState, PageHeader, Segmented, Skeleton } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, defaultModel } from "../lib/analytics.js";
import { classMeta } from "../lib/classes.js";
import { pct } from "../lib/format.js";

const CATEGORIES = ["All", "Fungal", "Oomycete", "Viral", "Insect pest", "Nutrient deficiency", "Healthy"];

export default function LibraryPage() {
  useDocumentTitle("Disease library");
  const { library, libraryError, libraryLoading, analytics } = useData();
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("All");
  const model = defaultModel(analytics, TASK);
  // Only the categories of the conditions in the library (the five diagnosed ones), so no filter can come up empty.
  const categories = useMemo(() => {
    const present = new Set((library?.classes || []).map((c) => classMeta(c.key).category));
    return CATEGORIES.filter((c) => c === "All" || present.has(c));
  }, [library]);

  const items = useMemo(() => {
    const list = library?.classes || [];
    const needle = q.trim().toLowerCase();
    return list.filter((c) => {
      const meta = classMeta(c.key);
      if (cat !== "All" && meta.category !== cat) return false;
      if (!needle) return true;
      return [c.name, c.summary, c.cause, ...(c.symptoms || [])].join(" ").toLowerCase().includes(needle);
    });
  }, [library, q, cat]);

  return (
    <div>
      <PageHeader
        eyebrow="Knowledge base"
        title="Disease library"
        description="The five leaf conditions Tomato Care diagnoses: how to recognise them, what causes them and how they are managed — with references to extension services and the models' own reliability for each condition."
      />
      <div className="mb-6 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <label className="relative block w-full max-w-md">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search symptoms, causes, names…"
            className="h-11 w-full rounded-xl border border-line-strong bg-surface pl-10 pr-3 text-sm text-ink outline-none placeholder:text-muted focus:border-leaf-500 focus:ring-2 focus:ring-leaf-500/20"
            aria-label="Search the disease library"
          />
        </label>
        <div className="overflow-x-auto scrollbar-thin">
          <Segmented size="sm" options={categories.map((c) => ({ value: c, label: c }))} value={cat} onChange={setCat} label="Category" />
        </div>
      </div>

      {libraryError && <ErrorState title="Library unavailable" error={libraryError} />}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {libraryLoading && !library && Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-80 rounded-2xl" />)}
        {items.map((c) => {
          const meta = classMeta(c.key);
          const Icon = meta.icon;
          const perf = model?.per_class?.find((p) => p.key === c.key);
          return (
            <Card key={c.key} as={Link} to={`/library/${c.key}`} interactive className="group flex flex-col overflow-hidden">
              <div className="relative aspect-[4/3] overflow-hidden bg-surface-3">
                {c.images?.[0] && <img src={c.images[0]} alt={c.name} className="size-full object-cover transition duration-500 group-hover:scale-105" loading="lazy" />}
                <span className="absolute left-3 top-3 grid size-8 place-items-center rounded-lg bg-surface/90 shadow-sm backdrop-blur" style={{ color: meta.color }}>
                  <Icon className="size-4" aria-hidden />
                </span>
              </div>
              <div className="flex flex-1 flex-col p-4">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="font-display font-semibold text-ink">{c.name}</h3>
                  <ArrowUpRight className="size-4 shrink-0 text-muted transition group-hover:text-ink" aria-hidden />
                </div>
                <p className="mt-1 text-xs font-medium" style={{ color: meta.color }}>
                  {c.category}
                </p>
                <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-muted">{c.summary}</p>
                {perf && (
                  <div className="mt-auto flex flex-wrap gap-1.5 pt-4">
                    <Badge tone="slate">recall {pct(perf.recall, 0)}</Badge>
                    <Badge tone={perf.support < 10 ? "sun" : "slate"}>{perf.support} test photos</Badge>
                  </div>
                )}
              </div>
            </Card>
          );
        })}
      </div>
      {library && !items.length && <EmptyState icon={BookOpenText} title="No match" description="Try another search term or category." />}
    </div>
  );
}
