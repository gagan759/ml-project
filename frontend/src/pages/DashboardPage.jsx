import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, ArrowRight, Cpu, Database, FlaskConical, Gauge, History, Layers3, ScanSearch, ShieldCheck, Sparkles, Target, Trash2 } from "lucide-react";
import { ClassDistribution } from "../components/charts/charts.jsx";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, Button, ButtonLink, Card, CardHeader, EmptyState, ErrorState, PageHeader, Skeleton, Stat } from "../components/ui/primitives.jsx";
import { useData } from "../context/DataContext.jsx";
import { TASK, classicalModel, comparisonsFor, defaultModel, isClassicalModel, previousDefault, ranked, servedName, taskClasses, taskInfo } from "../lib/analytics.js";
import { STATUS_STYLE, classMeta } from "../lib/classes.js";
import { ci, ms, pct, signedPct, timeAgo } from "../lib/format.js";
import { clearHistory, loadHistory } from "../lib/history.js";

function useHistory() {
  const [items, setItems] = useState(loadHistory);
  useEffect(() => {
    const update = () => setItems(loadHistory());
    window.addEventListener("tl-history", update);
    window.addEventListener("storage", update);
    return () => {
      window.removeEventListener("tl-history", update);
      window.removeEventListener("storage", update);
    };
  }, []);
  return items;
}

function RecentDiagnoses() {
  const items = useHistory();
  if (!items.length)
    return (
      <EmptyState
        icon={History}
        title="No diagnoses yet"
        description="Diagnoses you run are listed here (stored only in this browser)."
        action={<ButtonLink to="/diagnose" size="sm" icon={ScanSearch}>Diagnose a leaf</ButtonLink>}
        className="border-0 py-10"
      />
    );
  return (
    <div>
      <ul className="divide-y divide-line">
        {items.slice(0, 7).map((h) => {
          const meta = classMeta(h.prediction?.key);
          const tone = { Healthy: "leaf", Diseased: "tomato", Uncertain: "sun", "Unusual image": "slate" }[h.status] || "slate";
          return (
            <li key={h.id} className="flex items-center gap-3 px-5 py-3">
              {h.thumb ? <img src={h.thumb} alt="" className="size-11 rounded-lg object-cover" /> : <span className="size-11 rounded-lg bg-surface-3" />}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-ink">{h.status === "Healthy" || h.status === "Diseased" ? meta.name : `${h.status} (${meta.short}?)`}</p>
                <p className="truncate text-xs text-muted">
                  {h.model?.name} · {timeAgo(h.at)}
                  {h.trueKey && ` · true: ${classMeta(h.trueKey).short}`}
                </p>
              </div>
              <Badge tone={tone}>{pct(h.prediction?.confidence, 0)}</Badge>
            </li>
          );
        })}
      </ul>
      <div className="flex justify-end border-t border-line px-5 py-2.5">
        <Button variant="ghost" size="sm" icon={Trash2} onClick={clearHistory}>
          Clear history
        </Button>
      </div>
    </div>
  );
}

function Leaderboard({ analytics, family }) {
  const rows = ranked(analytics, TASK)
    .filter((m) => family === "all" || (family === "ml" ? isClassicalModel(m) : !isClassicalModel(m)))
    .slice(0, 5);
  return (
    <ul className="divide-y divide-line">
      {rows.map((m, i) => (
        <li key={m.key} className="flex items-center gap-3 px-5 py-3">
          <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-surface-2 font-mono text-xs font-semibold text-muted">{i + 1}</span>
          <div className="min-w-0 flex-1">
            <p className="flex items-center gap-1.5 truncate text-sm font-medium text-ink">
              {m.name}
              {m.generation === "upgrade" && <Sparkles className="size-3.5 shrink-0 text-leaf-600" aria-label="upgraded" />}
            </p>
            <p className="text-xs text-muted">{m.family}</p>
          </div>
          <div className="text-right">
            <p className="font-mono text-sm font-semibold tabular text-ink">{pct(m.test.f1_macro)}</p>
            <p className="text-[11px] text-muted">macro-F1</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function DashboardPage() {
  useDocumentTitle("Dashboard");
  const { analytics, analyticsError, analyticsLoading, reloadAnalytics, health, analyticsSource } = useData();
  const [family, setFamily] = useState("all");

  const d5 = defaultModel(analytics, TASK);
  const cm = classicalModel(analytics, TASK);
  const p5 = previousDefault(analytics, TASK);
  const gain = useMemo(() => comparisonsFor(analytics, TASK).find((c) => c.a === d5?.key && c.b === p5?.key), [analytics, d5, p5]);
  const classes = analytics?.dataset?.classes || [];
  const diagnosed = taskClasses(analytics, TASK);

  return (
    <div>
      <PageHeader
        eyebrow="Overview"
        title="Dashboard"
        description="Live system status, headline results of the deployed models on held-out test photos, and your recent diagnoses."
        actions={
          <>
            <ButtonLink to="/lab" variant="secondary" icon={FlaskConical}>
              Model lab
            </ButtonLink>
            <ButtonLink to="/diagnose" icon={ScanSearch}>
              New diagnosis
            </ButtonLink>
          </>
        }
      />

      {analyticsError && <ErrorState title="Could not load results" error={analyticsError} onRetry={reloadAnalytics} className="mb-6" />}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {analyticsLoading && !analytics ? (
          Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-[132px] rounded-2xl" />)
        ) : (
          <>
            <Stat
              icon={Target}
              label="Deep learning (default) · test accuracy"
              value={pct(d5?.test?.accuracy)}
              hint={`5 conditions · 95% CI ${ci(d5?.test?.accuracy_ci)}`}
            />
            <Stat
              icon={ShieldCheck}
              tone="sky"
              label="When it answers"
              value={pct(d5?.selective?.test?.accuracy_kept ?? d5?.selective?.accuracy_kept)}
              hint={d5?.selective ? `Answers ${pct(d5.selective.test?.coverage ?? d5.selective.coverage, 0)} of test photos; says “Uncertain” for the rest` : "Selective prediction not configured"}
            />
            <Stat
              icon={Layers3}
              tone="violet"
              label={`${servedName(cm)} · test accuracy`}
              value={pct(cm?.test?.accuracy)}
              hint={cm ? `No neural network · macro-F1 ${pct(cm.test?.f1_macro)} · 95% CI ${ci(cm.test?.accuracy_ci)}` : "Not in this results file"}
            />
            <Stat
              icon={Cpu}
              tone="sun"
              label={`${servedName(d5)} · CPU time`}
              value={ms(d5?.latency_ms)}
              hint={`Median per photo on a laptop CPU${cm?.latency_ms != null ? ` · ${servedName(cm)} ${ms(cm.latency_ms)}` : ""}`}
            />
          </>
        )}
      </div>

      {gain && (
        <Card className="mt-6 flex flex-col gap-4 p-5 sm:flex-row sm:items-center">
          <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-leaf-500/10 text-leaf-600">
            <Sparkles className="size-5" aria-hidden />
          </span>
          <div className="flex-1 text-sm leading-relaxed text-ink-2">
            <strong className="text-ink">Upgrade vs original default (same {taskInfo(analytics, TASK)?.test_photos ?? "—"} test photos):</strong> accuracy {signedPct(gain.accuracy_diff)}, macro-F1 {signedPct(gain.f1_diff)} (95% CI {signedPct(gain.f1_ci[0])} to {signedPct(gain.f1_ci[1])}).{" "}
            {gain.f1_ci[0] > 0 ? "The improvement is statistically reliable." : "The confidence interval includes zero, so this difference is not statistically reliable."}
          </div>
          <ButtonLink to="/lab" variant="ghost" size="sm" iconRight={ArrowRight}>
            Details
          </ButtonLink>
        </Card>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader
            icon={Gauge}
            title="Leaderboard (held-out test photos)"
            subtitle="Ranked by macro-F1, which weighs every class equally"
            action={
              <div className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5 text-xs">
                {[
                  ["all", "All"],
                  ["dl", "Deep learning"],
                  ["ml", "Classical ML"],
                ].map(([f, label]) => (
                  <button key={f} onClick={() => setFamily(f)} className={`rounded-md px-2.5 py-1 font-medium ${family === f ? "bg-surface text-ink shadow-sm" : "text-muted"}`}>
                    {label}
                  </button>
                ))}
              </div>
            }
          />
          {analytics ? <Leaderboard analytics={analytics} family={family} /> : <div className="p-5"><Skeleton className="h-48" /></div>}
          <div className="border-t border-line px-5 py-3 text-right">
            <Link to="/lab" className="inline-flex items-center gap-1 text-sm font-medium text-leaf-700 hover:underline dark:text-leaf-300">
              Open the model lab <ArrowRight className="size-4" aria-hidden />
            </Link>
          </div>
        </Card>

        <Card>
          <CardHeader icon={Activity} title="System status" />
          <dl className="divide-y divide-line text-sm">
            {[
              ["Diagnosis API", health.status],
              ["Models loaded", health.data ? `${health.data.loaded_models.length}` : "—"],
              ["Compute", health.data?.device?.toUpperCase() || "—"],
              ["API version", health.data?.version || "—"],
              ["Results source", analyticsSource === "api" ? "live API" : analyticsSource === "snapshot" ? "bundled snapshot" : "—"],
            ].map(([k, v]) => (
              <div key={k} className="flex items-center justify-between px-5 py-3">
                <dt className="text-muted">{k}</dt>
                <dd className="font-medium capitalize text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          {Object.keys(health.data?.load_errors || {}).length > 0 && (
            <p className="border-t border-line px-5 py-3 text-xs text-tomato-600">Load errors: {Object.keys(health.data.load_errors).join(", ")}</p>
          )}
        </Card>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader
            icon={Database}
            title="Dataset"
            subtitle={
              analytics
                ? `${analytics.dataset.photos.toLocaleString()} unique photos · ${analytics.dataset.files.toLocaleString()} files · leakage-free split by leaf group${diagnosed.length && classes.length ? ` · the app diagnoses ${diagnosed.length} of the ${classes.length} classes` : ""}`
                : ""
            }
          />
          <div className="p-4">{classes.length ? <ClassDistribution classes={classes} /> : <Skeleton className="h-64" />}</div>
        </Card>
        <Card>
          <CardHeader icon={History} title="Recent diagnoses" subtitle="On this device" />
          <RecentDiagnoses />
        </Card>
      </div>

      {analytics && (
        <Card className="mt-6 p-5">
          <p className="mb-3 text-sm font-semibold text-ink">Status meanings</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {Object.entries(STATUS_STYLE).map(([k, s]) => (
              <div key={k} className="rounded-xl bg-surface-2 p-3">
                <Badge tone={s.tone}>{s.label}</Badge>
                <p className="mt-2 text-xs leading-relaxed text-muted">{s.description}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
