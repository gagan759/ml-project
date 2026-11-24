import { useEffect, useState } from "react";
import { Check, Images } from "lucide-react";
import { getTestSamples } from "../../lib/api.js";
import { classMeta } from "../../lib/classes.js";
import { cx } from "../../lib/format.js";
import { Skeleton } from "../ui/primitives.jsx";

/** Demo examples: held-out test photographs (byte-identical copies, never used for training or selection) with their true labels. */
export default function SampleGallery({ onPick, activeKey, disabled, scope, columns = "grid-cols-4 sm:grid-cols-8" }) {
  const [state, setState] = useState({ samples: null, error: null });
  const [loading, setLoading] = useState(null);

  useEffect(() => {
    const c = new AbortController();
    getTestSamples(c.signal)
      .then((d) => setState({ samples: d.samples, error: null }))
      .catch((error) => !c.signal.aborted && setState({ samples: null, error }));
    return () => c.abort();
  }, []);

  const pick = async (s) => {
    if (disabled) return;
    setLoading(s.key);
    try {
      const res = await fetch(s.file);
      const blob = await res.blob();
      onPick(new File([blob], `${s.key}-test.jpg`, { type: blob.type || "image/jpeg" }), s);
    } finally {
      setLoading(null);
    }
  };

  if (state.error) return <p className="text-sm text-muted">Sample photos unavailable.</p>;
  return (
    <div>
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-ink">
        <Images className="size-4 text-muted" aria-hidden /> Example photos
        <span className="text-xs font-normal text-muted">· demo examples: held-out photos with a clear result, true label shown (accuracy: Model lab)</span>
      </div>
      <div className={cx("grid gap-2", columns)}>
        {!state.samples &&
          Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="aspect-square rounded-xl" />)}
        {state.samples?.map((s) => {
          const meta = classMeta(s.key);
          const out = scope && !scope.includes(s.key);
          return (
            <button
              key={s.key}
              type="button"
              onClick={() => pick(s)}
              disabled={disabled}
              title={`${meta.name}${out ? " (outside the selected model's classes)" : ""}`}
              className={cx(
                "group relative aspect-square overflow-hidden rounded-xl ring-2 transition",
                activeKey === s.key ? "ring-leaf-500" : "ring-transparent hover:ring-line-strong",
                disabled && "opacity-60",
              )}
            >
              <img src={s.file} alt={meta.name} className="size-full object-cover transition group-hover:scale-105" loading="lazy" />
              <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 to-transparent px-1.5 pb-1 pt-4 text-left text-[10px] font-semibold leading-tight text-white">
                {meta.abbr}
                {out && <span className="ml-1 font-normal opacity-80">· outside</span>}
              </span>
              {(activeKey === s.key || loading === s.key) && (
                <span className="absolute right-1 top-1 grid size-5 place-items-center rounded-full bg-leaf-500 text-white">
                  <Check className="size-3" aria-hidden />
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
