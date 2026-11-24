import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";
import { compare as compareApi, predict as predictApi } from "../lib/api.js";
import { addHistory, makeThumbnail } from "../lib/history.js";

const PredictionContext = createContext(null);

/** Holds the current photo, its diagnosis and an optional all-model comparison, shared across pages. */
export function PredictionProvider({ children }) {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [compareStatus, setCompareStatus] = useState("idle");
  const [error, setError] = useState(null);
  const [compareError, setCompareError] = useState(null);
  const controller = useRef(null);
  const compareController = useRef(null);

  const choose = useCallback((next) => {
    controller.current?.abort();
    compareController.current?.abort();
    setFile(next);
    setResult(null);
    setComparison(null);
    setStatus("idle");
    setCompareStatus("idle");
    setError(null);
    setCompareError(null);
  }, []);

  const run = useCallback(
    async (model, { source = "upload", trueKey = null } = {}) => {
      if (!file) return;
      controller.current?.abort();
      const c = new AbortController();
      controller.current = c;
      setStatus("running");
      setError(null);
      try {
        const res = await predictApi(file, { model, explain: true, signal: c.signal });
        if (c.signal.aborted) return;
        const enriched = { ...res, trueKey, source, fileName: file.name };
        setResult(enriched);
        setStatus("done");
        const thumb = await makeThumbnail(file);
        addHistory({
          thumb,
          fileName: file.name,
          model: res.model,
          prediction: res.prediction,
          status: res.prediction.status,
          trueKey,
          inference_ms: res.inference_ms,
        });
      } catch (err) {
        if (c.signal.aborted) return;
        setError(err);
        setStatus("error");
      }
    },
    [file],
  );

  const runCompare = useCallback(
    async ({ task, models } = {}) => {
      if (!file) return;
      compareController.current?.abort();
      const c = new AbortController();
      compareController.current = c;
      setCompareStatus("running");
      setCompareError(null);
      try {
        const res = await compareApi(file, { task, models, signal: c.signal });
        if (c.signal.aborted) return;
        setComparison(res);
        setCompareStatus("done");
      } catch (err) {
        if (c.signal.aborted) return;
        setCompareError(err);
        setCompareStatus("error");
      }
    },
    [file],
  );

  const value = useMemo(
    () => ({ file, choose, result, status, error, run, comparison, compareStatus, compareError, runCompare }),
    [file, choose, result, status, error, run, comparison, compareStatus, compareError, runCompare],
  );
  return <PredictionContext.Provider value={value}>{children}</PredictionContext.Provider>;
}

export function usePrediction() {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error("usePrediction must be used inside PredictionProvider");
  return ctx;
}
