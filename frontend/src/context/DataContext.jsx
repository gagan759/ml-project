import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { getAnalytics, getDiseaseLibrary, getHealth, getModels } from "../lib/api.js";

const DataContext = createContext(null);
const HEALTH_INTERVAL_MS = 30000;

function useResource(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    loader(controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => {
        if (!controller.signal.aborted) setState({ data: null, error, loading: false });
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps]);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { ...state, reload };
}

export function DataProvider({ children }) {
  const [health, setHealth] = useState({ status: "checking", data: null });
  const timer = useRef(null);

  const checkHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth({ status: data.model_loaded ? "online" : "degraded", data });
    } catch {
      setHealth({ status: "offline", data: null });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    timer.current = setInterval(checkHealth, HEALTH_INTERVAL_MS);
    return () => clearInterval(timer.current);
  }, [checkHealth]);

  const online = health.status === "online" || health.status === "degraded";
  const analytics = useResource(getAnalytics);
  const library = useResource(getDiseaseLibrary);
  const models = useResource((signal) => (online ? getModels(signal) : Promise.reject(new Error("offline"))), [online]);

  const value = useMemo(
    () => ({
      health,
      checkHealth,
      online,
      analytics: analytics.data?.data ?? null,
      analyticsSource: analytics.data?.source ?? null,
      analyticsError: analytics.error,
      analyticsLoading: analytics.loading,
      reloadAnalytics: analytics.reload,
      library: library.data,
      libraryError: library.error,
      libraryLoading: library.loading,
      models: models.data,
      modelsError: online ? models.error : null,
      modelsLoading: online && models.loading,
      reloadModels: models.reload,
    }),
    [health, checkHealth, online, analytics, library, models],
  );
  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useData() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useData must be used inside DataProvider");
  return ctx;
}
