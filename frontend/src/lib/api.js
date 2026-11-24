// Dev server: the FastAPI URL from .env. Production build: "/" (.env.production) -> "", i.e. same-origin /api/... calls.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");
const API_LABEL = API_BASE_URL || (typeof window !== "undefined" ? window.location.origin : "this server");

export class ApiError extends Error {
  constructor(message, { status = 0, kind = "http" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind; // "http" | "network" | "timeout"
  }
}

function detailOf(payload) {
  if (!payload) return null;
  const { detail } = payload;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d?.msg).filter(Boolean).join("; ");
  return null;
}

async function fetchJson(url, { method = "GET", body, signal, timeoutMs = 15000, networkMessage, timeoutHint = "" } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const forward = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", forward, { once: true });
  }
  try {
    const res = await fetch(url, { method, body, signal: controller.signal, headers: { Accept: "application/json" } });
    const text = await res.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    if (!res.ok) throw new ApiError(detailOf(payload) || `Request failed (HTTP ${res.status}).`, { status: res.status });
    if (payload === null) throw new ApiError("The server sent an empty or invalid response.", { status: res.status });
    return payload;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (timedOut) throw new ApiError(`The request timed out after ${Math.round(timeoutMs / 1000)} s.${timeoutHint}`, { kind: "timeout" });
    if (signal?.aborted) throw err;
    throw new ApiError(networkMessage || "Network request failed.", { kind: "network" });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forward);
  }
}

function api(path, options = {}) {
  return fetchJson(`${API_BASE_URL}${path}`, {
    networkMessage: `The diagnosis server is not reachable at ${API_LABEL}. Start it with "uvicorn backend.main:app" and try again.`,
    timeoutHint: ` The server at ${API_LABEL} may be busy.`,
    ...options,
  });
}

export const getHealth = (signal) => api("/api/health", { signal, timeoutMs: 8000 });
export const getModels = (signal) => api("/api/models", { signal });
export const getClasses = (signal) => api("/api/classes", { signal });

/** Analytics from the API; falls back to the snapshot bundled with the site (read-only pages work without a server). */
export async function getAnalytics(signal) {
  try {
    return { data: await api("/api/analytics", { signal, timeoutMs: 20000 }), source: "api" };
  } catch (err) {
    if (signal?.aborted) throw err;
    const data = await fetchJson(`${import.meta.env.BASE_URL}data/platform.json`, {
      signal,
      networkMessage: "Analytics are unavailable: the server is offline and no bundled snapshot was found.",
    });
    return { data, source: "snapshot" };
  }
}

export async function getDiseaseLibrary(signal) {
  try {
    return await getClasses(signal);
  } catch (err) {
    if (signal?.aborted) throw err;
    return fetchJson(`${import.meta.env.BASE_URL}data/disease_info.json`, {
      signal,
      networkMessage: "The disease library is unavailable.",
    });
  }
}

export const getTestSamples = (signal) =>
  fetchJson(`${import.meta.env.BASE_URL}test-samples/manifest.json`, {
    signal,
    networkMessage: "Could not load the held-out test-photo gallery.",
  });

export function predict(file, { model, explain = true, signal } = {}) {
  const form = new FormData();
  form.append("file", file, file.name || "photo.jpg");
  const params = new URLSearchParams();
  if (model) params.set("model", model);
  if (explain) params.set("explain", "true");
  const q = params.toString() ? `?${params}` : "";
  return api(`/api/predict${q}`, { method: "POST", body: form, signal, timeoutMs: 90000 });
}

export function compare(file, { models, task, signal } = {}) {
  const form = new FormData();
  form.append("file", file, file.name || "photo.jpg");
  const params = new URLSearchParams();
  if (models?.length) params.set("models", models.join(","));
  else if (task) params.set("task", task);
  const q = params.toString() ? `?${params}` : "";
  return api(`/api/compare${q}`, { method: "POST", body: form, signal, timeoutMs: 180000 });
}

export const API_DOCS_URL = `${API_BASE_URL}/api/docs`;
