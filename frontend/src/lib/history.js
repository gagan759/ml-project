// Recent diagnoses, kept only in this browser (localStorage). Thumbnails are small JPEGs made on the client.
const KEY = "tl-history-v2";
const MAX = 24;

function read() {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function write(items) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, MAX)));
    window.dispatchEvent(new Event("tl-history"));
  } catch {
    // storage full or blocked: history is a convenience only
  }
}

export const loadHistory = read;

export function addHistory(entry) {
  write([{ id: crypto.randomUUID?.() || String(Date.now()), at: new Date().toISOString(), ...entry }, ...read()]);
}

export function clearHistory() {
  write([]);
}

/** Downscale an image file to a small JPEG data URL for the history list. */
export function makeThumbnail(file, size = 96) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = size / Math.max(img.width, img.height);
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/jpeg", 0.7));
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(null);
    };
    img.src = url;
  });
}
