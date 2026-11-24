import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, ClipboardPaste, Focus, ImagePlus, Leaf, Maximize2, RefreshCw, StickyNote, Sun, UploadCloud, X } from "lucide-react";
import { bytes, cx } from "../../lib/format.js";
import { Button } from "../ui/primitives.jsx";

// How the training photos were taken; photos taken this way get the most reliable results.
const PHOTO_TIPS = [
  { icon: Leaf, text: "One leaf only" },
  { icon: StickyNote, text: "Plain, light paper" },
  { icon: Sun, text: "Daylight, no glare" },
  { icon: Maximize2, text: "Leaf fills the frame" },
  { icon: Focus, text: "Sharp, in focus" },
];
const MAX_BYTES = 10 * 1024 * 1024;
const TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

export function validateFile(file) {
  if (!file) return "No file selected.";
  if (file.type && !TYPES.includes(file.type)) {
    if (/heic|heif/i.test(file.type) || /\.hei[cf]$/i.test(file.name)) return "HEIC photos are not supported yet: export as JPEG (most phones: Settings → Camera → Formats → Most compatible).";
    return `Unsupported file type (${file.type}). Use a JPEG, PNG, WEBP or BMP image.`;
  }
  if (file.size > MAX_BYTES) return `File is ${bytes(file.size)}; the maximum is 10 MB.`;
  if (file.size === 0) return "The file is empty.";
  return null;
}

function useObjectUrl(file) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!file) return setUrl(null);
    const u = URL.createObjectURL(file);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);
  return url;
}

/** Drag-and-drop, browse, camera capture and clipboard paste, with client-side validation. */
export default function UploadPanel({ file, onFile, disabled = false }) {
  const inputRef = useRef(null);
  const cameraRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState(null);
  const preview = useObjectUrl(file);

  const accept = useCallback(
    (f) => {
      const problem = validateFile(f);
      setError(problem);
      if (!problem) onFile(f);
    },
    [onFile],
  );

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) {
        const blob = item.getAsFile();
        if (blob) accept(new File([blob], `pasted.${blob.type.split("/")[1] || "png"}`, { type: blob.type }));
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [accept]);

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    if (disabled) return;
    const f = e.dataTransfer.files?.[0];
    if (f) accept(f);
  };

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={cx(
          "relative overflow-hidden rounded-2xl border-2 border-dashed transition",
          drag ? "border-leaf-500 bg-leaf-500/5" : "border-line-strong bg-surface-2/50",
          preview ? "p-3" : "px-6 py-10",
        )}
      >
        {preview ? (
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
            <div className="relative mx-auto aspect-square w-full max-w-[220px] shrink-0 overflow-hidden rounded-xl bg-surface-3 sm:mx-0 sm:w-44">
              <img src={preview} alt="Selected leaf" className="size-full object-cover" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-ink" title={file.name}>
                {file.name}
              </p>
              <p className="mt-0.5 text-sm text-muted">
                {bytes(file.size)} · {file.type || "image"}
              </p>
              <p className="mt-3 text-xs leading-relaxed text-muted">
                The photo is resized to 256×256 the way the dataset was made (whole 3:4 frame, Display P3 colour, JPEG q90), so it reaches the model like a training photo.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button variant="secondary" size="sm" icon={RefreshCw} onClick={() => inputRef.current?.click()} disabled={disabled}>
                  Replace
                </Button>
                <Button variant="ghost" size="sm" icon={X} onClick={() => onFile(null)} disabled={disabled}>
                  Remove
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center text-center">
            <span className="grid size-14 place-items-center rounded-2xl bg-leaf-500/10 text-leaf-600 dark:text-leaf-400">
              <UploadCloud className="size-7" aria-hidden />
            </span>
            <p className="mt-4 font-display text-base font-semibold text-ink">Drop a leaf photo here</p>
            <p className="mt-1 max-w-sm text-sm text-muted">One detached tomato leaf on plain paper, in daylight, filling most of the frame. JPEG, PNG, WEBP or BMP up to 10 MB.</p>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              <Button icon={ImagePlus} onClick={() => inputRef.current?.click()} disabled={disabled}>
                Browse files
              </Button>
              <Button variant="secondary" icon={Camera} onClick={() => cameraRef.current?.click()} disabled={disabled} className="sm:hidden">
                Take photo
              </Button>
            </div>
            <ul className="mt-5 grid w-full max-w-md grid-cols-2 gap-2 text-left text-xs text-ink-2">
              {PHOTO_TIPS.map(({ icon: TipIcon, text }) => (
                <li key={text} className="flex items-center gap-2 rounded-lg bg-surface px-2.5 py-2 ring-1 ring-line">
                  <TipIcon className="size-4 shrink-0 text-leaf-600 dark:text-leaf-400" aria-hidden />
                  {text}
                </li>
              ))}
            </ul>
            <p className="mt-4 hidden items-center gap-1.5 text-xs text-muted sm:flex">
              <ClipboardPaste className="size-3.5" aria-hidden /> You can also paste an image (Ctrl/⌘ + V)
            </p>
          </div>
        )}
        <input ref={inputRef} type="file" accept={TYPES.join(",")} className="sr-only" onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])} tabIndex={-1} />
        <input ref={cameraRef} type="file" accept="image/*" capture="environment" className="sr-only" onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])} tabIndex={-1} />
      </div>
      {error && (
        <p role="alert" className="mt-2 text-sm text-tomato-600">
          {error}
        </p>
      )}
    </div>
  );
}
