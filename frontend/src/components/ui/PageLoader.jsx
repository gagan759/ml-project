import { Spinner } from "./primitives.jsx";

export default function PageLoader() {
  return (
    <div className="grid min-h-[50vh] place-items-center" role="status">
      <div className="flex items-center gap-3 text-sm text-muted">
        <Spinner />
        Loading…
      </div>
    </div>
  );
}
