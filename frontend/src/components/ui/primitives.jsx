import { forwardRef } from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Info, Loader2, RefreshCw } from "lucide-react";
import { cx } from "../../lib/format.js";

/* ---------------------------------------------------------------- Card */
export function Card({ as: Tag = "div", className = "", interactive = false, children, ...rest }) {
  return (
    <Tag
      className={cx(
        "rounded-2xl border border-line bg-surface shadow-card",
        interactive && "transition hover:-translate-y-0.5 hover:border-line-strong hover:shadow-pop",
        className,
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({ title, subtitle, icon: Icon, action, className = "" }) {
  return (
    <div className={cx("flex flex-wrap items-start justify-between gap-x-4 gap-y-3 border-b border-line px-5 py-4", className)}>
      <div className="flex min-w-[12rem] flex-1 items-start gap-3">
        {Icon && (
          <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-leaf-500/10 text-leaf-600 dark:text-leaf-400">
            <Icon className="size-4" aria-hidden />
          </span>
        )}
        <div className="min-w-0">
          <h3 className="font-display text-[15px] font-semibold text-ink">{title}</h3>
          {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
        </div>
      </div>
      {action && <div className="max-w-full shrink-0 overflow-x-auto scrollbar-thin">{action}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- Button */
const BTN_VARIANTS = {
  primary:
    "bg-leaf-600 text-white shadow-sm hover:bg-leaf-700 active:bg-leaf-800 disabled:bg-leaf-600/50 dark:bg-leaf-500 dark:hover:bg-leaf-400 dark:text-leaf-950",
  secondary: "bg-surface-2 text-ink hover:bg-surface-3 border border-line",
  outline: "border border-line-strong text-ink hover:bg-surface-2",
  ghost: "text-ink-2 hover:bg-surface-2 hover:text-ink",
  danger: "bg-tomato-600 text-white hover:bg-tomato-700",
  dark: "bg-ink text-canvas hover:opacity-90",
};
const BTN_SIZES = {
  sm: "h-8 px-3 text-[13px] gap-1.5 rounded-lg",
  md: "h-10 px-4 text-sm gap-2 rounded-xl",
  lg: "h-12 px-6 text-[15px] gap-2.5 rounded-xl",
  icon: "size-9 rounded-lg justify-center",
};

export const Button = forwardRef(function Button(
  { variant = "primary", size = "md", className = "", loading = false, icon: Icon, children, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cx(
        "inline-flex select-none items-center justify-center font-medium transition disabled:cursor-not-allowed disabled:opacity-60",
        BTN_VARIANTS[variant],
        BTN_SIZES[size],
        className,
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : Icon && <Icon className="size-4" aria-hidden />}
      {children}
    </button>
  );
});

export function ButtonLink({ to, href, variant = "primary", size = "md", className = "", icon: Icon, iconRight: IconRight, children, ...rest }) {
  const cls = cx(
    "inline-flex select-none items-center justify-center font-medium transition",
    BTN_VARIANTS[variant],
    BTN_SIZES[size],
    className,
  );
  const inner = (
    <>
      {Icon && <Icon className="size-4" aria-hidden />}
      {children}
      {IconRight && <IconRight className="size-4" aria-hidden />}
    </>
  );
  if (href)
    return (
      <a href={href} className={cls} target="_blank" rel="noreferrer" {...rest}>
        {inner}
      </a>
    );
  return (
    <Link to={to} className={cls} {...rest}>
      {inner}
    </Link>
  );
}

/* ---------------------------------------------------------------- Badge */
const TONES = {
  leaf: "bg-leaf-500/10 text-leaf-700 ring-leaf-600/20 dark:text-leaf-300",
  tomato: "bg-tomato-500/10 text-tomato-700 ring-tomato-600/20 dark:text-tomato-400",
  sun: "bg-sun-500/12 text-sun-700 ring-sun-600/25 dark:text-sun-400",
  slate: "bg-surface-3 text-ink-2 ring-line-strong",
  sky: "bg-sky-500/10 text-sky-700 ring-sky-600/20 dark:text-sky-300",
  violet: "bg-violet-500/10 text-violet-700 ring-violet-600/20 dark:text-violet-300",
};

export function Badge({ tone = "slate", icon: Icon, className = "", children, ...rest }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        TONES[tone] || TONES.slate,
        className,
      )}
      {...rest}
    >
      {Icon && <Icon className="size-3.5" aria-hidden />}
      {children}
    </span>
  );
}

/* ---------------------------------------------------------------- Segmented control */
export function Segmented({ options, value, onChange, size = "md", className = "", label }) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cx("inline-flex rounded-xl border border-line bg-surface-2 p-1", className)}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cx(
              "flex items-center gap-1.5 rounded-lg font-medium transition",
              size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
              active ? "bg-surface text-ink shadow-sm ring-1 ring-line" : "text-muted hover:text-ink",
            )}
          >
            {o.icon && <o.icon className="size-3.5" aria-hidden />}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- Stat */
export function Stat({ label, value, hint, icon: Icon, tone = "leaf", className = "" }) {
  const ring = { leaf: "text-leaf-600 bg-leaf-500/10 dark:text-leaf-400", tomato: "text-tomato-600 bg-tomato-500/10", sun: "text-sun-600 bg-sun-500/10", sky: "text-sky-600 bg-sky-500/10", violet: "text-violet-600 bg-violet-500/10" }[tone];
  return (
    <Card className={cx("p-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <p className="text-[13px] font-medium text-muted">{label}</p>
        {Icon && (
          <span className={cx("grid size-8 place-items-center rounded-lg", ring)}>
            <Icon className="size-4" aria-hidden />
          </span>
        )}
      </div>
      <p className="mt-2 font-display text-[28px] font-semibold leading-none tracking-tight text-ink tabular">{value}</p>
      {hint && <p className="mt-2 text-xs leading-relaxed text-muted">{hint}</p>}
    </Card>
  );
}

/* ---------------------------------------------------------------- Page header */
export function PageHeader({ eyebrow, title, description, actions, children }) {
  return (
    <header className="mb-8 flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl">
        {eyebrow && <p className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-leaf-600 dark:text-leaf-400">{eyebrow}</p>}
        <h1 className="font-display text-[28px] font-semibold leading-tight tracking-tight text-ink sm:text-[32px]">{title}</h1>
        {description && <p className="mt-3 text-[15px] leading-relaxed text-muted">{description}</p>}
        {children}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

/* ---------------------------------------------------------------- States */
export function Spinner({ className = "" }) {
  return <Loader2 className={cx("size-5 animate-spin text-leaf-600", className)} aria-label="Loading" />;
}

export function Skeleton({ className = "" }) {
  return <div className={cx("skeleton rounded-lg", className)} aria-hidden />;
}

export function EmptyState({ icon: Icon = Info, title, description, action, className = "" }) {
  return (
    <div className={cx("flex flex-col items-center justify-center rounded-2xl border border-dashed border-line-strong px-6 py-12 text-center", className)}>
      <span className="grid size-11 place-items-center rounded-xl bg-surface-2 text-muted">
        <Icon className="size-5" aria-hidden />
      </span>
      <p className="mt-4 font-display text-base font-semibold text-ink">{title}</p>
      {description && <p className="mt-1.5 max-w-md text-sm leading-relaxed text-muted">{description}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({ title = "Something went wrong", error, onRetry, className = "" }) {
  return (
    <div role="alert" className={cx("flex items-start gap-3 rounded-2xl border border-tomato-500/30 bg-tomato-500/5 p-4", className)}>
      <AlertTriangle className="mt-0.5 size-5 shrink-0 text-tomato-600" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-ink">{title}</p>
        <p className="mt-1 break-words text-sm text-ink-2">{error?.message || String(error || "Unknown error")}</p>
      </div>
      {onRetry && (
        <Button variant="secondary" size="sm" icon={RefreshCw} onClick={onRetry}>
          Retry
        </Button>
      )}
    </div>
  );
}

export function Callout({ tone = "leaf", icon: Icon = Info, title, children, className = "" }) {
  const styles = {
    leaf: "border-leaf-500/25 bg-leaf-500/[0.06] text-leaf-700 dark:text-leaf-300",
    sun: "border-sun-500/30 bg-sun-500/[0.07] text-sun-700 dark:text-sun-400",
    tomato: "border-tomato-500/30 bg-tomato-500/[0.06] text-tomato-700 dark:text-tomato-400",
    slate: "border-line-strong bg-surface-2 text-ink-2",
  }[tone];
  return (
    <div className={cx("flex gap-3 rounded-xl border p-4", styles, className)}>
      <Icon className="mt-0.5 size-[18px] shrink-0" aria-hidden />
      <div className="min-w-0 text-sm leading-relaxed">
        {title && <p className="mb-0.5 font-semibold">{title}</p>}
        <div className="text-ink-2">{children}</div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- Tooltip (hover + keyboard focus) */
export function Tip({ content, children, side = "top", align = "center", className = "" }) {
  // display:none until hover/focus, so an off-screen tooltip never widens the page on phones
  const pos = { center: "left-1/2 -translate-x-1/2", end: "right-0", start: "left-0" }[align];
  return (
    <span className={cx("group/tip relative inline-flex", className)}>
      {children}
      <span
        role="tooltip"
        className={cx(
          "pointer-events-none absolute z-50 hidden w-max max-w-[min(260px,80vw)] rounded-lg bg-ink px-2.5 py-1.5 text-xs leading-snug font-normal text-canvas shadow-pop group-hover/tip:block group-focus-within/tip:block",
          pos,
          side === "top" ? "bottom-full mb-2" : "top-full mt-2",
        )}
      >
        {content}
      </span>
    </span>
  );
}

export function InfoTip({ content, className = "" }) {
  return (
    <Tip content={content} className={className}>
      <button type="button" className="text-muted hover:text-ink" aria-label="More information">
        <Info className="size-3.5" aria-hidden />
      </button>
    </Tip>
  );
}

/* ---------------------------------------------------------------- Progress bar */
export function Meter({ value, color = "var(--color-leaf-500)", className = "", height = "h-2", label }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div className={cx("w-full overflow-hidden rounded-full bg-surface-3", height, className)} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} aria-label={label}>
      <div className="h-full rounded-full transition-[width] duration-700 ease-out" style={{ width: `${v * 100}%`, background: color }} />
    </div>
  );
}
