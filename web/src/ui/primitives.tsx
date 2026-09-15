import type { ButtonHTMLAttributes, PropsWithChildren, ReactNode } from "react";

export function Icon({ name, className = "" }: { name: string; className?: string }) {
  return <span className={`material-symbols-outlined select-none text-[20px] ${className}`}>{name}</span>;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 border-b border-border pb-6 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {eyebrow ? (
          <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-accent">{eyebrow}</p>
        ) : null}
        <h1 className="font-display text-3xl font-bold tracking-tight text-primary md:text-4xl">{title}</h1>
        {description ? <p className="mt-2 max-w-3xl text-sm leading-relaxed text-primary-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-3">{actions}</div> : null}
    </div>
  );
}

export function Panel({
  children,
  className = "",
}: PropsWithChildren<{ className?: string }>) {
  return (
    <section className={`min-w-0 border border-border bg-surface ${className}`}>
      {children}
    </section>
  );
}

export function Button({
  children,
  tone = "ghost",
  className = "",
  ...props
}: PropsWithChildren<{ tone?: "ghost" | "primary" | "secondary" | "danger"; className?: string } & ButtonHTMLAttributes<HTMLButtonElement>>) {
  const tones = {
    ghost: "border border-border bg-surface-low text-primary-muted hover:border-border-bright hover:text-primary hover:bg-surface-mid",
    primary: "border border-accent bg-accent text-background font-bold hover:bg-accent/90",
    secondary: "border border-border bg-surface-mid text-primary hover:border-accent/40 hover:text-accent",
    danger: "border border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 hover:border-rose-500/50",
  };
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-[0.14em] font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-40 ${tones[tone]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Chip({
  children,
  tone = "neutral",
  active = false,
}: PropsWithChildren<{ tone?: "neutral" | "primary" | "accent" | "success" | "warning" | "danger" | "violet"; active?: boolean }>) {
  const tones = {
    neutral: active ? "border-accent text-accent bg-accent/10" : "border-border bg-surface-low text-primary-muted",
    primary: "border-accent/50 bg-accent/10 text-accent",
    accent: "border-accent/50 bg-accent/10 text-accent",
    success: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
    warning: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    danger: "border-rose-500/40 bg-rose-500/10 text-rose-400",
    violet: "border-accent/30 bg-surface-mid text-accent",
  };
  return (
    <span className={`inline-flex max-w-full items-center border px-2.5 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function MetricCard({
  label,
  value,
  accent,
  icon,
  meta,
}: {
  label: string;
  value: string;
  accent?: ReactNode;
  icon?: string;
  meta?: ReactNode;
}) {
  return (
    <Panel className="relative p-6">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-primary-muted">{label}</p>
        {icon ? <Icon name={icon} className="text-primary-subtle" /> : null}
      </div>
      <div className="mt-3 flex items-baseline gap-3">
        <p className="font-display text-3xl font-bold tracking-tight text-primary md:text-4xl">{value}</p>
        {accent}
      </div>
      {meta ? <div className="mt-4 border-t border-border-muted pt-3">{meta}</div> : null}
    </Panel>
  );
}

export function ProgressBar({
  value,
  tone = "primary",
}: {
  value: number;
  tone?: "primary" | "warning" | "danger" | "violet";
}) {
  const toneClass = {
    primary: "bg-accent",
    warning: "bg-amber-400",
    danger: "bg-rose-400",
    violet: "bg-accent",
  }[tone];
  return (
    <div className="h-1 w-full bg-surface-high">
      <div className={`h-full ${toneClass}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function DataTable({
  headers,
  rows,
  dense = false,
}: {
  headers: string[];
  rows: Array<Array<ReactNode>>;
  dense?: boolean;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-left">
        <thead>
          <tr className="border-b border-border bg-surface-low font-mono text-[10px] font-bold uppercase tracking-[0.2em] text-primary-muted">
            {headers.map((header) => (
              <th key={header} className="px-4 py-3">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border-muted">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="transition-colors hover:bg-surface-low">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={`max-w-0 truncate px-4 ${dense ? "py-2.5" : "py-3.5"} text-sm text-primary`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Drawer({
  title,
  subtitle,
  children,
  footer,
}: PropsWithChildren<{ title: string; subtitle?: string; footer?: ReactNode }>) {
  return (
    <Panel className="flex h-full min-h-[520px] flex-col xl:max-h-[calc(100vh-10rem)]">
      <div className="border-b border-border bg-surface-low px-6 py-4">
        <h3 className="font-display text-lg font-bold text-primary">{title}</h3>
        {subtitle ? <p className="mt-1 truncate font-mono text-[11px] text-primary-muted">{subtitle}</p> : null}
      </div>
      <div className="flex-1 space-y-6 overflow-y-auto px-6 py-6">{children}</div>
      {footer ? <div className="grid grid-cols-1 gap-3 border-t border-border bg-surface-low p-4 sm:grid-cols-2">{footer}</div> : null}
    </Panel>
  );
}

export function LabeledValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-primary-muted">{label}</p>
      <div className="min-w-0 text-primary">{value}</div>
    </div>
  );
}

export function PathBlock({
  value,
  tone = "default",
  strike = false,
}: {
  value: string;
  tone?: "default" | "primary" | "success" | "danger";
  strike?: boolean;
}) {
  const tones = {
    default: "bg-surface-low border-border-muted text-primary-muted",
    primary: "bg-surface-low border-accent/30 text-accent",
    success: "bg-emerald-500/5 border-emerald-500/20 text-emerald-300",
    danger: "bg-rose-500/5 border-rose-500/20 text-rose-300",
  };
  return (
    <div className={`break-all border px-3 py-2 font-mono text-xs ${tones[tone]} ${strike ? "line-through opacity-70" : ""}`}>
      {value}
    </div>
  );
}
