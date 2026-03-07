import type { ButtonHTMLAttributes, PropsWithChildren, ReactNode } from "react";

export function Icon({ name, className = "" }: { name: string; className?: string }) {
  return <span className={`material-symbols-outlined ${className}`}>{name}</span>;
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
    <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {eyebrow ? <p className="mb-1 text-xs font-bold uppercase tracking-[0.28em] text-primary">{eyebrow}</p> : null}
        <h1 className="font-display text-3xl font-bold tracking-tight text-slate-100 md:text-4xl">{title}</h1>
        {description ? <p className="mt-2 max-w-3xl text-sm text-slate-400">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-3">{actions}</div> : null}
    </div>
  );
}

export function Panel({
  children,
  className = "",
}: PropsWithChildren<{ className?: string }>) {
  return (
    <section
      className={`min-w-0 rounded-xl border border-primary/10 bg-surface-dark/80 shadow-[0_0_30px_-18px_rgba(37,226,244,0.25)] backdrop-blur-xl ${className}`}
    >
      {children}
    </section>
  );
}

export function Button({
  children,
  tone = "ghost",
  className = "",
  ...props
}: PropsWithChildren<{ tone?: "ghost" | "primary" | "secondary"; className?: string } & ButtonHTMLAttributes<HTMLButtonElement>>) {
  const tones = {
    ghost: "border border-border-dark bg-surface-dark text-slate-200 hover:bg-border-dark focus-visible:border-primary/40",
    primary: "bg-primary text-background-dark hover:brightness-110 focus-visible:ring-primary/60",
    secondary: "border border-primary/20 bg-primary/10 text-primary hover:bg-primary/20 focus-visible:border-primary/40",
  };
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-bold transition-all focus-visible:outline-none focus-visible:ring-2 disabled:cursor-not-allowed disabled:opacity-50 ${tones[tone]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Chip({
  children,
  tone = "neutral",
  active = false,
}: PropsWithChildren<{ tone?: "neutral" | "primary" | "success" | "warning" | "danger" | "violet"; active?: boolean }>) {
  const tones = {
    neutral: active ? "border-slate-700 bg-slate-800 text-slate-200" : "border-border-dark bg-background-dark text-slate-400",
    primary: "border-primary/30 bg-primary/10 text-primary",
    success: "border-emerald-500/20 bg-emerald-500/10 text-emerald-400",
    warning: "border-amber-500/20 bg-amber-500/10 text-amber-400",
    danger: "border-rose-500/20 bg-rose-500/10 text-rose-400",
    violet: "border-secondary/20 bg-secondary/10 text-secondary",
  };
  return <span className={`inline-flex max-w-full items-center rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] ${tones[tone]}`}>{children}</span>;
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
    <Panel className="relative overflow-hidden p-6">
      {icon ? (
        <div className="absolute right-4 top-2 text-primary/10">
          <Icon name={icon} className="text-6xl" />
        </div>
      ) : null}
      <p className="text-xs font-bold uppercase tracking-[0.25em] text-slate-500">{label}</p>
      <div className="mt-3 flex items-end gap-3">
        <p className="font-display text-4xl font-bold tracking-tight text-slate-100">{value}</p>
        {accent}
      </div>
      {meta ? <div className="mt-4">{meta}</div> : null}
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
    primary: "bg-primary",
    warning: "bg-amber-500",
    danger: "bg-rose-500",
    violet: "bg-secondary",
  }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-border-dark">
      <div className={`h-full rounded-full ${toneClass}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
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
      <table className="w-full min-w-[640px] border-separate border-spacing-y-2 text-left">
        <thead>
          <tr className="text-[11px] font-bold uppercase tracking-[0.24em] text-slate-500">
            {headers.map((header) => (
              <th key={header} className="px-4 pb-2">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="rounded-2xl border border-border-dark bg-background-dark/70 transition-colors hover:border-primary/30">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={`max-w-0 truncate px-4 ${dense ? "py-3" : "py-4"} text-sm text-slate-200 first:rounded-l-2xl last:rounded-r-2xl`}>
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
    <Panel className="flex h-full min-h-[520px] flex-col overflow-hidden xl:max-h-[calc(100vh-10rem)]">
      <div className="border-b border-border-dark bg-background-dark/40 px-6 py-5">
        <h3 className="font-display text-lg font-bold text-slate-100">{title}</h3>
        {subtitle ? <p className="mt-1 truncate font-mono text-[11px] text-primary/70">{subtitle}</p> : null}
      </div>
      <div className="flex-1 space-y-6 overflow-y-auto px-6 py-6">{children}</div>
      {footer ? <div className="grid grid-cols-1 gap-4 border-t border-border-dark bg-background-dark/30 p-6 sm:grid-cols-2">{footer}</div> : null}
    </Panel>
  );
}

export function LabeledValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="space-y-1.5 min-w-0">
      <p className="text-[10px] font-bold uppercase tracking-[0.24em] text-slate-500">{label}</p>
      <div className="min-w-0">{value}</div>
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
    default: "bg-background-dark text-slate-400",
    primary: "bg-background-dark text-primary",
    success: "bg-emerald-500/5 text-slate-300",
    danger: "bg-rose-500/5 text-slate-300",
  };
  return (
    <div className={`break-all rounded-xl px-4 py-3 font-mono text-xs ${tones[tone]} ${strike ? "line-through opacity-70" : ""}`}>
      {value}
    </div>
  );
}
