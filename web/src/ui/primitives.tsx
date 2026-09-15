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
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-4 border-b border-white/[0.07] pb-6 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {eyebrow ? (
          <p className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-accent/80">{eyebrow}</p>
        ) : null}
        <h1 className="font-display text-2xl font-bold tracking-tight text-primary md:text-3xl">{title}</h1>
        {description ? <div className="mt-2 max-w-3xl text-sm leading-relaxed text-primary-muted">{description}</div> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2.5">{actions}</div> : null}
    </div>
  );
}

export function Panel({
  children,
  className = "",
}: PropsWithChildren<{ className?: string }>) {
  return (
    <section className={`min-w-0 rounded-[3px] border border-white/[0.07] bg-surface ${className}`}>
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
    ghost: "border border-white/[0.08] bg-surface-low text-primary-muted hover:border-white/[0.14] hover:text-primary hover:bg-surface-mid",
    primary: "border border-accent bg-accent text-background font-semibold hover:bg-accent/90 shadow-sm",
    secondary: "border border-white/[0.1] bg-white/[0.03] text-primary hover:border-accent/40 hover:text-accent hover:bg-white/[0.06]",
    danger: "border border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 hover:border-rose-500/50",
  };
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-2 rounded-[3px] px-3.5 py-1.5 font-sans text-xs font-medium tracking-normal transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-40 ${tones[tone]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Chip({
  children,
  tone = "neutral",
  active = false,
  className = "",
}: PropsWithChildren<{ tone?: "default" | "neutral" | "primary" | "accent" | "success" | "warning" | "danger" | "violet"; active?: boolean; className?: string }>) {
  const tones = {
    default: active ? "border-accent/40 text-accent bg-accent/10" : "border-white/[0.08] bg-white/[0.03] text-primary-muted",
    neutral: active ? "border-accent/40 text-accent bg-accent/10" : "border-white/[0.08] bg-white/[0.03] text-primary-muted",
    primary: "border-accent/30 bg-accent/10 text-accent",
    accent: "border-accent/30 bg-accent/10 text-accent",
    success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-400",
    danger: "border-rose-500/30 bg-rose-500/10 text-rose-400",
    violet: "border-accent/30 bg-surface-mid text-accent",
  };
  return (
    <span className={`inline-flex max-w-full items-center rounded-[3px] border px-2 py-0.5 font-sans text-[11px] font-medium tracking-normal ${tones[tone]} ${className}`}>
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
    <Panel className="relative p-5">
      <div className="flex items-center justify-between">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-primary-subtle">{label}</p>
        {icon ? <Icon name={icon} className="text-primary-subtle/70" /> : null}
      </div>
      <div className="mt-2.5 flex items-baseline gap-2.5">
        <p className="font-display text-2xl font-bold tracking-tight text-primary md:text-3xl">{value}</p>
        {accent}
      </div>
      {meta ? <div className="mt-3.5 border-t border-white/[0.05] pt-2.5">{meta}</div> : null}
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
    <div className="h-1 w-full overflow-hidden rounded-full bg-white/[0.06]">
      <div className={`h-full rounded-full transition-all duration-300 ${toneClass}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
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
          <tr className="border-b border-white/[0.07] bg-surface-low/80 font-sans text-[11px] font-medium text-primary-subtle">
            {headers.map((header) => (
              <th key={header} className="px-4 py-2.5">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.04]">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="transition-colors hover:bg-white/[0.02]">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={`max-w-0 truncate px-4 ${dense ? "py-2" : "py-3"} text-xs text-primary`}>
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
    <Panel className="flex h-full min-h-[520px] flex-col rounded-[3px] xl:max-h-[calc(100vh-10rem)]">
      <div className="border-b border-white/[0.07] bg-surface-low/80 px-5 py-3.5">
        <h3 className="font-display text-base font-bold text-primary">{title}</h3>
        {subtitle ? <p className="mt-0.5 truncate font-mono text-[10px] text-primary-subtle">{subtitle}</p> : null}
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">{children}</div>
      {footer ? <div className="grid grid-cols-1 gap-2.5 border-t border-white/[0.07] bg-surface-low/80 p-3.5 sm:grid-cols-2">{footer}</div> : null}
    </Panel>
  );
}

export function LabeledValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0 space-y-1">
      <p className="font-sans text-[11px] font-medium text-primary-subtle">{label}</p>
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
    default: "bg-surface-low border-white/[0.06] text-primary-muted",
    primary: "bg-surface-low border-accent/25 text-accent",
    success: "bg-emerald-500/[0.06] border-emerald-500/20 text-emerald-300",
    danger: "bg-rose-500/[0.06] border-rose-500/20 text-rose-300",
  };
  return (
    <div className={`break-all rounded-[3px] border px-2.5 py-1.5 font-mono text-xs ${tones[tone]} ${strike ? "line-through opacity-70" : ""}`}>
      {value}
    </div>
  );
}
