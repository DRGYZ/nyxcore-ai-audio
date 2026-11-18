import type { ButtonHTMLAttributes, PropsWithChildren, ReactNode } from "react";

export function Icon({ name, className = "" }: { name: string; className?: string }) {
  return <span aria-hidden="true" className={`material-symbols-outlined select-none text-[20px] ${className}`}>{name}</span>;
}

const SLEEVE_PALETTES = [
  { bg: "from-[#273142] to-[#171c26]", accent: "#e09f5a", ring: "rgba(224,159,90,0.34)" },
  { bg: "from-[#3b251e] to-[#1f1512]", accent: "#e5b57c", ring: "rgba(229,181,124,0.34)" },
  { bg: "from-[#213229] to-[#141d18]", accent: "#8bc4a6", ring: "rgba(139,196,166,0.32)" },
  { bg: "from-[#34232d] to-[#1b1318]", accent: "#e09f5a", ring: "rgba(224,159,90,0.32)" },
  { bg: "from-[#332c22] to-[#1a1712]", accent: "#d9c297", ring: "rgba(217,194,151,0.32)" },
] as const;

function hashSeed(input: string) {
  let hash = 0;
  for (let index = 0; index < input.length; index += 1) {
    hash = (hash * 31 + input.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function sleeveInitials(title?: string | null, artist?: string | null) {
  const source = (artist || title || "NC").trim();
  const parts = source.split(/[\s._-]+/).filter(Boolean);
  if (parts.length >= 2) {
    return (parts[0][0] + parts[1][0]).toUpperCase();
  }
  return source.slice(0, 2).toUpperCase();
}

export function SleeveArt({
  title,
  artist,
  format,
  hasArtwork,
  src,
  alt,
  size = "md",
  showVinyl = false,
  showMonogram = true,
  className = "",
}: {
  title?: string | null;
  artist?: string | null;
  format?: string | null;
  hasArtwork?: boolean;
  src?: string | null;
  alt?: string;
  size?: "sm" | "md" | "lg" | "xl";
  showVinyl?: boolean;
  showMonogram?: boolean;
  className?: string;
}) {
  const seed = hashSeed(`${artist ?? ""}|${title ?? ""}`);
  const palette = SLEEVE_PALETTES[seed % SLEEVE_PALETTES.length];
  const sizes = {
    sm: "size-11",
    md: "size-14",
    lg: "size-20",
    xl: "size-28",
  };
  const discSizes = {
    sm: "size-10 -right-2",
    md: "size-12 -right-2.5",
    lg: "size-[4.5rem] -right-3",
    xl: "size-24 -right-4",
  };
  const labelText = sleeveInitials(title, artist);
  const ext = format?.replace(/^\./, "").toUpperCase();

  return (
    <div className={`relative inline-flex shrink-0 items-center ${showVinyl ? "mr-2.5" : ""} ${className}`}>
      {showVinyl ? (
        <div
          aria-hidden="true"
          className={`absolute top-1/2 -translate-y-1/2 rounded-full border border-white/[0.12] bg-[radial-gradient(circle_at_center,#e09f5a_0_16%,#211e1b_17%_24%,#131210_25%_52%,#221f1c_53%_55%,#12110f_56%_100%)] shadow-md ${discSizes[size]}`}
        />
      ) : null}
      <div
        className={`relative overflow-hidden rounded-lg border border-white/[0.14] bg-gradient-to-br shadow-sleeve ${palette.bg} ${sizes[size]}`}
      >
        {src ? (
          <img src={src} alt={alt ?? title ?? "Album cover"} className="h-full w-full object-cover" />
        ) : (
          <div aria-hidden="true" className="relative flex h-full w-full flex-col justify-between p-1.5 select-none">
            <div
              className="pointer-events-none absolute -right-4 -top-4 size-12 rounded-full opacity-25"
              style={{ border: `1.5px solid ${palette.accent}` }}
            />
            <div className="flex items-center justify-between gap-1">
              <span
                className="size-1.5 rounded-full"
                style={{ backgroundColor: hasArtwork ? "#52b788" : palette.accent }}
              />
              {ext ? (
                <span className="font-mono text-[8px] font-semibold tracking-wider text-white/65">
                  {ext}
                </span>
              ) : null}
            </div>
            <div className="flex items-end justify-between">
              {showMonogram ? (
                <span
                  className={`font-editorial font-semibold leading-none tracking-tight text-white/90 ${
                    size === "xl" ? "text-xl" : size === "lg" ? "text-base" : "text-xs"
                  }`}
                >
                  {labelText}
                </span>
              ) : (
                <span className="material-symbols-outlined text-[18px] text-white/80">album</span>
              )}
              <span
                className="h-0.5 w-3.5 rounded-full opacity-80"
                style={{ backgroundColor: palette.accent }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
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
    <div className="flex flex-col gap-4 border-b border-white/[0.08] pb-5 md:flex-row md:items-end md:justify-between">
      <div className="min-w-0">
        {eyebrow ? (
          <p className="mb-1.5 inline-flex items-center gap-1.5 font-sans text-xs font-medium text-accent">
            <span className="size-1.5 rounded-full bg-accent/80" aria-hidden="true" />
            <span>{eyebrow}</span>
          </p>
        ) : null}
        <h1 className="font-display text-2xl font-semibold tracking-tight text-primary md:text-3xl">{title}</h1>
        {description ? <div className="mt-1.5 max-w-2xl text-sm leading-relaxed text-primary-muted">{description}</div> : null}
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
    <section className={`min-w-0 rounded-xl border border-white/[0.08] bg-surface shadow-[inset_0_1px_0_rgba(255,248,235,0.03)] ${className}`}>
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
    ghost: "border border-white/[0.09] bg-transparent text-primary-muted hover:border-white/[0.18] hover:text-primary hover:bg-white/[0.04]",
    primary: "border border-accent bg-accent text-[#14110e] font-semibold hover:bg-[#e8ae6d] shadow-sm",
    secondary: "border border-white/[0.12] bg-surface-mid text-primary hover:border-accent/50 hover:text-primary hover:bg-surface-high",
    danger: "border border-[#e06d53]/40 bg-[#e06d53]/14 text-[#f5a38f] hover:bg-[#e06d53]/24 hover:border-[#e06d53]/60",
  };
  return (
    <button
      {...props}
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 font-sans text-xs font-medium tracking-normal transition-colors active:translate-y-[0.5px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:active:translate-y-0 ${tones[tone]} ${className}`}
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
    default: active ? "border-accent/50 text-accent bg-accent/15" : "border-white/[0.09] bg-white/[0.04] text-primary-muted",
    neutral: active ? "border-accent/50 text-accent bg-accent/15" : "border-white/[0.09] bg-white/[0.04] text-primary-muted",
    primary: "border-accent/35 bg-accent/14 text-accent",
    accent: "border-accent/35 bg-accent/14 text-accent",
    success: "border-[#52b788]/35 bg-[#52b788]/14 text-[#7cd4aa]",
    warning: "border-[#e9b44c]/35 bg-[#e9b44c]/14 text-[#f2c86b]",
    danger: "border-[#e06d53]/35 bg-[#e06d53]/14 text-[#f59f8a]",
    violet: "border-accent/35 bg-surface-mid text-accent",
  };
  return (
    <span className={`inline-flex max-w-full items-center rounded-full border px-2.5 py-0.5 font-sans text-[11px] font-medium tracking-normal ${tones[tone]} ${className}`}>
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
    <Panel className="relative p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2">
        <p className="font-sans text-xs font-medium text-primary-muted">{label}</p>
        {icon ? (
          <span className="flex size-7 items-center justify-center rounded-lg bg-white/[0.04] text-accent/80">
            <Icon name={icon} className="text-[17px]" />
          </span>
        ) : null}
      </div>
      <div className="mt-2.5 flex items-baseline gap-2.5">
        <p className="font-display text-2xl font-semibold tracking-tight text-primary md:text-[1.75rem]">{value}</p>
        {accent}
      </div>
      {meta ? <div className="mt-3 border-t border-white/[0.06] pt-2.5 text-xs text-primary-subtle">{meta}</div> : null}
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
    warning: "bg-[#e9b44c]",
    danger: "bg-[#e06d53]",
    violet: "bg-[#52b788]",
  }[tone];
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.07]">
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
          <tr className="border-b border-white/[0.08] bg-surface-low/80 font-sans text-xs font-medium text-primary-subtle">
            {headers.map((header) => (
              <th key={header} className="px-4 py-2.5">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/[0.05]">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="transition-colors hover:bg-white/[0.025]">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className={`max-w-0 truncate px-4 ${dense ? "py-2.5" : "py-3.5"} text-xs text-primary`}>
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
  action,
}: PropsWithChildren<{ title: string; subtitle?: string; footer?: ReactNode; action?: ReactNode }>) {
  return (
    <Panel className="flex h-full min-h-[520px] flex-col overflow-hidden xl:max-h-[calc(100vh-10rem)]">
      <div className="flex items-start justify-between border-b border-white/[0.08] bg-surface-low/85 px-5 py-4">
        <div className="min-w-0">
          <h3 className="font-display text-base font-semibold text-primary">{title}</h3>
          {subtitle ? <p className="mt-0.5 truncate font-mono text-[11px] text-primary-subtle">{subtitle}</p> : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">{children}</div>
      {footer ? <div className="grid grid-cols-1 gap-2.5 border-t border-white/[0.08] bg-surface-low/85 p-4 sm:grid-cols-2">{footer}</div> : null}
    </Panel>
  );
}

export function LabeledValue({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0 space-y-1.5">
      <p className="font-sans text-xs font-medium text-primary-subtle">{label}</p>
      <div className="min-w-0 text-primary">{value}</div>
    </div>
  );
}

export function PathBlock({
  value,
  tone = "default",
  strike = false,
  className = "",
}: {
  value: string;
  tone?: "default" | "primary" | "success" | "danger";
  strike?: boolean;
  className?: string;
}) {
  const tones = {
    default: "bg-surface-low/90 border-white/[0.07] text-primary-muted",
    primary: "bg-accent/[0.08] border-accent/25 text-accent",
    success: "bg-[#52b788]/[0.08] border-[#52b788]/25 text-[#88dcb3]",
    danger: "bg-[#e06d53]/[0.08] border-[#e06d53]/25 text-[#f59f8a]",
  };
  return (
    <div
      title={value}
      className={`min-w-0 break-words [overflow-wrap:anywhere] rounded-lg border px-3 py-1.5 font-mono text-[11.5px] leading-relaxed ${tones[tone]} ${strike ? "line-through opacity-70" : ""} ${className}`}
    >
      {value}
    </div>
  );
}
