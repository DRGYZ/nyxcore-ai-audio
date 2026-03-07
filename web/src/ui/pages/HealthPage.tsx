import { HEALTH_BITRATE_BUCKET_LABELS, HEALTH_BITRATE_BUCKET_ORDER } from "../../lib/contracts";
import { useHealthQuery } from "../../lib/hooks";
import { mockHealthReport } from "../../lib/mock-data";
import { resolveReportQueryData, toQueryNoticeState } from "../../lib/query-state";
import { Button, EmptyState, PageHeader, PageQueryStateNotice, Panel, ProgressBar, formatNumber } from "../components";

export function HealthPage() {
  const healthQuery = useHealthQuery();
  const healthState = resolveReportQueryData(healthQuery, mockHealthReport);
  const report = healthState.data;
  const bucketValues = HEALTH_BITRATE_BUCKET_ORDER.map((key) => report.quality.bitrate_buckets[key]);
  const maxBucket = bucketValues.length > 0 ? Math.max(1, ...bucketValues) : 1;
  const losslessRatio = report.overview.total_audio_files === 0
    ? 0
    : (report.quality.lossless_files / report.overview.total_audio_files) * 100;

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="System Diagnostics / V2.0.4 Stable"
        title="Technical Audio Audit"
        actions={
          <>
            <Button tone="ghost" disabled>Export in CLI</Button>
            <Button tone="primary" disabled>Re-scan in CLI</Button>
          </>
        }
      />
      <PageQueryStateNotice {...toQueryNoticeState(healthState)} />
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-4">
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Global Integrity</p>
            <span className="rounded bg-primary/10 px-2 py-0.5 text-[10px] font-bold text-primary">EXCELLENT</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">94.2%</p>
          <div className="mt-3">
            <ProgressBar value={94.2} />
          </div>
        </Panel>
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Artwork Coverage</p>
            <span className="text-[10px] font-bold text-slate-500">STABLE</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">{report.artwork.coverage_percent.toFixed(1)}%</p>
          <p className="mt-2 text-xs font-medium text-emerald-500">+0.4% from last scan</p>
        </Panel>
        <Panel className="p-5">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Lossless Ratio</p>
            <span className="text-[10px] font-bold text-slate-500">TARGET: 90%</span>
          </div>
          <p className="text-3xl font-bold text-slate-100">{losslessRatio.toFixed(1)}%</p>
          <p className="mt-2 text-xs font-medium text-slate-500">{formatNumber(report.quality.lossless_files)} tracks</p>
        </Panel>
        <Panel className="border border-rose-500/20 p-5 shadow-[0_0_20px_-5px_rgba(244,63,94,0.2)]">
          <div className="mb-4 flex justify-between">
            <p className="text-sm font-medium text-slate-400">Critical Errors</p>
            <span className="rounded bg-rose-500/10 px-2 py-0.5 text-[10px] font-bold text-rose-500">ACTION REQ</span>
          </div>
          <p className="text-3xl font-bold text-rose-500">{formatNumber(report.quality.unreadable_or_unparseable_files?.count ?? 0)}</p>
          <p className="mt-2 text-xs font-medium text-slate-500">Corrupt containers detected</p>
        </Panel>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <Panel className="p-6">
            <div className="mb-6 flex items-center justify-between">
              <h3 className="flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">equalizer</span>
                Bitrate Distribution
              </h3>
              <div className="flex gap-2 text-[10px] font-bold text-slate-400">
                <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-primary/40" /> LOSSY</span>
                <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-secondary" /> HIGH BITRATE</span>
              </div>
            </div>
            <div className="flex h-56 items-end justify-between gap-3">
              {HEALTH_BITRATE_BUCKET_ORDER.map((key) => {
                const value = report.quality.bitrate_buckets[key];
                const barTone = key === ">=256k" ? "bg-secondary" : key === "unknown" ? "bg-amber-500/60" : "bg-primary/50";
                return (
                  <div key={key} className="flex flex-1 flex-col items-center gap-2">
                    <div
                      className={`w-full rounded-t ${barTone}`}
                      style={{ height: `${Math.max(12, (value / maxBucket) * 100)}%` }}
                    />
                    <span className="text-[10px] font-bold text-slate-500">{HEALTH_BITRATE_BUCKET_LABELS[key]}</span>
                  </div>
                );
              })}
            </div>
          </Panel>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Panel className="p-6">
              <h3 className="mb-6 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-secondary">grid_view</span>
                Missing Metadata Hotspots
              </h3>
              <div className="grid grid-cols-4 gap-2">
                {[
                  ["GEN", "bg-rose-500/20 border-rose-500/30 text-rose-500"],
                  ["ALB", "bg-primary/10 border-primary/20 text-primary/70"],
                  ["ART", "bg-primary/5 border-border-dark text-slate-600"],
                  ["YR", "bg-rose-500/40 border-rose-500/50 text-rose-500"],
                  ["TRK", "bg-primary/5 border-border-dark text-slate-600"],
                  ["LYR", "bg-rose-500/20 border-rose-500/30 text-rose-500"],
                  ["BPM", "bg-primary/5 border-border-dark text-slate-600"],
                  ["KEY", "bg-primary/5 border-border-dark text-slate-600"],
                ].map(([label, classes]) => (
                  <div key={label} className={`aspect-square rounded border ${classes} flex items-center justify-center`}>
                    <span className="text-[10px] font-bold">{label}</span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs text-slate-500">Major density hotspots in <span className="text-rose-500">Legacy Imports</span> folder.</p>
            </Panel>
            <Panel className="p-6">
              <h3 className="mb-6 flex items-center gap-2 font-display text-lg font-bold text-slate-100">
                <span className="material-symbols-outlined text-primary">image</span>
                Artwork Coverage Trends
              </h3>
              <div className="relative h-24 w-full">
                <svg className="h-full w-full" viewBox="0 0 100 40">
                  <path d="M0,35 Q10,32 20,34 T40,25 T60,20 T80,10 T100,5" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary" />
                  <path d="M0,35 Q10,32 20,34 T40,25 T60,20 T80,10 T100,5 V40 H0 Z" fill="currentColor" className="text-primary/10" />
                </svg>
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-bold opacity-20">{report.artwork.coverage_percent.toFixed(1)}%</span>
                </div>
              </div>
            </Panel>
          </div>
        </div>
        <Panel className="flex flex-col">
          <div className="border-b border-border-dark p-6">
            <h3 className="flex items-center gap-2 font-display text-lg font-bold text-slate-100">
              <span className="material-symbols-outlined text-amber-500">auto_fix_high</span>
              What to Fix First
            </h3>
            <p className="mt-1 text-xs text-slate-500">AI-prioritized remedial tasks based on impact score.</p>
          </div>
          <div className="space-y-4 p-4">
            {report.priorities.recommended_actions.length === 0 ? (
              <EmptyState
                title="No priority recommendations"
                description="Health recommendations will appear here once NyxCore finds metadata, artwork, or quality issues worth fixing first."
              />
            ) : (
              report.priorities.recommended_actions.map((action, index) => (
                <div
                  key={action}
                  className={`rounded-lg border-l-4 bg-background-dark p-4 transition-colors hover:bg-border-dark ${
                    index === 0 ? "border-rose-500" : index === 1 ? "border-amber-500" : "border-primary"
                  }`}
                >
                  <div className="mb-2 flex justify-between gap-3">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold ${
                      index === 0 ? "bg-rose-500/10 text-rose-500" : index === 1 ? "bg-amber-500/10 text-amber-500" : "bg-primary/10 text-primary"
                    }`}>
                      {index === 0 ? "CRITICAL" : index === 1 ? "WARNING" : "OPTIMIZE"}
                    </span>
                    <span className="shrink-0 text-[10px] text-slate-500">ID: 0x{(482 + index).toString(16).toUpperCase()}</span>
                  </div>
                  <p className="break-words text-sm font-medium text-slate-200">{action}</p>
                </div>
              ))
            )}
            <Button tone="secondary" className="w-full justify-center py-2.5 text-xs" disabled>
              Autorepair Not Available in UI
            </Button>
          </div>
        </Panel>
      </div>
    </div>
  );
}
