import { Button } from "./primitives";

export function ListPages({
  label,
  count,
  page,
  pageSize,
  onPageChange,
}: {
  label: string;
  count: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}) {
  if (count <= pageSize) return null;
  const first = page * pageSize + 1;
  const last = Math.min(first + pageSize - 1, count);
  return (
    <nav aria-label={`${label} pages`} className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/[0.08] bg-surface px-4 py-2.5">
      <p aria-live="polite" className="font-sans text-xs text-primary-muted">
        {first}–{last} of {count} {label}
      </p>
      <div className="flex items-center gap-2">
        <Button tone="secondary" disabled={page === 0} onClick={() => onPageChange(page - 1)}>Previous page</Button>
        <Button tone="secondary" disabled={last === count} onClick={() => onPageChange(page + 1)}>Next page</Button>
      </div>
    </nav>
  );
}
