import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchMetadataIssue } from "../../lib/api";
import { pathFilename } from "../../lib/library-path";
import { ISSUE_LABELS, type MetadataIssue } from "../../lib/metadata";
import { ActionBanner, Button, PathBlock } from "../components";
import { MetadataEditButton } from "./MetadataEditor";

const PAGE_SIZE = 25;

export function MetadataIssueList({ issue }: { issue: MetadataIssue }) {
  const [page, setPage] = useState(0);
  const query = useQuery({
    queryKey: ["metadata", "issues", issue, page],
    queryFn: () => fetchMetadataIssue(issue, page * PAGE_SIZE), retry: false,
  });
  const data = query.data;
  return (
    <section className="mt-4 space-y-3" aria-label={`${ISSUE_LABELS[issue]} affected tracks`}>
      <h4 className="text-sm font-medium text-primary">{ISSUE_LABELS[issue]} · affected tracks</h4>
      {query.isLoading && <p role="status" className="text-xs text-primary-muted">Loading tracks…</p>}
      {query.error && <ActionBanner tone="error" message={query.error.message} />}
      {data && (
        <>
          {data.total ? <ul className="divide-y divide-white/10 border-y border-white/10">
            {data.paths.map((path) => (
              <li key={path} className="space-y-1.5 py-2.5">
                <p className="text-xs font-semibold text-primary [overflow-wrap:anywhere]">{pathFilename(path)}</p>
                <PathBlock value={path} />
                <MetadataEditButton path={path} />
              </li>
            ))}
          </ul> : <p className="text-xs text-primary-muted">No current tracks match this issue.</p>}
          {(page > 0 || data.total > PAGE_SIZE) && <nav aria-label="Affected tracks pages" className="flex flex-wrap items-center gap-2 text-xs text-primary-muted">
            <Button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</Button>
            <span>{data.paths.length ? `${page * PAGE_SIZE + 1}–${Math.min(data.total, (page + 1) * PAGE_SIZE)}` : "0"} of {data.total}</span>
            <Button type="button" disabled={(page + 1) * PAGE_SIZE >= data.total} onClick={() => setPage(page + 1)}>Next</Button>
          </nav>}
        </>
      )}
    </section>
  );
}
