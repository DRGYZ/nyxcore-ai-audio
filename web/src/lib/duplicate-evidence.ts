import type { ActionPlanReport, ApiMeta, DuplicateGroup, DuplicateReport, ReviewItem } from "./types";

export type DuplicateKind = "exact" | "likely";

export type DuplicateEvidence =
  | { kind: DuplicateKind; status: "loading" | "unavailable" }
  | { kind: DuplicateKind; status: "matched"; group: DuplicateGroup; root: string };

export function duplicateKind(itemType?: string): DuplicateKind | undefined {
  if (itemType === "exact_duplicate_group") return "exact";
  if (itemType === "likely_duplicate_group") return "likely";
}

export function sameDuplicateLibrary(left?: ApiMeta, right?: ApiMeta) {
  if (!left?.music_path || !right || left.mode !== right.mode
      || left.music_path !== right.music_path || left.active_profile !== right.active_profile) {
    return false;
  }
  return left.mode === "live" || (left.mode === "snapshot"
    && Boolean(left.analysis_generation) && left.analysis_generation === right.analysis_generation);
}

function sameMembers(left: readonly unknown[], right: readonly unknown[]) {
  const members = new Set(left);
  return members.size === right.length && members.size === left.length
    && new Set(right).size === right.length && right.every((entry) => members.has(entry));
}

export function matchesDuplicateFinding(group: DuplicateGroup, item: ReviewItem, kind: DuplicateKind) {
  const affected = item.affected_paths;
  if (duplicateKind(item.item_type) !== kind || !affected || affected.length < 2
      || !sameMembers(affected, group.files.map((file) => file.path))
      || (item.preferred_path && item.preferred_path !== group.preferred.path)) {
    return false;
  }
  const recorded = item.details;
  if (Array.isArray(recorded?.preferred_reasons) && !sameMembers(recorded.preferred_reasons, group.preferred.reasons)) {
    return false;
  }
  if (kind === "exact") {
    return Boolean(group.content_hash) && (recorded?.content_hash == null || recorded.content_hash === group.content_hash);
  }
  return (recorded?.relationship_count == null || recorded.relationship_count === group.relationships?.length)
    && (item.confidence == null || item.confidence === group.confidence);
}

export function findDuplicateGroup(kind: DuplicateKind, item: ReviewItem, report: DuplicateReport) {
  const groups = kind === "exact" ? report.exact_duplicates : report.likely_duplicates;
  const matches = groups.filter((group) => matchesDuplicateFinding(group, item, kind));
  return matches.length === 1 ? matches[0] : undefined;
}

export function planMatchesDuplicateEvidence(report: ActionPlanReport, itemId: string, group: DuplicateGroup, kind: DuplicateKind) {
  if (!sameMembers(report.source_review_item_ids, [itemId])) return false;
  if (kind === "likely") {
    return report.plans.length === 0 && report.unsupported_items.some((item) => item.source_review_item_id === itemId);
  }
  const [plan] = report.plans;
  if (report.plans.length !== 1 || plan.action_type !== "exact_duplicate_keep_plan"
      || !sameMembers(plan.source_review_item_ids, [itemId])) {
    return false;
  }
  const paths = group.files.map((file) => file.path);
  return sameMembers(plan.affected_files, paths)
    && sameMembers(plan.proposed_operations.map((operation) => operation.path), paths)
    && plan.proposed_operations.every((operation) =>
      operation.expected_hash === group.content_hash
      && operation.operation_type === (operation.path === group.preferred.path ? "keep_preferred" : "quarantine_move"),
    );
}
