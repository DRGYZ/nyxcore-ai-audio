import type { ActionPlan, ActionPlanOperation, ActionPlanReport } from "./types";

export const FALLBACK_MAX_SELECTED_OPERATIONS = 50;

const SELECTABLE_OPERATION_TYPES = new Set([
  "write_metadata",
  "rename_file",
  "quarantine_move",
]);

export function isSelectableActionOperation(operation: ActionPlanOperation) {
  return SELECTABLE_OPERATION_TYPES.has(operation.operation_type);
}

export function actionOperationSelectionKey(planId: string, operationId: string) {
  return `${planId}:${operationId}`;
}

export function isActionOperationSelectable(plan: ActionPlan, operation: ActionPlanOperation) {
  return plan.apply_supported && operation.apply_supported && isSelectableActionOperation(operation);
}

export function selectedOperationCount(selectedOperationIds: ReadonlySet<string>) {
  return selectedOperationIds.size;
}

export function setActionOperationSelected(
  selectedOperationIds: Set<string>,
  report: ActionPlanReport,
  planId: string,
  operationId: string,
  selected: boolean,
): Set<string> {
  const maximum = report.summary.max_automatic_operations ?? FALLBACK_MAX_SELECTED_OPERATIONS;
  const plan = report.plans.find((candidate) => candidate.plan_id === planId);
  const target = plan?.proposed_operations.find((operation) => operation.operation_id === operationId);
  if (!plan || !target || !isActionOperationSelectable(plan, target)) return selectedOperationIds;

  const key = actionOperationSelectionKey(planId, operationId);
  if (selected && !selectedOperationIds.has(key) && selectedOperationIds.size >= maximum) {
    return selectedOperationIds;
  }

  const next = new Set(selectedOperationIds);
  if (selected) next.add(key);
  else next.delete(key);
  return next;
}

export function buildSelectedActionPlanReport(
  report: ActionPlanReport,
  selectedOperationIds: ReadonlySet<string>,
) {
  const plans = report.plans
    .map((plan): ActionPlan => {
      if (!plan.apply_supported) return { ...plan, proposed_operations: [] };
      const proposedOperations = plan.proposed_operations.filter(
        (operation) =>
          operation.operation_type === "keep_preferred"
          || (
            isActionOperationSelectable(plan, operation)
            && selectedOperationIds.has(actionOperationSelectionKey(plan.plan_id, operation.operation_id))
          ),
      );
      const selectedMutations = proposedOperations.filter(isSelectableActionOperation);
      const selectableMutations = plan.proposed_operations.filter(isSelectableActionOperation);
      const allMutationsSelected = selectableMutations.length > 0 && selectableMutations.every(
        (operation) =>
          isActionOperationSelectable(plan, operation)
          && selectedOperationIds.has(actionOperationSelectionKey(plan.plan_id, operation.operation_id)),
      );
      return {
        ...plan,
        affected_files: [
          ...new Set(
            proposedOperations
              .map((operation) => operation.path)
              .filter((path): path is string => Boolean(path)),
          ),
        ].sort(),
        proposed_operations: proposedOperations,
        apply_supported: selectedMutations.length > 0,
        resolves_review_items: plan.resolves_review_items && allMutationsSelected,
      };
    })
    .filter((plan) => plan.apply_supported);

  return {
    ...report,
    plans,
    summary: {
      ...report.summary,
      generated_plan_count: plans.length,
      apply_supported_plan_count: plans.length,
    },
  };
}
