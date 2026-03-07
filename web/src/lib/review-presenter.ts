import type { PriorityBand, ReviewStatus } from "./types";

export type ReviewTone = "neutral" | "primary" | "success" | "warning" | "danger";

export function reviewPriorityTone(priority: PriorityBand): ReviewTone {
  if (priority === "high") return "danger";
  if (priority === "medium") return "warning";
  return "neutral";
}

export function reviewStatusTone(status: ReviewStatus): ReviewTone {
  if (status === "new") return "primary";
  if (status === "resolved") return "success";
  if (status === "snoozed") return "warning";
  return "neutral";
}
