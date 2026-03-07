import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  applyReviewPlan,
  fetchDuplicates,
  fetchHealth,
  fetchHistory,
  fetchPlaylists,
  fetchReview,
  fetchStatus,
  generateReviewPlan,
  mutateReviewState,
  restoreHistoryBatch,
  undoHistoryBatch,
} from "./api";
import type { ReportEnvelope, ReviewReport } from "./types";

export function useStatusQuery() {
  return useQuery({ queryKey: ["status"], queryFn: fetchStatus });
}

export function useHealthQuery() {
  return useQuery({ queryKey: ["health"], queryFn: fetchHealth });
}

export function useReviewQuery() {
  return useQuery({ queryKey: ["review"], queryFn: fetchReview });
}

export function useDuplicatesQuery() {
  return useQuery({ queryKey: ["duplicates"], queryFn: fetchDuplicates });
}

export function usePlaylistsQuery() {
  return useQuery({ queryKey: ["playlists"], queryFn: fetchPlaylists });
}

export function useHistoryQuery() {
  return useQuery({ queryKey: ["history"], queryFn: fetchHistory });
}

export function useReviewStateMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: mutateReviewState,
    onMutate: async (variables) => {
      await queryClient.cancelQueries({ queryKey: ["review"] });
      const previous = queryClient.getQueryData<ReportEnvelope<ReviewReport>>(["review"]);
      if (previous) {
        queryClient.setQueryData<ReportEnvelope<ReviewReport>>(["review"], {
          ...previous,
          data: {
            ...previous.data,
            items: previous.data.items.map((item) =>
              variables.item_ids.includes(item.item_id)
                ? { ...item, review_status: variables.action as typeof item.review_status }
                : item,
            ),
          },
        });
      }
      return { previous };
    },
    onError: (_error, _variables, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["review"], context.previous);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["review"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
}

export function useGenerateReviewPlanMutation() {
  return useMutation({ mutationFn: generateReviewPlan });
}

export function useApplyReviewPlanMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: applyReviewPlan,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["review"] });
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
}

export function useRestoreHistoryMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ batchId, target_path, alternate_restore_dir }: { batchId: string; target_path?: string; alternate_restore_dir?: string }) =>
      restoreHistoryBatch(batchId, { target_path, alternate_restore_dir }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["review"] });
    },
  });
}

export function useUndoHistoryMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ batchId, target_path, alternate_restore_dir }: { batchId: string; target_path?: string; alternate_restore_dir?: string }) =>
      undoHistoryBatch(batchId, { target_path, alternate_restore_dir }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["review"] });
    },
  });
}
