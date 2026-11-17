import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  applyReviewPlan,
  createPlaylist,
  fetchDuplicates,
  fetchHealth,
  fetchHistory,
  fetchLibraryReports,
  fetchPlaylists,
  fetchReview,
  fetchArchiveSearch,
  fetchStatus,
  generateReviewPlan,
  mutateReviewState,
  refreshLibrary,
  refreshPlaylist,
  restoreHistoryBatch,
  undoHistoryBatch,
} from "./api";
import type { LibraryReports, ReportEnvelope, ReviewReport } from "./types";

const libraryQuery = {
  queryKey: ["library"],
  queryFn: fetchLibraryReports,
  staleTime: 30_000,
  retry: false,
};

export function useCheckConnection() {
  const queryClient = useQueryClient();
  const [checking, setChecking] = useState(false);

  const checkConnection = useCallback(async () => {
    setChecking(true);
    try {
      await Promise.allSettled([
        queryClient.invalidateQueries({ queryKey: ["status"] }),
        queryClient.invalidateQueries({ queryKey: ["health"] }),
        queryClient.invalidateQueries({ queryKey: ["review"] }),
        queryClient.invalidateQueries({ queryKey: ["duplicates"] }),
        queryClient.invalidateQueries({ queryKey: ["history"] }),
        queryClient.invalidateQueries({ queryKey: libraryQuery.queryKey }),
      ]);
    } finally {
      setChecking(false);
    }
  }, [queryClient]);

  return { checkConnection, checking };
}

export function useStatusQuery() {
  return useQuery({ queryKey: ["status"], queryFn: fetchStatus, retry: 1 });
}

export function useLibraryAnalysis() {
  const reviewOnly = useStatusQuery().data?.review_only === true;
  return useQuery({ ...libraryQuery, enabled: reviewOnly });
}

export function useRefreshLibrary() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => {
      await refreshLibrary();
      await queryClient.cancelQueries({ queryKey: libraryQuery.queryKey });
      try {
        await queryClient.invalidateQueries({ queryKey: libraryQuery.queryKey }, { throwOnError: true });
      } catch {
        throw new Error("The library was scanned, but the updated reports could not be loaded. Try Refresh Library again.");
      }
    },
  });
}

function useLibraryReport<T>(
  key: string,
  legacyFetch: () => Promise<ReportEnvelope<T>>,
  select: (library: LibraryReports) => ReportEnvelope<T>,
  enabled = true,
) {
  const status = useStatusQuery();
  const shared = status.data?.review_only === true;
  const snapshot = useQuery({ ...libraryQuery, enabled: enabled && shared, select });
  const individual = useQuery({
    queryKey: [key], queryFn: legacyFetch, enabled: enabled && status.isFetched && !shared, retry: 1,
  });
  return shared ? snapshot : individual;
}

const selectHealth = ({ meta, health }: LibraryReports) => ({ meta, data: health });
const selectReview = ({ meta, review }: LibraryReports) => ({ meta, data: review });
const selectDuplicates = ({ meta, duplicates }: LibraryReports) => ({ meta, data: duplicates });

export function useHealthQuery() {
  return useLibraryReport("health", fetchHealth, selectHealth);
}

export function useReviewQuery() {
  return useLibraryReport("review", fetchReview, selectReview);
}

export function useDuplicatesQuery(enabled = true) {
  return useLibraryReport("duplicates", fetchDuplicates, selectDuplicates, enabled);
}

export function useArchiveSearchQuery(query: string) {
  const normalizedQuery = query.trim();
  const reviewOnly = useStatusQuery().data?.review_only === true;
  const generation = useLibraryAnalysis().data?.meta.analysis_generation;
  return useQuery({
    queryKey: ["search", generation, normalizedQuery],
    queryFn: () => fetchArchiveSearch(normalizedQuery),
    enabled: normalizedQuery.length >= 2 && (!reviewOnly || Boolean(generation)),
  });
}

export function usePlaylistsQuery() {
  return useQuery({ queryKey: ["playlists"], queryFn: fetchPlaylists });
}

export function useCreatePlaylistMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createPlaylist,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["playlists"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
}

export function useRefreshPlaylistMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ playlistId, export_m3u }: { playlistId: string; export_m3u?: boolean }) =>
      refreshPlaylist(playlistId, { export_m3u }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["playlists"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
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
    onSettled: (_data, error, variables) => {
      queryClient.invalidateQueries({ queryKey: libraryQuery.queryKey });
      // Keep a successful resolution visible until the next explicit or natural
      // refresh. That refresh is the scan boundary that may reactivate it.
      if (error || variables.action !== "resolved") {
        queryClient.invalidateQueries({ queryKey: ["review"] });
      }
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
      queryClient.invalidateQueries({ queryKey: ["health"] });
      queryClient.invalidateQueries({ queryKey: ["duplicates"] });
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
      queryClient.invalidateQueries({ queryKey: ["health"] });
      queryClient.invalidateQueries({ queryKey: ["duplicates"] });
      queryClient.invalidateQueries({ queryKey: ["review"] });
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
}


export function useUndoHistoryMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ batchId, target_path, alternate_restore_dir }: { batchId: string; target_path?: string; alternate_restore_dir?: string }) =>
      undoHistoryBatch(batchId, { target_path, alternate_restore_dir }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["health"] });
      queryClient.invalidateQueries({ queryKey: ["duplicates"] });
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["review"] });
    },
  });
}

export function useReverseHistoryMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ batchId, target_path, alternate_restore_dir }: { batchId: string; target_path?: string; alternate_restore_dir?: string }) =>
      restoreHistoryBatch(batchId, { target_path, alternate_restore_dir }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["health"] });
      queryClient.invalidateQueries({ queryKey: ["duplicates"] });
      queryClient.invalidateQueries({ queryKey: ["history"] });
      queryClient.invalidateQueries({ queryKey: ["review"] });
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
}
