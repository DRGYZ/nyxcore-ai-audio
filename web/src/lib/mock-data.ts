import type {
  DuplicateReport,
  HealthReport,
  HistoryResponse,
  PlaylistsResponse,
  ReviewReport,
  StatusResponse,
} from "./types";

export const mockStatus: StatusResponse = {
  status: "ok",
  service: "nyxcore-webapi",
  music_path: "/Volumes/Media/Audio_Archive",
  out_path: "/Volumes/Media/Audio_Archive/.nyxcore",
  active_profile: "default",
  review_state_exists: true,
  history_exists: true,
  saved_playlist_count: 3,
};

export const mockReviewReport: ReviewReport = {
  summary: {
    total_items: 4,
    counts_by_type: { exact_duplicate_group: 1, missing_metadata: 1, low_quality_audio: 1, folder_hotspot: 1 },
    counts_by_priority_band: { high: 2, medium: 1, low: 1 },
    counts_by_state: { new: 2, seen: 1, snoozed: 1 },
    total_files_referenced: 10,
  },
  metadata: {
    active_profile: "default",
    generation_mode: "mock",
  },
  items: [
    {
      item_id: "exact_duplicate_group-7abb604ec008",
      item_type: "exact_duplicate_group",
      priority_band: "high",
      priority_score: 98,
      summary: "Exact Duplicate Found: 'Blue Train.flac'",
      reason_summary: "Confirmed content hash match with 2 redundant copies.",
      review_status: "new",
      affected_paths: ["/library/jazz/Blue Train.flac", "/downloads/Blue Train Copy.flac"],
      sample_paths: ["/library/jazz/Blue Train.flac"],
      preferred_path: "/library/jazz/Blue Train.flac",
      reclaimable_bytes: 8415928320,
    },
    {
      item_id: "missing_metadata-bbf0fa8ec1c4",
      item_type: "missing_metadata",
      priority_band: "medium",
      priority_score: 85,
      summary: "Incomplete ID3 tags: 'A Love Supreme'",
      reason_summary: "Missing album and year metadata across 4 files.",
      review_status: "seen",
      affected_paths: ["/library/jazz/A Love Supreme.flac"],
      sample_paths: ["/library/jazz/A Love Supreme.flac"],
      confidence: 0.85,
    },
    {
      item_id: "low_quality_audio-724be03f4e2b",
      item_type: "low_quality_audio",
      priority_band: "high",
      priority_score: 91,
      summary: "Upscaled lossy source suspected",
      reason_summary: "218 tracks are under the low bitrate threshold.",
      review_status: "new",
      affected_paths: ["/library/imports/legacy/Track 01.mp3"],
      sample_paths: ["/library/imports/legacy/Track 01.mp3"],
    },
    {
      item_id: "folder_hotspot-a07b80ff0d9f",
      item_type: "folder_hotspot",
      priority_band: "low",
      priority_score: 72,
      summary: "Legacy Imports has concentrated issues",
      reason_summary: "Missing metadata and artwork cluster in one folder.",
      review_status: "snoozed",
      folder: "/library/imports/legacy",
      sample_paths: ["/library/imports/legacy"],
    },
  ],
};

export const mockDuplicateReport: DuplicateReport = {
  summary: {
    total_tracks: 42804,
    exact_group_count: 42,
    likely_group_count: 18,
    exact_duplicate_file_count: 126,
    likely_duplicate_file_count: 54,
  },
  exact_duplicates: [
    {
      group_id: "exact-001",
      content_hash: "sha256:9f6b1d1e01f6",
      preferred: {
        path: "/Volumes/Media/Jazz/Blue Train.flac",
        reasons: ["preferred_lossless_format", "metadata_richer"],
      },
      files: [
        { path: "/Volumes/Media/Jazz/Blue Train.flac", file_size_bytes: 42000000, extension: ".flac", has_cover_art: true, metadata_fields_present: 7 },
        { path: "/Downloads/Imports/Blue Train copy.mp3", file_size_bytes: 12000000, extension: ".mp3", has_cover_art: false, metadata_fields_present: 4 },
        { path: "/Desktop/Transfers/Blue Train duplicate.m4a", file_size_bytes: 11800000, extension: ".m4a", has_cover_art: false, metadata_fields_present: 4 },
      ],
      reclaimable_bytes: 23800000,
      reasons: ["full_hash_match"],
    },
  ],
  likely_duplicates: [
    {
      group_id: "likely-001",
      preferred: {
        path: "/Volumes/Media/Electronic/Night Drive.flac",
        reasons: ["metadata_richer", "cover_art_present"],
      },
      files: [
        { path: "/Volumes/Media/Electronic/Night Drive.flac", file_size_bytes: 28600000, extension: ".flac", has_cover_art: true, metadata_fields_present: 7 },
        { path: "/Downloads/Night Drive (Final).mp3", file_size_bytes: 9800000, extension: ".mp3", has_cover_art: false, metadata_fields_present: 5 },
      ],
      confidence: 0.82,
      reasons: ["title_match", "duration_close"],
      relationships: [
        {
          paths: ["/Volumes/Media/Electronic/Night Drive.flac", "/Downloads/Night Drive (Final).mp3"],
          confidence: 0.82,
          reasons: ["filename_similarity", "metadata_near_match"],
        },
      ],
    },
  ],
};

export const mockHealthReport: HealthReport = {
  overview: {
    total_audio_files: 42804,
    total_folders_touched: 812,
    format_breakdown: {
      ".flac": 24520,
      ".mp3": 12340,
      ".m4a": 4120,
      ".wav": 1824,
    },
    total_library_size_bytes: 1024000000000,
    duplicate_exact_groups: 42,
    duplicate_likely_groups: 18,
  },
  metadata: {
    missing_title: { count: 88 },
    missing_artist: { count: 124 },
    missing_album: { count: 240 },
    placeholder_metadata: { count: 5200 },
    suspicious_title_artist_swaps: { count: 17 },
  },
  artwork: {
    coverage_percent: 92.1,
    with_artwork: 39421,
    without_artwork: 3383,
  },
  quality: {
    bitrate_buckets: {
      unknown: 12,
      "<128k": 89,
      "128k-191k": 1200,
      "192k-255k": 6200,
      ">=256k": 35303,
    },
    low_bitrate_files: { count: 89 },
    lossless_files: 24520,
    lossy_files: 18284,
    duration_outliers: { count: 42 },
    unreadable_or_unparseable_files: { count: 12 },
  },
  naming: {
    noisy_filenames: { count: 312 },
    long_filenames: { count: 104 },
    filename_metadata_disagreement: { count: 67 },
    high_issue_folders: [
      { folder: "imports/legacy", issue_count: 94 },
      { folder: "downloads/staging", issue_count: 58 },
    ],
  },
  duplicates: {
    exact_duplicate_groups: 42,
    likely_duplicate_groups: 18,
    files_in_exact_duplicates: 126,
    files_in_likely_duplicates: 54,
    total_files_in_duplicates: 180,
    reclaimable_bytes_exact: 6815744000,
  },
  priorities: {
    top_issue_categories: [
      { category: "placeholder_metadata", count: 5200, action: "Repair placeholder tags in high-volume folders" },
      { category: "exact_duplicates", count: 42, action: "Review duplicate keep plans first" },
    ],
    top_problematic_folders: [
      { folder: "imports/legacy", issue_count: 94 },
      { folder: "downloads/staging", issue_count: 58 },
    ],
    recommended_actions: [
      "Review exact duplicates first",
      "Repair missing artist tags in Legacy Imports",
      "Re-scan corrupt FLAC containers in Vault",
    ],
  },
};

export const mockPlaylistsResponse: PlaylistsResponse = {
  meta: {
    music_path: "/Volumes/Media/Audio_Archive",
    out_path: "/Volumes/Media/Audio_Archive/.nyxcore",
    active_profile: "default",
    mode: "mock",
  },
  items: [
    {
      playlist_id: "smart-pl-0921-x",
      name: "70s Hard Bop High-Res",
      profile: "collector",
      query: "hard bop 1970s flac",
      last_refreshed_at: "2026-03-06T18:22:00+00:00",
      track_count: 1242,
      latest_summary: { track_count: 1242, estimated_total_duration_seconds: 235980 },
      latest_refresh_diff: {
        tracks_added: ["/music/jazz/Blue Train (2024 Remaster).flac", "/music/jazz/The Sidewinder.flac"],
        tracks_removed: ["/music/jazz/Moanin'.flac"],
        track_count_delta: 11,
        estimated_duration_delta_seconds: 1640,
      },
    },
    {
      playlist_id: "ambient-focus-2cf21a",
      name: "Ambient Focus - No Lyrics",
      profile: "default",
      query: "focus music no vocals bpm under 100",
      last_refreshed_at: "2026-03-05T09:18:00+00:00",
      track_count: 458,
      latest_summary: { track_count: 458, estimated_total_duration_seconds: 88210 },
      latest_refresh_diff: {
        tracks_added: ["/music/ambient/Stillness.wav"],
        tracks_removed: [],
        track_count_delta: 1,
        estimated_duration_delta_seconds: 381,
      },
    },
  ],
};

export const mockHistoryResponse: HistoryResponse = {
  meta: {
    music_path: "/Volumes/Media/Audio_Archive",
    out_path: "/Volumes/Media/Audio_Archive/.nyxcore",
    active_profile: "default",
    mode: "mock",
  },
  items: [
    {
      batch_id: "batch-ff2b6ac418a4",
      applied_at: "2026-03-06T14:22:01+00:00",
      action_types: ["metadata_fix_plan"],
      reversible: true,
      affected_count: 12,
      source_plan_ids: ["plan-001"],
      source_review_item_ids: ["missing_metadata-bbf0fa8ec1c4"],
      operations: [
        {
          operation_id: "op-1",
          operation_type: "write_metadata",
          status: "ok",
          reversible: true,
          original_path: "/music/jazz/A Love Supreme.flac",
          current_path: "/music/jazz/A Love Supreme.flac",
        },
      ],
    },
    {
      batch_id: "batch-b0c88f1082c1",
      applied_at: "2026-03-06T13:05:45+00:00",
      action_types: ["exact_duplicate_keep_plan"],
      reversible: true,
      affected_count: 3,
      source_plan_ids: ["plan-002"],
      source_review_item_ids: ["exact_duplicate_group-7abb604ec008"],
      operations: [
        {
          operation_id: "op-2",
          operation_type: "quarantine_move",
          status: "ok",
          reversible: true,
          original_path: "/music/imports/Blue Train copy.mp3",
          current_path: "/music/.nyxcore_quarantine/imports/Blue Train copy.mp3",
        },
      ],
    },
  ],
};
