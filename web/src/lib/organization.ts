export type OrganizationStatus = "ready" | "needs_review" | "collision" | "unchanged";
export type OrganizationView = "files" | "folders";

export type OrganizationProposal = {
  path: string;
  source: string;
  status: OrganizationStatus;
  reasons: string[];
  destination: [artist: string, album: string, filename: string] | null;
  inspectable: boolean;
};

export type OrganizationPreview = {
  id: string;
  root: string;
  counts: Record<OrganizationStatus, number>;
  apply_enabled: boolean;
};

export type OrganizationApproval = {
  approval: string;
  moves: { source: string; destination: string }[];
};

export type OrganizationResult = {
  batch_id: string;
  operations: { operation_id: string; path: string; destination_path: string; status: "ok" | "error" | "skipped"; message: string }[];
};

export type OrganizationPageData = {
  total: number;
  page_size: number;
  rows: OrganizationProposal[];
};
