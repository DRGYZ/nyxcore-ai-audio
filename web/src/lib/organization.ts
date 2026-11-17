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
};

export type OrganizationPageData = {
  total: number;
  page_size: number;
  rows: OrganizationProposal[];
};
