import { Route, Routes } from "react-router-dom";
import type { DesktopSession } from "../lib/api";
import { DashboardPage } from "./pages/DashboardPage";
import { DuplicatesPage } from "./pages/DuplicatesPage";
import { HealthPage } from "./pages/HealthPage";
import { HistoryPage } from "./pages/HistoryPage";
import { MetadataPage } from "./pages/MetadataPage";
import { MetadataEditor } from "./metadata/MetadataEditor";
import { OrganizationPage } from "./pages/OrganizationPage";
import { PlaylistsPage } from "./pages/PlaylistsPage";
import { ReviewPage } from "./pages/ReviewPage";
import { SearchPage } from "./pages/SearchPage";
import { AppShell } from "./shell";

export function App({ desktop = false, desktopSession }: { desktop?: boolean; desktopSession?: DesktopSession }) {
  return (
    <AppShell desktop={desktop} demo={desktopSession?.demo}>
      <Routes>
        <Route path="/" element={<DashboardPage desktop={desktop} />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/playlists" element={<PlaylistsPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/duplicates" element={<DuplicatesPage />} />
        <Route path="/health" element={<HealthPage desktop={desktop} />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/metadata" element={<MetadataPage />} />
        <Route path="/organization" element={<OrganizationPage />} />
      </Routes>
      <MetadataEditor />
    </AppShell>
  );
}
