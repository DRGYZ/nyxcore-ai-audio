import { Route, Routes } from "react-router-dom";
import { DashboardPage } from "./pages/DashboardPage";
import { DuplicatesPage } from "./pages/DuplicatesPage";
import { HealthPage } from "./pages/HealthPage";
import { HistoryPage } from "./pages/HistoryPage";
import { PlaylistsPage } from "./pages/PlaylistsPage";
import { ReviewPage } from "./pages/ReviewPage";
import { AppShell } from "./shell";

export function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/playlists" element={<PlaylistsPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/duplicates" element={<DuplicatesPage />} />
        <Route path="/health" element={<HealthPage />} />
      </Routes>
    </AppShell>
  );
}
