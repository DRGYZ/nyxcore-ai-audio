# Changelog

All notable changes to the NyxCore project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-16

### Added
- **Editorial Noir Web UI**: Implemented high-contrast editorial design system across six primary web surfaces (Overview, Archive Search, Review Inbox, Duplicates, Library Health, History).
- **Accessibility Hardening**:
  - Semantic keyboard-navigable interactive controls for review queues, duplicates, and history rows.
  - Centralized focus trapping, Escape handling, and safe focus restoration in `Modal` dialogs (`PlanReportModal`, batch reversal confirmation).
  - ARIA status regions, visible focus outlines across all interactive elements, and accessible error boundaries.
- **Frontend Test Suite**: Added automated component and interaction test harness using Vitest and `@testing-library/react`.
- **Responsive Hardening**: Responsive layout adaptations across desktop, tablet, and mobile viewports with table scroll containers, flex wraps, and adaptive drawer/modal sizing.
- **CI Automation**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) covering backend pytest suites and frontend Vitest runs and production builds.
- **Curated Screenshots**: High-resolution 1440px desktop showcase and 768px tablet verification captures cataloged under `screenshots/`.

### Changed
- **Root Documentation**: Comprehensive rewrite of `README.md` with technical workflow architecture, safety guarantees and limitations, quickstart guide, and screenshot gallery.
- **WSL2 Guide**: Rewrote `INSTALL_WSL.md` with generic paths and accurate dependency requirements.
- **Dependency Clarity**: Clarified that pure Python `mutagen` powers core metadata operations; `ffmpeg` is strictly optional for synthetic audio fixtures.

### Safety
- **Explicit Review-First Workflow**: Enforced `scan → detect → review → plan → apply → history → reverse` lifecycle.
- **Serialized Action Plans**: Strict preflight validation, SHA-256 content fingerprints, and 50-operation safety thresholds.
- **No-Clobber Quarantine**: Non-preferred duplicate relocation into `.nyxcore_quarantine` with collision-safe naming.
- **Durable Intent Journaling & Single-Writer Locking**: Interruption-safe operation journaling and library-level lockfile protection.
- **Crash Recovery Inspection**: `recover-review-action --action inspect` for safe classification of interrupted batches.
- **Disabled Legacy Commands**: Unreviewed mutation commands (`apply`, `rename --apply`, `rename-undo`, `apply-ai`, `apply-judge`) are disabled in this release.

## [0.2.0] - 2026-08-15

### Added
- Initial local FastAPI backend and web prototype.
- Duplicate detection (exact hash and fuzzy title/duration matching).
- Library health diagnostics (missing tags, placeholder detection, bitrate analysis).
- Basic CLI commands for scan, duplicates, and health reporting.
