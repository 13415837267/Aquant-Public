# Changelog

Every code change is recorded here with the corresponding Git commit or pull request. Automated data-only refreshes use the `data:` commit prefix.

## [0.1.0] — 2026-09-30
- Initial public runtime: Next.js 16 App Router candidate-pool dashboard.
- Added AKShare/Eastmoney A-share spot ingestion with trading-day guard.
- Added cross-sectional factors: momentum, liquidity, value, risk and activity.
- Added daily 18:00 Beijing-time GitHub Actions refresh (`10:00 UTC`).
- Added `/api/candidates` and `/api/health`.
- Added a pull-request quality gate requiring changelog entries for code/workflow changes.
- Added repository maintenance and architecture documentation.
- Removed the unused Vercel Cron/refresh endpoint so GitHub Actions remains the single source of truth for daily data updates.
- Added one immutable `data/history/YYYY-MM-DD.json` snapshot per trading day for incremental history.

## Unreleased — GitHub Pages
- Switched the Next.js runtime to static export for GitHub Pages.
- Added GitHub Pages build/deploy workflow.
- Removed server-only API routes; the dashboard reads the generated candidate snapshot at build time.

## Unreleased — Cloud-only runtime boundary
- Formalized Aquant-Public as the exclusive cloud build/runtime surface.
- Clarified that Aquant-Private does not participate in CI, scheduled jobs, production backtests, or site builds.
- Removed stale API/Vercel deployment references from the public runtime documentation.

## Unreleased — Cloud build hardening
- Changed the GitHub Pages workflow from npm ci to npm install because the public repository intentionally has no committed npm lockfile yet.
- Kept the entire production build on GitHub-hosted runners without requiring local dependency installation.
