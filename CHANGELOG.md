# Changelog

All code changes are recorded here with the corresponding Git commit or PR. Automated data-only refreshes use the `data:` commit prefix.

## [0.1.0] — 2026-09-30
- Initial public release: Next.js 16 dashboard for the daily candidate pool.
- Added AKShare/Eastmoney A-share spot ingestion with a trading-day guard.
- Added explainable cross-sectional factors: momentum, liquidity, value, risk and activity.
- Added daily 18:00 Beijing-time GitHub Actions refresh (`10:00 UTC`).
- Added health and candidate JSON API routes.
- Established the private-research / public-runtime repository split.
