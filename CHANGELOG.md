# Changelog

## 2026-10-03 — Short-term system pivot 2.0.0

- Re-scoped the production system to a 1–5 trading-session short-term signal horizon.
- Replaced the medium-term 126-session/value-led candidate model with short-horizon momentum, volume activity, price strength, liquidity and safety.
- Added a market breadth no-trade gate and dynamic Top-3 admission.
- Added T_close -> T+1_open -> max five-session research with explicit +6% target and -3% stop reference.
- Updated GitHub Pages to show short-term signal features.
- Replaced production readiness checks so old medium-term backtests are no longer the release gate.

## 2026-10-03 — Candidate pool focus cleanup

- Kept historical data and PIT fundamentals as research infrastructure.
- Removed obsolete portfolio/execution outputs from the current candidate-pool production target.
