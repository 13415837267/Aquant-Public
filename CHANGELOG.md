## 2026-10-04 — Candidate coverage optimization 2.3.0

- Retained the primary 0.89 precision gate and added a 0.88 research-backed daily Top-1 fallback when no primary candidate is available in a non-risk-off regime.
- Added explicit admission-tier fields and validation for fallback provenance.
- Kept hard short-term risk controls and A-share T_close -> T+1_open -> T+2 earliest-exit semantics unchanged.
- Candidate coverage is improved without lowering the primary quality gate; research metrics still do not equal an execution-constrained 80%+ production guarantee.

# Changelog

## 2026-10-03 — Runtime cleanup and A-share UI normalization

- Consolidated the repository around the `main` branch only.
- Removed obsolete admission/variant/next-generation research workflows, scripts and stale research artifacts.
- Kept historical data, PIT fundamentals, candidate generation, short-term research, audit and Pages as the core runtime.
- Restored the A-share color convention to red-up and green-down across directional return metrics.
- Added stale-result protection so an old cloud research run cannot write results after the source strategy changes.
- Standardized Aquant-Private/main as the sole strategy source; current baseline strategy is 2.1.0.

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
