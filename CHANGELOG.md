## 2026-10-03 — Candidate pool focus cleanup

- 将生产目标重新收敛为“每日候选池”，组合与执行层不再属于当前生产产物。
- 删除重复的执行/组合 Workflow、执行回测模块及对应生成结果。
- 新增候选池独立质量门与按交易日不可覆盖的候选历史归档。
- 网页改为直接读取生产候选快照中的正式因子权重。
- 统一保留一个候选池研究入口，避免重复运行研究任务。
- 合并后再次清理研究与每日生产流程中的旧组合检查和重复候选校验，并将每日任务限制为工作日触发。

## 2026-10-03

- 完成共享组合权重接口，统一生产组合与约束回测的逆波动率配置、5%单票上限和5%现金缓冲。
- 每日云端生产链已打通：候选池 → 目标组合 → 执行计划 → 跨层审计。
- 增加五年 PIT 基本面云端回补与 `report_date/pub_date` 完整性校验。
- 增加跨层系统就绪审计，并坚持 GitHub Actions 为唯一生产/研究计算环境。
- 修正约束回测工作流默认参数与组合逻辑变更触发规则。

# Changelog

## Unreleased — Production-weight constrained backtest
- Added a shared inverse-volatility allocator in `scripts/portfolio.py` so production portfolio construction and the constrained backtest use identical 5% single-name caps and 5% cash buffers.
- Reworked `scripts/backtest_constrained.py` to persistent T+1 holdings instead of forced end-of-day flattening.
- Added 100-share lot handling, a configurable 30% daily turnover cap, cash-floor enforcement, and small non-target residual cleanup within unused turnover capacity.
- Corrected performance statistics to exclude the feature warm-up period while retaining it in the full daily audit trail.

## Unreleased — Portfolio execution planning
- Added `scripts/execution_plan.py` to convert the production portfolio into a next-open execution plan.
- Enforced 100-share lot rounding, T+1 available-share limits, paused/ST checks, aggregate turnover caps, cash-floor checks, and configurable commission/stamp-duty/slippage assumptions.
- Latest-close prices are explicitly reference-only; every order carries a next-open recheck gate for price-limit and execution-state validation.

## Unreleased — Database scope and Shanghai/Shenzhen main-board candidates
- Added a metadata reconciliation script and manual finalizer workflow so the historical coverage manifest is rebuilt from the actual Git-tracked daily files after a multi-year rebuild.
- Kept the database broader than the production candidate pool so Shanghai/Shenzhen/Beijing stock rows can remain available for research.
- Historical yearly backfill no longer uses the current active strategy universe as the database row filter; candidate generation is the production boundary for the Shanghai/Shenzhen main-board scope.
- Added --rebuild and a workflow input so existing historical files can be re-downloaded under the database stock scope.
- Candidate generation enforces the main-board scope both while reading the history window and inside the candidate builder.
- 60-day momentum and 20-day volatility now use the provider daily pct_chg return series; the future-sensitive adjustment factor is retained as data but is not used to construct the signal.
- Candidate generation excludes stocks without a complete 60-trading-day momentum window.

Every code change is recorded here with the corresponding Git commit or pull request. Automated data-only refreshes use the `data:` commit prefix.

## Unreleased — Checkpoint push resilience
- Historical-data Git checkpoints now rebase onto the current remote main branch before pushing, so unrelated concurrent candidate/data commits do not abort a long backfill run.

## Unreleased — Database-driven candidate engine
- Replaced live AKShare/Eastmoney spot input in `scripts/update_candidates.py` with the committed `data/history/YYYY-MM-DD.csv.gz` database.
- Added 20/60 trading-day momentum, 20-day realized volatility, 20-day average traded value and 20-day average turnover features.
- Corrected the risk-factor direction so lower realized volatility receives the stronger rank.
- Kept the strategy universe aligned with the historical database: non-ST, non-delisted, non-paused, price above 2 yuan and recent traded value above 20 million yuan.
- Candidate snapshots now record the database date and strategy version.

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