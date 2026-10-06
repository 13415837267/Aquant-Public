# Architecture

## Production flow

```text
zzshare
   |
   v
Shanghai/Shenzhen main-board historical database
   |
   +--> daily incremental data
   |
   +--> historical/fundamental research data
   |
   v
scripts/update_candidates.py
   |
   +--> hard eligibility filters
   +--> cross-sectional factor scoring from Aquant-Private/main
   +--> candidate admission policy
   |
   v
data/candidates.json
   |
   v
scripts/portfolio.py
   |
   +--> deterministic inverse-volatility allocation
   +--> 5% single-name cap / 5% cash buffer
   |
   v
data/portfolio.json
   |
   v
scripts/execution_plan.py
   |
   +--> lot/T+1/cash/turnover checks
   +--> next-open recheck gate
   |
   v
data/execution_plan.json
   |
   +--> historical backtest / constrained execution research
   |
   v
Next.js / GitHub Pages
```

## Repository responsibilities

- **Aquant-Private**: the single source of truth for production strategy code, factors, parameters, and strategy version.
- **Aquant-Public**: production data, database collection, quality checks, candidate generation, backtesting infrastructure, Actions, and Pages.
- Public does **not** maintain an independent strategy implementation.

## Database

Daily market and valuation data are persisted as:

`data/history/YYYY/YYYY-MM-DD.csv.gz`

Quarterly financial data are persisted under:

`data/fundamentals/{indicator,income,balance,cash_flow}/`

`scripts/pit_fundamentals.py` provides the canonical point-in-time read path: only publications with `pub_date <= trade_date` can enter a historical information set, and the latest available publication is selected per security.

Candidate generation reads the persisted database rather than rebuilding five years of history from the provider on every run.

## Model lifecycle and daily inference

第一版正式模型权重固定保存于：

`data/models/production_v1.json`

模型资产包含模型版本、模型代码提交号、训练窗口、特征空间和完整权重。生产候选生成必须先校验模型资产与 `config/production_release_v1.json` 一致；日常运行**禁止重新训练**。

日常交易日流程为：

1. `02` 仅增量维护最新市场数据并完成数据校验。
2. `03` 读取当前固定模型，不重新训练。
3. 为最新交易日建立特征时，只使用该交易日及此前可见的数据；滚动特征只读取所需历史窗口。
4. 固定概率阈值和候选数量规则后生成候选池。
5. 只有新的研究版本完成验证并被正式发布时，才允许替换固定模型权重。

历史日期回放也读取同一个固定模型，但输入数据截断到目标信号日；因此不会读取目标日期之后的行情数据。需要区分：这种回放用于回答“当前固定模型在历史截面上的评分”，并不等同于“站在历史日期重新训练模型后的完全无未来模型回测”。

Private 仓库仍作为策略源和版本审计来源；Public 不日常重新实现或训练一套独立策略。

## Runtime boundary

Production computation covers data collection, candidate generation, portfolio construction, execution-plan preparation, and historical execution-constrained research. Broker/OMS submission remains a separate external service; `data/execution_plan.json` is an auditable plan and not a broker fill.
