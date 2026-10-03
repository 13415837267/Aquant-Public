# Aquant-Public

个人量化交易系统的**短线候选池生产与研究仓库**。

当前生产目标：**1–5 个交易日短线**。

交易时序：`T 收盘信号 → T+1 开盘进入 → 最长 5 个交易日 → 退出`

核心因子：

- 短线动量 35%
- 量能活跃度 25%
- 价格强度 15%
- 流动性 15%
- 安全 10%

生产候选最多 3 个；市场广度过弱时允许 0 个候选。

## 数据与策略

`Aquant-Private` 是唯一策略源，当前版本为 2.0.0。

`Aquant-Public` 负责历史数据、候选生产、短线研究、质量门、GitHub Actions 和 GitHub Pages。

生产运行直接 checkout `Aquant-Private/main`，并在候选快照中记录策略版本与 commit。

## 短线研究

`scripts/short_term_research.py` 使用历史日线进行 T 收盘信号、T+1 开盘执行、最长 5 日持有的研究，并输出：

- `data/backtest/short_term_latest.json`
- `data/backtest/short_term_sensitivity.json`
- `data/backtest/short_term_exit_diagnostics.json`

研究同时检查未来函数、交易成本、止盈止损假设和候选样本覆盖。

## 网页

https://13415837267.github.io/Aquant-Public/

网页展示最新生产候选、短线因子、市场状态和策略版本，不连接券商、不自动下单。

## 阶段边界

当前生产层是**短线候选池 + 研究验证 + 网页展示**。

券商接入、纸上持仓账本和自动下单属于后续执行层，不与候选池生产混在一起。
