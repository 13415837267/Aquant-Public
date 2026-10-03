# Aquant-Public

个人量化交易系统的**短线候选池生产与研究仓库**。

当前生产目标：**1–5 个交易日短线**。

交易时序：`T 收盘信号 → T+1 开盘进入 → 最长 5 个交易日 → 退出`

## 数据与策略

`Aquant-Private/main` 是唯一策略源，当前策略版本为 **2.1.0**。

`Aquant-Public` 负责历史数据、候选生产、短线研究、质量门、GitHub Actions 和 GitHub Pages。

所有项目计算统一通过 GitHub Actions 云端运行；本地设备不是项目运行环境。仓库执行分支只保留 `main`。

生产运行直接 checkout `Aquant-Private/main`，并在候选快照与研究结果中记录策略版本和 commit。

## 短线研究

`scripts/short_term_research.py` 使用历史日线进行 T 收盘信号、T+1 开盘执行、最长 5 日持有的研究，并输出：

- `data/backtest/short_term_latest.json`
- `data/backtest/short_term_sensitivity.json`
- `data/backtest/short_term_exit_diagnostics.json`
- `data/backtest/short_term_release_validation.json`

研究检查未来函数、交易成本、止盈止损假设、候选覆盖和策略来源一致性。

## 网页

https://13415837267.github.io/Aquant-Public/

网页统一采用中国 A 股显示习惯：**红涨、绿跌、平盘中性**。当前网页不连接券商、不自动下单。

## 系统边界

生产层只负责短线候选池、研究验证、审计和网页展示。

券商接入、纸上持仓账本和自动下单属于独立后续执行层，不与候选池生产混在一起。
