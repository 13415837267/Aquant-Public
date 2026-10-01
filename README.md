# Aquant-Public

个人量化交易系统的**公开数据与生产运行仓库**。

本仓库负责公开、可复现的量化基础设施：全沪深京 A 股历史数据库、数据采集、数据质量检查、候选股计算、回测基础设施、GitHub Actions 与 GitHub Pages。

## 一、两个仓库的分工

| 仓库 | 定位 | 主要内容 | 是否生产运行 |
|---|---|---|---|
| **Aquant-Private** | 私有策略研究母版 | 策略研究、因子实验、参数、未公开逻辑 | 否 |
| **Aquant-Public** | 公开生产运行面 | 数据库、数据管线、发布版策略、候选股、回测基础设施、Pages | 是 |

核心原则：

> **Private 负责研究和策略母版，Public 负责公开的发布版策略与全部云端运行。**

Public 不直接依赖 Private 仓库运行。策略经过研究、审核后，以**发布版快照**进入 Public；之后 GitHub Actions 只从 Public 自身代码和数据库计算候选股。

## 二、当前候选股是怎么产生的

当前候选股由：

`scripts/update_candidates.py`

直接读取：

`data/history/YYYY-MM-DD.csv.gz`

计算最近 60 个交易日的因子并生成：

`data/candidates.json`

当前包含：

- 动量
- 流动性
- 估值
- 风险
- 活跃度

并执行非 ST、非退市相关、非停牌、价格和成交额等基础过滤。

**重要：当前版本不是运行时跨仓库调用 Aquant-Private。**

Private 是策略研究母版；Public 中保存的是经过发布边界处理后的运行代码。后续如果 Private 策略升级，必须经过发布流程更新 Public，不能让生产任务隐式读取 Private。

## 三、历史数据库

数据库按**交易日逐日保存**，采用“近到远”的方式回补。

每个交易日：

`交易日 → 全市场 Bulk 日线 → 策略宇宙过滤 → 估值快照 → 数据质量检查 → 日文件 → checkpoint`

历史回补从最近交易日向前推进，目标覆盖五年，并支持断点续传。

### 每日历史文件

`data/history/YYYY-MM-DD.csv.gz`

一只股票一行，一个交易日一个文件。

主要字段：

- 身份：`symbol`、`date`
- 行情：`open`、`high`、`low`、`close`、`pre_close`、`change`、`pct_chg`、`volume`、`amount`
- 市场状态：`factor`、`high_limit`、`low_limit`、`turnover_pct`、`amplitude_pct`、`is_paused`、`is_st`
- 市值：`capitalization`、`circulating_cap`、`market_cap`、`circulating_market_cap`
- 估值：`turnover_ratio`、`pe_ratio`、`pe_ratio_lyr`、`pb_ratio`、`ps_ratio`、`pcf_ratio`

数据源没有提供的估值字段保留为空，不人为填补。

## 四、季度基本面

位置：

`data/fundamentals/{indicator,income,balance,cash_flow}/YYYYqN.csv.gz`

保留：

- `report_date`
- `pub_date`

用于 point-in-time join，避免回测中的未来函数。

## 五、数据源

当前生产主数据源为 **zzshare**，依赖版本锁定在：

`zzshare>=0.4.11,<0.5`

数据接口采用 provider 隔离设计，后续可以加入备用数据源进行故障切换和交叉校验。

## 六、断点续传

状态文件：

`data/history/_BACKFILL_STATE.json`

完成标记：

`data/history/_BACKFILL_COMPLETE`

每个交易日先完成文件写入和质量检查，再更新 checkpoint。

GitHub Actions 使用小批次 checkpoint，避免长时间任务因为单次提交失败而丢失进度。

## 七、运行环境

- **GitHub Actions**：唯一生产运行环境
- **GitHub Pages**：公开静态展示
- **本地电脑**：只用于编辑、提交和控制
- **Aquant-Private**：不参与生产构建和生产运行

## 八、数据质量要求

正式历史库要求：

- 全沪深京市场覆盖
- 交易日与文件日期一致
- `symbol + date` 不重复
- 收盘价不为空
- 每日文件保持完整字段结构
- 估值与交易日正确对应
- 财务数据保留真实披露日期
- 策略运行不使用未来数据

## 九、项目结构

```text
data/
  history/
    YYYY-MM-DD.csv.gz
    _BACKFILL_STATE.json
    _BACKFILL_COMPLETE
  fundamentals/
    indicator/
    income/
    balance/
    cash_flow/
  candidates.json

scripts/
  backfill_history.py
  update_candidates.py

app/
  # GitHub Pages 静态站点
```

## 十、GitHub Pages

公开网站：

https://13415837267.github.io/Aquant-Public/

Pages 使用 Next.js 静态导出。

## 十一、维护规则

- 数据提交使用 `data:` 前缀
- 策略发布使用明确的版本号
- 代码和工作流变更同步记录到 `CHANGELOG.md`
- Public 不保存私有策略研究、账户凭证或 API 密钥
- 生产任务不得隐式依赖 Private 仓库

## 十二、系统目标

最终形成：

`数据采集 → 历史数据库 → 因子计算 → 候选股 → 回测 → 风控 → 组合 → 研究展示`

其中：

> **Aquant-Private 是研究与策略母版；Aquant-Public 是数据、发布版策略和生产运行平台。**
