# Aquant-Public

个人量化交易系统的公开数据与运行仓库，负责全沪深京 A 股历史数据库、每日数据更新、质量检查和 GitHub Pages。

## 数据库

数据库按**交易日逐日保存**，不是按个股逐只抓取。

每个交易日执行：

`交易日 → 全市场 Bulk 日线 → 当前非 ST/退市策略宇宙过滤 → 全市场估值快照 → 数据校验 → YYYY-MM-DD.csv.gz → checkpoint`

历史回补采用**近到远**顺序，从最近交易日向前推进至五年前，并支持断点续传。

### 每日历史文件

位置：

`data/history/YYYY-MM-DD.csv.gz`

每个交易日文件以股票为行，包含：

| 类别 | 字段 |
|---|---|
| 身份 | symbol, date |
| 行情 | open, high, low, close, pre_close, volume, amount, pct_chg, change |
| 行情衍生 | factor, high_limit, low_limit, turnover_pct, amplitude_pct, is_paused, is_st |
| 估值 | capitalization, circulating_cap, market_cap, circulating_market_cap |
| 估值倍数 | turnover_ratio, pe_ratio, pe_ratio_lyr, pb_ratio, ps_ratio, pcf_ratio |

估值指标来自 zzshare 的日频估值表。对于数据源本身没有值的股票，数据库保留 NaN，不做人为填补。

### 季度基本面

位置：

`data/fundamentals/{indicator,income,balance,cash_flow}/YYYYqN.csv.gz`

保存报告期和实际披露日期（`report_date` / `pub_date`），用于回测时做 point-in-time join，避免未来函数。

数据源提供日频估值以及财务指标、利润表、资产负债表、现金流量表等接口；同时提供 PIT 查询以支持严格的时间点回测。 

## 数据源

当前生产主源为 **zzshare**，版本锁定在 `>=0.4.11,<0.5`。该项目公开提供 A 股行情与基本面接口。 

本系统仍保留 provider 隔离设计，后续可以加入备用源进行故障切换和交叉校验。

## Universe

当前策略宇宙由股票基础资料生成，过滤名称匹配 `ST|退` 的标的；当前约 5,370 只活跃非 ST 股票。

历史行情请求仍从全市场 Bulk 获取，再按策略宇宙过滤后写入数据库。

## 断点续传

状态文件：

`data/history/_BACKFILL_STATE.json`

完成标记：

`data/history/_BACKFILL_COMPLETE`

每个交易日先完成本地文件与质量检查，再更新 checkpoint。Git 远端采用小批次 checkpoint，减少长任务中的远端 push 次数；中断后从最近已确认的日期继续。

## 运行环境

- GitHub Actions：唯一生产运行面
- GitHub Pages：静态研究/数据展示
- 本地电脑：仅编辑、提交、控制，不作为生产采集节点
- Private 仓库：保存策略研究母版，不参与生产构建

## 数据质量目标

正式历史库要求：

- 全沪深京市场覆盖
- 交易日与文件日期一致
- `symbol + date` 不重复
- 收盘价不为空
- 每日文件保留完整字段结构
- 估值数据与交易日一一对应
- 财务数据保留真实披露日期，避免 look-ahead bias

## 项目结构

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

scripts/
  backfill_history.py

Aquant-Private/
  strategy research only
```

## GitHub Pages

网站：

https://13415837267.github.io/Aquant-Public/

Pages 使用 Next.js 静态导出。

## 维护规则

代码与工作流变更同步记录到 `CHANGELOG.md`。数据刷新使用 `data:` 提交前缀。
