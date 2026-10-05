"""使用当前近期训练窗口模型重新生成研究候选池。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.high_precision_profit_mining import NonlinearModel
from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_short_term_model import FEATURES, build_targets, make_features

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = "2022-12-30"
MODEL_CODE_COMMIT = "3df9ef34ee23a86d6f844ceb9e96bf4b4f387592"
MODEL_VERSION = "近期训练窗口研究版"
MAX_FORWARD_SESSIONS = 5
TOP_K = 2


def train_model(files, start):
    dates = [p.name[:10] for p in files]
    start_i = dates.index(start)
    end_i = dates.index(TRAIN_END)
    model = NonlinearModel(len(FEATURES))
    state = FeatureState()
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        return cache[i]

    days = samples = 0
    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > TRAIN_END or frame.empty:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x = make_features(frame)
        labels, _, _, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        model.update(x[keep], labels[keep].astype(np.float64))
        days += 1
        samples += len(keep)
        if days % 50 == 0:
            print(f"[候选基准训练] 已处理{days}日，样本{samples}", flush=True)
    if samples < 5000:
        raise RuntimeError(f"训练样本不足: {samples}")
    print(f"[候选基准训练] 完成：{days}日，样本{samples}", flush=True)
    return model


def build_latest_frame(files):
    state = FeatureState()
    frame = None
    for path in files:
        frame = state.build(read_daily(path))
    if frame is None or frame.empty:
        raise RuntimeError("最新交易日没有可评分股票")
    return frame, files[-1].name[:10]


def payload_row(row, rank):
    flags = []
    if float(row["volatility_10d_pct"]) > 8:
        flags.append("高波动")
    if float(row["change_pct"]) < -7:
        flags.append("当日跌幅<-7%")
    if float(row["return_5d_pct"]) > 15:
        flags.append("5日过热")
    return {
        "rank": rank,
        "symbol": str(row["symbol"]).zfill(6),
        "name": str(row["name"]),
        "price": round(float(row["close"]), 3),
        "change_pct": round(float(row["change_pct"]), 3),
        "return_3d_pct": round(float(row["return_3d_pct"]), 3),
        "return_5d_pct": round(float(row["return_5d_pct"]), 3),
        "return_10d_pct": round(float(row["return_10d_pct"]), 3),
        "volume_ratio_5d": round(float(row["volume_ratio_5d"]), 3),
        "turnover_pct": round(float(row["turnover_pct"]), 3),
        "amount": round(float(row["amount"]), 2),
        "volatility_10d_pct": round(float(row["volatility_10d_pct"]), 3),
        "close_strength": round(float(row["close_strength"]), 3),
        "overnight_1d_pct": round(float(row["overnight_1d_pct"]), 3),
        "overnight_3d_pct": round(float(row["overnight_3d_pct"]), 3),
        "overnight_5d_pct": round(float(row["overnight_5d_pct"]), 3),
        "intraday_return_pct": round(float(row["intraday_return_pct"]), 3),
        "limit_up_5d_count": round(float(row["limit_up_5d_count"]), 3),
        "score": round(float(row["score"]), 3),
        "precision_probability": round(float(row["precision_probability"]), 6),
        "admission_tier": "model_threshold_060_top2",
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-02")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output", default="data/candidates.json")
    args = parser.parse_args()

    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or TRAIN_END not in dates:
        raise ValueError("训练区间不在历史数据范围内")
    if not 0 < args.threshold < 1:
        raise ValueError("概率阈值必须在0和1之间")

    model = train_model(files, args.start)
    frame, signal_date = build_latest_frame(files)
    probabilities = model.predict(make_features(frame))
    scored = frame.copy()
    scored["precision_probability"] = probabilities
    scored["score"] = probabilities * 100.0
    scored = scored.sort_values(
        ["precision_probability", "symbol"], ascending=[False, True]
    ).reset_index(drop=True)

    eligible = scored.loc[scored["precision_probability"] >= args.threshold]
    selected = eligible.head(TOP_K)
    candidates = [payload_row(row, i + 1) for i, (_, row) in enumerate(selected.iterrows())]
    observation = [payload_row(row, i + 1) for i, (_, row) in enumerate(eligible.head(10).iterrows())]

    breadth = float(frame["market_breadth_pct"].iloc[0])
    median = float(frame["market_median_return_pct"].iloc[0])
    regime = "risk_on" if breadth >= 55 and median > 0.3 else (
        "neutral" if breadth >= 35 and median >= -0.3 else "risk_off"
    )

    result = {
        "as_of": f"{signal_date}T18:00:00+08:00",
        "timezone": "Asia/Shanghai",
        "source": "Aquant-Public data/history",
        "status": "ready",
        "strategy_source": "04历史研究流水线/近期训练窗口模型",
        "strategy_version": MODEL_VERSION,
        "strategy_commit": MODEL_CODE_COMMIT,
        "model_definition": {
            "type": "numpy_two_layer_mlp",
            "training_start": args.start,
            "training_end": TRAIN_END,
            "data_cutoff": "2026-09-30",
            "probability_threshold": args.threshold,
            "top_k": TOP_K,
            "objective": "单笔净利润达到+1%才计为胜，最长5个交易日",
        },
        "market_scope": "沪深主板：000001-004999.SZ（排除001001-001199 CDR）+ 600/601/603/605.SH",
        "universe": "训练模型可评分范围；排除 ST/退市相关标的、停牌、价格≤2元、最近交易日成交额<3000万元",
        "lookback_trading_days": 20,
        "signal_horizon": "T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日",
        "holding_window_sessions": [2, 5],
        "risk_controls": {
            "net_win_threshold_pct": 1.0,
            "stop_loss_pct": 3.0,
            "max_positions": TOP_K,
            "sell_start_session": 2,
        },
        "market": {
            "breadth_pct": round(breadth, 3),
            "median_return_pct": round(median, 3),
            "regime": regime,
        },
        "diagnostics": {
            "scorable_rows": int(len(frame)),
            "threshold_eligible_rows": int(len(eligible)),
            "candidate_count": len(candidates),
            "candidate_probability_min": min((r["precision_probability"] for r in candidates), default=None),
            "candidate_probability_max": max((r["precision_probability"] for r in candidates), default=None),
            "risk_off_no_trade": regime == "risk_off",
        },
        "candidates": candidates,
        "research_observation_candidates": observation,
        "factor_weights": None,
        "future_function": False,
        "candidate_admission_policy": "top_2_by_recent_window_model_probability_threshold_0.60",
        "audit": {
            "hard_eligibility_applied_before_model_scoring": True,
            "model_source_locked_to_recent_window_research": True,
            "short_term_features_only": True,
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "strict_profit_label": True,
            "final_holdout_not_used_for_candidate_selection": True,
        },
        "history_files_used": len(files),
        "history_window_start": dates[0],
        "history_window_end": signal_date,
    }

    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "ready",
        "signal_date": signal_date,
        "candidate_count": len(candidates),
        "candidates": [
            {"symbol": r["symbol"], "name": r["name"], "probability": r["precision_probability"]}
            for r in candidates
        ],
        "strategy_version": MODEL_VERSION,
        "strategy_commit": MODEL_CODE_COMMIT,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
