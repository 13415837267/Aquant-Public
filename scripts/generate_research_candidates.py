"""使用当前近期训练窗口模型重新生成研究候选池。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.production_model import load_model, DEFAULT_MODEL_PATH
from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_short_term_model import make_features

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 2


def load_fixed_model(model_path: Path):
    release = json.loads((ROOT / "config" / "production_release_v1.json").read_text(encoding="utf-8"))
    baseline = release["model_baseline"]
    model, payload = load_model(
        model_path,
        expected_version=baseline["version"],
        expected_commit=baseline["model_code_commit"],
    )
    if payload["training_start"] != baseline["training_start"] or payload["training_end"] != baseline["training_end"]:
        raise RuntimeError("正式模型训练窗口与生产基准不一致")
    if float(baseline["probability_threshold"]) != 0.60 or int(baseline["top_k"]) != 2:
        raise RuntimeError("生产候选参数不是第一版正式基准")
    return model, payload

def build_latest_frame(files, signal_date):
    dates = [p.name[:10] for p in files]
    end_i = dates.index(signal_date)
    start_i = max(0, end_i - 20)
    state = FeatureState()
    frame = None
    for path in files[start_i:end_i + 1]:
        frame = state.build(read_daily(path))
    if frame is None or frame.empty:
        raise RuntimeError("最新交易日没有可评分股票")
    return frame, signal_date


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
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--output", default="data/candidates.json")
    parser.add_argument("--model", default=str(ROOT / DEFAULT_MODEL_PATH))
    parser.add_argument("--signal-date", default=None)
    args = parser.parse_args()

    files = history_files()
    dates = [p.name[:10] for p in files]
    signal_date = args.signal_date or dates[-1]
    if signal_date not in dates:
        raise ValueError("训练区间不在历史数据范围内")
    if not 0 < args.threshold < 1:
        raise ValueError("概率阈值必须在0和1之间")

    model, model_payload = load_fixed_model(Path(args.model))
    frame, signal_date = build_latest_frame(files, signal_date)
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
        "strategy_version": model_payload["strategy_version"],
        "strategy_commit": model_payload["model_code_commit"],
        "model_definition": {
            "type": "numpy_two_layer_mlp",
            "training_start": model_payload["training_start"],
            "training_end": model_payload["training_end"],
            "model_data_cutoff": model_payload["data_cutoff"],
            "signal_data_cutoff": signal_date,
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
            "fixed_model_weights_used": True,
            "daily_retraining": False,
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
        "inference_data_only_to_signal_date": True,
        "model_artifact": str(Path(args.model).as_posix()),
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
        "strategy_version": model_payload["strategy_version"],
        "strategy_commit": MODEL_CODE_COMMIT,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
