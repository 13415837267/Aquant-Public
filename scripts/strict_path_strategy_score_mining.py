"""严格交易胜率下的当前策略评分阈值研究。

真实胜定义：
1. T日产生信号；
2. T+1开盘买入；
3. 10bp双边成本后净利润达到至少+1%；
4. 未来5个交易日内，+1%必须先于-3%止损触发；
5. 同一日同时触及止盈和止损时按止损优先；
6. T+1涨停无法成交的样本剔除。

研究仅用于研究候选，不直接改变正式生产状态。
训练区间使用2015-2022；阈值只在2023-2024验证集选择；2025-2026只做最终留出检验。
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_path_aware_model import path_targets

OUT_DIR = ROOT / "data" / "backtest"
TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"
MAX_FORWARD_SESSIONS = 5
NET_WIN_THRESHOLD_PCT = 1.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MIN_VALIDATION_SAMPLES = 200
MIN_FINAL_SAMPLES = 100
MIN_DAILY_TOPK_DAYS = 100
TOP_K = (1, 2, 3, 5, 10)

# score is 0-100 in the current private strategy.
SCORE_THRESHOLDS = tuple(float(x) for x in np.arange(70.0, 99.5, 0.5))


def load_strategy():
    root = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not root:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return model, str(version.STRATEGY_VERSION), commit


def wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def collect(files, start, end, strategy_model):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    state = FeatureState()
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    days = []
    samples = 0

    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end or frame.empty:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files):
            continue

        scored = strategy_model.score_universe(frame)
        if scored.empty:
            continue

        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        symbols = scored["symbol"].astype(str).str.zfill(6).tolist()
        y, best, close5, complete = path_targets(symbols, futures)
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue

        probs = pd.to_numeric(
            scored["precision_probability"], errors="coerce"
        ).to_numpy(dtype=float)[keep] if "precision_probability" in scored.columns else (
            pd.to_numeric(scored["score"], errors="coerce").to_numpy(dtype=float)[keep] / 100.0
        )
        scores = pd.to_numeric(scored["score"], errors="coerce").to_numpy(dtype=float)[keep]

        days.append((date, scores, probs, y[keep].astype(np.uint8), best[keep], close5[keep]))
        samples += len(keep)

        if len(days) % 50 == 0:
            print(f"[严格策略研究] {start}-{end} 已处理{len(days)}日，样本{samples}", flush=True)

    return days, len(days), samples


def threshold_rows(days):
    rows = []
    for threshold in SCORE_THRESHOLDS:
        ss, yy, bb, cc = [], [], [], []
        day_count = 0
        for date, scores, probs, y, best, close5 in days:
            mask = scores >= threshold
            if not mask.any():
                continue
            day_count += 1
            ss.append(scores[mask])
            yy.append(y[mask])
            bb.append(best[mask])
            cc.append(close5[mask])
        if not yy:
            rows.append({
                "score_threshold": threshold,
                "samples": 0,
                "days": 0,
                "win_rate_pct": None,
                "wilson_lower_pct": None,
                "mean_best_return_pct": None,
                "mean_5d_close_return_pct": None,
            })
            continue
        y = np.concatenate(yy)
        b = np.concatenate(bb)
        c = np.concatenate(cc)
        n = int(len(y))
        w = int(y.sum())
        rows.append({
            "score_threshold": threshold,
            "samples": n,
            "days": day_count,
            "win_rate_pct": float(w / n * 100.0),
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0),
            "mean_best_return_pct": float(b.mean()),
            "mean_5d_close_return_pct": float(c.mean()),
        })
    return rows


def topk_rows(days):
    out = []
    for k in TOP_K:
        ys, bs, cs = [], [], []
        day_count = 0
        for date, scores, probs, y, best, close5 in days:
            order = np.argsort(-scores, kind="stable")
            take = order[: min(k, len(order))]
            if len(take) == 0:
                continue
            day_count += 1
            ys.append(y[take])
            bs.append(best[take])
            cs.append(close5[take])
        if not ys:
            out.append({
                "top_k_per_day": k, "days": 0, "samples": 0,
                "days_meet_minimum": False, "win_rate_pct": None,
                "wilson_lower_pct": None, "mean_best_return_pct": None,
                "mean_5d_close_return_pct": None,
            })
            continue
        y = np.concatenate(ys)
        b = np.concatenate(bs)
        c = np.concatenate(cs)
        n = len(y)
        w = int(y.sum())
        out.append({
            "top_k_per_day": k,
            "days": day_count,
            "samples": int(n),
            "days_meet_minimum": bool(day_count >= MIN_DAILY_TOPK_DAYS),
            "win_rate_pct": float(w / n * 100.0),
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0),
            "mean_best_return_pct": float(b.mean()),
            "mean_5d_close_return_pct": float(c.mean()),
        })
    return out


def select_highest(validation_rows):
    eligible = [
        r for r in validation_rows
        if r["samples"] >= MIN_VALIDATION_SAMPLES and r["days"] >= MIN_DAILY_TOPK_DAYS
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda r: (
            r["win_rate_pct"],
            r["wilson_lower_pct"],
            r["samples"],
        ),
    )


def select_preferred(validation_rows):
    eligible = [
        r for r in validation_rows
        if r["samples"] >= max(MIN_VALIDATION_SAMPLES, 1000)
        and r["days"] >= MIN_DAILY_TOPK_DAYS
    ]
    if not eligible:
        return None
    # Prefer the highest precision among sufficiently large validation samples.
    return max(
        eligible,
        key=lambda r: (
            r["win_rate_pct"],
            r["wilson_lower_pct"],
            r["samples"],
        ),
    )


def yearly_rows(days, threshold):
    buckets = {}
    for date, scores, probs, y, best, close5 in days:
        year = date[:4]
        m = scores >= threshold
        if not m.any():
            continue
        item = buckets.setdefault(year, [[], [], []])
        item[0].append(y[m])
        item[1].append(best[m])
        item[2].append(close5[m])
    out = []
    for year in sorted(buckets):
        y = np.concatenate(buckets[year][0])
        b = np.concatenate(buckets[year][1])
        c = np.concatenate(buckets[year][2])
        n = len(y)
        w = int(y.sum())
        out.append({
            "year": year,
            "samples": int(n),
            "win_rate_pct": float(w / n * 100.0) if n else None,
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0) if n else None,
            "mean_best_return_pct": float(b.mean()) if n else None,
            "mean_5d_close_return_pct": float(c.mean()) if n else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "strict_path_strategy_score_mining_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("研究区间必须落在历史数据范围内")

    model, version, commit = load_strategy()

    # The current private strategy is already fixed before validation/final.
    # We use 2015-2022 only as historical diagnostic, never to tune the final holdout.
    train_days, train_processed, train_samples = collect(
        files, args.start, TRAIN_END, model
    )
    validation_days, validation_processed, validation_samples = collect(
        files, VALIDATION_START, VALIDATION_END, model
    )
    final_days, final_processed, final_samples = collect(
        files, FINAL_START, args.final_end, model
    )

    train_thresholds = threshold_rows(train_days)
    validation_thresholds = threshold_rows(validation_days)
    final_thresholds = threshold_rows(final_days)
    validation_topk = topk_rows(validation_days)
    final_topk = topk_rows(final_days)

    highest = select_highest(validation_thresholds)
    preferred = select_preferred(validation_thresholds)

    highest_final = None
    preferred_final = None
    if highest:
        highest_final = next(
            r for r in final_thresholds
            if r["score_threshold"] == highest["score_threshold"]
        )
    if preferred:
        preferred_final = next(
            r for r in final_thresholds
            if r["score_threshold"] == preferred["score_threshold"]
        )

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "strict_path_strategy_score_threshold_mining",
        "objective": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": version,
        "strategy_commit": commit,
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "parameters": {
            "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
            "minimum_validation_samples": MIN_VALIDATION_SAMPLES,
            "minimum_final_samples": MIN_FINAL_SAMPLES,
            "minimum_daily_days": MIN_DAILY_TOPK_DAYS,
            "score_thresholds": list(SCORE_THRESHOLDS),
            "top_k_per_day": list(TOP_K),
        },
        "train": {
            "processed_days": train_processed,
            "samples": train_samples,
            "thresholds": train_thresholds,
            "daily_topk": topk_rows(train_days),
        },
        "validation": {
            "processed_days": validation_processed,
            "samples": validation_samples,
            "thresholds": validation_thresholds,
            "daily_topk": validation_topk,
            "highest_observed": highest,
            "preferred": preferred,
        },
        "final": {
            "processed_days": final_processed,
            "samples": final_samples,
            "thresholds": final_thresholds,
            "daily_topk": final_topk,
            "highest_observed_at_validation_threshold": highest_final,
            "preferred_at_validation_threshold": preferred_final,
            "highest_observed_yearly": (
                yearly_rows(final_days, highest["score_threshold"]) if highest else []
            ),
            "preferred_yearly": (
                yearly_rows(final_days, preferred["score_threshold"]) if preferred else []
            ),
        },
        "decision_policy": {
            "target_win_rate_pct": 80.0,
            "below_target_use_highest_stable_research_rate": True,
            "final_holdout_is_evaluation_only": True,
            "production_release": False,
        },
        "audit": {
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "target_is_net_profit_at_least_1pct": True,
            "target_gross_equivalent_pct": 1.1,
            "stop_loss_gross_pct": STOP_LOSS_PCT,
            "same_day_stop_first": True,
            "threshold_selection_only_on_validation": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "strategy_version": version,
        "strategy_commit": commit,
        "validation_highest": highest,
        "validation_preferred": preferred,
        "final_at_validation_threshold": preferred_final,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
