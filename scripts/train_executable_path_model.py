"""Executable path-aware research for the +3% short-term trading opportunity.

Research only. A positive label means the T+1 entry price can reach at least
+3% net profit from T+2 onward before a -3% stop, so every positive label
represents a sellable outcome under the A-share T+1 rule.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, FEATURE_WARMUP_SESSIONS, history_files, read_daily
from scripts.selection_factor_catalog import active_market_factors, active_stock_factors, load_research_config, transform_market_feature

OUT_DIR = ROOT / "data" / "backtest"
RESEARCH_CONFIG = load_research_config()
STOCK_RANK_FEATURES = active_stock_factors(RESEARCH_CONFIG)
MARKET_FEATURES = active_market_factors(RESEARCH_CONFIG)
FEATURES = STOCK_RANK_FEATURES + MARKET_FEATURES
REGIME_NAMES = ("risk_off", "neutral", "risk_on")
REGIME_BREADTH_CUTOFFS = (0.35, 0.65)
FEATURE_NAMES = list(FEATURES)

TRAIN_END = "2024-12-31"
VALIDATION_START = "2025-01-02"
VALIDATION_END = "2025-12-31"
FINAL_START = "2026-01-05"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = float(RESEARCH_CONFIG["高收益目标净收益百分比"])
STOP_LOSS_PCT = float(RESEARCH_CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(RESEARCH_CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(RESEARCH_CONFIG["最大前瞻交易日数"])
MIN_SELECTION_SAMPLES = 5000

LEARNING_RATE = 0.04
L2 = 0.02
EPOCHS_PER_DAY = 2
THRESHOLDS = [float(x) for x in RESEARCH_CONFIG["高收益模型概率阈值网格"]]
POSITIVE_CLASS_WEIGHT = 4.0
TOP_K_VALUES = (2, 5, 10)
ABSTAIN_TOP_PROBABILITY_VALUES = (0.55, 0.60, 0.65)
# 单变量实验：按T+2起可兑现的最大净盈利空间增加正样本训练权重。
UPSIDE_WEIGHT_SLOPE = 0.50
UPSIDE_WEIGHT_MAX_MULTIPLIER = 2.00
MAX_ABSTAIN_SHARE_PCT = 15.0


def percentile_rank(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_features(frame):
    if frame.empty:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    cols = [percentile_rank(frame[name]) for name in STOCK_RANK_FEATURES]
    cols.extend(transform_market_feature(frame, name) for name in MARKET_FEATURES)
    return np.column_stack(cols)


def path_targets(symbols, future_days):
    """生成 +3% 机会标签，并把 3% 止盈与 3% 止损纳入路径判断。"""
    n = len(symbols)
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)

    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    entry = pd.to_numeric(future_days[0].set_index("symbol")["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)
    target_gross_pct = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    target = entry * (1.0 + target_gross_pct / 100.0)
    stop = entry * (1.0 - STOP_LOSS_PCT / 100.0)

    opens, highs, lows, closes = [], [], [], []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        idx = day.set_index("symbol")
        opens.append(pd.to_numeric(idx["open"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        highs.append(pd.to_numeric(idx["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(idx["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(idx["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))
    opens, highs, lows, closes = map(np.column_stack, (opens, highs, lows, closes))
    complete = (
        np.isfinite(entry) & (entry > 0) &
        np.isfinite(opens).all(axis=1) &
        np.isfinite(highs).all(axis=1) &
        np.isfinite(lows).all(axis=1) &
        np.isfinite(closes).all(axis=1)
    )

    target_hit = np.zeros(n, dtype=bool)
    stopped_before_target = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)

    for row in np.flatnonzero(complete):
        best_net[row] = np.max(highs[row, 1:]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        for d in range(1, MAX_FORWARD_SESSIONS):
            # T+1 仅允许建仓，不能卖出；从 T+2 起才允许触发止盈或止损。
            # 同一日同时触及目标和止损时按止损优先。
            if lows[row, d] <= stop[row]:
                stopped_before_target[row] = True
                break
            if highs[row, d] >= target[row]:
                target_hit[row] = True
                break

    # +3%目标只在T+2至T+5可退出的时段判断；若先触发止损则标记为负样本。
    labels = target_hit & ~stopped_before_target
    return labels, best_net, close_net, complete



class LogisticModel:
    def __init__(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

    def update(self, x, y, upside_multiplier=None):
        if len(y) == 0:
            return
        z = np.clip(x @ self.w + self.b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        if upside_multiplier is None:
            upside_multiplier = np.ones(len(y), dtype=np.float64)
        sample_weight = np.where(
            y > 0.5,
            POSITIVE_CLASS_WEIGHT * upside_multiplier,
            1.0,
        )
        err = (p - y) * sample_weight
        weight_mean = float(sample_weight.mean())
        self.w -= LEARNING_RATE * ((x.T @ err) / len(y) / weight_mean + L2 * self.w)
        self.b -= LEARNING_RATE * float(err.mean() / weight_mean)

    def predict(self, x):
        z = np.clip(x @ self.w + self.b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


def collect(files, start, end, state, model=None, train=False):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        return cache[i]

    metrics = []
    samples = 0
    processed = 0

    for i in range(max(0, start_i - FEATURE_WARMUP_SESSIONS), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue

        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_all = make_features(frame)
        y, best, close5, complete = path_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue

        x = x_all[keep]
        yy = y[keep].astype(np.float64)
        bb = best[keep]
        cc = close5[keep]
        samples += len(yy)
        processed += 1

        breadth = np.clip(frame["market_breadth_pct"].to_numpy(dtype=np.float64)[keep] / 100.0, 0.0, 1.0)
        regime_index = np.select(
            [breadth < REGIME_BREADTH_CUTOFFS[0], breadth < REGIME_BREADTH_CUTOFFS[1]],
            [0, 1],
            default=2,
        ).astype(np.int64)
        if train:
            for regime_idx, regime in enumerate(REGIME_NAMES):
                mask = regime_index == regime_idx
                if not mask.any():
                    continue
                for _ in range(EPOCHS_PER_DAY):
                    upside = np.nan_to_num(bb[mask], nan=NET_WIN_THRESHOLD_PCT)
                    upside_ratio = np.clip(
                        (upside - NET_WIN_THRESHOLD_PCT) / NET_WIN_THRESHOLD_PCT,
                        0.0,
                        UPSIDE_WEIGHT_MAX_MULTIPLIER,
                    )
                    upside_multiplier = 1.0 + UPSIDE_WEIGHT_SLOPE * upside_ratio
                    model[regime].update(x[mask], yy[mask], upside_multiplier)
        else:
            probs = np.zeros(len(yy), dtype=np.float64)
            for regime_idx, regime in enumerate(REGIME_NAMES):
                mask = regime_index == regime_idx
                if mask.any():
                    probs[mask] = model[regime].predict(x[mask])
            metrics.append((date, probs, yy, bb, cc))

        if processed % 50 == 0:
            print(f"[路径模型] {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    return metrics, processed, samples


def summarize(metrics):
    if not metrics:
        return {"samples": 0, "thresholds": [], "yearly": []}

    probs = np.concatenate([m[1] for m in metrics])
    y = np.concatenate([m[2] for m in metrics])
    best = np.concatenate([m[3] for m in metrics])
    close5 = np.concatenate([m[4] for m in metrics])

    thresholds = []
    for t in THRESHOLDS:
        mask = probs >= t
        n = int(mask.sum())
        thresholds.append({
            "probability_threshold": t,
            "samples": n,
            "sample_share_pct": float(n / len(y) * 100.0),
            "path_win_3pct_rate_pct": float(y[mask].mean() * 100.0) if n else None,
            "mean_best_return_pct": float(best[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(close5[mask].mean()) if n else None,
        })

    top_k = []
    for k in TOP_K_VALUES:
        selected_y, selected_best, selected_close, selected_probs = [], [], [], []
        for _, p, yy, bb, cc in metrics:
            order = np.argsort(-p)[:min(k, len(p))]
            selected_probs.append(p[order])
            selected_y.append(yy[order])
            selected_best.append(bb[order])
            selected_close.append(cc[order])
        if selected_y:
            pp = np.concatenate(selected_probs)
            yy = np.concatenate(selected_y)
            bb = np.concatenate(selected_best)
            cc = np.concatenate(selected_close)
            top_k.append({
                "top_k": k,
                "samples": int(len(yy)),
                "path_win_3pct_rate_pct": float(yy.mean() * 100.0),
                "mean_model_probability_pct": float(pp.mean() * 100.0),
                "mean_best_return_pct": float(bb.mean()),
                "mean_5d_close_return_pct": float(cc.mean()),
            })

    abstain_days = []
    for threshold in ABSTAIN_TOP_PROBABILITY_VALUES:
        selected_y, selected_best, selected_close = [], [], []
        eligible_days = 0
        skipped_days = 0
        for _, p, yy, bb, cc in metrics:
            eligible_days += 1
            if len(p) == 0 or float(np.max(p)) < threshold:
                skipped_days += 1
                continue
            order = np.argsort(-p)[:min(TOP_K_VALUES[0], len(p))]
            selected_y.append(yy[order])
            selected_best.append(bb[order])
            selected_close.append(cc[order])
        if selected_y:
            yy = np.concatenate(selected_y)
            bb = np.concatenate(selected_best)
            cc = np.concatenate(selected_close)
            share = skipped_days / eligible_days * 100.0 if eligible_days else 0.0
            abstain_days.append({
                "top_probability_floor": threshold,
                "eligible_days": int(eligible_days),
                "abstained_days": int(skipped_days),
                "abstain_share_pct": float(share),
                "selected_samples": int(len(yy)),
                "path_win_3pct_rate_pct": float(yy.mean() * 100.0),
                "mean_best_return_pct": float(bb.mean()),
                "mean_5d_close_return_pct": float(cc.mean()),
            })

    yearly_map = {}
    for date, p, yy, bb, cc in metrics:
        year = date[:4]
        item = yearly_map.setdefault(year, [[], [], [], []])
        item[0].append(p)
        item[1].append(yy)
        item[2].append(bb)
        item[3].append(cc)

    yearly = []
    for year in sorted(yearly_map):
        pp = np.concatenate(yearly_map[year][0])
        yy = np.concatenate(yearly_map[year][1])
        bb = np.concatenate(yearly_map[year][2])
        cc = np.concatenate(yearly_map[year][3])
        row = {
            "year": year,
            "samples": int(len(yy)),
            "base_path_win_3pct_rate_pct": float(yy.mean() * 100.0),
        }
        for t in (0.80, 0.85):
            mask = pp >= t
            row[f"threshold_{str(t).replace('.', '_')}_samples"] = int(mask.sum())
            row[f"threshold_{str(t).replace('.', '_')}_path_win_3pct_rate_pct"] = (
                float(yy[mask].mean() * 100.0) if mask.any() else None
            )
        yearly.append(row)

    return {
        "samples": int(len(y)),
        "base_path_win_3pct_rate_pct": float(y.mean() * 100.0),
        "base_mean_best_return_pct": float(best.mean()),
        "base_mean_5d_close_return_pct": float(close5.mean()),
        "thresholds": thresholds,
        "daily_top_k": top_k,
        "daily_top_k_with_abstention": abstain_days,
        "yearly": yearly,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "executable_path_profit_training_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据文件范围内")

    model = {regime: LogisticModel(len(FEATURE_NAMES)) for regime in REGIME_NAMES}

    train_state = FeatureState()
    _, train_days, train_samples = collect(
        files, args.start, TRAIN_END, train_state, model=model, train=True
    )
    if train_samples < MIN_SELECTION_SAMPLES:
        raise RuntimeError(f"训练样本不足: {train_samples}")

    validation_state = FeatureState()
    validation_metrics, validation_days, validation_samples = collect(
        files, VALIDATION_START, VALIDATION_END, validation_state, model=model, train=False
    )

    final_state = FeatureState()
    final_metrics, final_days, final_samples = collect(
        files, FINAL_START, args.final_end, final_state, model=model, train=False
    )

    validation = summarize(validation_metrics)
    final = summarize(final_metrics)

    selected = None
    for row in validation["thresholds"]:
        if (
            row["samples"] >= MIN_SELECTION_SAMPLES
            and row["path_win_3pct_rate_pct"] is not None
            and row["path_win_3pct_rate_pct"] >= 80.0
        ):
            selected = row
            break
    if selected is None:
        eligible = [r for r in validation["thresholds"] if r["samples"] >= MIN_SELECTION_SAMPLES]
        selected = (
            max(eligible, key=lambda r: (r["path_win_3pct_rate_pct"] or -1, r["samples"]))
            if eligible else None
        )

    abstention_candidates = [row for row in validation["daily_top_k_with_abstention"] if row["abstain_share_pct"] <= MAX_ABSTAIN_SHARE_PCT]
    selected_abstention = (
        max(abstention_candidates, key=lambda r: (r["path_win_3pct_rate_pct"], -r["abstain_share_pct"], r["selected_samples"]))
        if abstention_candidates else None
    )

    result = {
        "schema_version": 3,
        "status": "research_only",
        "method": "full_market_path_aware_short_term_training",
        "objective": "net_profit_at_least_3pct_opportunity_before_3pct_stop_within_5_sessions_with_sellable_upside_quality_weighting",
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "features": FEATURE_NAMES,
        "base_features": FEATURES,
        "market_regime_routing": {
            "type": "breadth_three_state_routed_models",
            "states": list(REGIME_NAMES),
            "cutoffs": list(REGIME_BREADTH_CUTOFFS),
            "rule": "market_breadth_pct < 35% risk_off; 35%-65% neutral; >=65% risk_on"
        },
        "stock_features_are_cross_sectional_percentile_ranked": True,
        "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
        "take_profit_pct": NET_WIN_THRESHOLD_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "model": {
            "type": "logistic_regression_sgd_weighted_regime_routed",
            "positive_class_weight": POSITIVE_CLASS_WEIGHT,
            "upside_weight_slope": UPSIDE_WEIGHT_SLOPE,
            "upside_weight_max_multiplier": UPSIDE_WEIGHT_MAX_MULTIPLIER,
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "epochs_per_day": EPOCHS_PER_DAY,
            "routing": "breadth_three_state_routed_models",
            "regime_models": {
                regime: {
                    "intercept": float(model[regime].b),
                    "coefficients": {name: float(value) for name, value in zip(FEATURE_NAMES, model[regime].w)},
                }
                for regime in REGIME_NAMES
            },
        },
        "train": {
            "processed_days": train_days,
            "samples": train_samples,
        },
        "validation": {
            "processed_days": validation_days,
            **validation,
        },
        "final": {
            "processed_days": final_days,
            **final,
        },
        "selected_validation_operating_point": selected,
        "selected_validation_abstention_point": selected_abstention,
        "audit": {
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "target_is_net_profit_at_least_3pct_opportunity": True,
            "take_profit_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_gross_pct": STOP_LOSS_PCT,
            "t_plus_1_target_counts_as_win": True,
            "stop_loss_applies_from_t_plus_2": True,
            "same_day_stop_first": True,
            "same_day_target_stop_ambiguity": "daily_bar_conservative_stop_first",
            "formal_production_changed": False,
            "market_regime_routing_research_only": True,
            "upside_quality_weighting_uses_future_training_labels_only": True,
            "best_profit_metric_starts_t_plus_2": True,
            "t_plus_1_target_not_counted_as_win": True,
            "sellable_win_starts_t_plus_2": True,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "final_samples": final_samples,
        "selected_validation_operating_point": selected,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
