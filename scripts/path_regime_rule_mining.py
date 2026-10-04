"""Market-regime conditioned rule mining for the strict 1% path objective.

Research only. Discovery uses 2015-2022, rule selection uses 2023-2024,
and the final holdout is 2025-2026-09-30. No production strategy changes.

Trade label:
T+1 open entry, executable only when not already at the upper limit.
Within the next five sessions, net +1% must be reached before a gross -3%
stop. When both are touched in the same daily bar, stop is conservatively
treated as first.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily

OUT_DIR = ROOT / "data" / "backtest"

STOCK_FEATURES = [
    "return_1d_pct", "return_3d_pct", "return_5d_pct", "return_10d_pct",
    "return_20d_pct", "overnight_1d_pct", "overnight_3d_pct",
    "overnight_5d_pct", "overnight_10d_pct", "volume_ratio_5d",
    "amount_20d", "volatility_10d_pct", "close_strength",
    "intraday_return_pct", "limit_up_5d_count", "turnover_pct", "change_pct",
]
RANK_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
MIN_TRAIN_RULE_SAMPLES = 10_000
MIN_VALIDATION_RULE_SAMPLES = 5_000
MIN_REGIME_SAMPLES = 25_000
TOP_STOCK_ATOMICS_PER_REGIME = 10
TOP_OUTPUT_RULES = 30
TARGET_WIN_RATE_PCT = 80.0

NET_WIN_THRESHOLD_PCT = 1.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
ENTRY_LIMIT_UP_BLOCK = True

# Non-overlapping market regimes. Their boundaries are centralized and are
# intentionally independent of the final stock rule selection.
BREADTH_BINS = (-np.inf, 40.0, 50.0, 60.0, 70.0, np.inf)
MEDIAN_BINS = (-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf)


def percentile_rank(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_stock_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([percentile_rank(frame[name]) for name in STOCK_FEATURES])


def regime_defs() -> list[dict]:
    out = []
    for bi in range(len(BREADTH_BINS) - 1):
        for mi in range(len(MEDIAN_BINS) - 1):
            out.append({
                "regime_index": len(out),
                "breadth_low": None if not np.isfinite(BREADTH_BINS[bi]) else float(BREADTH_BINS[bi]),
                "breadth_high": None if not np.isfinite(BREADTH_BINS[bi + 1]) else float(BREADTH_BINS[bi + 1]),
                "median_low": None if not np.isfinite(MEDIAN_BINS[mi]) else float(MEDIAN_BINS[mi]),
                "median_high": None if not np.isfinite(MEDIAN_BINS[mi + 1]) else float(MEDIAN_BINS[mi + 1]),
            })
    return out


REGIMES = regime_defs()


def regime_masks(frame: pd.DataFrame) -> np.ndarray:
    breadth = pd.to_numeric(frame["market_breadth_pct"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
    median = pd.to_numeric(frame["market_median_return_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    masks = []
    for rule in REGIMES:
        b = np.ones(len(frame), dtype=bool)
        m = np.ones(len(frame), dtype=bool)
        if rule["breadth_low"] is not None:
            b &= breadth >= rule["breadth_low"]
        if rule["breadth_high"] is not None:
            b &= breadth < rule["breadth_high"]
        if rule["median_low"] is not None:
            m &= median >= rule["median_low"]
        if rule["median_high"] is not None:
            m &= median < rule["median_high"]
        masks.append(b & m)
    return np.column_stack(masks).astype(np.uint8)


def stock_rule_defs() -> list[dict]:
    out = []
    for feature_idx, feature in enumerate(STOCK_FEATURES):
        for threshold in RANK_THRESHOLDS:
            out.append({"rule_index": len(out), "feature": feature, "feature_index": feature_idx, "op": ">=", "threshold": threshold})
            out.append({"rule_index": len(out), "feature": feature, "feature_index": feature_idx, "op": "<=", "threshold": threshold})
    return out


RULES = stock_rule_defs()


def rule_label(rule: dict) -> str:
    return f'{rule["feature"]}{rule["op"]}{rule["threshold"]:g}'


def wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def path_targets(symbols: list[str], future_days: list[pd.DataFrame]):
    n = len(symbols)
    empty = (
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return empty

    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    first = future_days[0].set_index("symbol")
    entry = pd.to_numeric(first["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)
    if ENTRY_LIMIT_UP_BLOCK and "high_limit" in first.columns:
        high_limit = pd.to_numeric(first["high_limit"], errors="coerce").reindex(keys).to_numpy(dtype=float)
        executable = ~np.isfinite(high_limit) | (entry < high_limit * (1.0 - 1e-6))
    else:
        executable = np.ones(n, dtype=bool)

    target_gross = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    target = entry * (1.0 + target_gross / 100.0)
    stop = entry * (1.0 - STOP_LOSS_PCT / 100.0)

    opens, highs, lows, closes = [], [], [], []
    for day in future_days:
        idx = day.set_index("symbol")
        opens.append(pd.to_numeric(idx["open"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        highs.append(pd.to_numeric(idx["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(idx["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(idx["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))

    opens = np.column_stack(opens)
    highs = np.column_stack(highs)
    lows = np.column_stack(lows)
    closes = np.column_stack(closes)
    complete = (
        np.isfinite(entry) & (entry > 0) & executable
        & np.isfinite(opens).all(axis=1)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
        & np.isfinite(closes).all(axis=1)
    )

    win = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    target_day = np.full(n, np.nan)
    stop_day = np.full(n, np.nan)

    valid_rows = np.flatnonzero(complete)
    for row in valid_rows:
        best_net[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        mae[row] = np.min(lows[row]) / entry[row] * 100.0 - 100.0
        mfe[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0
        for d in range(MAX_FORWARD_SESSIONS):
            day_open, day_high, day_low = opens[row, d], highs[row, d], lows[row, d]
            if day_open <= stop[row] or day_low <= stop[row]:
                stop_day[row] = d + 1
                break
            if day_high >= target[row]:
                target_day[row] = d + 1
                win[row] = True
                break
    return win, best_net, close_net, complete, mae, mfe, target_day, stop_day


def accumulate(acc: dict, mask: np.ndarray, y: np.ndarray, best: np.ndarray, close: np.ndarray, mae: np.ndarray, mfe: np.ndarray, target_day: np.ndarray, stop_day: np.ndarray):
    m = mask.astype(np.uint8)
    valid_target = np.isfinite(target_day) & (target_day > 0)
    valid_stop = np.isfinite(stop_day) & (stop_day > 0)
    target_clean = np.where(valid_target, target_day, 0.0)
    stop_clean = np.where(valid_stop, stop_day, 0.0)
    acc["samples"] += m.sum(axis=0, dtype=np.int64)
    acc["wins"] += (m * y[:, None]).sum(axis=0, dtype=np.int64)
    acc["best_sum"] += (m * best[:, None]).sum(axis=0)
    acc["close_sum"] += (m * close[:, None]).sum(axis=0)
    acc["mae_sum"] += (m * mae[:, None]).sum(axis=0)
    acc["mfe_sum"] += (m * mfe[:, None]).sum(axis=0)
    acc["target_day_sum"] += (m * target_clean[:, None]).sum(axis=0)
    acc["target_day_count"] += (m * valid_target[:, None]).sum(axis=0, dtype=np.int64)
    acc["stop_day_sum"] += (m * stop_clean[:, None]).sum(axis=0)
    acc["stop_day_count"] += (m * valid_stop[:, None]).sum(axis=0, dtype=np.int64)


def new_acc(n: int) -> dict:
    return {
        "samples": np.zeros(n, dtype=np.int64),
        "wins": np.zeros(n, dtype=np.int64),
        "best_sum": np.zeros(n, dtype=np.float64),
        "close_sum": np.zeros(n, dtype=np.float64),
        "mae_sum": np.zeros(n, dtype=np.float64),
        "mfe_sum": np.zeros(n, dtype=np.float64),
        "target_day_sum": np.zeros(n, dtype=np.float64),
        "target_day_count": np.zeros(n, dtype=np.int64),
        "stop_day_sum": np.zeros(n, dtype=np.float64),
        "stop_day_count": np.zeros(n, dtype=np.int64),
    }


def row_from_acc(rule_type: str, rule_payload: dict, acc: dict, pos: int, regime: dict | None = None) -> dict:
    n = int(acc["samples"][pos])
    w = int(acc["wins"][pos])
    payload = {
        "rule_type": rule_type,
        "samples": n,
        "wins": w,
        "path_win_1pct_rate_pct": w / n * 100.0 if n else None,
        "wilson_lower_pct": wilson_lower_bound(w, n) * 100.0 if n else None,
        "mean_best_return_pct": acc["best_sum"][pos] / n if n else None,
        "mean_5d_close_return_pct": acc["close_sum"][pos] / n if n else None,
        "mean_mae_pct": acc["mae_sum"][pos] / n if n else None,
        "mean_mfe_pct": acc["mfe_sum"][pos] / n if n else None,
        "target_day_mean": acc["target_day_sum"][pos] / acc["target_day_count"][pos] if acc["target_day_count"][pos] else None,
        "stop_day_mean": acc["stop_day_sum"][pos] / acc["stop_day_count"][pos] if acc["stop_day_count"][pos] else None,
    }
    payload.update(rule_payload)
    if regime is not None:
        payload["regime"] = regime
    return payload


def iter_split(files, start, end, state):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if frame.empty or i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x = make_stock_features(frame)
        y, best, close, complete, mae, mfe, target_day, stop_day = path_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if not len(keep):
            continue
        yield (
            date, x[keep], y[keep], best[keep], close[keep], mae[keep], mfe[keep],
            target_day[keep], stop_day[keep], regime_masks(frame)[keep]
        )


def discover_train(files, start, end):
    state = FeatureState()
    rule_acc = [new_acc(len(RULES)) for _ in REGIMES]
    regime_counts = np.zeros(len(REGIMES), dtype=np.int64)
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = np.column_stack([
            (x[:, rule["feature_index"]] >= rule["threshold"] if rule["op"] == ">=" else x[:, rule["feature_index"]] <= rule["threshold"]).astype(np.uint8)
            for rule in RULES
        ])
        for ridx in range(len(REGIMES)):
            rm = rmask[:, ridx].astype(bool)
            if not rm.any():
                continue
            regime_counts[ridx] += int(rm.sum())
            accumulate(rule_acc[ridx], stock_masks[rm], y[rm], best[rm], close[rm], mae[rm], mfe[rm], target_day[rm], stop_day[rm])
        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[状态挖掘] 训练 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    candidates_by_regime = {}
    for ridx, acc in enumerate(rule_acc):
        rows = []
        for pos, rule in enumerate(RULES):
            n = int(acc["samples"][pos])
            if n < max(MIN_TRAIN_RULE_SAMPLES, MIN_REGIME_SAMPLES):
                continue
            row = row_from_acc(
                "regime_stock_atomic",
                {
                    "rule_index": rule["rule_index"],
                    "stock_rule": rule,
                    "stock_rule_label": rule_label(rule),
                },
                acc, pos, REGIMES[ridx],
            )
            rows.append(row)
        rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
        candidates_by_regime[ridx] = rows[:TOP_STOCK_ATOMICS_PER_REGIME]
    return candidates_by_regime, regime_counts, processed, samples


def pair_train(files, start, end, candidates_by_regime):
    specs = []
    for ridx, rows in candidates_by_regime.items():
        indices = [row["rule_index"] for row in rows]
        for pos, left in enumerate(indices):
            for right in indices[pos + 1:]:
                specs.append((ridx, left, right))
    if not specs:
        return [], {}, 0, 0

    state = FeatureState()
    acc = new_acc(len(specs))
    spec_pos = {spec: pos for pos, spec in enumerate(specs)}
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = np.column_stack([
            (x[:, rule["feature_index"]] >= rule["threshold"] if rule["op"] == ">=" else x[:, rule["feature_index"]] <= rule["threshold"]).astype(np.uint8)
            for rule in RULES
        ])
        for ridx, left, right in specs:
            rm = rmask[:, ridx].astype(bool)
            if not rm.any():
                continue
            mask = rm & (stock_masks[:, left].astype(bool)) & (stock_masks[:, right].astype(bool))
            local = np.zeros(len(y), dtype=np.uint8)
            local[mask] = 1
            pos = spec_pos[(ridx, left, right)]
            accumulate(acc, local[:, None], y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[状态挖掘] 三条件训练 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    rows = []
    for pos, spec in enumerate(specs):
        ridx, left, right = spec
        left_rule, right_rule = RULES[left], RULES[right]
        n = int(acc["samples"][pos])
        if n < MIN_TRAIN_RULE_SAMPLES:
            continue
        rows.append(row_from_acc(
            "regime_stock_pair",
            {
                "rule_indices": [int(left), int(right)],
                "stock_rules": [left_rule, right_rule],
                "stock_rule_labels": [rule_label(left_rule), rule_label(right_rule)],
                "rule_label": f'{REGIMES[ridx]} | {rule_label(left_rule)} AND {rule_label(right_rule)}',
            },
            acc, pos, REGIMES[ridx],
        ))
    rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
    return rows, {spec: pos for pos, spec in enumerate(specs)}, processed, samples


def evaluate_specs(files, start, end, specs):
    if not specs:
        return []
    state = FeatureState()
    acc = new_acc(len(specs))
    spec_pos = {spec: pos for pos, spec in enumerate(specs)}
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = np.column_stack([
            (x[:, rule["feature_index"]] >= rule["threshold"] if rule["op"] == ">=" else x[:, rule["feature_index"]] <= rule["threshold"]).astype(np.uint8)
            for rule in RULES
        ])
        for pos, spec in enumerate(specs):
            ridx, left, right = spec
            mask = rmask[:, ridx].astype(bool) & stock_masks[:, left].astype(bool) & stock_masks[:, right].astype(bool)
            local = np.zeros(len(y), dtype=np.uint8)
            local[mask] = 1
            accumulate(acc, local[:, None], y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)

    rows = []
    for pos, (ridx, left, right) in enumerate(specs):
        left_rule, right_rule = RULES[left], RULES[right]
        rows.append(row_from_acc(
            "regime_stock_pair",
            {
                "rule_indices": [int(left), int(right)],
                "stock_rules": [left_rule, right_rule],
                "stock_rule_labels": [rule_label(left_rule), rule_label(right_rule)],
                "rule_label": f'regime_{ridx} | {rule_label(left_rule)} AND {rule_label(right_rule)}',
            },
            acc, pos, REGIMES[ridx],
        ))
    return rows


def evaluate_atomic(files, start, end, specs):
    if not specs:
        return []
    state = FeatureState()
    acc = new_acc(len(specs))
    processed = samples = 0
    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = np.column_stack([
            (x[:, rule["feature_index"]] >= rule["threshold"] if rule["op"] == ">=" else x[:, rule["feature_index"]] <= rule["threshold"]).astype(np.uint8)
            for rule in RULES
        ])
        for pos, (ridx, rule_idx) in enumerate(specs):
            mask = rmask[:, ridx].astype(bool) & stock_masks[:, rule_idx].astype(bool)
            local = np.zeros(len(y), dtype=np.uint8)
            local[mask] = 1
            accumulate(acc, local[:, None], y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)
    rows = []
    for pos, (ridx, rule_idx) in enumerate(specs):
        rule = RULES[rule_idx]
        rows.append(row_from_acc(
            "regime_stock_atomic",
            {
                "rule_index": int(rule_idx),
                "stock_rule": rule,
                "stock_rule_label": rule_label(rule),
                "rule_label": f'regime_{ridx} | {rule_label(rule)}',
            },
            acc, pos, REGIMES[ridx],
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "path_regime_rule_mining_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据范围内")

    train_candidates, regime_counts, train_days, train_samples = discover_train(files, args.start, TRAIN_END)
    train_specs = []
    for ridx, rows in train_candidates.items():
        train_specs.extend((ridx, row["rule_index"]) for row in rows)

    # Build regime-conditioned stock-pair candidates from training only.
    pair_train_rows, _, pair_days, pair_samples = pair_train(files, args.start, TRAIN_END, train_candidates)
    pair_train_rows = pair_train_rows[:TOP_OUTPUT_RULES * 4]
    pair_specs = []
    for row in pair_train_rows:
        ridx = REGIMES.index(row["regime"])
        pair_specs.append((ridx, row["rule_indices"][0], row["rule_indices"][1]))

    validation_atomic = evaluate_atomic(files, VALIDATION_START, VALIDATION_END, train_specs)
    validation_pair = evaluate_specs(files, VALIDATION_START, VALIDATION_END, pair_specs)
    validation_rows = validation_atomic + validation_pair
    validation_rows = [
        row for row in validation_rows
        if row["samples"] >= MIN_VALIDATION_RULE_SAMPLES
    ]
    validation_rows.sort(key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]), reverse=True)
    selected = validation_rows[:TOP_OUTPUT_RULES]

    selected_atomic_specs = []
    selected_pair_specs = []
    selected_source = []
    for row in selected:
        regime = row["regime"]
        ridx = REGIMES.index(regime)
        if row["rule_type"] == "regime_stock_atomic":
            selected_atomic_specs.append((ridx, row["rule_index"]))
        else:
            selected_pair_specs.append((ridx, row["rule_indices"][0], row["rule_indices"][1]))
        selected_source.append(row)

    final_atomic = evaluate_atomic(files, FINAL_START, args.final_end, selected_atomic_specs)
    final_pairs = evaluate_specs(files, FINAL_START, args.final_end, selected_pair_specs)
    final_rows = final_atomic + final_pairs
    final_map = {}
    for row in final_rows:
        if row["rule_type"] == "regime_stock_atomic":
            key = ("atomic", REGIMES.index(row["regime"]), row["rule_index"])
        else:
            key = ("pair", REGIMES.index(row["regime"]), *row["rule_indices"])
        final_map[key] = row

    selected_output = []
    for row in selected_source:
        ridx = REGIMES.index(row["regime"])
        if row["rule_type"] == "regime_stock_atomic":
            key = ("atomic", ridx, row["rule_index"])
        else:
            key = ("pair", ridx, *row["rule_indices"])
        selected_output.append({
            "rule_type": row["rule_type"],
            "regime": row["regime"],
            "rule_label": row["rule_label"],
            "train": next((x for x in pair_train_rows + sum(train_candidates.values(), []) if x.get("rule_label") == row["rule_label"] or (
                x["rule_type"] == row["rule_type"] and REGIMES.index(x["regime"]) == ridx and (
                    x.get("rule_index") == row.get("rule_index") or x.get("rule_indices") == row.get("rule_indices")
                )
            )), None),
            "validation": row,
            "final": final_map.get(key),
            "validation_target_80pct": row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT,
        })

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "market_regime_conditioned_three_condition_rule_mining",
        "objective": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "entry_rule": "T_plus_1_open; limit_up_entry_blocked",
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
            "min_train_rule_samples": MIN_TRAIN_RULE_SAMPLES,
            "min_validation_rule_samples": MIN_VALIDATION_RULE_SAMPLES,
            "min_regime_samples": MIN_REGIME_SAMPLES,
            "top_stock_atomics_per_regime": TOP_STOCK_ATOMICS_PER_REGIME,
            "top_output_rules": TOP_OUTPUT_RULES,
            "rank_thresholds": list(RANK_THRESHOLDS),
            "breadth_bins": list(BREADTH_BINS),
            "median_bins": list(MEDIAN_BINS),
        },
        "regimes": REGIMES,
        "train": {
            "samples": train_samples,
            "processed_days": train_days,
            "regime_sample_counts": regime_counts.tolist(),
        },
        "train_top_atomic_by_regime": train_candidates,
        "train_top_pairs": pair_train_rows,
        "validation_top_rules": selected,
        "validation_qualified_rules_ge_80pct": [
            row for row in selected if row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
        ],
        "selected_rules": selected_output,
        "final_qualified_rules_ge_80pct": [
            row for row in final_rows
            if row["samples"] >= MIN_VALIDATION_RULE_SAMPLES
            and row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
        ],
        "audit": {
            "no_future_features": True,
            "path_label_target_first": True,
            "same_day_stop_first": True,
            "entry_is_T_plus_1_open": True,
            "limit_up_entry_blocked": ENTRY_LIMIT_UP_BLOCK,
            "final_holdout_used_only_after_validation_selection": True,
            "rule_discovery_split": "train",
            "rule_selection_split": "validation",
            "final_holdout_split": "final",
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(__import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(__import__("json").dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_rules": len(validation_rows),
        "validation_ge_80pct": len(result["validation_qualified_rules_ge_80pct"]),
        "final_ge_80pct": len(result["final_qualified_rules_ge_80pct"]),
        "top_validation_rule": selected[0]["rule_label"] if selected else None,
        "top_validation_rate": selected[0]["path_win_1pct_rate_pct"] if selected else None,
        "top_final_rate": selected_output[0]["final"]["path_win_1pct_rate_pct"] if selected_output and selected_output[0]["final"] else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
