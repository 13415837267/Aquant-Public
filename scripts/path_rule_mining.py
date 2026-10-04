"""Path-aware conditional rule mining for the 1% short-term objective.

Research only. Rules use information available at signal close T. Rule families
are discovered on 2015-2022, ranked on 2023-2024, and tested on 2025-2026-09-30.
The trade label requires net +1% before a gross -3% stop within five sessions.
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
MARKET_FEATURES = ["market_breadth_pct", "market_median_return_pct"]
ALL_FEATURES = STOCK_FEATURES + MARKET_FEATURES

TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = 1.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
ENTRY_LIMIT_UP_BLOCK = True

RANK_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
BREADTH_THRESHOLDS = (40.0, 45.0, 50.0, 55.0, 60.0, 65.0)
MEDIAN_THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)

MIN_TRAIN_RULE_SAMPLES = 10_000
MIN_VALIDATION_RULE_SAMPLES = 5_000
MAX_ATOMIC_CANDIDATES = 16
MAX_PAIR_CANDIDATES = 24
TOP_OUTPUT_RULES = 20
TARGET_WIN_RATE_PCT = 80.0


def percentile_rank(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_features(frame):
    stock = [percentile_rank(frame[name]) for name in STOCK_FEATURES]
    breadth = np.clip(
        pd.to_numeric(frame["market_breadth_pct"], errors="coerce")
        .fillna(50.0).to_numpy(dtype=float), 0.0, 100.0
    )
    median_ret = pd.to_numeric(
        frame["market_median_return_pct"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    return np.column_stack(stock + [breadth, median_ret])


def feature_defs():
    defs = []
    for name in STOCK_FEATURES:
        for threshold in RANK_THRESHOLDS:
            defs.append({"feature": name, "op": ">=", "threshold": threshold, "scale": "rank"})
            defs.append({"feature": name, "op": "<=", "threshold": threshold, "scale": "rank"})
    for threshold in BREADTH_THRESHOLDS:
        defs.append({"feature": "market_breadth_pct", "op": ">=", "threshold": threshold, "scale": "pct"})
        defs.append({"feature": "market_breadth_pct", "op": "<=", "threshold": threshold, "scale": "pct"})
    for threshold in MEDIAN_THRESHOLDS:
        defs.append({"feature": "market_median_return_pct", "op": ">=", "threshold": threshold, "scale": "pct"})
        defs.append({"feature": "market_median_return_pct", "op": "<=", "threshold": threshold, "scale": "pct"})
    return defs


RULE_DEFS = feature_defs()
FEATURE_INDEX = {name: idx for idx, name in enumerate(ALL_FEATURES)}


def rule_label(rule):
    return f'{rule["feature"]}{rule["op"]}{rule["threshold"]:g}'


def wilson_lower_bound(wins, samples, z=1.96):
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def empty_targets(n):
    return (
        np.zeros(n, dtype=bool),
        np.full(n, np.nan), np.full(n, np.nan),
        np.zeros(n, dtype=bool),
        np.full(n, np.nan), np.full(n, np.nan),
        np.full(n, np.nan), np.full(n, np.nan),
    )


def path_targets(symbols, future_days):
    n = len(symbols)
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return empty_targets(n)

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
    for day in future_days[:MAX_FORWARD_SESSIONS]:
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

    path_win = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    target_day = np.full(n, np.nan)
    stop_day = np.full(n, np.nan)

    for row in np.flatnonzero(complete):
        best_net[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        mae[row] = np.min(lows[row]) / entry[row] * 100.0 - 100.0
        mfe[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0

        stopped = False
        for d in range(MAX_FORWARD_SESSIONS):
            # T+1 为建仓日，严格禁止在 T+1 触发卖出；T+2 才是最早退出日。
            if d == 0:
                continue
            day_open, day_high, day_low = opens[row, d], highs[row, d], lows[row]
            if day_open <= stop[row] or day_low <= stop[row]:
                stop_day[row] = d + 1
                stopped = True
                break
            if day_high >= target[row]:
                target_day[row] = d + 1
                break
        path_win[row] = target_day[row] > 0 and not stopped

    return path_win, best_net, close_net, complete, mae, mfe, target_day, stop_day


def atomic_masks(x, rule_indices=None):
    indices = list(range(len(RULE_DEFS))) if rule_indices is None else list(rule_indices)
    masks = np.empty((len(x), len(indices)), dtype=np.uint8)
    for pos, rule_idx in enumerate(indices):
        rule = RULE_DEFS[rule_idx]
        values = x[:, FEATURE_INDEX[rule["feature"]]]
        threshold = rule["threshold"]
        masks[:, pos] = (
            values >= threshold if rule["op"] == ">=" else values <= threshold
        ).astype(np.uint8)
    return masks


def new_accumulator(count):
    return {
        "samples": np.zeros(count, dtype=np.int64),
        "wins": np.zeros(count, dtype=np.int64),
        "best_sum": np.zeros(count, dtype=np.float64),
        "close_sum": np.zeros(count, dtype=np.float64),
        "mae_sum": np.zeros(count, dtype=np.float64),
        "mfe_sum": np.zeros(count, dtype=np.float64),
        "target_day_sum": np.zeros(count, dtype=np.float64),
        "target_day_count": np.zeros(count, dtype=np.int64),
        "stop_day_sum": np.zeros(count, dtype=np.float64),
        "stop_day_count": np.zeros(count, dtype=np.int64),
    }


def accumulate_masks(acc, masks, y, best, close, mae, mfe, target_day, stop_day):
    acc["samples"] += masks.sum(axis=0, dtype=np.int64)
    acc["wins"] += (masks * y[:, None]).sum(axis=0, dtype=np.int64)
    acc["best_sum"] += (masks * best[:, None]).sum(axis=0)
    acc["close_sum"] += (masks * close[:, None]).sum(axis=0)
    acc["mae_sum"] += (masks * mae[:, None]).sum(axis=0)
    acc["mfe_sum"] += (masks * mfe[:, None]).sum(axis=0)
    target_valid = (target_day > 0).astype(np.float64)
    stop_valid = (stop_day > 0).astype(np.float64)
    acc["target_day_sum"] += (masks * (target_day * target_valid)[:, None]).sum(axis=0)
    acc["target_day_count"] += (masks * target_valid[:, None]).sum(axis=0, dtype=np.int64)
    acc["stop_day_sum"] += (masks * (stop_day * stop_valid)[:, None]).sum(axis=0)
    acc["stop_day_count"] += (masks * stop_valid[:, None]).sum(axis=0, dtype=np.int64)


def rule_row(rule_type, rule_indices, acc, pos):
    n = int(acc["samples"][pos])
    w = int(acc["wins"][pos])
    item = {
        "rule_type": rule_type,
        "rule_indices": [int(x) for x in rule_indices],
        "rules": [RULE_DEFS[int(x)] for x in rule_indices],
        "rule_label": " AND ".join(rule_label(RULE_DEFS[int(x)]) for x in rule_indices),
        "samples": n,
        "wins": w,
        "path_win_1pct_rate_pct": (w / n * 100.0) if n else None,
        "wilson_lower_pct": (wilson_lower_bound(w, n) * 100.0) if n else None,
        "mean_best_return_pct": (acc["best_sum"][pos] / n) if n else None,
        "mean_5d_close_return_pct": (acc["close_sum"][pos] / n) if n else None,
        "mean_mae_pct": (acc["mae_sum"][pos] / n) if n else None,
        "mean_mfe_pct": (acc["mfe_sum"][pos] / n) if n else None,
        "target_day_mean": (
            acc["target_day_sum"][pos] / acc["target_day_count"][pos]
            if acc["target_day_count"][pos] else None
        ),
        "stop_day_mean": (
            acc["stop_day_sum"][pos] / acc["stop_day_count"][pos]
            if acc["stop_day_count"][pos] else None
        ),
    }
    return item


def summarize_atomic(acc):
    rows = []
    for idx, rule in enumerate(RULE_DEFS):
        n = int(acc["samples"][idx])
        if n < MIN_TRAIN_RULE_SAMPLES:
            continue
        rows.append(rule_row("atomic", [idx], acc, idx))
    rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
    return rows


def summarize_selected(acc, rule_specs):
    rows = []
    for pos, (rule_type, indices) in enumerate(rule_specs):
        rows.append(rule_row(rule_type, indices, acc, pos))
    rows.sort(key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]), reverse=True)
    return rows


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
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_full = make_features(frame)
        symbols = frame["symbol"].astype(str).str.zfill(6).tolist()
        y, best, close, complete, mae, mfe, target_day, stop_day = path_targets(symbols, futures)
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        yield (
            date,
            x_full[keep],
            y[keep],
            best[keep],
            close[keep],
            mae[keep],
            mfe[keep],
            target_day[keep],
            stop_day[keep],
        )


def scan_atomic(files, start, end):
    state = FeatureState()
    acc = new_accumulator(len(RULE_DEFS))
    processed = samples = 0
    for date, x, y, best, close, mae, mfe, target_day, stop_day in iter_split(files, start, end, state):
        masks = atomic_masks(x)
        accumulate_masks(acc, masks, y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[条件挖掘] 原子规则 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    return acc, processed, samples


def scan_pairs(files, start, end, atomic_indices, pair_indices):
    state = FeatureState()
    rule_specs = [("pair", pair) for pair in pair_indices]
    acc = new_accumulator(len(rule_specs))
    processed = samples = 0
    for date, x, y, best, close, mae, mfe, target_day, stop_day in iter_split(files, start, end, state):
        base_masks = atomic_masks(x, atomic_indices)
        local_index = {rule_idx: pos for pos, rule_idx in enumerate(atomic_indices)}
        masks = np.empty((len(x), len(pair_indices)), dtype=np.uint8)
        for pos, (left, right) in enumerate(pair_indices):
            masks[:, pos] = base_masks[:, local_index[left]] & base_masks[:, local_index[right]]
        accumulate_masks(acc, masks, y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[条件挖掘] 配对规则 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    return rule_specs, acc, processed, samples


def scan_selected(files, start, end, atomic_indices, pair_indices):
    rule_specs = [("atomic", (idx,)) for idx in atomic_indices] + [("pair", pair) for pair in pair_indices]
    state = FeatureState()
    acc = new_accumulator(len(rule_specs))
    processed = samples = 0
    atomic_pos = {idx: pos for pos, idx in enumerate(atomic_indices)}
    for date, x, y, best, close, mae, mfe, target_day, stop_day in iter_split(files, start, end, state):
        base_masks = atomic_masks(x, atomic_indices)
        masks = np.empty((len(x), len(rule_specs)), dtype=np.uint8)
        for pos, idx in enumerate(atomic_indices):
            masks[:, pos] = base_masks[:, atomic_pos[idx]]
        for offset, (left, right) in enumerate(pair_indices, len(atomic_indices)):
            masks[:, offset] = base_masks[:, atomic_pos[left]] & base_masks[:, atomic_pos[right]]
        accumulate_masks(acc, masks, y, best, close, mae, mfe, target_day, stop_day)
        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[条件挖掘] 选择规则 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    return rule_specs, acc, processed, samples


def split_summary(acc):
    total = int(acc["samples"].sum())
    wins = int(acc["wins"].sum())
    if total == 0:
        return {"samples": 0, "path_win_1pct_rate_pct": None}
    return {
        "samples": total,
        "path_win_1pct_rate_pct": wins / total * 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "path_rule_mining_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据范围内")

    train_atomic_acc, train_days, train_samples = scan_atomic(files, args.start, TRAIN_END)
    train_atomic = summarize_atomic(train_atomic_acc)
    top_atomic = train_atomic[:MAX_ATOMIC_CANDIDATES]
    atomic_indices = [row["rule_indices"][0] for row in top_atomic]
    if not atomic_indices:
        raise RuntimeError("没有满足训练样本门槛的原子条件")

    candidate_pairs = [
        (left, right)
        for pos, left in enumerate(atomic_indices)
        for right in atomic_indices[pos + 1:]
    ]
    _, train_pair_acc, train_pair_days, _ = scan_pairs(
        files, args.start, TRAIN_END, atomic_indices, candidate_pairs
    )
    train_pairs_all = [
        rule_row("pair", pair, train_pair_acc, pos)
        for pos, pair in enumerate(candidate_pairs)
        if int(train_pair_acc["samples"][pos]) >= MIN_TRAIN_RULE_SAMPLES
    ]
    train_pairs_all.sort(
        key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]),
        reverse=True,
    )
    top_pairs = train_pairs_all[:MAX_PAIR_CANDIDATES]
    pair_indices = [tuple(r["rule_indices"]) for r in top_pairs]

    validation_specs, validation_acc, validation_days, validation_samples = scan_selected(
        files, VALIDATION_START, VALIDATION_END, atomic_indices, pair_indices
    )
    validation_rows_all = summarize_selected(validation_acc, validation_specs)
    eligible_validation = [
        row for row in validation_rows_all if row["samples"] >= MIN_VALIDATION_RULE_SAMPLES
    ]
    eligible_validation.sort(
        key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]),
        reverse=True,
    )
    selected = eligible_validation[:TOP_OUTPUT_RULES]

    selected_specs = [(row["rule_type"], tuple(row["rule_indices"])) for row in selected]
    # Pair rules depend on both underlying atomic rules. Keep their components
    # available during final evaluation even when the atomic rules themselves
    # were not selected as standalone outputs.
    final_atomic = sorted({
        idx
        for typ, indices in selected_specs
        for idx in indices
    })
    final_pairs = [indices for typ, indices in selected_specs if typ == "pair"]

    final_specs, final_acc, final_days, final_samples = scan_selected(
        files, FINAL_START, args.final_end, final_atomic, final_pairs
    )
    final_rows = summarize_selected(final_acc, final_specs)
    final_map = {tuple(row["rule_indices"]): row for row in final_rows}
    validation_map = {tuple(row["rule_indices"]): row for row in selected}
    train_map = {}

    selected_output = []
    for row in selected:
        key = tuple(row["rule_indices"])
        train_source = top_atomic if row["rule_type"] == "atomic" else top_pairs
        for candidate in train_source:
            if tuple(candidate["rule_indices"]) == key:
                train_map[key] = candidate
                break
        selected_output.append({
            "rule_type": row["rule_type"],
            "rule_indices": list(key),
            "rules": row["rules"],
            "rule_label": row["rule_label"],
            "train": train_map.get(key),
            "validation": validation_map.get(key),
            "final": final_map.get(key),
            "validation_target_80pct": (
                row["path_win_1pct_rate_pct"] is not None
                and row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
            ),
        })

    validation_qualified = [
        row for row in selected
        if row["path_win_1pct_rate_pct"] is not None
        and row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
    ]
    final_qualified = [
        row for row in final_rows
        if row["path_win_1pct_rate_pct"] is not None
        and row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
        and row["samples"] >= MIN_VALIDATION_RULE_SAMPLES
    ]

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "path_aware_conditional_rule_mining",
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
            "max_atomic_candidates": MAX_ATOMIC_CANDIDATES,
            "max_pair_candidates": MAX_PAIR_CANDIDATES,
            "top_output_rules": TOP_OUTPUT_RULES,
            "target_win_rate_pct": TARGET_WIN_RATE_PCT,
            "rank_thresholds": list(RANK_THRESHOLDS),
            "breadth_thresholds": list(BREADTH_THRESHOLDS),
            "median_thresholds": list(MEDIAN_THRESHOLDS),
        },
        "base": {
            "train": {
                "samples": train_samples,
                "path_win_1pct_rate_pct": float(train_atomic_acc["wins"].sum() / train_atomic_acc["samples"].sum() * 100.0)
                if train_samples else None,
            },
            "validation": {
                "samples": validation_samples,
                "path_win_1pct_rate_pct": float(sum(validation_map[key]["wins"] for key in validation_map) / max(1, sum(validation_map[key]["samples"] for key in validation_map)) * 100.0)
                if validation_map else None,
            },
            "final": {
                "samples": final_samples,
                "path_win_1pct_rate_pct": float(sum(final_map[key]["wins"] for key in final_map) / max(1, sum(final_map[key]["samples"] for key in final_map)) * 100.0)
                if final_map else None,
            },
        },
        "train_top_atomic": train_atomic[:TOP_OUTPUT_RULES],
        "train_top_pairs": top_pairs[:TOP_OUTPUT_RULES],
        "validation_top_rules": selected,
        "validation_qualified_rules_ge_80pct": validation_qualified,
        "selected_rules": selected_output,
        "final_qualified_rules_ge_80pct": final_qualified,
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
    Path(args.output).write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "final_samples": final_samples,
        "validation_rules_ge_80pct": len(validation_qualified),
        "final_rules_ge_80pct": len(final_qualified),
        "top_validation_rule": selected[0]["rule_label"] if selected else None,
        "top_validation_rate": selected[0]["path_win_1pct_rate_pct"] if selected else None,
        "top_final_rate": final_map.get(tuple(selected[0]["rule_indices"]), {}).get("path_win_1pct_rate_pct") if selected else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
