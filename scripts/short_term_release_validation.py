"""Release validation with an untouched final 2026 holdout.

The strategy is loaded from Aquant-Private/main. No parameter selection is
performed here. The final_holdout period is evaluated only after the strategy
version is fixed in the private repository.
"""
from __future__ import annotations

import argparse
import importlib
import time
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, managed_trade, read_daily, _state_to_jsonable, _state_from_jsonable

GATE = {
    "forward_3d_mean_return_pct": 0.10,
    "forward_5d_mean_return_pct": 0.10,
    "forward_5d_positive_rate_pct": 50.0,
    "managed_trade_mean_return_pct": 0.10,
    "managed_trade_threshold_win_rate_pct": 80.0,
    "max_managed_trade_drawdown_pct": -40.0,
    "minimum_managed_trade_samples": 100,
}


def _config_value(text: str, key: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped.split(":", 1)[1].strip().strip('"')
    raise RuntimeError(f"strategy config missing: {key}")


def validate_private_hard_rules(root: Path) -> None:
    config = root / "config" / "strategy.yaml"
    if not config.exists():
        raise RuntimeError("strategy config is missing")
    text = config.read_text(encoding="utf-8")
    expected = {
        "signal_at": "T_close",
        "entry": "T+1_open",
        "earliest_exit": "T+2",
        "maximum_exit_session_after_signal": "5",
        "max_holding_sessions": "5",
        "target_return_pct": "1.0",
        "stop_loss_pct": "3.0",
        "entry_limit_up_block": "true",
    }
    for key, value in expected.items():
        if _config_value(text, key) != value:
            raise RuntimeError(f"strategy hard rule mismatch: {key}")


def load_strategy():
    root = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not root:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root = str(Path(root).resolve())
    validate_private_hard_rules(Path(root))
    sys.path.insert(0, root)
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    return model, str(version.STRATEGY_VERSION), commit


def forward_return(future_days, symbol, horizon):
    if not future_days:
        return None
    entry_row = future_days[0].loc[future_days[0]["symbol"].eq(symbol)]
    if entry_row.empty or pd.isna(entry_row.iloc[0].get("open")):
        return None
    entry = float(entry_row.iloc[0]["open"])
    target_index = horizon - 1
    if target_index >= len(future_days):
        return None
    target_row = future_days[target_index].loc[future_days[target_index]["symbol"].eq(symbol)]
    if target_row.empty or pd.isna(target_row.iloc[0].get("close")):
        return None
    close = float(target_row.iloc[0]["close"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(close):
        return None
    return (close / entry - 1.0) * 100.0


def summarize(data, cost_bps, slippage_bps):
    r3 = np.asarray(data["forward_3d"], dtype=float)
    r5 = np.asarray(data["forward_5d"], dtype=float)
    trades = data["trades"]
    gross = pd.to_numeric(pd.Series([x["gross_return_pct"] for x in trades]), errors="coerce")
    net = gross - 2 * (cost_bps + slippage_bps) / 100.0 if len(gross) else gross
    if len(net):
        eq = (1.0 + net / 100.0).cumprod()
        dd = eq / eq.cummax() - 1.0
        trade_stats = {
            "samples": int(len(net)),
            "win_rate_pct": float((net >= 1.0).mean() * 100.0),
            "positive_rate_pct": float((net > 0).mean() * 100.0),
            "threshold_win_rate_pct": float((net >= 1.0).mean() * 100.0),
            "win_threshold_pct": 1.0,
            "mean_return_pct": float(net.mean()),
            "median_return_pct": float(net.median()),
            "max_drawdown_pct": float(dd.min() * 100.0),
            "mean_holding_days": float(pd.to_numeric(pd.Series([x["holding_days"] for x in trades])).mean()),
        }
    else:
        trade_stats = {
            "samples": 0, "win_rate_pct": None, "positive_rate_pct": None,
            "threshold_win_rate_pct": None, "win_threshold_pct": 1.0, "mean_return_pct": None,
            "median_return_pct": None, "max_drawdown_pct": None, "mean_holding_days": None,
        }
    result = {
        "signal_days": int(data["signal_days"]),
        "candidate_days": int(data["candidate_days"]),
        "candidate_day_rate_pct": float(data["candidate_days"] / data["signal_days"] * 100.0) if data["signal_days"] else 0.0,
        "forward_3d": {
            "samples": int(len(r3)),
            "mean_return_pct": float(r3.mean()) if len(r3) else None,
            "positive_rate_pct": float((r3 > 0).mean() * 100.0) if len(r3) else None,
        },
        "forward_5d": {
            "samples": int(len(r5)),
            "mean_return_pct": float(r5.mean()) if len(r5) else None,
            "positive_rate_pct": float((r5 > 0).mean() * 100.0) if len(r5) else None,
        },
        "managed_trade": trade_stats,
    }
    result["production_gate_passed"] = bool(
        result["forward_3d"]["mean_return_pct"] is not None
        and result["forward_5d"]["mean_return_pct"] is not None
        and trade_stats["mean_return_pct"] is not None
        and trade_stats["threshold_win_rate_pct"] is not None
        and trade_stats["max_drawdown_pct"] is not None
        and trade_stats["samples"] >= GATE["minimum_managed_trade_samples"]
        and result["forward_3d"]["mean_return_pct"] >= GATE["forward_3d_mean_return_pct"]
        and result["forward_5d"]["mean_return_pct"] >= GATE["forward_5d_mean_return_pct"]
        and result["forward_5d"]["positive_rate_pct"] >= GATE["forward_5d_positive_rate_pct"]
        and trade_stats["mean_return_pct"] >= GATE["managed_trade_mean_return_pct"]
        and trade_stats["threshold_win_rate_pct"] >= GATE["managed_trade_threshold_win_rate_pct"]
        and trade_stats["max_drawdown_pct"] >= GATE["max_managed_trade_drawdown_pct"]
    )
    return result




RESEARCH_CHECKPOINT_DAYS = 5


def _repo_root_from_env() -> Path:
    value = os.environ.get("AQUANT_CHECKPOINT_REPO_ROOT", "").strip()
    return Path(value).resolve() if value else ROOT


def research_root_from_args(args) -> Path:
    value = args.research_root or str(ROOT / "data" / "research" / "short_term_release_validation")
    return Path(value).resolve()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_progress(path: Path, event: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[研究进度] {event} {json.dumps(fields, ensure_ascii=False)}", flush=True)


def load_research_state(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_research_state(path: Path, state_payload: dict) -> None:
    write_json(path, state_payload)


def checkpoint_git(paths: list[str], message: str) -> None:
    repo = _repo_root_from_env()
    subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)
    if subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "aquant-bot"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "commit", "-m", message], check=True)
    for attempt in range(1, 4):
        subprocess.run(["git", "-C", str(repo), "fetch", "origin", "main"], check=True)
        subprocess.run(["git", "-C", str(repo), "rebase", "origin/main"], check=True)
        try:
            subprocess.run(["git", "-C", str(repo), "push", "origin", "HEAD:main"], check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(min(10, attempt * 2))


def run(args):
    model, version, commit = load_strategy()
    files = history_files()
    dates = [p.name[:10] for p in files]
    windows = {
        "development": (args.development_start, args.development_end),
        "validation": (args.validation_start, args.validation_end),
        "final_holdout": (args.final_start, args.final_end),
    }
    indices = {}
    for name, (start_date, end_date) in windows.items():
        if start_date not in dates or end_date not in dates:
            raise ValueError(f"{name} dates must be trading dates")
        indices[name] = (dates.index(start_date), dates.index(end_date))

    begin = max(0, indices["development"][0] - 20)
    end_i = indices["final_holdout"][1]
    active = files[begin : end_i + 6]
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(active[i])
        return cache[i]

    state = FeatureState()
    buckets = {name: {"signal_days": 0, "candidate_days": 0, "forward_3d": [], "forward_5d": [], "trades": []} for name in windows}
    research_root = research_root_from_args(args)
    state_file = research_root / "_RESEARCH_STATE.json"
    progress_file = research_root / "_PROGRESS.jsonl"
    completed_dates = set()
    resume_i = 0

    checkpoint = load_research_state(state_file)
    expected = {
        "schema_version": 2,
        "windows": {k: list(v) for k, v in windows.items()},
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "strategy_version": version,
        "strategy_commit": commit,
        "checkpoint_days": RESEARCH_CHECKPOINT_DAYS,
    }
    if checkpoint:
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(f"研究断点参数不一致: {key}")
        _state_from_jsonable(state, checkpoint["state"])
        buckets = checkpoint["buckets"]
        completed_dates = set(checkpoint.get("completed_dates", []))
        resume_i = int(checkpoint.get("next_active_index", 0))
        append_progress(
            progress_file,
            "断点续传",
            completed_days=len(completed_dates),
            resume_index=resume_i,
            total_active_days=len(active),
        )
    else:
        research_root.mkdir(parents=True, exist_ok=True)
        append_progress(
            progress_file,
            "研究开始",
            total_active_days=len(active),
            strategy_version=version,
            strategy_commit=commit,
            checkpoint_days=RESEARCH_CHECKPOINT_DAYS,
        )

    def persist_checkpoint(next_i: int, signal_date: str | None = None, force_git: bool = False):
        payload = {
            **expected,
            "status": "running",
            "next_active_index": next_i,
            "completed_dates": sorted(completed_dates),
            "state": _state_to_jsonable(state),
            "buckets": buckets,
            "last_signal_date": signal_date,
        }
        save_research_state(state_file, payload)
        if force_git:
            day_files = list(research_root.rglob(f"{signal_date}.json"))
            if not day_files:
                raise RuntimeError(f"研究日期文件不存在: {signal_date}")
            checkpoint_git(
                [
                    str(state_file.relative_to(_repo_root_from_env())),
                    str(progress_file.relative_to(_repo_root_from_env())),
                    str(day_files[-1].relative_to(_repo_root_from_env())),
                ],
                f"研究：历史验证检查点至 {signal_date}",
            )

    started = time.time()
    pending_checkpoint_days = 0
    for i in range(resume_i, len(active) - 5):
        signal_date = active[i].name[:10]
        if signal_date in completed_dates:
            continue

        frame = state.build(get(i))
        period = None
        for name, (start_date, end_date) in windows.items():
            if start_date <= signal_date <= end_date:
                end_rel = indices[name][1] - begin
                if i + 5 <= end_rel:
                    period = name
                break

        day_payload = {
            "schema_version": 1,
            "signal_date": signal_date,
            "period": period,
            "strategy_version": version,
            "strategy_commit": commit,
            "candidate_count": 0,
            "candidates": [],
        }

        if period is not None and not frame.empty:
            scored = model.score_universe(frame)
            selected = model.admit_candidates(scored)
            bucket = buckets[period]
            bucket["signal_days"] += 1
            if not selected.empty:
                bucket["candidate_days"] += 1
                future = [get(i + j) for j in range(1, 6)]
                day_payload["candidate_count"] = int(len(selected))
                day_payload["candidates"] = selected.to_dict(orient="records")
                for symbol in selected["symbol"].astype(str).str.zfill(6):
                    r3 = forward_return(future, symbol, 3)
                    r5 = forward_return(future, symbol, 5)
                    if r3 is not None:
                        bucket["forward_3d"].append(r3)
                    if r5 is not None:
                        bucket["forward_5d"].append(r5)
                    trade = managed_trade(
                        symbol,
                        future,
                        round_trip_cost_bps=2 * (args.cost_bps + args.slippage_bps),
                    )
                    if trade:
                        bucket["trades"].append(trade)

        day_file = research_root / (period or "out_of_window") / signal_date[:4] / f"{signal_date}.json"
        write_json(day_file, day_payload)
        completed_dates.add(signal_date)
        pending_checkpoint_days += 1

        append_progress(
            progress_file,
            "日期完成",
            signal_date=signal_date,
            active_index=i,
            completed_days=len(completed_dates),
            progress_pct=round((i + 1) / max(1, len(active)) * 100.0, 2),
            candidate_days={k: v["candidate_days"] for k, v in buckets.items()},
            elapsed_seconds=round(time.time() - started, 2),
        )

        force_git = pending_checkpoint_days >= RESEARCH_CHECKPOINT_DAYS
        persist_checkpoint(i + 1, signal_date, force_git=force_git)
        if force_git:
            pending_checkpoint_days = 0

    output = {
        "schema_version": 1,
        "status": "ready",
        "method": "short_term_release_validation",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": version,
        "strategy_commit": commit,
        "future_function": False,
        "entry": "T+1_open",
        "max_holding_sessions": 5,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "production_gate": GATE,
        "selection_rule": "strategy version and admission parameters are fixed before final_holdout evaluation",
        "win_definition": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "fallback_policy": "below_80_pct_use_highest_stable_research_operating_point; do_not_force_production_gate",
    }
    for name in windows:
        output[name] = summarize(buckets[name], args.cost_bps, args.slippage_bps)
    output["release_gate"] = output["final_holdout"]["production_gate_passed"]
    output["release_gate_scope"] = "final_holdout_only"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    completed_payload = {
        **expected,
        "status": "completed",
        "next_active_index": len(active),
        "completed_dates": sorted(completed_dates),
        "state": _state_to_jsonable(state),
        "buckets": buckets,
        "last_signal_date": sorted(completed_dates)[-1] if completed_dates else None,
    }
    save_research_state(state_file, completed_payload)
    if pending_checkpoint_days:
        checkpoint_git(
            [
                str(state_file.relative_to(_repo_root_from_env())),
                str(progress_file.relative_to(_repo_root_from_env())),
                str(output_path.relative_to(_repo_root_from_env())),
                str(research_root.relative_to(_repo_root_from_env())),
            ],
            f"研究：完成历史验证至 {completed_payload['last_signal_date']}",
        )
    append_progress(progress_file, "研究完成", elapsed_seconds=round(time.time() - started, 2), release_gate=output["release_gate"])
    print(json.dumps({
        "status": "ready",
        "strategy_version": version,
        "strategy_commit": commit,
        "release_gate": output["release_gate"],
        "output": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-start", default="2018-01-05")
    ap.add_argument("--development-end", default="2024-12-20")
    ap.add_argument("--validation-start", default="2025-01-02")
    ap.add_argument("--validation-end", default="2025-12-31")
    ap.add_argument("--final-start", default="2026-01-05")
    ap.add_argument("--final-end", default="2026-09-30")
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--output", default=str(ROOT / "data/backtest/short_term_release_validation.json"))
    ap.add_argument("--research-root", default=str(ROOT / "data/research" / "short_term_release_validation"))
    run(ap.parse_args())
