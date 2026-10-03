from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import date
from pathlib import Path

from scripts.market_scope import is_main_board_symbol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.json"
REQUIRED_CANDIDATE_FIELDS = {
    "rank",
    "symbol",
    "name",
    "price",
    "change_pct",
    "momentum_60d",
    "turnover_pct",
    "amount",
    "volatility_proxy",
    "score",
}
EXPECTED_WEIGHTS = {"momentum", "liquidity", "value", "safety"}
MAX_CANDIDATES = 3


def _finite(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{field} is not finite: {value!r}")
    return number


def _date_only(value: object, field: str) -> str:
    text = str(value or "")
    if len(text) < 10:
        raise RuntimeError(f"{field} is not a valid ISO timestamp/date: {value!r}")
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError as exc:
        raise RuntimeError(f"{field} has invalid date: {value!r}") from exc


def _private_provenance() -> tuple[str | None, str | None]:
    root_text = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH")
    if not root_text:
        return None, None
    root = Path(root_text).resolve()
    version_file = root / "strategy" / "version.py"
    if not version_file.exists():
        raise RuntimeError(f"Private strategy version file not found: {version_file}")

    version = None
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("STRATEGY_VERSION") and "=" in line:
            version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not version:
        raise RuntimeError("Private strategy STRATEGY_VERSION not found")

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve private strategy commit: {exc}") from exc
    return version, commit


def validate_candidates(payload: dict, private_version: str | None = None, private_commit: str | None = None) -> dict:
    if not isinstance(payload, dict):
        raise RuntimeError("candidate snapshot must be an object")
    if payload.get("status") != "ready":
        raise RuntimeError("candidate snapshot status must be ready")
    if payload.get("future_function") is not False:
        raise RuntimeError("candidate snapshot future_function audit failed")

    as_of = str(payload.get("as_of") or "")
    as_of_day = _date_only(as_of, "as_of")
    if not as_of.endswith("T18:00:00+08:00"):
        raise RuntimeError("as_of must use the production 18:00 Asia/Shanghai timestamp")
    window_end = _date_only(payload.get("history_window_end"), "history_window_end")
    window_start = _date_only(payload.get("history_window_start"), "history_window_start")
    if as_of_day != window_end:
        raise RuntimeError("as_of must match history_window_end")
    if window_start > window_end:
        raise RuntimeError("history_window_start must not be after history_window_end")

    files_used = payload.get("history_files_used")
    if not isinstance(files_used, int) or isinstance(files_used, bool) or files_used < 126:
        raise RuntimeError("history_files_used must be an integer >= 126")

    strategy_source = payload.get("strategy_source")
    if strategy_source != "Aquant-Private/main":
        raise RuntimeError("strategy_source must be Aquant-Private/main")

    strategy_version = str(payload.get("strategy_version") or "")
    strategy_commit = str(payload.get("strategy_commit") or "")
    if not strategy_version or not strategy_commit:
        raise RuntimeError("strategy version/commit provenance is missing")
    if private_version is not None and strategy_version != private_version:
        raise RuntimeError(
            f"candidate strategy version mismatch: snapshot={strategy_version}, private={private_version}"
        )
    if private_commit is not None and strategy_commit != private_commit:
        raise RuntimeError(
            f"candidate strategy commit mismatch: snapshot={strategy_commit}, private={private_commit}"
        )

    market_scope = str(payload.get("market_scope") or "")
    universe = str(payload.get("universe") or "")
    if "沪深主板" not in market_scope or "排除" not in universe:
        raise RuntimeError("candidate market scope/universe metadata is incomplete")

    weights = payload.get("factor_weights")
    if not isinstance(weights, dict) or set(weights) != EXPECTED_WEIGHTS:
        raise RuntimeError(f"factor_weights must contain exactly {sorted(EXPECTED_WEIGHTS)}")
    weight_sum = sum(_finite(weights[key], f"factor_weights.{key}") for key in EXPECTED_WEIGHTS)
    if abs(weight_sum - 1.0) > 1e-9:
        raise RuntimeError(f"factor weights must sum to 1.0, got {weight_sum}")
    expected_weights = {"momentum": 0.35, "liquidity": 0.15, "value": 0.30, "safety": 0.20}
    for key, expected in expected_weights.items():
        actual = _finite(weights[key], f"factor_weights.{key}")
        if abs(actual - expected) > 1e-9:
            raise RuntimeError(
                f"factor_weights.{key} mismatch: expected={expected}, got={actual}"
            )

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("candidates must be a list")
    if len(candidates) > MAX_CANDIDATES:
        raise RuntimeError(f"candidate count exceeds production maximum {MAX_CANDIDATES}")

    symbols: list[str] = []
    previous_score = math.inf
    for expected_rank, row in enumerate(candidates, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(f"candidate #{expected_rank} must be an object")
        missing = REQUIRED_CANDIDATE_FIELDS - set(row)
        if missing:
            raise RuntimeError(f"candidate #{expected_rank} missing fields: {sorted(missing)}")

        rank = row["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank != expected_rank:
            raise RuntimeError(f"candidate #{expected_rank} has invalid rank: {rank!r}")

        symbol = str(row["symbol"]).zfill(6)
        if not is_main_board_symbol(symbol):
            raise RuntimeError(f"candidate {symbol} is outside the production main-board scope")
        symbols.append(symbol)

        name = str(row["name"] or "").strip()
        if not name:
            raise RuntimeError(f"candidate {symbol} has empty name")

        price = _finite(row["price"], f"{symbol}.price")
        if price <= 0:
            raise RuntimeError(f"candidate {symbol} price must be positive")
        amount = _finite(row["amount"], f"{symbol}.amount")
        if amount < 0:
            raise RuntimeError(f"candidate {symbol} amount must be non-negative")
        score = _finite(row["score"], f"{symbol}.score")
        if not 0 <= score <= 100:
            raise RuntimeError(f"candidate {symbol} score must be within 0..100")
        if score > previous_score + 1e-9:
            raise RuntimeError("candidates are not sorted by descending score")
        previous_score = score

    if len(symbols) != len(set(symbols)):
        raise RuntimeError("candidate symbols contain duplicates")

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("diagnostics metadata is missing")
    reported_count = diagnostics.get("candidate_count")
    if reported_count != len(candidates):
        raise RuntimeError("diagnostics.candidate_count does not match candidates length")

    admission_policy = payload.get("candidate_admission_policy")
    if admission_policy != "top_score_3_max":
        raise RuntimeError("unexpected candidate admission policy")

    audit = payload.get("audit")
    if not isinstance(audit, dict):
        raise RuntimeError("candidate audit metadata is missing")
    if audit.get("hard_eligibility_applied_before_scoring") is not True:
        raise RuntimeError("hard eligibility audit failed")
    if audit.get("strategy_source_locked_to_private") is not True:
        raise RuntimeError("private strategy lock audit failed")
    if audit.get("top_n_is_not_a_score_threshold") is not True:
        raise RuntimeError("top-N admission audit failed")

    return {
        "status": "pass",
        "as_of": as_of,
        "candidate_count": len(candidates),
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "history_files_used": files_used,
        "max_candidates": MAX_CANDIDATES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    private_version, private_commit = _private_provenance()
    result = validate_candidates(payload, private_version, private_commit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
