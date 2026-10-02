from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "candidates.json"
OUTPUT = ROOT / "data" / "portfolio.json"


def allocate_weights(
    targets: pd.DataFrame,
    max_weight: float = 0.05,
    cash_buffer: float = 0.05,
) -> dict[str, float]:
    """Allocate deterministic inverse-volatility target weights.

    This pure function is shared by production portfolio construction and
    execution-constrained historical backtests so both layers use identical
    position sizing assumptions.
    """
    rows = targets.copy()
    if rows.empty:
        return {}
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must be in [0, 1)")

    rows["volatility_proxy"] = pd.to_numeric(rows["volatility_proxy"], errors="coerce")
    rows["score"] = pd.to_numeric(rows.get("score", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    rows["symbol"] = rows["symbol"].astype(str).str.zfill(6)
    rows = rows.loc[rows["symbol"].notna() & rows["volatility_proxy"].notna()].copy()
    if rows.empty:
        return {}

    vol = rows["volatility_proxy"].clip(lower=0.5)
    raw = 1.0 / vol.replace(0, np.nan)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0:
        return {}

    investable = 1.0 - cash_buffer
    base_weights = raw / raw.sum() * investable

    active = pd.Series(True, index=rows.index)
    final = pd.Series(0.0, index=rows.index)
    remaining = investable
    while active.any():
        proposal = base_weights.loc[active]
        if proposal.sum() <= 0:
            break
        scaled = proposal / proposal.sum() * remaining
        capped = scaled > max_weight + 1e-12
        if not capped.any():
            final.loc[active] = scaled
            break
        capped_idx = scaled.index[capped]
        final.loc[capped_idx] = max_weight
        remaining -= max_weight * len(capped_idx)
        active.loc[capped_idx] = False
        if remaining <= 1e-12:
            break

    return {
        str(symbol).zfill(6): float(weight)
        for symbol, weight in zip(rows["symbol"], final)
        if float(weight) > 0
    }


def build_portfolio(snapshot: dict, max_weight: float = 0.05, cash_buffer: float = 0.05) -> dict:
    if snapshot.get("status") != "ready":
        raise RuntimeError("candidate snapshot is not ready")
    rows = pd.DataFrame(snapshot.get("candidates", []))
    if rows.empty:
        raise RuntimeError("candidate snapshot has no candidates")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1]")
    if not 0 <= cash_buffer < 1:
        raise ValueError("cash_buffer must be in [0, 1)")

    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
    rows["symbol"] = rows["symbol"].astype(str).str.zfill(6)
    rows = rows.loc[rows["symbol"].notna() & rows["score"].notna()].copy()

    allocation = allocate_weights(
        rows,
        max_weight=max_weight,
        cash_buffer=cash_buffer,
    )
    rows["target_weight"] = rows["symbol"].map(allocation).fillna(0.0)

    positions = [
        {
            "rank": int(row["rank"]),
            "symbol": row["symbol"],
            "name": row["name"],
            "score": float(row["score"]),
            "volatility_proxy": float(row["volatility_proxy"]),
            "target_weight": round(float(row["target_weight"]), 8),
        }
        for _, row in rows.iterrows()
    ]
    return {
        "schema_version": 1,
        "status": "ready",
        "as_of": snapshot["as_of"],
        "strategy_source": snapshot["strategy_source"],
        "strategy_version": snapshot["strategy_version"],
        "strategy_commit": snapshot["strategy_commit"],
        "construction": "inverse_volatility_with_position_cap",
        "max_position_weight": max_weight,
        "cash_buffer": cash_buffer,
        "gross_target_weight": round(sum(p["target_weight"] for p in positions), 8),
        "position_count": len(positions),
        "positions": positions,
        "audit": {
            "long_only": True,
            "leverage": False,
            "future_function": False,
            "source_candidates": len(snapshot["candidates"]),
        },
    }


def main() -> None:
    snapshot = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    payload = build_portfolio(snapshot)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "position_count": payload["position_count"],
        "gross_target_weight": payload["gross_target_weight"],
        "cash_buffer": payload["cash_buffer"],
        "strategy_commit": payload["strategy_commit"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
