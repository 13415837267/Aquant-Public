"""Summarize a paper replay report for research/UI consumption.

No trading side effects. Computes only descriptive performance statistics from
an already generated paper replay equity curve.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


class ReportError(ValueError):
    pass


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("mode") != "paper_replay":
        raise ReportError("report mode must be paper_replay")
    curve = report.get("equity_curve")
    if not isinstance(curve, list) or not curve:
        raise ReportError("equity_curve must be a non-empty list")

    equities = [float(row["equity"]) for row in curve]
    if any((not math.isfinite(x) or x <= 0) for x in equities):
        raise ReportError("equity curve contains invalid values")

    start = equities[0]
    end = equities[-1]
    peak = start
    max_drawdown = 0.0
    for equity in equities:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)

    cycle_returns = [
        equities[i] / equities[i - 1] - 1.0
        for i in range(1, len(equities))
    ]
    positive = sum(r > 0 for r in cycle_returns)
    negative = sum(r < 0 for r in cycle_returns)

    return {
        "schema_version": 1,
        "mode": "paper_replay_summary",
        "status": "complete",
        "cycle_count": len(curve),
        "start_date": curve[0]["execution_date"],
        "end_date": curve[-1]["execution_date"],
        "start_equity": round(start, 6),
        "end_equity": round(end, 6),
        "total_return": round(end / start - 1.0, 8),
        "max_drawdown": round(max_drawdown, 8),
        "positive_cycles": positive,
        "negative_cycles": negative,
        "audit": {
            "descriptive_only": True,
            "broker_api_used": False,
            "live_order_submission": False,
            "future_function": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper replay")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = summarize(report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
