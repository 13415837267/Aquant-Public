from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "history"
STATE = HISTORY / "_BACKFILL_STATE.json"
COMPLETE = HISTORY / "_BACKFILL_COMPLETE"
TZ = ZoneInfo("Asia/Shanghai")
PATTERN = re.compile(r"data/history/(\d{4})/(\d{4}-\d{2}-\d{2})\.csv\.gz$")


def git_history_files() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "data/history"],
        text=True,
    )
    return [line.strip() for line in raw.splitlines() if PATTERN.fullmatch(line.strip())]


def main() -> None:
    files = git_history_files()
    matches = [PATTERN.fullmatch(path) for path in files]
    dates = sorted({m.group(2) for m in matches if m}, reverse=True)
    if not dates:
        raise RuntimeError("no historical daily files found")
    years = Counter(m.group(1) for m in matches if m)
    expected_years = set(str(y) for y in range(2015, 2027))
    missing_years = sorted(expected_years - set(years))
    if missing_years:
        raise RuntimeError(f"historical database missing years: {missing_years}")

    now = datetime.now(TZ).isoformat()
    payload = {
        "status": "complete",
        "completed_at": now,
        "target_start": dates[-1],
        "end": dates[0],
        "database_scope": "DATABASE_STOCK_ROWS",
        "candidate_scope": "CN_A_MAINBOARD",
        "years": sorted(int(year) for year in years),
        "year_count": len(years),
        "trading_days": len(dates),
        "days_completed": len(dates),
        "files_by_year": {year: years[year] for year in sorted(years)},
        "source": "zzshare daily bulk + daily valuation",
        "format": "daily CSV gzip",
    }
    HISTORY.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    COMPLETE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
