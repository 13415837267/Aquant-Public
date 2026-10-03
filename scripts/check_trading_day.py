"""Check whether the current Shanghai date is an A-share trading day."""
from __future__ import annotations
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backfill_history import api_client, load_trade_days

def main() -> int:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    day = today.strftime("%Y%m%d")
    start = (today - timedelta(days=7)).strftime("%Y%m%d")
    end = (today + timedelta(days=7)).strftime("%Y%m%d")
    trading_days = set(load_trade_days(api_client(), start=start, end=end))
    is_trading = day in trading_days
    print(f"TRADE_DATE={day}")
    print(f"IS_TRADING_DAY={'true' if is_trading else 'false'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
