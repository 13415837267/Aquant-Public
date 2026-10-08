"""检查当前北京时间对应的A股交易日。"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_history import api_client, load_trade_days, request_bulk_day


def main() -> int:
    tz = ZoneInfo("Asia/Shanghai")
    today = datetime.now(tz).date()
    day = today.strftime("%Y%m%d")
    start = (today - timedelta(days=7)).strftime("%Y%m%d")
    end = (today + timedelta(days=7)).strftime("%Y%m%d")

    api = api_client()
    trading_days = set(load_trade_days(api, start=start, end=end))
    is_trading = day in trading_days
    source = "trade_calendar"

    # 交易日历可能存在发布延迟。若日历暂时没有今天，直接检查今天的实际日线数据；
    # 只有拿到有效行情时才将今天认定为交易日。
    if not is_trading:
        try:
            daily = request_bulk_day(api, today.strftime("%Y-%m-%d"))
            is_trading = daily is not None and not daily.empty
            if is_trading:
                source = "daily_market_data_fallback"
        except Exception as exc:
            print(f"TRADE_DAY_FALLBACK_ERROR={type(exc).__name__}: {exc}")

    print(f"TRADE_DATE={day}")
    print(f"IS_TRADING_DAY={'true' if is_trading else 'false'}")
    print(f"DETECTION_SOURCE={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
