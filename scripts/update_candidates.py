from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "candidates.json"
TZ = ZoneInfo("Asia/Shanghai")
STRATEGY_VERSION = "0.1.0"

WEIGHTS = {
    "momentum": 0.30,
    "liquidity": 0.20,
    "value": 0.20,
    "risk": 0.15,
    "activity": 0.15,
}


def _num(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    clean = s.replace([np.inf, -np.inf], np.nan)
    return clean.rank(pct=True, ascending=ascending, method="average").fillna(0.5)


def _fetch_spot() -> tuple[pd.DataFrame, str]:
    import akshare as ak

    try:
        return ak.stock_zh_a_spot_em(), "AKShare / Eastmoney A-share spot"
    except Exception as first_error:
        frames = [ak.stock_sh_a_spot_em(), ak.stock_sz_a_spot_em(), ak.stock_bj_a_spot_em()]
        if not frames:
            raise RuntimeError(f"A-share spot data fetch failed: {first_error}") from first_error
        return pd.concat(frames, ignore_index=True), "AKShare / Eastmoney exchange spot"


def build_candidates(raw: pd.DataFrame, as_of: str) -> dict:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    code = df.get("代码", pd.Series(index=df.index, dtype="object")).astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    name = df.get("名称", pd.Series(index=df.index, dtype="object")).astype(str)
    price = _num(df, "最新价", "现价")
    change = _num(df, "涨跌幅")
    momentum = _num(df, "60日涨跌幅")
    turnover = _num(df, "换手率")
    pb = _num(df, "市净率", "PB")
    pe = _num(df, "市盈率-动态", "市盈率(TTM)", "市盈率")
    amount = _num(df, "成交额")
    high = _num(df, "最高")
    low = _num(df, "最低")
    prev_close = _num(df, "昨收")
    volume_ratio = _num(df, "量比")

    out = pd.DataFrame({
        "symbol": code,
        "name": name,
        "price": price,
        "change_pct": change,
        "momentum_60d": momentum,
        "turnover_pct": turnover,
        "pb": pb,
        "pe": pe,
        "amount": amount,
        "volatility_proxy": ((high - low) / prev_close.replace(0, np.nan) * 100),
        "volume_ratio": volume_ratio,
    })

    excluded = out["name"].str.contains(r"ST|退", na=False)
    liquid = out["amount"].fillna(0).ge(2e7)
    usable_price = out["price"].gt(2)
    out = out.loc[~excluded & liquid & usable_price].copy()
    if out.empty:
        raise RuntimeError("No usable A-share rows after universe filters")

    out["f_momentum"] = _pct_rank(out["momentum_60d"])
    out["f_liquidity"] = _pct_rank(np.log1p(out["amount"].clip(lower=0)))
    value_base = ((1 / out["pb"].where(out["pb"] > 0)) + (1 / out["pe"].where(out["pe"] > 0))) / 2
    out["f_value"] = _pct_rank(value_base)
    out["f_risk"] = _pct_rank(out["volatility_proxy"], ascending=False)
    out["f_activity"] = _pct_rank(out["volume_ratio"].fillna(1) * out["turnover_pct"].fillna(0).clip(lower=0))

    out["score"] = 100 * (
        WEIGHTS["momentum"] * out["f_momentum"]
        + WEIGHTS["liquidity"] * out["f_liquidity"]
        + WEIGHTS["value"] * out["f_value"]
        + WEIGHTS["risk"] * out["f_risk"]
        + WEIGHTS["activity"] * out["f_activity"]
    )
    out["score"] -= np.where(out["volatility_proxy"] > 8, 5, 0)
    out["score"] -= np.where(out["change_pct"] < -7, 3, 0)
    out["score"] = out["score"].clip(0, 100)
    out = out.sort_values(["score", "amount"], ascending=[False, False]).head(30).reset_index(drop=True)

    rows = []
    for idx, row in out.iterrows():
        flags = []
        if row["volatility_proxy"] > 8:
            flags.append("高波动")
        if row["change_pct"] < -7:
            flags.append("单日大跌")
        if row["momentum_60d"] < 0:
            flags.append("60日动量<0")
        rows.append({
            "rank": idx + 1,
            "symbol": row["symbol"],
            "name": row["name"],
            "price": round(float(row["price"]), 3),
            "change_pct": round(float(row["change_pct"]), 3) if pd.notna(row["change_pct"]) else 0,
            "momentum_60d": round(float(row["momentum_60d"]), 3) if pd.notna(row["momentum_60d"]) else 0,
            "turnover_pct": round(float(row["turnover_pct"]), 3) if pd.notna(row["turnover_pct"]) else 0,
            "pb": round(float(row["pb"]), 3) if pd.notna(row["pb"]) and row["pb"] > 0 else None,
            "pe": round(float(row["pe"]), 3) if pd.notna(row["pe"]) and row["pe"] > 0 else None,
            "amount": round(float(row["amount"]), 2) if pd.notna(row["amount"]) else 0,
            "volatility_proxy": round(float(row["volatility_proxy"]), 3) if pd.notna(row["volatility_proxy"]) else 0,
            "score": round(float(row["score"]), 3),
            "flags": flags,
        })

    return {
        "as_of": as_of,
        "timezone": "Asia/Shanghai",
        "source": "AKShare / Eastmoney A-share spot",
        "status": "ready",
        "strategy_version": STRATEGY_VERSION,
        "universe": "沪深京 A 股；排除 ST/退市相关标的、价格≤2元、成交额<2000万元",
        "candidates": rows,
        "factor_weights": WEIGHTS,
    }


def main() -> None:
    import akshare as ak

    today = datetime.now(TZ).date()
    try:
        trade_dates = set(pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).dt.date.tolist())
        if today not in trade_dates:
            print(json.dumps({"changed": False, "skipped": True, "reason": "non-trading-day", "date": str(today)}, ensure_ascii=False))
            return
    except Exception as exc:
        print(f"trade calendar unavailable, continuing: {exc}")

    as_of = f"{today.isoformat()}T18:00:00+08:00"
    raw, source = _fetch_spot()
    snapshot = build_candidates(raw, as_of)
    snapshot["source"] = source

    previous = None
    if DATA_FILE.exists():
        try:
            previous = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "changed": previous != snapshot,
        "rows": len(snapshot["candidates"]),
        "as_of": as_of,
        "source": source,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
