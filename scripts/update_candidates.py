from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORY_DIR = DATA_DIR / "history"
DATA_FILE = DATA_DIR / "candidates.json"
STRATEGY_VERSION = "0.2.0"

WEIGHTS = {"momentum": 0.30, "liquidity": 0.20, "value": 0.20, "risk": 0.15, "activity": 0.15}
REQUIRED_COLUMNS = {"symbol","date","close","pre_close","volume","amount","pct_chg","turnover_pct","amplitude_pct","is_paused","is_st","market_cap","circulating_market_cap","pe_ratio","pb_ratio"}

def _pct_rank(s: pd.Series, ascending: bool = True) -> pd.Series:
    clean = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return clean.rank(pct=True, ascending=ascending, method="average").fillna(0.5)

def _history_files() -> list[Path]:
    files = sorted(HISTORY_DIR.glob("*.csv.gz"), reverse=True)
    if not files:
        raise RuntimeError("No historical daily files found in data/history")
    return files

def _read_history_window(max_files: int = 61) -> tuple[pd.DataFrame, list[str]]:
    files = _history_files()[:max_files]
    frames, dates = [], []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                frame = pd.read_csv(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {path.name}: {exc}") from exc
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} missing required columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        dates.append(path.name[:10])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    return data.sort_values(["symbol", "date"]), dates

def build_candidates(history: pd.DataFrame) -> dict:
    latest_date = history["date"].dropna().max()
    latest = history.loc[history["date"].eq(latest_date)].copy()
    if latest.empty:
        raise RuntimeError("Latest history date has no rows")

    latest["name"] = latest.get("name", latest["symbol"]).astype(str)
    for col in ["close", "amount", "turnover_pct", "is_paused", "is_st"]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    latest["is_paused"] = latest["is_paused"].fillna(0)
    latest["is_st"] = latest["is_st"].fillna(0)

    excluded_name = latest["name"].str.contains(r"ST|退", case=False, na=False)
    usable = (~excluded_name & latest["is_st"].eq(0) & latest["is_paused"].eq(0)
              & latest["close"].gt(2) & latest["amount"].ge(2e7))
    latest = latest.loc[usable].copy()
    if latest.empty:
        raise RuntimeError("No usable stocks after strategy universe filters")

    hist = history.copy()
    numeric_cols = ["close","amount","turnover_pct","pct_chg","market_cap","circulating_market_cap","pe_ratio","pb_ratio"]
    for col in numeric_cols:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.sort_values(["symbol", "date"])
    grouped = hist.groupby("symbol", sort=False)
    hist["ret_1d"] = grouped["close"].pct_change()
    hist["ret_20d"] = grouped["close"].pct_change(20)
    hist["ret_60d"] = grouped["close"].pct_change(60)
    hist["vol_20d"] = (hist.groupby("symbol", sort=False)["ret_1d"].rolling(20, min_periods=10)
                       .std().reset_index(level=0, drop=True).mul(np.sqrt(252) * 100))
    hist["avg_amount_20d"] = (hist.groupby("symbol", sort=False)["amount"].rolling(20, min_periods=10)
                              .mean().reset_index(level=0, drop=True))
    hist["avg_turnover_20d"] = (hist.groupby("symbol", sort=False)["turnover_pct"].rolling(20, min_periods=10)
                                .mean().reset_index(level=0, drop=True))

    features = hist.loc[hist["date"].eq(latest_date),
                        ["symbol","ret_20d","ret_60d","vol_20d","avg_amount_20d","avg_turnover_20d"]].copy()
    out = latest.merge(features, on="symbol", how="left")
    out["momentum_20d"] = out["ret_20d"] * 100
    out["momentum_60d"] = out["ret_60d"] * 100
    out["volatility_20d"] = out["vol_20d"]
    out["avg_amount_20d"] = out["avg_amount_20d"].fillna(out["amount"])
    out["avg_turnover_20d"] = out["avg_turnover_20d"].fillna(out["turnover_pct"])

    out["f_momentum"] = _pct_rank(out["momentum_60d"])
    out["f_liquidity"] = _pct_rank(np.log1p(out["avg_amount_20d"].clip(lower=0)))
    value_base = (0.5 / out["pb_ratio"].where(out["pb_ratio"] > 0)
                  + 0.5 / out["pe_ratio"].where(out["pe_ratio"] > 0))
    out["f_value"] = _pct_rank(value_base)
    out["f_risk"] = _pct_rank(out["volatility_20d"], ascending=False)
    out["f_activity"] = _pct_rank(out["avg_turnover_20d"].clip(lower=0))

    out["score"] = 100 * sum([
        WEIGHTS["momentum"] * out["f_momentum"],
        WEIGHTS["liquidity"] * out["f_liquidity"],
        WEIGHTS["value"] * out["f_value"],
        WEIGHTS["risk"] * out["f_risk"],
        WEIGHTS["activity"] * out["f_activity"],
    ])
    out.loc[out["volatility_20d"] > 60, "score"] -= 5
    out.loc[out["momentum_60d"] < 0, "score"] -= 3
    out.loc[out["ret_20d"] < -0.15, "score"] -= 3
    out["score"] = out["score"].clip(0, 100)

    out = out.sort_values(["score","avg_amount_20d"], ascending=[False, False]).head(30).reset_index(drop=True)
    rows = []
    for idx, row in out.iterrows():
        flags = []
        if pd.notna(row["volatility_20d"]) and row["volatility_20d"] > 60: flags.append("高波动")
        if pd.notna(row["momentum_60d"]) and row["momentum_60d"] < 0: flags.append("60日动量<0")
        if pd.notna(row["ret_20d"]) and row["ret_20d"] < -0.15: flags.append("20日跌幅>15%")
        rows.append({
            "rank": idx + 1, "symbol": row["symbol"], "name": row["name"],
            "price": round(float(row["close"]), 3),
            "change_pct": round(float(row["pct_chg"]), 3) if pd.notna(row["pct_chg"]) else None,
            "momentum_20d": round(float(row["momentum_20d"]), 3) if pd.notna(row["momentum_20d"]) else None,
            "momentum_60d": round(float(row["momentum_60d"]), 3) if pd.notna(row["momentum_60d"]) else None,
            "turnover_pct": round(float(row["turnover_pct"]), 3) if pd.notna(row["turnover_pct"]) else None,
            "avg_turnover_20d": round(float(row["avg_turnover_20d"]), 3) if pd.notna(row["avg_turnover_20d"]) else None,
            "pb": round(float(row["pb_ratio"]), 3) if pd.notna(row["pb_ratio"]) and row["pb_ratio"] > 0 else None,
            "pe": round(float(row["pe_ratio"]), 3) if pd.notna(row["pe_ratio"]) and row["pe_ratio"] > 0 else None,
            "avg_amount_20d": round(float(row["avg_amount_20d"]), 2) if pd.notna(row["avg_amount_20d"]) else None,
            "volatility_20d": round(float(row["volatility_20d"]), 3) if pd.notna(row["volatility_20d"]) else None,
            "score": round(float(row["score"]), 3), "flags": flags,
        })
    return {
        "as_of": f"{latest_date}T18:00:00+08:00", "timezone": "Asia/Shanghai",
        "source": "Aquant-Public data/history", "status": "ready",
        "strategy_version": STRATEGY_VERSION,
        "universe": "沪深京 A 股；排除 ST/退市相关标的、停牌、价格≤2元、最近交易日成交额<2000万元",
        "lookback_trading_days": 60, "candidates": rows, "factor_weights": WEIGHTS,
    }

def main() -> None:
    history, dates = _read_history_window(61)
    snapshot = build_candidates(history)
    previous = None
    if DATA_FILE.exists():
        try: previous = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError: pass
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed": previous != snapshot, "rows": len(snapshot["candidates"]),
                      "as_of": snapshot["as_of"], "source": snapshot["source"],
                      "history_files_used": len(dates), "oldest_file_used": dates[-1]}, ensure_ascii=False))

if __name__ == "__main__":
    main()
