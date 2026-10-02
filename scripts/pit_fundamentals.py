from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FUNDAMENTALS = ROOT / "data" / "fundamentals"
TABLES = ("indicator", "income", "balance", "cash_flow")


def _quarter_files(table: str) -> list[Path]:
    if table not in TABLES:
        raise ValueError(f"unsupported fundamentals table: {table}")
    return sorted((FUNDAMENTALS / table).glob("????q?.csv.gz"))


def _read_files(files: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            frame = pd.read_csv(fh)
        if frame.empty:
            continue
        required = {"symbol", "report_date", "pub_date"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path}: missing PIT columns {sorted(missing)}")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["symbol", "report_date", "pub_date"])
    out = pd.concat(frames, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    out["report_date"] = pd.to_datetime(out["report_date"], errors="coerce")
    out["pub_date"] = pd.to_datetime(out["pub_date"], errors="coerce")
    return out.dropna(subset=["symbol"]).drop_duplicates().reset_index(drop=True)


def load_table_asof(table: str, trade_date: str, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Return the latest publication available on trade_date for each symbol."""
    asof = pd.Timestamp(trade_date)
    raw = _read_files(_quarter_files(table))
    if raw.empty:
        return raw
    raw = raw.loc[raw["pub_date"].notna() & (raw["pub_date"] <= asof)].copy()
    if symbols is not None:
        wanted = {str(s).zfill(6) for s in symbols}
        raw = raw.loc[raw["symbol"].isin(wanted)].copy()
    if raw.empty:
        return raw
    raw = raw.sort_values(["symbol", "pub_date", "report_date"])
    return raw.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def build_pit_snapshot(
    trade_date: str,
    symbols: Iterable[str] | None = None,
    tables: Iterable[str] = TABLES,
) -> pd.DataFrame:
    """Join latest available publications from multiple tables by symbol."""
    result: pd.DataFrame | None = None
    for table in tables:
        frame = load_table_asof(table, trade_date, symbols)
        if frame.empty:
            continue
        prefix = f"{table}_"
        rename = {
            column: f"{prefix}{column}"
            for column in frame.columns
            if column not in {"symbol", "report_date", "pub_date"}
        }
        frame = frame.rename(columns=rename)
        if result is None:
            result = frame
        else:
            frame = frame.drop(columns=["report_date", "pub_date"], errors="ignore")
            result = result.merge(frame, on="symbol", how="outer", validate="one_to_one")
    if result is None:
        return pd.DataFrame(columns=["symbol"])
    return result.sort_values("symbol").reset_index(drop=True)


def validate_pit_frame(frame: pd.DataFrame, trade_date: str) -> None:
    if frame["symbol"].duplicated().any():
        raise ValueError("PIT frame contains duplicate symbols")
    cutoff = pd.Timestamp(trade_date)
    pub_cols = [c for c in frame.columns if c.endswith("_pub_date")]
    for column in pub_cols:
        if pd.to_datetime(frame[column], errors="coerce").gt(cutoff).any():
            raise ValueError(f"future publication detected in {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read persisted fundamentals as a point-in-time snapshot")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--table", action="append", choices=TABLES)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = build_pit_snapshot(args.trade_date, tables=tuple(args.table) if args.table else TABLES)
    validate_pit_frame(frame, args.trade_date)
    summary = {
        "status": "ready",
        "trade_date": args.trade_date,
        "tables": list(tuple(args.table) if args.table else TABLES),
        "symbols": int(len(frame)),
        "future_function": False,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    print(summary)


if __name__ == "__main__":
    main()
