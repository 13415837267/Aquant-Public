import pandas as pd

from scripts.market_scope import is_main_board_symbol


def test_main_board_symbols():
    assert is_main_board_symbol("000001.SZ")
    assert is_main_board_symbol("002594.SZ")
    assert is_main_board_symbol("600519.SH")
    assert is_main_board_symbol("605499.SH")
    assert is_main_board_symbol("000001")


def test_non_main_board_symbols():
    assert not is_main_board_symbol("300750.SZ")
    assert not is_main_board_symbol("301001.SZ")
    assert not is_main_board_symbol("688981.SH")
    assert not is_main_board_symbol("830799.BJ")
    assert not is_main_board_symbol("900901.SH")

def test_pit_fundamentals_excludes_future_publications_and_keeps_latest_available(tmp_path, monkeypatch):
    import gzip
    from pathlib import Path
    from scripts import pit_fundamentals as pit

    root = tmp_path / "fundamentals"
    table_dir = root / "indicator"
    table_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "report_date": "2025-12-31", "pub_date": "2026-02-01", "roe": 10.0},
            {"symbol": "000001.SZ", "report_date": "2026-03-31", "pub_date": "2026-04-30", "roe": 12.0},
            {"symbol": "000002.SZ", "report_date": "2026-03-31", "pub_date": "2026-10-02", "roe": 20.0},
        ]
    )
    path = table_dir / "2026q1.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        frame.to_csv(fh, index=False)

    monkeypatch.setattr(pit, "FUNDAMENTALS", Path(root))
    result = pit.load_table_asof("indicator", "2026-09-30")

    assert result["symbol"].tolist() == ["000001"]
    assert float(result.iloc[0]["roe"]) == 12.0
    assert result.iloc[0]["pub_date"].strftime("%Y-%m-%d") == "2026-04-30"


def test_pit_snapshot_preserves_publication_dates(tmp_path, monkeypatch):
    import gzip
    from pathlib import Path
    from scripts import pit_fundamentals as pit

    root = tmp_path / "fundamentals"
    for table, value in [("indicator", 12.0), ("income", 100.0)]:
        table_dir = root / table
        table_dir.mkdir(parents=True)
        frame = pd.DataFrame(
            [{"symbol": "000001.SZ", "report_date": "2026-03-31", "pub_date": "2026-04-30", "value": value}]
        )
        with gzip.open(table_dir / "2026q1.csv.gz", "wt", encoding="utf-8") as fh:
            frame.to_csv(fh, index=False)

    monkeypatch.setattr(pit, "FUNDAMENTALS", Path(root))
    result = pit.build_pit_snapshot("2026-09-30")
    pit.validate_pit_frame(result, "2026-09-30")

    assert "indicator_pub_date" in result.columns
    assert "income_pub_date" in result.columns
    assert result.iloc[0]["indicator_pub_date"].strftime("%Y-%m-%d") == "2026-04-30"

