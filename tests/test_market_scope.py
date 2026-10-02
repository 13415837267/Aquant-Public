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
