"""Shared market-scope rules for the production quant system."""
from __future__ import annotations

import re

_SH_MAIN_BOARD_PREFIXES = ("600", "601", "603", "605")
_SZ_MAIN_BOARD_MIN = 1
_SZ_MAIN_BOARD_MAX = 4999
_SZ_MAIN_BOARD_CDR_MIN = 1001
_SZ_MAIN_BOARD_CDR_MAX = 1199

_TS_CODE_RE = re.compile(r"^(\d{6})(?:\.(SH|SZ|BJ))?$", re.IGNORECASE)


def is_main_board_symbol(value: object) -> bool:
    """Return True only for Shanghai/Shenzhen main-board A-share stocks.

    Shenzhen's 001001-001199 range is reserved for main-board CDRs, so it is
    deliberately excluded from the stock candidate universe.
    """
    if value is None:
        return False

    code = str(value).strip().upper()
    match = _TS_CODE_RE.fullmatch(code)
    if not match:
        return False

    digits, exchange = match.groups()
    number = int(digits)

    is_sh_main = digits.startswith(_SH_MAIN_BOARD_PREFIXES)
    is_sz_main = (
        _SZ_MAIN_BOARD_MIN <= number <= _SZ_MAIN_BOARD_MAX
        and not (_SZ_MAIN_BOARD_CDR_MIN <= number <= _SZ_MAIN_BOARD_CDR_MAX)
    )

    if exchange == "SH":
        return is_sh_main
    if exchange == "SZ":
        return is_sz_main
    if exchange is None:
        # The candidate builder normalizes ts_code to a bare six-digit symbol.
        return is_sh_main or is_sz_main

    return False
