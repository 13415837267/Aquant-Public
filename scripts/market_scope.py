"""Shared market-scope rules for the production quant system."""
from __future__ import annotations

import re

SH_MAIN_BOARD_PREFIXES = ("600", "601", "603", "605")
SZ_MAIN_BOARD_PREFIXES = ("000", "001", "002", "003")
MAIN_BOARD_PREFIXES = SH_MAIN_BOARD_PREFIXES + SZ_MAIN_BOARD_PREFIXES

_TS_CODE_RE = re.compile(r"^(\d{6})(?:\.(SH|SZ|BJ))?$", re.IGNORECASE)


def is_main_board_symbol(value: object) -> bool:
    """Return True only for Shanghai/Shenzhen A-share main-board symbols."""
    if value is None:
        return False
    code = str(value).strip().upper()
    match = _TS_CODE_RE.fullmatch(code)
    if not match:
        return False

    prefix, exchange = match.groups()
    if prefix.startswith(SH_MAIN_BOARD_PREFIXES):
        return exchange in (None, "SH")
    if prefix.startswith(SZ_MAIN_BOARD_PREFIXES):
        return exchange in (None, "SZ")
    return False
