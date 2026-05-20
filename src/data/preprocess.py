from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import DataContractError


REQUIRED_PANEL_COLUMNS = {"trade_date", "ts_code", "open", "high", "low", "close", "vol"}
WIDE_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "log_return",
    "amount",
    "vol",
    "turnover_rate",
)


def validate_panel_schema(panel: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PANEL_COLUMNS - set(panel.columns))
    if missing:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: missing panel columns {missing}",
        )


def build_return_fields(panel: pd.DataFrame) -> pd.DataFrame:
    validate_panel_schema(panel)
    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    result["pre_close"] = result.groupby("ts_code", sort=False)["close"].shift(1)
    result["pct_chg"] = result["close"] / result["pre_close"] - 1.0
    result["log_return"] = np.log(result["close"] / result["pre_close"])
    return result


def detect_amount_proxy(panel: pd.DataFrame, rtol: float = 1.0e-10, atol: float = 1.0e-8) -> tuple[pd.DataFrame, bool]:
    validate_panel_schema(panel)
    result = panel.copy()
    proxy_amount = result["close"] * result["vol"] * 100.0
    if "amount" not in result.columns or result["amount"].isna().all():
        result["amount"] = proxy_amount
        return result, True

    comparable = result["amount"].notna() & proxy_amount.notna()
    is_proxy = bool(np.allclose(result.loc[comparable, "amount"], proxy_amount.loc[comparable], rtol=rtol, atol=atol))
    return result, is_proxy


def panel_to_wide_tables(panel: pd.DataFrame, asset_order: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    validate_panel_schema(panel)
    enriched = build_return_fields(panel)
    enriched, _ = detect_amount_proxy(enriched)
    if "turnover_rate" not in enriched.columns:
        enriched["turnover_rate"] = np.nan

    enriched["trade_date"] = pd.to_datetime(enriched["trade_date"])
    if asset_order is None:
        asset_order = list(dict.fromkeys(enriched.sort_values(["trade_date", "ts_code"])["ts_code"]))
    date_index = pd.DatetimeIndex(sorted(enriched["trade_date"].dropna().unique()))

    wide: dict[str, pd.DataFrame] = {}
    for field in WIDE_FIELDS:
        table = enriched.pivot(index="trade_date", columns="ts_code", values=field)
        table.index = pd.to_datetime(table.index)
        wide[field] = table.reindex(index=date_index, columns=list(asset_order)).sort_index()
    return wide
