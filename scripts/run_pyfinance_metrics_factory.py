from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path("/Users/chenjunming/Desktop/DRL_PM")
PYFINANCE_ROOT = ROOT / "vendor" / "pyfinance"
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "metrics_factory"
REPORTS_DIR = ROOT / "data" / "reports"

sys.path.insert(0, str(PYFINANCE_ROOT))

from MetricsFactory.metrics_factory import compute_metrics_for_period_initialize  # noqa: E402
from MetricsFactory.metrics_cal_config import create_rolling_metrics_map  # noqa: E402
from MetricsFactory.rolling_metrics_cal import CalRollingMetrics  # noqa: E402


def load_wide(name: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"wide_{name}.parquet")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def as_c_array(df: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(df.to_numpy(dtype=np.float64, copy=True))


def run_period_metrics(wide: dict[str, pd.DataFrame], funds: list[str]) -> None:
    log_return = wide["log_return"].copy()
    for code in funds:
        first_valid = log_return[code].first_valid_index()
        if first_valid is None:
            continue
        valid_price_mask = wide["close"][code].notna() & (log_return.index >= first_valid)
        log_return.loc[valid_price_mask, code] = log_return.loc[valid_price_mask, code].fillna(0.0)
    compute_metrics_for_period_initialize(
        log_return,
        wide["close"],
        wide["high"],
        wide["low"],
        wide["vol"],
        str(OUT_DIR),
        p_list=None,
        metrics_list=None,
        fund_list=funds,
        spec_end_date=None,
        num_workers=min(8, os.cpu_count() or 1),
        multi_process=True,
        min_data_required=2,
    )


def run_rolling_metrics(wide: dict[str, pd.DataFrame], funds: list[str]) -> pd.DataFrame:
    rolling_map = create_rolling_metrics_map()
    dates = pd.to_datetime(wide["close"].index.to_numpy())
    funds_array = np.array(funds)
    close = as_c_array(wide["close"][funds])
    open_ = as_c_array(wide["open"][funds])
    high = as_c_array(wide["high"][funds])
    low = as_c_array(wide["low"][funds])
    vol = as_c_array(wide["vol"][funds])

    final_df: pd.DataFrame | None = None
    for window in tqdm(list(rolling_map.keys()), desc="PyFinance rolling metrics"):
        calculator = CalRollingMetrics(
            funds_array,
            close,
            open_,
            high,
            low,
            vol,
            int(window),
            dates,
        )
        sub_df = calculator.cal_all_metrics(rolling_map[window])
        if sub_df is None or sub_df.empty:
            continue
        final_df = sub_df if final_df is None else final_df.merge(sub_df, on=["date", "ts_code"], how="inner")
    if final_df is None:
        raise RuntimeError("PyFinance rolling metrics returned no data.")
    final_df.to_parquet(OUT_DIR / "rolling_metrics.parquet")
    return final_df


def combine_all_metrics() -> pd.DataFrame:
    files = sorted(
        p for p in OUT_DIR.glob("*.parquet")
        if p.name != "all_metrics_features.parquet"
    )
    frames = []
    for path in files:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on=["date", "ts_code"], how="outer")
    merged = merged.sort_values(["date", "ts_code"]).reset_index(drop=True)
    merged.to_parquet(OUT_DIR / "all_metrics_features.parquet")
    return merged


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    wide = {
        "log_return": load_wide("log_return"),
        "open": load_wide("open"),
        "close": load_wide("close"),
        "high": load_wide("high"),
        "low": load_wide("low"),
        "vol": load_wide("vol"),
    }
    funds = list(wide["close"].columns)

    run_period_metrics(wide, funds)
    rolling = run_rolling_metrics(wide, funds)
    all_metrics = combine_all_metrics()

    period_files = sorted(
        p.name for p in OUT_DIR.glob("*.parquet")
        if p.name not in {"rolling_metrics.parquet", "all_metrics_features.parquet"}
    )
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "PyFinance MetricsFactory with C-contiguous rolling input compatibility wrapper",
        "pyfinance_root": str(PYFINANCE_ROOT),
        "asset_count": len(funds),
        "period_files": period_files,
        "rolling_rows": int(len(rolling)),
        "all_metrics_rows": int(len(all_metrics)),
        "all_metrics_columns": int(len(all_metrics.columns)),
        "feature_columns": int(len([c for c in all_metrics.columns if c not in {"date", "ts_code"}])),
        "uses_future_data": False,
        "historical_window_ending_at_feature_date": True,
        "notes": [
            "Rolling metrics call CalRollingMetrics directly because pandas 3 returns read-only Fortran arrays via DataFrame.values.",
            "Input arrays are converted to writable C-contiguous float64 arrays before calling numba-backed indicators.",
        ],
    }
    (REPORTS_DIR / "metrics_factory_all_features_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[done] PyFinance MetricsFactory")
    print(f"rolling_shape={rolling.shape}")
    print(f"all_metrics_shape={all_metrics.shape}")


if __name__ == "__main__":
    main()
