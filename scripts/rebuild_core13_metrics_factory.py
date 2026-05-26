from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "data" / "metrics_factory"
REPORTS_DIR = ROOT / "data" / "reports"
PERIODS = (5, 20, 60, 120)
METRIC_NAMES = (
    "TotalReturn",
    "AnnualizedReturn",
    "AverageDailyReturn",
    "MedianDailyReturn",
    "Volatility",
    "AnnualizedVolatility",
    "MaxGain",
    "MaxLoss",
    "ReturnRange",
    "MeanAbsoluteDeviation",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric_frame(log_return: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for days in PERIODS:
        rolling = log_return.rolling(days, min_periods=2)
        total_return = rolling.sum()
        annualized_return = annualized_return_frame(total_return, days)
        average_daily_return = rolling.mean()
        median_daily_return = rolling.median()
        volatility = rolling.std(ddof=1)
        annualized_volatility = volatility * np.sqrt(252.0)
        max_gain = rolling.max()
        max_loss = rolling.min()
        return_range = max_gain - max_loss
        mean_absolute_deviation = log_return.rolling(days, min_periods=2).apply(
            lambda values: float(np.nanmean(np.abs(values - np.nanmean(values)))),
            raw=True,
        )
        values = {
            "TotalReturn": total_return,
            "AnnualizedReturn": annualized_return,
            "AverageDailyReturn": average_daily_return,
            "MedianDailyReturn": median_daily_return,
            "Volatility": volatility,
            "AnnualizedVolatility": annualized_volatility,
            "MaxGain": max_gain,
            "MaxLoss": max_loss,
            "ReturnRange": return_range,
            "MeanAbsoluteDeviation": mean_absolute_deviation,
        }
        for name in METRIC_NAMES:
            frame = values[name].copy()
            frame.index.name = "date"
            frame.columns.name = "ts_code"
            frames.append(frame.stack().rename(f"{name}:{days}d"))
    return pd.concat(frames, axis=1).reset_index()


def annualized_return_frame(total_return: pd.DataFrame, days: int) -> pd.DataFrame:
    result = pd.DataFrame(np.nan, index=total_return.index, columns=total_return.columns, dtype=float)
    index = pd.DatetimeIndex(total_return.index)
    for position in range(days, len(index)):
        nature_days = max((pd.Timestamp(index[position]) - pd.Timestamp(index[position - days])).days, 1)
        result.iloc[position] = total_return.iloc[position] / nature_days * 365.0
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Core-13 MetricsFactory-compatible features from adj_nav returns.")
    parser.add_argument("--wide-log-return", default=str(PROCESSED_DIR / "core13_wide_log_return.parquet"))
    parser.add_argument("--output", default=str(METRICS_DIR / "core13_all_metrics_features.parquet"))
    parser.add_argument("--manifest-output", default=str(REPORTS_DIR / "core13_metrics_factory_manifest.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    wide_log_return_path = Path(args.wide_log_return)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest_output)
    if not wide_log_return_path.exists():
        raise FileNotFoundError(f"CORE13_WIDE_LOG_RETURN_MISSING: {wide_log_return_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    log_return = pd.read_parquet(wide_log_return_path).copy()
    log_return.index = pd.DatetimeIndex(pd.to_datetime(log_return.index))
    features = metric_frame(log_return)
    features.to_parquet(output_path, index=False)
    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": "core13_wide_log_return",
        "wide_log_return_path": str(wide_log_return_path),
        "output_path": str(output_path),
        "row_count": int(len(features)),
        "feature_count": int(len([column for column in features.columns if column not in {"date", "ts_code"}])),
        "periods": list(PERIODS),
        "metric_names": list(METRIC_NAMES),
        "return_source": "adj_nav",
        "files": {
            project_relative(wide_log_return_path): file_sha256(wide_log_return_path),
            project_relative(output_path): file_sha256(output_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output_path={output_path}")
    print(f"manifest_path={manifest_path}")
    print(f"row_count={manifest['row_count']}")
    print(f"feature_count={manifest['feature_count']}")


if __name__ == "__main__":
    main()
