from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METRICS_DIR = DATA_DIR / "metrics_factory"
REPORTS_DIR = DATA_DIR / "reports"
DEFAULT_UNIVERSE_PATH = ROOT / "configs" / "data" / "etf_lof_universe.yaml"

START_DATE = "2005-02-23"
END_DATE = "2026-05-09"


@dataclass(frozen=True)
class Asset:
    ts_code: str
    symbol: str
    name: str
    asset_type: str
    pool: str

    @property
    def market_symbol(self) -> str:
        prefix = "sh" if self.ts_code.endswith(".SH") else "sz"
        return f"{prefix}{self.symbol}"


def load_universe(path: Path) -> tuple[list[Asset], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"UNIVERSE_CONFIG_NOT_FOUND: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = payload.get("assets")
    if not isinstance(records, list) or not records:
        raise ValueError("UNIVERSE_CONFIG_INVALID: assets must be a non-empty list")

    assets: list[Asset] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"UNIVERSE_ASSET_INVALID: assets[{index}] must be a mapping")
        ts_code = str(record.get("ts_code", "")).strip()
        if not ts_code:
            raise ValueError(f"UNIVERSE_ASSET_INVALID: assets[{index}].ts_code is required")
        if not (ts_code.endswith(".SH") or ts_code.endswith(".SZ")):
            raise ValueError(f"UNIVERSE_ASSET_INVALID: unsupported ts_code suffix {ts_code}")
        if ts_code in seen:
            raise ValueError(f"UNIVERSE_ASSET_DUPLICATE: {ts_code}")
        seen.add(ts_code)
        symbol = str(record.get("symbol") or ts_code.split(".", maxsplit=1)[0]).strip()
        name = str(record.get("name") or ts_code).strip()
        asset_type = str(record.get("asset_type") or record.get("type") or "unknown").strip()
        pool = str(record.get("pool") or "unknown").strip()
        assets.append(Asset(ts_code=ts_code, symbol=symbol, name=name, asset_type=asset_type, pool=pool))
    return assets, payload


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, METRICS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def date_chunks(start: str, end: str, days: int = 700) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=days), end_ts)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_tencent(asset: Asset, start_date: str, end_date: str) -> pd.DataFrame:
    import requests

    session = requests.Session()
    rows: list[list[str]] = []
    for start, end in date_chunks(start_date, end_date):
        param = (
            f"{asset.market_symbol},day,"
            f"{start.strftime('%Y-%m-%d')},{end.strftime('%Y-%m-%d')},640,qfq"
        )
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = session.get(
                    url,
                    params={"param": param},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", {}).get(asset.market_symbol, {})
                arr = data.get("qfqday") or data.get("day") or []
                rows.extend(arr)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.5 * attempt)
        else:
            raise RuntimeError(f"FETCH_FAILED {asset.ts_code}: {last_error}")
        time.sleep(0.15)

    if not rows:
        raise RuntimeError(f"NO_DATA {asset.ts_code}")

    df = pd.DataFrame(rows, columns=["trade_date", "open", "close", "high", "low", "vol"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ["open", "close", "high", "low", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date", "close"])
    df = df.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    df["ts_code"] = asset.ts_code
    df["symbol"] = asset.symbol
    df["asset_name"] = asset.name
    df["asset_type"] = asset.asset_type
    df["pool"] = asset.pool
    df["pre_close"] = df["close"].shift(1)
    df["change"] = df["close"] - df["pre_close"]
    df["pct_chg"] = df["close"] / df["pre_close"] - 1.0
    df["log_return"] = np.log(df["close"] / df["pre_close"])
    df["pct_chg_percent_raw"] = df["pct_chg"] * 100.0
    df["amplitude"] = (df["high"] - df["low"]) / df["pre_close"] * 100.0
    # Tencent volume is reported in lots for A-share style quotes. Amount is a proxy.
    df["amount"] = df["close"] * df["vol"] * 100.0
    df["turnover_rate"] = np.nan
    ordered = [
        "ts_code",
        "symbol",
        "asset_name",
        "asset_type",
        "pool",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct_chg_percent_raw",
        "vol",
        "amount",
        "amplitude",
        "turnover_rate",
        "pre_close",
        "pct_chg",
        "log_return",
    ]
    return df[ordered]


def write_wide(panel: pd.DataFrame, assets: Sequence[Asset]) -> dict[str, pd.DataFrame]:
    fields = [
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
    ]
    wide = {}
    dates = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    asset_order = [asset.ts_code for asset in assets]
    for field in fields:
        table = panel.pivot(index="trade_date", columns="ts_code", values=field)
        table = table.reindex(index=dates, columns=asset_order)
        wide[field] = table
        table.to_parquet(PROCESSED_DIR / f"wide_{field}.parquet")
        # Compatibility with PyFinance naming.
        table.to_parquet(PROCESSED_DIR / f"wide_{field}_df.parquet")
    return wide


def write_asset_universe(panel: pd.DataFrame, assets: Sequence[Asset]) -> pd.DataFrame:
    rows = []
    for asset in assets:
        sub = panel.loc[panel["ts_code"] == asset.ts_code]
        rows.append(
            {
                "ts_code": asset.ts_code,
                "symbol": asset.symbol,
                "name": asset.name,
                "type": asset.asset_type,
                "pool": asset.pool,
                "status": "ok" if not sub.empty else "missing",
                "rows": int(len(sub)),
                "first_date": sub["trade_date"].min().date().isoformat() if not sub.empty else "",
                "last_date": sub["trade_date"].max().date().isoformat() if not sub.empty else "",
                "median_amount_last_252": float(sub.tail(252)["amount"].median()) if not sub.empty else np.nan,
                "raw_path": str(RAW_DIR / f"{asset.ts_code}_daily.parquet"),
            }
        )
    universe = pd.DataFrame(rows)
    universe.to_csv(PROCESSED_DIR / "asset_universe.csv", index=False)
    return universe


def max_drawdown_from_returns(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    nav = (1.0 + values).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def cvar_loss(values: pd.Series, alpha: float = 0.95) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    losses = -values
    var = losses.quantile(alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def max_drawdown_from_returns_np(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    nav = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(nav)
    return float(np.min(nav / running_max - 1.0))


def cvar_loss_np(values: np.ndarray, alpha: float = 0.95) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.nan
    losses = -values
    var = np.nanquantile(losses, alpha)
    tail = losses[losses >= var]
    return float(np.nanmean(tail)) if tail.size else float(var)


def build_rolling_metrics(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pct = wide["pct_chg"]
    log_ret = wide["log_return"]
    close = wide["close"]
    amount = wide["amount"]
    vol = wide["vol"]
    windows = [5, 10, 20, 60, 120, 252]
    features: dict[str, pd.DataFrame] = {}
    for w in windows:
        mean = pct.rolling(w, min_periods=max(2, min(w, 20))).mean()
        std = pct.rolling(w, min_periods=max(2, min(w, 20))).std()
        downside = pct.where(pct < 0).rolling(w, min_periods=max(2, min(w, 20))).std()
        rolling_max = close.rolling(w, min_periods=max(2, min(w, 20))).max()
        drawdown = close / rolling_max - 1.0
        features[f"RollingMeanReturn:{w}"] = mean
        features[f"RollingLogReturnSum:{w}"] = log_ret.rolling(w, min_periods=max(2, min(w, 20))).sum()
        features[f"RollingVolatility:{w}"] = std
        features[f"AnnualizedVolatility:{w}"] = std * math.sqrt(252)
        features[f"RollingSharpeRatio:{w}"] = mean / std.replace(0, np.nan) * math.sqrt(252)
        features[f"RollingDownsideVolatility:{w}"] = downside
        features[f"RollingSortinoRatio:{w}"] = mean / downside.replace(0, np.nan) * math.sqrt(252)
        features[f"RollingMaxDrawDown:{w}"] = drawdown.rolling(w, min_periods=1).min()
        features[f"CurrentDrawDown:{w}"] = drawdown
        features[f"Momentum:{w}"] = np.log(close / close.shift(w))
        features[f"ReturnSkewness:{w}"] = pct.rolling(w, min_periods=max(3, min(w, 20))).skew()
        features[f"ReturnKurtosis:{w}"] = pct.rolling(w, min_periods=max(4, min(w, 20))).kurt()
        features[f"MeanAmount:{w}"] = amount.rolling(w, min_periods=max(2, min(w, 20))).mean()
        features[f"MeanVolume:{w}"] = vol.rolling(w, min_periods=max(2, min(w, 20))).mean()
        features[f"CVaRLoss95:{w}"] = pct.rolling(w, min_periods=min(w, 20)).apply(cvar_loss_np, raw=True)
    return features_to_long(features)


def build_period_metrics(wide: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    pct = wide["pct_chg"]
    close = wide["close"]
    periods = {
        "2d": 2,
        "3d": 3,
        "5d": 5,
        "10d": 10,
        "20d": 20,
        "25d": 25,
        "50d": 50,
        "75d": 75,
        "6m": 126,
        "12m": 252,
        "2y": 504,
        "3y": 756,
        "5y": 1260,
    }
    all_frames = []
    per_period = {}
    for name, w in periods.items():
        min_periods = max(2, min(w, 20))
        total_return = close / close.shift(w) - 1.0
        mean = pct.rolling(w, min_periods=min_periods).mean()
        std = pct.rolling(w, min_periods=min_periods).std()
        downside = pct.where(pct < 0).rolling(w, min_periods=min_periods).std()
        features = {
            f"TotalReturn:{name}": total_return,
            f"AnnualizedReturn:{name}": (1.0 + total_return).pow(252 / w) - 1.0,
            f"AverageDailyReturn:{name}": mean,
            f"Volatility:{name}": std,
            f"AnnualizedVolatility:{name}": std * math.sqrt(252),
            f"SharpeRatio:{name}": mean / std.replace(0, np.nan) * math.sqrt(252),
            f"SortinoRatio:{name}": mean / downside.replace(0, np.nan) * math.sqrt(252),
            f"MaxDrawDown:{name}": pct.rolling(w, min_periods=min_periods).apply(
                max_drawdown_from_returns_np,
                raw=True,
            ),
            f"CVaRLoss95:{name}": pct.rolling(w, min_periods=min_periods).apply(
                cvar_loss_np,
                raw=True,
            ),
        }
        frame = features_to_long(features)
        per_period[name] = frame
        frame.to_parquet(METRICS_DIR / f"{name}.parquet")
        all_frames.append(frame)
    return pd.concat(all_frames, axis=1).loc[:, lambda x: ~x.columns.duplicated()], per_period


def features_to_long(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    long = None
    for name, table in features.items():
        stacked = table.stack().rename(name)
        stacked.index = stacked.index.set_names(["date", "ts_code"])
        if long is None:
            long = stacked.to_frame()
        else:
            long = long.join(stacked, how="outer")
    if long is None:
        return pd.DataFrame(columns=["date", "ts_code"])
    return long.reset_index()


def write_metrics(wide: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rolling = build_rolling_metrics(wide)
    rolling.to_parquet(METRICS_DIR / "rolling_metrics.parquet")
    period_all, _ = build_period_metrics(wide)
    metrics = rolling.merge(period_all, on=["date", "ts_code"], how="outer")
    metrics = metrics.sort_values(["date", "ts_code"]).reset_index(drop=True)
    metrics.to_parquet(METRICS_DIR / "all_metrics_features.parquet")
    return metrics


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    universe_path: Path,
    start_date: str,
    end_date: str,
) -> None:
    files = [
        PROCESSED_DIR / "asset_universe.csv",
        PROCESSED_DIR / "etf_lof_daily_panel.parquet",
        METRICS_DIR / "all_metrics_features.parquet",
    ]
    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": "Tencent web.ifzq.gtimg.cn fqkline get",
        "adjust": "qfq",
        "universe_config_path": str(universe_path.relative_to(ROOT) if universe_path.is_relative_to(ROOT) else universe_path),
        "start_date_requested": start_date,
        "end_date_requested": end_date,
        "asset_count": int(len(universe)),
        "ok_asset_count": int((universe["status"] == "ok").sum()),
        "panel_rows": int(len(panel)),
        "panel_start": panel["trade_date"].min().date().isoformat(),
        "panel_end": panel["trade_date"].max().date().isoformat(),
        "metrics_rows": int(len(metrics)),
        "metrics_feature_columns": int(len([c for c in metrics.columns if c not in {"date", "ts_code"}])),
        "files": {str(path.relative_to(ROOT)): file_sha256(path) for path in files if path.exists()},
        "notes": [
            "amount is proxied as close * volume * 100 because Tencent kline returns volume but not turnover amount.",
            "turnover_rate is unavailable from Tencent kline and stored as NaN.",
        ],
    }
    (REPORTS_DIR / "akshare_etf_lof_download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_manifest = {
        "generated_at": manifest["generated_at"],
        "source": "local rolling/period metrics compatible with MetricsFactory feature table contract",
        "output_file": str((METRICS_DIR / "all_metrics_features.parquet").relative_to(ROOT)),
        "rows": manifest["metrics_rows"],
        "feature_columns": manifest["metrics_feature_columns"],
        "end_date_equals_feature_date": True,
        "historical_window_ending_at_feature_date": True,
        "uses_future_data": False,
        "uses_full_sample_statistics": False,
        "source_files": [
            {"file": "rolling_metrics.parquet"},
            *[
                {"file": f"{period}.parquet"}
                for period in ["2d", "3d", "5d", "10d", "20d", "25d", "50d", "75d", "6m", "12m", "2y", "3y", "5y"]
            ],
        ],
    }
    (REPORTS_DIR / "metrics_factory_all_features_manifest.json").write_text(
        json.dumps(metrics_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and rebuild ETF/LOF platform data.")
    parser.add_argument(
        "--universe",
        default=str(DEFAULT_UNIVERSE_PATH),
        help="YAML file defining the ETF/LOF universe to download.",
    )
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--refresh", action="store_true", help="Ignore cached raw parquet files and re-download.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    universe_path = Path(args.universe).expanduser().resolve()
    assets, _ = load_universe(universe_path)
    ensure_dirs()
    frames = []
    for asset in assets:
        raw_path = RAW_DIR / f"{asset.ts_code}_daily.parquet"
        if raw_path.exists() and not args.refresh:
            print(f"[load] {asset.ts_code} {asset.name}", flush=True)
            df = pd.read_parquet(raw_path)
        else:
            print(f"[download] {asset.ts_code} {asset.name}", flush=True)
            df = fetch_tencent(asset, args.start_date, args.end_date)
            df.to_parquet(raw_path)
        print(f"  rows={len(df)} {df['trade_date'].min().date()} -> {df['trade_date'].max().date()}", flush=True)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    panel.to_parquet(PROCESSED_DIR / "etf_lof_daily_panel.parquet")
    wide = write_wide(panel, assets)
    universe = write_asset_universe(panel, assets)
    metrics = write_metrics(wide)
    write_manifest(
        panel,
        metrics,
        universe,
        universe_path=universe_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print("[done]")
    print(f"panel_rows={len(panel)}")
    print(f"panel_date_range={panel['trade_date'].min().date()} -> {panel['trade_date'].max().date()}")
    print(f"asset_count={len(universe)} ok={(universe['status'] == 'ok').sum()}")
    print(f"metrics_shape={metrics.shape}")


if __name__ == "__main__":
    main()
