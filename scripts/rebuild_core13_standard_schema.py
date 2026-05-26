from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE_PATH = ROOT / "configs" / "data" / "core13_universe.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "data" / "reports"
WIDE_FIELDS = ("open", "high", "low", "close", "pre_close", "pct_chg", "log_return", "amount", "vol", "turnover_rate")
REQUIRED_COLUMNS = ("adj_nav", "open", "high", "low", "close", "amount", "vol", "pct_chg", "log_return")


@dataclass(frozen=True)
class Asset:
    ts_code: str
    symbol: str
    name: str
    asset_type: str
    pool: str
    asset_bucket: str


def load_universe(path: Path) -> list[Asset]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = payload.get("assets")
    if not isinstance(records, list) or not records:
        raise ValueError("CORE13_UNIVERSE_INVALID: assets must be a non-empty list")
    assets: list[Asset] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"CORE13_UNIVERSE_INVALID: assets[{index}] must be a mapping")
        ts_code = str(record.get("ts_code", "")).strip()
        if not ts_code:
            raise ValueError(f"CORE13_UNIVERSE_INVALID: assets[{index}].ts_code")
        if ts_code in seen:
            raise ValueError(f"CORE13_UNIVERSE_DUPLICATE: {ts_code}")
        seen.add(ts_code)
        assets.append(
            Asset(
                ts_code=ts_code,
                symbol=str(record.get("symbol") or ts_code.split(".", maxsplit=1)[0]),
                name=str(record.get("name") or ts_code),
                asset_type=str(record.get("asset_type") or record.get("type") or ""),
                pool=str(record.get("pool") or ""),
                asset_bucket=str(record.get("asset_bucket") or ""),
            )
        )
    return assets


def load_nav(path: Path, asset_order: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CORE13_NAV_LONG_MISSING: {path}")
    frame = pd.read_parquet(path).copy()
    required = {"ts_code", "nav_date", "adj_nav"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CORE13_NAV_SCHEMA_MISMATCH: missing {missing}")
    frame["trade_date"] = pd.to_datetime(frame["nav_date"])
    frame["adj_nav"] = pd.to_numeric(frame["adj_nav"], errors="coerce")
    frame = frame.loc[frame["ts_code"].astype(str).isin(asset_order)].copy()
    frame = frame.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")
    frame["pct_chg_from_adj_nav"] = frame.groupby("ts_code", sort=False)["adj_nav"].pct_change()
    frame["log_return_from_adj_nav"] = np.log(frame.groupby("ts_code", sort=False)["adj_nav"].pct_change() + 1.0)
    return frame[["ts_code", "trade_date", "adj_nav", "pct_chg_from_adj_nav", "log_return_from_adj_nav"]]


def ohlcv_path(raw_dir: Path, ts_code: str) -> Path:
    preferred = raw_dir / f"core13_{ts_code}_daily.parquet"
    return preferred if preferred.exists() else raw_dir / f"{ts_code}_daily.parquet"


def load_ohlcv(raw_dir: Path, assets: Sequence[Asset], *, allow_nav_only_proxy: bool, nav: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for asset in assets:
        path = ohlcv_path(raw_dir, asset.ts_code)
        if not path.exists():
            missing.append(asset.ts_code)
            continue
        frame = pd.read_parquet(path).copy()
        required = {"trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"}
        absent = sorted(required - set(frame.columns))
        if absent:
            raise ValueError(f"CORE13_OHLCV_SCHEMA_MISMATCH: {path}: missing {absent}")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        for column in ["open", "high", "low", "close", "vol", "amount", "pre_close", "turnover_rate"]:
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["ts_code"] = asset.ts_code
        frames.append(frame[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol", "turnover_rate"]])
    if missing and not allow_nav_only_proxy:
        raise FileNotFoundError(f"CORE13_OHLCV_MISSING: {missing}")
    if missing:
        proxy = nav.loc[nav["ts_code"].isin(missing), ["ts_code", "trade_date", "adj_nav"]].copy()
        for field in ["open", "high", "low", "close"]:
            proxy[field] = proxy["adj_nav"]
        proxy["pre_close"] = proxy.groupby("ts_code", sort=False)["adj_nav"].shift(1)
        proxy["vol"] = 1.0
        proxy["amount"] = proxy["adj_nav"] * 100.0
        proxy["turnover_rate"] = np.nan
        frames.append(proxy[["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "amount", "vol", "turnover_rate"]])
    if not frames:
        raise FileNotFoundError("CORE13_OHLCV_MISSING: no OHLCV frames")
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"], keep="last")


def build_panel(nav: pd.DataFrame, ohlcv: pd.DataFrame, assets: Sequence[Asset]) -> pd.DataFrame:
    panel = nav.merge(ohlcv, on=["ts_code", "trade_date"], how="inner")
    panel["pct_chg"] = panel["pct_chg_from_adj_nav"]
    panel["log_return"] = panel["log_return_from_adj_nav"]
    meta = pd.DataFrame(
        [
            {
                "ts_code": asset.ts_code,
                "symbol": asset.symbol,
                "asset_name": asset.name,
                "asset_type": asset.asset_type,
                "pool": asset.pool,
                "asset_bucket": asset.asset_bucket,
            }
            for asset in assets
        ]
    )
    panel = panel.merge(meta, on="ts_code", how="left")
    ordered = [
        "ts_code",
        "symbol",
        "asset_name",
        "asset_type",
        "pool",
        "asset_bucket",
        "trade_date",
        "adj_nav",
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
    return panel.loc[:, ordered].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def write_wide(panel: pd.DataFrame, asset_order: Sequence[str], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    dates = pd.DatetimeIndex(sorted(panel["trade_date"].dropna().unique()))
    for field in WIDE_FIELDS:
        table = panel.pivot(index="trade_date", columns="ts_code", values=field)
        table = table.reindex(index=dates, columns=list(asset_order)).sort_index()
        table.index = pd.DatetimeIndex(table.index)
        path = output_dir / f"core13_wide_{field}.parquet"
        table.to_parquet(path)
        paths[field] = path
    return paths


def write_asset_universe(panel: pd.DataFrame, assets: Sequence[Asset], raw_dir: Path, output_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asset in assets:
        sub = panel.loc[panel["ts_code"] == asset.ts_code]
        raw_path = ohlcv_path(raw_dir, asset.ts_code)
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
                "raw_path": str(raw_path),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def calendar_loss_report(panel: pd.DataFrame, asset_order: Sequence[str], start_date: str) -> pd.DataFrame:
    candidate_dates = pd.DatetimeIndex(sorted(panel.loc[panel["trade_date"] >= pd.Timestamp(start_date), "trade_date"].unique()))
    rows: list[dict[str, Any]] = []
    for date in candidate_dates:
        sub = panel.loc[panel["trade_date"] == date]
        counts = {column: int(sub.loc[sub[column].notna(), "ts_code"].nunique()) for column in REQUIRED_COLUMNS}
        all_required = set(asset_order)
        for column in REQUIRED_COLUMNS:
            all_required &= set(sub.loc[sub[column].notna(), "ts_code"].astype(str))
        retained = len(all_required) == len(asset_order)
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "n_assets_with_adj_nav": counts["adj_nav"],
                "n_assets_with_ohlcv": min(counts["open"], counts["high"], counts["low"], counts["close"]),
                "n_assets_with_amount": counts["amount"],
                "n_assets_all_required_available": len(all_required),
                "dropped_by_strict_common_history": not retained,
                "drop_reason_top": "" if retained else "missing_required_field",
            }
        )
    return pd.DataFrame(rows)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Core-13 loader-standard schema from NAV and OHLCV inputs.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--nav-long", default=str(PROCESSED_DIR / "core13_etf_lof_fund_nav_tushare_long.parquet"))
    parser.add_argument("--ohlcv-dir", default=str(RAW_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--data-mode", choices=["strict_common_history", "availability_mask"], default="strict_common_history")
    parser.add_argument("--allow-nav-only-proxy", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    universe_path = Path(args.universe)
    nav_long_path = Path(args.nav_long)
    raw_dir = Path(args.ohlcv_dir)
    processed_dir = Path(args.processed_dir)
    reports_dir = Path(args.reports_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets = load_universe(universe_path)
    asset_order = [asset.ts_code for asset in assets]
    nav = load_nav(nav_long_path, asset_order)
    ohlcv = load_ohlcv(raw_dir, assets, allow_nav_only_proxy=args.allow_nav_only_proxy, nav=nav)
    panel = build_panel(nav, ohlcv, assets)
    panel_path = processed_dir / "core13_etf_lof_daily_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    asset_universe_path = processed_dir / "core13_asset_universe.csv"
    asset_universe = write_asset_universe(panel, assets, raw_dir, asset_universe_path)
    wide_paths = write_wide(panel, asset_order, processed_dir)
    loss_report = calendar_loss_report(panel, asset_order, args.start_date)
    loss_report_path = reports_dir / "core13_calendar_loss_report.csv"
    loss_report.to_csv(loss_report_path, index=False)
    retained = int((~loss_report["dropped_by_strict_common_history"]).sum()) if not loss_report.empty else 0
    total = int(len(loss_report))
    retention_ratio = float(retained / total) if total else 0.0
    data_mode = "nav_only_execution_proxy" if args.allow_nav_only_proxy else args.data_mode
    strict_common_passed = retention_ratio >= 0.90
    calendar_loss = {
        "total_candidate_dates": total,
        "retained_dates": retained,
        "dropped_dates": total - retained,
        "calendar_loss_retention_ratio": retention_ratio,
        "data_mode": data_mode,
        "pass_threshold": 0.90,
        "strict_common_history_passed": strict_common_passed,
        "passed": strict_common_passed or data_mode == "availability_mask",
    }
    summary_path = reports_dir / "core13_calendar_loss_summary.json"
    summary_path.write_text(json.dumps(calendar_loss, ensure_ascii=False, indent=2), encoding="utf-8")
    output_files = [panel_path, asset_universe_path, loss_report_path, summary_path, *wide_paths.values()]
    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "universe_path": str(universe_path),
        "nav_long_path": str(nav_long_path),
        "ohlcv_dir": str(raw_dir),
        "asset_count": len(assets),
        "canonical_asset_order": asset_order,
        "asset_universe_rows": int(len(asset_universe)),
        "panel_rows": int(len(panel)),
        "configured_start_date": args.start_date,
        "date_start": loss_report.loc[~loss_report["dropped_by_strict_common_history"], "date"].min()
        if not loss_report.empty
        else None,
        "date_end": loss_report.loc[~loss_report["dropped_by_strict_common_history"], "date"].max()
        if not loss_report.empty
        else None,
        "return_source": "adj_nav",
        "execution_price_source": "ohlcv" if not args.allow_nav_only_proxy else "adj_nav_proxy",
        "valuation_table": "core13_adj_nav",
        "execution_price_table": "core13_ohlcv" if not args.allow_nav_only_proxy else "core13_adj_nav_proxy",
        "valuation_execution_split": not args.allow_nav_only_proxy,
        "calendar_loss": calendar_loss,
        "files": {project_relative(path): file_sha256(path) for path in output_files},
    }
    manifest_path = reports_dir / "core13_data_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if data_mode == "strict_common_history" and retention_ratio < 0.90:
        raise RuntimeError(f"CORE13_CALENDAR_LOSS_THRESHOLD_FAILED: {retention_ratio:.6f}")
    print(f"panel_path={panel_path}")
    print(f"asset_universe_path={asset_universe_path}")
    print(f"manifest_path={manifest_path}")
    print(f"calendar_loss_retention_ratio={retention_ratio:.6f}")


if __name__ == "__main__":
    main()
