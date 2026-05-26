from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE_PATH = ROOT / "configs" / "data" / "core13_universe.yaml"
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "data" / "reports"


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
            )
        )
    return assets


def date_chunks(start: str, end: str, days: int = 700) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=days), end_ts)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_tencent(asset: Asset, start_date: str, end_date: str, *, retries: int, retry_sleep: float) -> pd.DataFrame:
    import requests

    session = requests.Session()
    session.trust_env = False
    rows: list[list[str]] = []
    for start, end in date_chunks(start_date, end_date):
        param = f"{asset.market_symbol},day,{start:%Y-%m-%d},{end:%Y-%m-%d},640,qfq"
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(
                    "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                    params={"param": param},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", {}).get(asset.market_symbol, {})
                rows.extend(data.get("qfqday") or data.get("day") or [])
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_sleep * attempt)
        else:
            raise RuntimeError(f"FETCH_CORE13_OHLCV_FAILED {asset.ts_code}: {last_error}")
        time.sleep(0.15)
    if not rows:
        raise RuntimeError(f"FETCH_CORE13_OHLCV_EMPTY {asset.ts_code}")
    return normalize_frame(pd.DataFrame(rows), asset)


def normalize_frame(frame: pd.DataFrame, asset: Asset) -> pd.DataFrame:
    result = frame.iloc[:, :6].copy()
    result.columns = ["trade_date", "open", "close", "high", "low", "vol"]
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    for column in ["open", "close", "high", "low", "vol"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["trade_date", "close"]).drop_duplicates("trade_date", keep="last")
    result = result.sort_values("trade_date").reset_index(drop=True)
    result["ts_code"] = asset.ts_code
    result["symbol"] = asset.symbol
    result["asset_name"] = asset.name
    result["asset_type"] = asset.asset_type
    result["pool"] = asset.pool
    result["pre_close"] = result["close"].shift(1)
    result["change"] = result["close"] - result["pre_close"]
    result["pct_chg"] = result["close"] / result["pre_close"] - 1.0
    result["log_return"] = np.log(result["close"] / result["pre_close"])
    result["amount"] = result["close"] * result["vol"] * 100.0
    result["turnover_rate"] = np.nan
    return result[
        [
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
            "vol",
            "amount",
            "turnover_rate",
            "pre_close",
            "pct_chg",
            "log_return",
        ]
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Core-13 ETF/LOF OHLCV data from Tencent quotes.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH))
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d"))
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--manifest-output", default=str(REPORTS_DIR / "core13_ohlcv_download_manifest.json"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    universe_path = Path(args.universe)
    raw_dir = Path(args.raw_dir)
    manifest_output = Path(args.manifest_output)
    assets = load_universe(universe_path)
    if args.dry_run:
        for asset in assets:
            print(f"{asset.ts_code} -> {raw_dir / f'core13_{asset.ts_code}_daily.parquet'}")
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, asset in enumerate(assets, start=1):
        print(f"[{index:02d}/{len(assets)}] download {asset.ts_code} {asset.name}", flush=True)
        try:
            frame = fetch_tencent(asset, args.start_date, args.end_date, retries=args.retries, retry_sleep=args.retry_sleep)
            output_path = raw_dir / f"core13_{asset.ts_code}_daily.parquet"
            frame.to_parquet(output_path, index=False)
            rows.append(
                {
                    "ts_code": asset.ts_code,
                    "rows": int(len(frame)),
                    "first_date": frame["trade_date"].min().date().isoformat(),
                    "last_date": frame["trade_date"].max().date().isoformat(),
                    "output_path": str(output_path),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"ts_code": asset.ts_code, "error": str(exc)})
            print(f"    ERROR {exc}", flush=True)
    manifest = {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "source": "Tencent fqkline qfqday",
        "universe_path": str(universe_path),
        "start_date_requested": args.start_date,
        "end_date_requested": args.end_date,
        "asset_count_requested": len(assets),
        "asset_count_ok": len(rows),
        "assets": rows,
        "errors": errors,
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError(f"CORE13_OHLCV_INCOMPLETE: {len(errors)} failed")
    print(f"manifest_output={manifest_output}")


if __name__ == "__main__":
    main()
