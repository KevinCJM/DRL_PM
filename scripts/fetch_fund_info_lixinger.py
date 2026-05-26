from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "data" / "reports"
DEFAULT_API_URL = "https://open.lixinger.com/api/cn/fund"
DEFAULT_CSV_OUTPUT = REPORTS_DIR / "lixinger_fund_info.csv"
DEFAULT_EXCEL_OUTPUT = REPORTS_DIR / "lixinger_fund_info.xlsx"
DEFAULT_MANIFEST_OUTPUT = REPORTS_DIR / "lixinger_fund_info_manifest.json"

FUND_SECOND_LEVEL_CN = {
    "company": "股票型",
    "hybrid": "混合型",
    "bond": "债券型",
    "QDII": "QDII",
    "reit": "REIT",
    "fof": "FOF",
    "commodity": "商品基金",
}
FUND_FIRST_LEVEL_CN = {
    "mutual_recognition": "互认基金",
}


def normalize_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    return text.zfill(6)


def normalize_iso_date(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    return text[:10]


def is_truthy_name_match(row: dict[str, Any], keyword: str) -> bool:
    values = [normalize_text(row.get("name")), normalize_text(row.get("shortName"))]
    return any(keyword.upper() in value.upper() for value in values if value)


def flatten_fund_record(row: dict[str, Any]) -> dict[str, Any]:
    exchange = normalize_text(row.get("exchange"))
    flat = {
        "stock_code": normalize_code(row.get("stockCode")),
        "name": row.get("name"),
        "short_name": row.get("shortName"),
        "fund_first_level": row.get("fundFirstLevel"),
        "fund_first_level_cn": FUND_FIRST_LEVEL_CN.get(str(row.get("fundFirstLevel"))),
        "fund_second_level": row.get("fundSecondLevel"),
        "fund_second_level_cn": FUND_SECOND_LEVEL_CN.get(str(row.get("fundSecondLevel"))),
        "area_code": row.get("areaCode"),
        "market": row.get("market"),
        "exchange": exchange,
        "inception_datetime": row.get("inceptionDate"),
        "inception_date": normalize_iso_date(row.get("inceptionDate")),
        "delisted_datetime": row.get("delistedDate"),
        "delisted_date": normalize_iso_date(row.get("delistedDate")),
        "is_exchange_traded": exchange in {"sh", "sz"},
        "is_jj_market": exchange == "jj",
        "is_etf_name": is_truthy_name_match(row, "ETF"),
        "is_lof_name": is_truthy_name_match(row, "LOF"),
    }
    flat["fund_market_type"] = resolve_fund_market_type(flat)
    return flat


def resolve_fund_market_type(row: dict[str, Any]) -> str:
    if bool(row.get("is_etf_name")):
        return "ETF"
    if bool(row.get("is_lof_name")):
        return "LOF"
    if row.get("exchange") in {"sh", "sz"}:
        return "exchange_traded_fund"
    return "off_exchange_fund"


def post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != 1:
        raise RuntimeError(f"LIXINGER_API_ERROR: code={data.get('code')} message={data.get('message')}")
    return data


def fetch_fund_info(
    *,
    token: str,
    api_url: str,
    stock_codes: Sequence[str] | None,
    start_page: int,
    max_pages: int | None,
    timeout: float,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total: int | None = None
    pages_fetched = 0
    session = requests.Session()
    normalized_codes = [normalize_code(code) for code in stock_codes] if stock_codes else None

    page_index = max(0, int(start_page))
    while True:
        payload: dict[str, Any] = {"token": token, "pageIndex": page_index}
        if normalized_codes:
            payload["stockCodes"] = normalized_codes
        data = post_json(session, api_url, payload, timeout)
        total = int(data.get("total", 0) or 0)
        page_rows = data.get("data") or []
        rows.extend(flatten_fund_record(row) for row in page_rows)
        pages_fetched += 1

        print(
            f"[page] index={page_index} rows={len(page_rows)} collected={len(rows)} total={total}",
            flush=True,
        )
        if not page_rows:
            break
        if total and len(rows) >= total:
            break
        if max_pages is not None and pages_fetched >= max_pages:
            break
        page_index += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    frame = pd.DataFrame(rows).drop_duplicates("stock_code", keep="last")
    if not frame.empty:
        frame = frame.sort_values(["inception_date", "stock_code"], na_position="last").reset_index(drop=True)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts.fetch_fund_info_lixinger",
        "api_url": api_url,
        "doc_url": "https://www.lixinger.com/api/open-api/html-doc/cn/fund",
        "token_source": "argument_or_LIXINGER_TOKEN",
        "row_count": int(len(frame)),
        "api_total": total,
        "pages_fetched": pages_fetched,
        "stock_codes_filter_count": len(normalized_codes or []),
        "inception_date_non_empty": int(frame["inception_date"].notna().sum()) if "inception_date" in frame else 0,
    }
    return frame, manifest


def make_excel_safe(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(lambda column: column.map(excel_safe_value))


def excel_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_outputs(frame: pd.DataFrame, manifest: dict[str, Any], csv_output: Path, excel_output: Path, manifest_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        make_excel_safe(frame).to_excel(writer, index=False, sheet_name="fund_info")
        pd.DataFrame(
            [
                {"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                for key, value in manifest.items()
            ]
        ).to_excel(writer, index=False, sheet_name="manifest")


def parse_stock_codes(value: str | None) -> list[str] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch mainland China fund information from Lixinger Open API.")
    parser.add_argument("--token", default=os.environ.get("LIXINGER_TOKEN"), help="Lixinger token. Prefer LIXINGER_TOKEN.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--stock-codes", default=None, help="Comma separated fund codes or a txt file path.")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--csv-output", default=str(DEFAULT_CSV_OUTPUT))
    parser.add_argument("--excel-output", default=str(DEFAULT_EXCEL_OUTPUT))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_OUTPUT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.token:
        raise SystemExit("LIXINGER_TOKEN_REQUIRED: set LIXINGER_TOKEN or pass --token")
    frame, manifest = fetch_fund_info(
        token=str(args.token),
        api_url=str(args.api_url),
        stock_codes=parse_stock_codes(args.stock_codes),
        start_page=int(args.start_page),
        max_pages=args.max_pages,
        timeout=float(args.timeout),
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
    )
    write_outputs(
        frame,
        manifest,
        Path(args.csv_output).expanduser().resolve(),
        Path(args.excel_output).expanduser().resolve(),
        Path(args.manifest_output).expanduser().resolve(),
    )
    print(f"[done] rows={len(frame)}")
    print(f"csv={Path(args.csv_output)}")
    print(f"excel={Path(args.excel_output)}")
    print(f"manifest={Path(args.manifest_output)}")


if __name__ == "__main__":
    main()
