from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rebuild_etf_lof_data import DEFAULT_UNIVERSE_PATH, REPORTS_DIR, ROOT, Asset, load_universe


DEFAULT_CSV_OUTPUT = REPORTS_DIR / "etf_lof_basic_info_akshare.csv"
DEFAULT_EXCEL_OUTPUT = REPORTS_DIR / "etf_lof_basic_info_akshare.xlsx"
DEFAULT_MANIFEST_OUTPUT = REPORTS_DIR / "etf_lof_basic_info_akshare_manifest.json"
DEFAULT_ALL_MARKET_CSV_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare.csv"
DEFAULT_ALL_MARKET_EXCEL_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare.xlsx"
DEFAULT_ALL_MARKET_MANIFEST_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare_manifest.json"
DEFAULT_ENRICHED_CSV_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare_with_establish_dates.csv"
DEFAULT_ENRICHED_EXCEL_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare_with_establish_dates.xlsx"
DEFAULT_ENRICHED_MANIFEST_OUTPUT = REPORTS_DIR / "all_fund_basic_info_akshare_with_establish_dates_manifest.json"
DEFAULT_DETAIL_CACHE_OUTPUT = REPORTS_DIR / "fund_establish_date_detail_cache.csv"
SINA_OPEN_SCALE_TYPES = ["股票型基金", "混合型基金", "债券型基金", "货币型基金", "QDII基金"]


def normalize_code(value: Any) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", maxsplit=1)[0]
    return text.zfill(6)


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_chinese_date(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return text


def split_establish_date_scale(value: Any) -> tuple[str | None, str | None]:
    text = normalize_text(value)
    if not text:
        return None, None
    parts = [part.strip() for part in text.split("/", maxsplit=1)]
    date = normalize_chinese_date(parts[0])
    scale = parts[1] if len(parts) > 1 and parts[1] else None
    return date, scale


def frame_match_by_code(frame: pd.DataFrame, code_column: str, code: str) -> dict[str, Any]:
    if frame.empty or code_column not in frame.columns:
        return {}
    series = frame[code_column].astype(str).map(normalize_code)
    matched = frame.loc[series == normalize_code(code)]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def kv_frame_to_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or not {"字段", "值"}.issubset(frame.columns):
        return {}
    output: dict[str, Any] = {}
    for _, row in frame.iterrows():
        key = normalize_text(row.get("字段"))
        if key:
            output[key] = row.get("值")
    return output


def safe_call(
    name: str,
    fn: Callable[[], pd.DataFrame],
    *,
    attempts: int = 1,
    retry_sleep_seconds: float = 2.0,
    timeout_seconds: float | None = None,
) -> tuple[pd.DataFrame, str | None]:
    last_error: str | None = None
    for attempt in range(1, max(1, attempts) + 1):
        old_handler: Any = None
        if timeout_seconds is not None and timeout_seconds > 0:
            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001
            last_error = f"{name}: attempt={attempt}: {type(exc).__name__}: {exc}"
            if attempt < attempts and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
        finally:
            if timeout_seconds is not None and timeout_seconds > 0:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
    return pd.DataFrame(), last_error


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise TimeoutError("AKSHARE_CALL_TIMEOUT")


def load_global_sources() -> tuple[dict[str, pd.DataFrame], dict[str, str | None]]:
    import akshare as ak

    calls: dict[str, Callable[[], pd.DataFrame]] = {
        "fund_name_em": ak.fund_name_em,
        "fund_etf_spot_em": ak.fund_etf_spot_em,
        "fund_lof_spot_em": ak.fund_lof_spot_em,
        "fund_etf_category_ths_ETF": lambda: ak.fund_etf_category_ths(symbol="ETF"),
        "fund_etf_category_ths_LOF": lambda: ak.fund_etf_category_ths(symbol="LOF"),
    }
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str | None] = {}
    for name, fn in calls.items():
        frame, error = safe_call(name, fn, attempts=3, timeout_seconds=120)
        frames[name] = frame
        errors[name] = error
    return frames, errors


def load_all_market_sources() -> tuple[dict[str, pd.DataFrame], dict[str, str | None]]:
    import akshare as ak

    calls: dict[str, Callable[[], pd.DataFrame]] = {
        "fund_name_em": ak.fund_name_em,
        "fund_purchase_em": ak.fund_purchase_em,
        "fund_open_fund_daily_em": ak.fund_open_fund_daily_em,
        "fund_money_fund_daily_em": ak.fund_money_fund_daily_em,
        "fund_etf_fund_daily_em": ak.fund_etf_fund_daily_em,
        "fund_new_found_em": ak.fund_new_found_em,
        "fund_exchange_rank_em": ak.fund_exchange_rank_em,
        "fund_scale_close_sina": ak.fund_scale_close_sina,
        "fund_scale_structured_sina": ak.fund_scale_structured_sina,
        "fund_etf_spot_em": ak.fund_etf_spot_em,
        "fund_lof_spot_em": ak.fund_lof_spot_em,
        "fund_etf_category_ths_ETF": lambda: ak.fund_etf_category_ths(symbol="ETF"),
        "fund_etf_category_ths_LOF": lambda: ak.fund_etf_category_ths(symbol="LOF"),
    }
    for scale_type in SINA_OPEN_SCALE_TYPES:
        calls[f"fund_scale_open_sina:{scale_type}"] = (
            lambda scale_type=scale_type: ak.fund_scale_open_sina(symbol=scale_type).assign(
                scale_open_sina_type=scale_type
            )
        )
    frames: dict[str, pd.DataFrame] = {}
    errors: dict[str, str | None] = {}
    for name, fn in calls.items():
        frame, error = safe_call(name, fn, attempts=2, timeout_seconds=120)
        frames[name] = frame
        errors[name] = error
    return frames, errors


def fetch_asset_detail(asset: Asset, sleep_seconds: float) -> tuple[dict[str, Any], dict[str, str | None]]:
    import akshare as ak

    code = normalize_code(asset.symbol)
    info_frame, info_error = safe_call(f"fund_info_ths({code})", lambda: ak.fund_info_ths(symbol=code))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    overview_frame, overview_error = safe_call(f"fund_overview_em({code})", lambda: ak.fund_overview_em(symbol=code))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    info = kv_frame_to_dict(info_frame)
    overview = overview_frame.iloc[0].to_dict() if not overview_frame.empty else {}
    return {**{f"ths_{key}": value for key, value in info.items()}, **overview}, {
        "fund_info_ths_error": info_error,
        "fund_overview_em_error": overview_error,
    }


def fetch_establish_date_detail(code: str) -> dict[str, Any]:
    import akshare as ak

    normalized_code = normalize_code(code)
    fetched_at = datetime.now(UTC).isoformat()
    overview_frame, overview_error = safe_call(
        f"fund_overview_em({normalized_code})",
        lambda: ak.fund_overview_em(symbol=normalized_code),
        attempts=2,
        timeout_seconds=45,
    )
    if not overview_frame.empty:
        overview = overview_frame.iloc[0].to_dict()
        overview_date, overview_scale = split_establish_date_scale(overview.get("成立日期/规模"))
        if overview_date:
            return {
                "code": normalized_code,
                "detail_establish_date": overview_date,
                "detail_establish_date_source": "fund_overview_em",
                "detail_establish_scale": overview_scale,
                "detail_fund_full_name": overview.get("基金全称"),
                "detail_fund_short_name": overview.get("基金简称"),
                "detail_fetch_error": None,
                "detail_fetched_at": fetched_at,
            }

    info_frame, info_error = safe_call(
        f"fund_info_ths({normalized_code})",
        lambda: ak.fund_info_ths(symbol=normalized_code),
        attempts=1,
        timeout_seconds=30,
    )
    info = kv_frame_to_dict(info_frame)
    info_date = normalize_chinese_date(info.get("成立日期"))
    if info_date:
        return {
            "code": normalized_code,
            "detail_establish_date": info_date,
            "detail_establish_date_source": "fund_info_ths",
            "detail_establish_scale": info.get("成立规模"),
            "detail_fund_full_name": info.get("基金全称"),
            "detail_fund_short_name": info.get("基金简称"),
            "detail_fetch_error": None,
            "detail_fetched_at": fetched_at,
        }

    xq_frame, xq_error = safe_call(
        f"fund_individual_basic_info_xq({normalized_code})",
        lambda: ak.fund_individual_basic_info_xq(symbol=normalized_code, timeout=10),
        attempts=1,
        timeout_seconds=20,
    )
    xq = xq_frame_to_dict(xq_frame)
    xq_date = normalize_chinese_date(xq.get("成立时间"))
    if xq_date:
        return {
            "code": normalized_code,
            "detail_establish_date": xq_date,
            "detail_establish_date_source": "fund_individual_basic_info_xq",
            "detail_establish_scale": None,
            "detail_fund_full_name": xq.get("基金全称"),
            "detail_fund_short_name": xq.get("基金名称"),
            "detail_fetch_error": None,
            "detail_fetched_at": fetched_at,
        }

    errors = [error for error in [overview_error, info_error, xq_error] if error]
    return {
        "code": normalized_code,
        "detail_establish_date": None,
        "detail_establish_date_source": None,
        "detail_establish_scale": None,
        "detail_fund_full_name": None,
        "detail_fund_short_name": None,
        "detail_fetch_error": " | ".join(errors) if errors else "DETAIL_DATE_NOT_FOUND",
        "detail_fetched_at": fetched_at,
    }


def xq_frame_to_dict(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or not {"item", "value"}.issubset(frame.columns):
        return {}
    output: dict[str, Any] = {}
    for _, row in frame.iterrows():
        key = normalize_text(row.get("item"))
        if key:
            output[key] = row.get("value")
    return output


def build_basic_info(
    assets: Sequence[Asset],
    global_sources: dict[str, pd.DataFrame],
    global_errors: dict[str, str | None],
    *,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    category = pd.concat(
        [
            global_sources.get("fund_etf_category_ths_ETF", pd.DataFrame()),
            global_sources.get("fund_etf_category_ths_LOF", pd.DataFrame()),
        ],
        ignore_index=True,
    )
    etf_spot = global_sources.get("fund_etf_spot_em", pd.DataFrame())
    lof_spot = global_sources.get("fund_lof_spot_em", pd.DataFrame())
    fund_names = global_sources.get("fund_name_em", pd.DataFrame())

    rows: list[dict[str, Any]] = []
    row_errors: dict[str, dict[str, str | None]] = {}
    for index, asset in enumerate(assets, start=1):
        code = normalize_code(asset.symbol)
        detail, detail_errors = fetch_asset_detail(asset, sleep_seconds)
        row_errors[asset.ts_code] = detail_errors

        name_match = frame_match_by_code(fund_names, "基金代码", code)
        category_match = frame_match_by_code(category, "基金代码", code)
        etf_spot_match = frame_match_by_code(etf_spot, "代码", code)
        lof_spot_match = frame_match_by_code(lof_spot, "代码", code)
        spot_match = etf_spot_match or lof_spot_match

        overview_establish_date, overview_establish_scale = split_establish_date_scale(detail.get("成立日期/规模"))
        ths_establish_date = normalize_chinese_date(detail.get("ths_成立日期"))
        issue_date = normalize_chinese_date(detail.get("发行日期"))

        row = {
            "order": index,
            "ts_code": asset.ts_code,
            "code": code,
            "configured_name": asset.name,
            "configured_type": asset.asset_type,
            "configured_pool": asset.pool,
            "ak_name_em": name_match.get("基金简称"),
            "ak_type_em": name_match.get("基金类型"),
            "ak_pinyin_abbr": name_match.get("拼音缩写"),
            "ak_full_name": detail.get("基金全称") or detail.get("ths_基金全称"),
            "ak_short_name": detail.get("基金简称") or detail.get("ths_基金简称"),
            "ak_type_overview": detail.get("基金类型"),
            "ak_type_ths": detail.get("ths_基金类型"),
            "ak_investment_type_ths": detail.get("ths_投资类型"),
            "issue_date": issue_date,
            "establish_date": ths_establish_date or overview_establish_date,
            "establish_date_ths": ths_establish_date,
            "establish_date_overview": overview_establish_date,
            "establish_scale": detail.get("ths_成立规模") or overview_establish_scale,
            "establish_scale_overview": overview_establish_scale,
            "net_asset_size": detail.get("净资产规模"),
            "share_size": detail.get("份额规模") or detail.get("ths_份额规模"),
            "fund_company": detail.get("基金管理人") or detail.get("ths_基金管理人"),
            "custodian": detail.get("基金托管人") or detail.get("ths_基金托管人"),
            "fund_manager": detail.get("基金经理人") or detail.get("ths_基金经理"),
            "management_fee": detail.get("管理费率") or detail.get("ths_管理费"),
            "custody_fee": detail.get("托管费率") or detail.get("ths_托管费"),
            "sales_service_fee": detail.get("销售服务费率"),
            "benchmark": detail.get("业绩比较基准") or detail.get("ths_业绩比较基准"),
            "tracking_index": detail.get("跟踪标的"),
            "purchase_status": category_match.get("申购状态"),
            "redeem_status": category_match.get("赎回状态"),
            "category_fund_type": category_match.get("基金类型"),
            "latest_nav_date": category_match.get("最新-交易日"),
            "latest_unit_nav": category_match.get("最新-单位净值"),
            "latest_acc_nav": category_match.get("最新-累计净值"),
            "category_query_date": category_match.get("查询日期"),
            "is_in_etf_spot_list": bool(etf_spot_match),
            "is_in_lof_spot_list": bool(lof_spot_match),
            "is_in_exchange_spot_list": bool(spot_match),
            "spot_source": "ETF" if etf_spot_match else "LOF" if lof_spot_match else None,
            "spot_name": spot_match.get("名称"),
            "spot_latest_price": spot_match.get("最新价"),
            "spot_amount": spot_match.get("成交额"),
            "spot_turnover_rate": spot_match.get("换手率"),
            "spot_data_date": spot_match.get("数据日期"),
            "spot_update_time": spot_match.get("更新时间"),
            "fund_info_ths_error": detail_errors["fund_info_ths_error"],
            "fund_overview_em_error": detail_errors["fund_overview_em_error"],
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts.fetch_etf_lof_basic_info_akshare",
        "asset_count": len(assets),
        "row_count": int(len(frame)),
        "global_source_errors": global_errors,
        "row_errors": row_errors,
        "outputs_note": (
            "is_in_exchange_spot_list means the code appeared in AKShare ETF/LOF spot quote lists "
            "at fetch time; it is not a formal legal listing-status field."
        ),
    }
    return frame, manifest


def build_all_market_basic_info(
    global_sources: dict[str, pd.DataFrame],
    global_errors: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fund_names = global_sources.get("fund_name_em", pd.DataFrame())
    purchase = global_sources.get("fund_purchase_em", pd.DataFrame())
    if fund_names.empty and purchase.empty:
        raise RuntimeError(
            "AKSHARE_REQUIRED_SOURCE_EMPTY: "
            f"fund_name_em={global_errors.get('fund_name_em')}; "
            f"fund_purchase_em={global_errors.get('fund_purchase_em')}"
        )
    base = fund_names if not fund_names.empty else purchase
    base_source = "fund_name_em" if not fund_names.empty else "fund_purchase_em"

    open_daily = global_sources.get("fund_open_fund_daily_em", pd.DataFrame())
    money_daily = global_sources.get("fund_money_fund_daily_em", pd.DataFrame())
    etf_daily = global_sources.get("fund_etf_fund_daily_em", pd.DataFrame())
    new_found = global_sources.get("fund_new_found_em", pd.DataFrame())
    exchange_rank = global_sources.get("fund_exchange_rank_em", pd.DataFrame())
    scale_open = concat_sources(global_sources, "fund_scale_open_sina:")
    scale_close = global_sources.get("fund_scale_close_sina", pd.DataFrame())
    scale_structured = global_sources.get("fund_scale_structured_sina", pd.DataFrame())
    etf_spot = global_sources.get("fund_etf_spot_em", pd.DataFrame())
    lof_spot = global_sources.get("fund_lof_spot_em", pd.DataFrame())
    category = pd.concat(
        [
            global_sources.get("fund_etf_category_ths_ETF", pd.DataFrame()),
            global_sources.get("fund_etf_category_ths_LOF", pd.DataFrame()),
        ],
        ignore_index=True,
    )

    purchase_map = code_map(purchase, "基金代码")
    open_map = code_map(open_daily, "基金代码")
    money_map = code_map(money_daily, "基金代码")
    etf_daily_map = code_map(etf_daily, "基金代码")
    new_found_map = code_map(new_found, "基金代码")
    exchange_rank_map = code_map(exchange_rank, "基金代码")
    scale_open_map = code_map(scale_open, "基金代码")
    scale_close_map = code_map(scale_close, "基金代码")
    scale_structured_map = code_map(scale_structured, "基金代码")
    category_map = code_map(category, "基金代码")
    etf_spot_map = code_map(etf_spot, "代码")
    lof_spot_map = code_map(lof_spot, "代码")

    open_cols = dated_nav_columns(open_daily, "单位净值", "累计净值")
    money_cols = dated_money_columns(money_daily)
    etf_cols = dated_nav_columns(etf_daily, "单位净值", "累计净值")

    rows: list[dict[str, Any]] = []
    for index, (_, source_row) in enumerate(base.iterrows(), start=1):
        code = normalize_code(source_row.get("基金代码"))
        purchase_row = purchase_map.get(code, {})
        open_row = open_map.get(code, {})
        money_row = money_map.get(code, {})
        etf_daily_row = etf_daily_map.get(code, {})
        new_found_row = new_found_map.get(code, {})
        exchange_rank_row = exchange_rank_map.get(code, {})
        scale_open_row = scale_open_map.get(code, {})
        scale_close_row = scale_close_map.get(code, {})
        scale_structured_row = scale_structured_map.get(code, {})
        category_row = category_map.get(code, {})
        etf_spot_row = etf_spot_map.get(code, {})
        lof_spot_row = lof_spot_map.get(code, {})
        spot_row = etf_spot_row or lof_spot_row
        establish_date, establish_date_source = first_date_with_source(
            ("fund_exchange_rank_em", exchange_rank_row.get("成立日期")),
            ("fund_scale_open_sina", scale_open_row.get("成立日期")),
            ("fund_scale_close_sina", scale_close_row.get("成立日期")),
            ("fund_scale_structured_sina", scale_structured_row.get("成立日期")),
            ("fund_money_fund_daily_em", money_row.get("成立日期")),
            ("fund_new_found_em", new_found_row.get("成立日期")),
        )

        row = {
            "order": index,
            "code": code,
            "base_source": base_source,
            "fund_name": source_row.get("基金简称"),
            "fund_type": source_row.get("基金类型"),
            "pinyin_abbr": source_row.get("拼音缩写"),
            "pinyin_full": source_row.get("拼音全称"),
            "establish_date": establish_date,
            "establish_date_source": establish_date_source,
            "establish_date_bulk": normalize_chinese_date(
                first_non_empty(money_row.get("成立日期"), new_found_row.get("成立日期"))
            ),
            "fund_manager_bulk": first_non_empty(money_row.get("基金经理"), new_found_row.get("基金经理")),
            "purchase_name": purchase_row.get("基金简称"),
            "purchase_type": purchase_row.get("基金类型"),
            "purchase_latest_value": purchase_row.get("最新净值/万份收益"),
            "purchase_latest_report_time": purchase_row.get("最新净值/万份收益-报告时间"),
            "purchase_status": purchase_row.get("申购状态"),
            "redeem_status": purchase_row.get("赎回状态"),
            "next_open_date": purchase_row.get("下一开放日"),
            "min_purchase_amount": purchase_row.get("购买起点"),
            "daily_purchase_limit": purchase_row.get("日累计限定金额"),
            "purchase_fee": purchase_row.get("手续费"),
            "open_latest_date": open_cols["latest_date"],
            "open_latest_unit_nav": open_row.get(open_cols["latest_unit_nav"]),
            "open_latest_acc_nav": open_row.get(open_cols["latest_acc_nav"]),
            "open_previous_date": open_cols["previous_date"],
            "open_previous_unit_nav": open_row.get(open_cols["previous_unit_nav"]),
            "open_previous_acc_nav": open_row.get(open_cols["previous_acc_nav"]),
            "open_daily_growth_value": open_row.get("日增长值"),
            "open_daily_growth_rate": open_row.get("日增长率"),
            "open_purchase_status": open_row.get("申购状态"),
            "open_redeem_status": open_row.get("赎回状态"),
            "open_fee": open_row.get("手续费"),
            "money_latest_date": money_cols["latest_date"],
            "money_latest_income": money_row.get(money_cols["latest_income"]),
            "money_latest_7d_annualized": money_row.get(money_cols["latest_7d_annualized"]),
            "money_latest_unit_nav": money_row.get(money_cols["latest_unit_nav"]),
            "money_previous_date": money_cols["previous_date"],
            "money_previous_income": money_row.get(money_cols["previous_income"]),
            "money_previous_7d_annualized": money_row.get(money_cols["previous_7d_annualized"]),
            "money_previous_unit_nav": money_row.get(money_cols["previous_unit_nav"]),
            "money_daily_growth_rate": money_row.get("日涨幅"),
            "money_establish_date": normalize_chinese_date(money_row.get("成立日期")),
            "money_manager": money_row.get("基金经理"),
            "money_fee": money_row.get("手续费"),
            "money_available_for_purchase": money_row.get("可购全部"),
            "exchange_rank_establish_date": normalize_chinese_date(exchange_rank_row.get("成立日期")),
            "exchange_rank_type": exchange_rank_row.get("类型"),
            "exchange_rank_date": exchange_rank_row.get("日期"),
            "exchange_rank_unit_nav": exchange_rank_row.get("单位净值"),
            "exchange_rank_acc_nav": exchange_rank_row.get("累计净值"),
            "etf_daily_type": etf_daily_row.get("类型"),
            "etf_latest_date": etf_cols["latest_date"],
            "etf_latest_unit_nav": etf_daily_row.get(etf_cols["latest_unit_nav"]),
            "etf_latest_acc_nav": etf_daily_row.get(etf_cols["latest_acc_nav"]),
            "etf_previous_date": etf_cols["previous_date"],
            "etf_previous_unit_nav": etf_daily_row.get(etf_cols["previous_unit_nav"]),
            "etf_previous_acc_nav": etf_daily_row.get(etf_cols["previous_acc_nav"]),
            "etf_growth_value": etf_daily_row.get("增长值"),
            "etf_growth_rate": etf_daily_row.get("增长率"),
            "etf_market_price": etf_daily_row.get("市价"),
            "etf_discount_rate": etf_daily_row.get("折价率"),
            "new_found_company": new_found_row.get("发行公司"),
            "new_found_type": new_found_row.get("基金类型"),
            "new_found_subscription_period": new_found_row.get("集中认购期"),
            "new_found_raised_shares": new_found_row.get("募集份额"),
            "new_found_establish_date": normalize_chinese_date(new_found_row.get("成立日期")),
            "new_found_return_since_establish": new_found_row.get("成立来涨幅"),
            "new_found_manager": new_found_row.get("基金经理"),
            "new_found_purchase_status": new_found_row.get("申购状态"),
            "new_found_discount_fee": new_found_row.get("优惠费率"),
            "scale_open_establish_date": normalize_chinese_date(scale_open_row.get("成立日期")),
            "scale_open_type": scale_open_row.get("scale_open_sina_type"),
            "scale_open_total_raised": scale_open_row.get("总募集规模"),
            "scale_open_latest_share": scale_open_row.get("最近总份额"),
            "scale_open_manager": scale_open_row.get("基金经理"),
            "scale_open_update_date": scale_open_row.get("更新日期"),
            "scale_close_establish_date": normalize_chinese_date(scale_close_row.get("成立日期")),
            "scale_close_total_raised": scale_close_row.get("总募集规模"),
            "scale_close_latest_share": scale_close_row.get("最近总份额"),
            "scale_close_manager": scale_close_row.get("基金经理"),
            "scale_close_update_date": scale_close_row.get("更新日期"),
            "scale_structured_establish_date": normalize_chinese_date(scale_structured_row.get("成立日期")),
            "scale_structured_total_raised": scale_structured_row.get("总募集规模"),
            "scale_structured_latest_share": scale_structured_row.get("最近总份额"),
            "scale_structured_manager": scale_structured_row.get("基金经理"),
            "scale_structured_update_date": scale_structured_row.get("更新日期"),
            "category_purchase_status": category_row.get("申购状态"),
            "category_redeem_status": category_row.get("赎回状态"),
            "category_fund_type": category_row.get("基金类型"),
            "category_latest_nav_date": category_row.get("最新-交易日"),
            "category_latest_unit_nav": category_row.get("最新-单位净值"),
            "category_latest_acc_nav": category_row.get("最新-累计净值"),
            "category_query_date": category_row.get("查询日期"),
            "is_in_purchase_list": bool(purchase_row),
            "is_in_open_fund_daily": bool(open_row),
            "is_in_money_fund_daily": bool(money_row),
            "is_in_etf_fund_daily": bool(etf_daily_row),
            "is_in_new_found_list": bool(new_found_row),
            "is_in_exchange_rank": bool(exchange_rank_row),
            "is_in_scale_open_sina": bool(scale_open_row),
            "is_in_scale_close_sina": bool(scale_close_row),
            "is_in_scale_structured_sina": bool(scale_structured_row),
            "is_in_etf_spot_list": bool(etf_spot_row),
            "is_in_lof_spot_list": bool(lof_spot_row),
            "is_in_exchange_spot_list": bool(spot_row),
            "spot_source": "ETF" if etf_spot_row else "LOF" if lof_spot_row else None,
            "spot_name": spot_row.get("名称"),
            "spot_latest_price": spot_row.get("最新价"),
            "spot_amount": spot_row.get("成交额"),
            "spot_turnover_rate": spot_row.get("换手率"),
            "spot_data_date": spot_row.get("数据日期"),
            "spot_update_time": spot_row.get("更新时间"),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts.fetch_etf_lof_basic_info_akshare",
        "scope": "all_market",
        "base_source": base_source,
        "row_count": int(len(frame)),
        "global_source_errors": global_errors,
        "source_shapes": {name: list(source.shape) for name, source in global_sources.items()},
        "establish_date_non_empty": int(frame["establish_date"].notna().sum()) if "establish_date" in frame else 0,
        "outputs_note": (
            "This all-market file is built from AKShare batch endpoints. "
            "establish_date is populated from batch sources when available: "
            "fund_exchange_rank_em, fund_scale_open_sina, fund_scale_close_sina, "
            "fund_scale_structured_sina, fund_money_fund_daily_em, and fund_new_found_em. "
            "is_in_exchange_spot_list means the code appeared in AKShare ETF/LOF spot quote lists "
            "at fetch time; it is not a formal legal listing-status field."
        ),
    }
    return frame, manifest


def code_map(frame: pd.DataFrame, code_column: str) -> dict[str, dict[str, Any]]:
    if frame.empty or code_column not in frame.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        code = normalize_code(row.get(code_column))
        output.setdefault(code, row.to_dict())
    return output


def concat_sources(global_sources: dict[str, pd.DataFrame], prefix: str) -> pd.DataFrame:
    frames = [frame for name, frame in global_sources.items() if name.startswith(prefix) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def first_date_with_source(*items: tuple[str, Any]) -> tuple[str | None, str | None]:
    for source, value in items:
        date = normalize_chinese_date(value)
        if date:
            return date, source
    return None, None


def enrich_missing_establish_dates(
    input_path: Path,
    cache_path: Path,
    *,
    sleep_seconds: float,
    max_detail_requests: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = read_table(input_path)
    if "code" not in frame.columns:
        raise ValueError("INPUT_TABLE_MISSING_CODE_COLUMN")
    if "establish_date" not in frame.columns:
        frame["establish_date"] = None
    if "establish_date_source" not in frame.columns:
        frame["establish_date_source"] = None

    frame["code"] = frame["code"].map(normalize_code)
    cache = read_cache(cache_path)
    cache_map = {normalize_code(row["code"]): row.to_dict() for _, row in cache.iterrows()} if not cache.empty else {}
    missing_codes = [
        code
        for code in frame.loc[frame["establish_date"].map(normalize_text).isna(), "code"].dropna().map(normalize_code)
        if code not in cache_map or not normalize_text(cache_map[code].get("detail_establish_date"))
    ]
    if max_detail_requests is not None and max_detail_requests >= 0:
        missing_codes = missing_codes[:max_detail_requests]

    fetched_rows: list[dict[str, Any]] = []
    for index, code in enumerate(missing_codes, start=1):
        result = fetch_establish_date_detail(code)
        cache_map[code] = result
        fetched_rows.append(result)
        if index % 25 == 0 or index == len(missing_codes):
            write_cache(cache_map, cache_path)
            found = sum(1 for row in fetched_rows if normalize_text(row.get("detail_establish_date")))
            print(f"[detail] fetched={index}/{len(missing_codes)} found={found}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_cache(cache_map, cache_path)
    enriched = apply_detail_cache(frame, cache_map)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "scripts.fetch_etf_lof_basic_info_akshare",
        "scope": "enrich-dates",
        "input_path": str(input_path),
        "cache_path": str(cache_path),
        "row_count": int(len(enriched)),
        "input_establish_date_non_empty": int(frame["establish_date"].map(normalize_text).notna().sum()),
        "output_establish_date_non_empty": int(enriched["establish_date"].map(normalize_text).notna().sum()),
        "detail_cache_rows": int(len(cache_map)),
        "detail_requests_this_run": int(len(missing_codes)),
        "detail_found_this_run": int(
            sum(1 for row in fetched_rows if normalize_text(row.get("detail_establish_date")))
        ),
        "detail_error_rows_total": int(
            sum(
                1
                for row in cache_map.values()
                if not normalize_text(row.get("detail_establish_date"))
                and normalize_text(row.get("detail_fetch_error"))
            )
        ),
    }
    return enriched, manifest


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="basic_info", dtype={"code": str})
    return pd.read_csv(path, dtype={"code": str})


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"code": str})


def write_cache(cache_map: dict[str, dict[str, Any]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = pd.DataFrame(list(cache_map.values()))
    if not cache.empty:
        cache["code"] = cache["code"].map(normalize_code)
        cache = cache.sort_values("code")
    cache.to_csv(cache_path, index=False, encoding="utf-8-sig")


def apply_detail_cache(frame: pd.DataFrame, cache_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    enriched = frame.copy()
    for column in [
        "detail_establish_date",
        "detail_establish_date_source",
        "detail_establish_scale",
        "detail_fund_full_name",
        "detail_fund_short_name",
        "detail_fetch_error",
        "detail_fetched_at",
    ]:
        if column not in enriched.columns:
            enriched[column] = None

    for row_index, row in enriched.iterrows():
        code = normalize_code(row["code"])
        detail = cache_map.get(code)
        if not detail:
            continue
        for column in [
            "detail_establish_date",
            "detail_establish_date_source",
            "detail_establish_scale",
            "detail_fund_full_name",
            "detail_fund_short_name",
            "detail_fetch_error",
            "detail_fetched_at",
        ]:
            enriched.at[row_index, column] = detail.get(column)
        detail_date = normalize_text(detail.get("detail_establish_date"))
        if detail_date and not normalize_text(row.get("establish_date")):
            enriched.at[row_index, "establish_date"] = detail_date
            enriched.at[row_index, "establish_date_source"] = detail.get("detail_establish_date_source")
    return enriched


def dated_nav_columns(frame: pd.DataFrame, unit_suffix: str, acc_suffix: str) -> dict[str, str | None]:
    unit_cols = dated_columns(frame, unit_suffix)
    acc_cols = dated_columns(frame, acc_suffix)
    latest_date, latest_unit = unit_cols[0] if unit_cols else (None, None)
    previous_date, previous_unit = unit_cols[1] if len(unit_cols) > 1 else (None, None)
    latest_acc = next((column for date, column in acc_cols if date == latest_date), None)
    previous_acc = next((column for date, column in acc_cols if date == previous_date), None)
    return {
        "latest_date": latest_date,
        "latest_unit_nav": latest_unit,
        "latest_acc_nav": latest_acc,
        "previous_date": previous_date,
        "previous_unit_nav": previous_unit,
        "previous_acc_nav": previous_acc,
    }


def dated_money_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    income_cols = dated_columns(frame, "万份收益")
    annual_cols = dated_columns(frame, "7日年化%")
    unit_cols = dated_columns(frame, "单位净值")
    latest_date, latest_income = income_cols[0] if income_cols else (None, None)
    previous_date, previous_income = income_cols[1] if len(income_cols) > 1 else (None, None)
    return {
        "latest_date": latest_date,
        "latest_income": latest_income,
        "latest_7d_annualized": next((column for date, column in annual_cols if date == latest_date), None),
        "latest_unit_nav": next((column for date, column in unit_cols if date == latest_date), None),
        "previous_date": previous_date,
        "previous_income": previous_income,
        "previous_7d_annualized": next((column for date, column in annual_cols if date == previous_date), None),
        "previous_unit_nav": next((column for date, column in unit_cols if date == previous_date), None),
    }


def dated_columns(frame: pd.DataFrame, suffix: str) -> list[tuple[str, str]]:
    if frame.empty:
        return []
    matches: list[tuple[str, str]] = []
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})-" + re.escape(suffix) + r"$")
    for column in frame.columns:
        match = pattern.match(str(column))
        if match:
            matches.append((match.group(1), str(column)))
    return sorted(matches, reverse=True)


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if normalize_text(value):
            return value
    return None


def write_outputs(frame: pd.DataFrame, manifest: dict[str, Any], csv_output: Path, excel_output: Path, manifest_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    excel_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    frame.to_csv(csv_output, index=False, encoding="utf-8-sig")
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    excel_frame = make_excel_safe(frame)
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        excel_frame.to_excel(writer, index=False, sheet_name="basic_info")
        pd.DataFrame(
            [
                {"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                for key, value in manifest.items()
            ]
        ).to_excel(writer, index=False, sheet_name="manifest")


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch fund basic information from AKShare.")
    parser.add_argument(
        "--scope",
        choices=["universe", "all-market", "enrich-dates"],
        default="universe",
        help=(
            "universe fetches configured ETF/LOF details; all-market fetches all funds from batch endpoints; "
            "enrich-dates fills missing establish_date from per-fund detail endpoints."
        ),
    )
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE_PATH), help="ETF/LOF universe YAML.")
    parser.add_argument("--input", default=str(DEFAULT_ALL_MARKET_EXCEL_OUTPUT), help="Input table for enrich-dates.")
    parser.add_argument("--detail-cache", default=str(DEFAULT_DETAIL_CACHE_OUTPUT), help="Detail-date cache CSV path.")
    parser.add_argument("--max-detail-requests", type=int, default=None, help="Limit detail calls for enrich-dates.")
    parser.add_argument("--csv-output", default=None, help="CSV output path.")
    parser.add_argument("--excel-output", default=None, help="Excel output path.")
    parser.add_argument("--manifest-output", default=None, help="JSON manifest output path.")
    parser.add_argument("--sleep-seconds", type=float, default=0.3, help="Delay between per-asset detail requests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_default = DEFAULT_ALL_MARKET_CSV_OUTPUT if args.scope == "all-market" else DEFAULT_CSV_OUTPUT
    excel_default = DEFAULT_ALL_MARKET_EXCEL_OUTPUT if args.scope == "all-market" else DEFAULT_EXCEL_OUTPUT
    manifest_default = DEFAULT_ALL_MARKET_MANIFEST_OUTPUT if args.scope == "all-market" else DEFAULT_MANIFEST_OUTPUT
    if args.scope == "enrich-dates":
        csv_default = DEFAULT_ENRICHED_CSV_OUTPUT
        excel_default = DEFAULT_ENRICHED_EXCEL_OUTPUT
        manifest_default = DEFAULT_ENRICHED_MANIFEST_OUTPUT
    csv_output = Path(args.csv_output or csv_default).expanduser().resolve()
    excel_output = Path(args.excel_output or excel_default).expanduser().resolve()
    manifest_output = Path(args.manifest_output or manifest_default).expanduser().resolve()

    if args.scope == "all-market":
        global_sources, global_errors = load_all_market_sources()
        frame, manifest = build_all_market_basic_info(global_sources, global_errors)
    elif args.scope == "enrich-dates":
        frame, manifest = enrich_missing_establish_dates(
            Path(args.input).expanduser().resolve(),
            Path(args.detail_cache).expanduser().resolve(),
            sleep_seconds=max(0.0, float(args.sleep_seconds)),
            max_detail_requests=args.max_detail_requests,
        )
    else:
        universe_path = Path(args.universe).expanduser().resolve()
        assets, _ = load_universe(universe_path)
        global_sources, global_errors = load_global_sources()
        frame, manifest = build_basic_info(
            assets,
            global_sources,
            global_errors,
            sleep_seconds=max(0.0, float(args.sleep_seconds)),
        )
    write_outputs(frame, manifest, csv_output, excel_output, manifest_output)

    def display(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    print(f"[done] rows={len(frame)}")
    print(f"csv={display(csv_output)}")
    print(f"excel={display(excel_output)}")
    print(f"manifest={display(manifest_output)}")


if __name__ == "__main__":
    main()
