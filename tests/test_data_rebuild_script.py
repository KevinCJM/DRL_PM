from __future__ import annotations

import pytest

from scripts.validate_paper_config_data_paths import validate_config
from scripts.validate_core13_data_contract import _normalize_date_token
from scripts.rebuild_etf_lof_data import load_universe, parse_args


def test_rebuild_script_loads_configured_universe(tmp_path):
    universe = tmp_path / "universe.yaml"
    universe.write_text(
        """
assets:
  - ts_code: 510050.SH
    name: SSE 50 ETF
    asset_type: ETF
    pool: equity
  - ts_code: 159915.SZ
    symbol: "159915"
    name: ChiNext ETF
    asset_type: ETF
    pool: equity
""",
        encoding="utf-8",
    )

    assets, _ = load_universe(universe)

    assert [asset.ts_code for asset in assets] == ["510050.SH", "159915.SZ"]
    assert assets[0].symbol == "510050"
    assert assets[1].market_symbol == "sz159915"


def test_rebuild_script_rejects_duplicate_assets(tmp_path):
    universe = tmp_path / "universe.yaml"
    universe.write_text(
        """
assets:
  - ts_code: 510050.SH
  - ts_code: 510050.SH
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="UNIVERSE_ASSET_DUPLICATE"):
        load_universe(universe)


def test_rebuild_script_accepts_custom_universe_argument(tmp_path):
    universe = tmp_path / "custom.yaml"
    args = parse_args(["--universe", str(universe), "--start-date", "2020-01-01", "--end-date", "2020-12-31"])

    assert args.universe == str(universe)
    assert args.start_date == "2020-01-01"
    assert args.end_date == "2020-12-31"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("20260520", "2026-05-20"),
        ("2026-05-20", "2026-05-20"),
        ("", ""),
    ],
)
def test_core13_cutoff_date_normalization(raw, expected):
    assert _normalize_date_token(raw) == expected


def test_paper_config_validator_requires_core13_protocol_and_governance(tmp_path):
    config = tmp_path / "paper.yaml"
    config.write_text(
        """
data:
  asset_universe_path: data/processed/core13_asset_universe.csv
  panel_path: data/processed/core13_etf_lof_daily_panel.parquet
  wide_open_path: data/processed/core13_wide_open.parquet
  wide_high_path: data/processed/core13_wide_high.parquet
  wide_low_path: data/processed/core13_wide_low.parquet
  wide_close_path: data/processed/core13_wide_close.parquet
  wide_adj_nav_path: data/processed/core13_wide_adj_nav_tushare.parquet
  wide_pre_close_path: data/processed/core13_wide_pre_close.parquet
  wide_pct_chg_path: data/processed/core13_wide_pct_chg.parquet
  wide_log_return_path: data/processed/core13_wide_log_return.parquet
  wide_amount_path: data/processed/core13_wide_amount.parquet
  wide_vol_path: data/processed/core13_wide_vol.parquet
  wide_turnover_rate_path: data/processed/core13_wide_turnover_rate.parquet
  all_metrics_features_path: data/metrics_factory/core13_all_metrics_features.parquet
  download_manifest_path: data/reports/core13_data_download_manifest.json
  metrics_manifest_path: data/reports/core13_metrics_factory_manifest.json
  metrics_factory:
    all_metrics_features_path: data/metrics_factory/core13_all_metrics_features.parquet
  data_mode: availability_mask
protocol:
  protocol_id: core13_v2_full_reset_20260522
  asset_universe_id: core13_v2
  data_cutoff_date: "2026-05-20"
data_governance:
  return_source: adj_nav
  valuation_source: adj_nav
  reward_return_source: adj_nav
  metrics_return_source: adj_nav
  execution_price_source: ohlcv
  valuation_table: core13_adj_nav
  execution_price_table: core13_ohlcv
  valuation_execution_split: true
  reward_valuation_split: true
""",
        encoding="utf-8",
    )

    assert validate_config(config, require_core13=True) == []

    text = config.read_text(encoding="utf-8").replace("valuation_execution_split: true", "valuation_execution_split: false")
    config.write_text(text, encoding="utf-8")

    errors = validate_config(config, require_core13=True)
    assert any(error["key"] == "data_governance.valuation_execution_split" for error in errors)
