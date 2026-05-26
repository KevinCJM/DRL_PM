from __future__ import annotations

from scripts.fetch_fund_info_lixinger import flatten_fund_record, parse_stock_codes


def test_flatten_fund_record_normalizes_dates_and_etf_type():
    row = flatten_fund_record(
        {
            "stockCode": "588510",
            "fundSecondLevel": "company",
            "areaCode": "cn",
            "market": "a",
            "exchange": "sh",
            "shortName": "华夏中证科创创业人工智能ETF(588510)",
            "name": "华夏中证科创创业人工智能交易型开放式指数证券投资基金",
            "inceptionDate": "2026-05-15T00:00:00+08:00",
        }
    )

    assert row["stock_code"] == "588510"
    assert row["inception_date"] == "2026-05-15"
    assert row["fund_second_level_cn"] == "股票型"
    assert row["is_exchange_traded"] is True
    assert row["fund_market_type"] == "ETF"


def test_flatten_fund_record_marks_off_exchange_fund():
    row = flatten_fund_record(
        {
            "stockCode": "27328",
            "fundSecondLevel": "hybrid",
            "exchange": "jj",
            "shortName": "浦银安盛半导体产业混合(027328)",
            "name": "浦银安盛半导体产业混合型发起式证券投资基金",
        }
    )

    assert row["stock_code"] == "027328"
    assert row["is_jj_market"] is True
    assert row["fund_market_type"] == "off_exchange_fund"


def test_parse_stock_codes_from_csv_string():
    assert parse_stock_codes("510050, 159915") == ["510050", "159915"]
