from __future__ import annotations

import os

import pytest

from scripts import fetch_core13_fund_nav_tushare as script


def test_resolve_token_uses_environment_without_plaintext_argument(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "unit-test-token")

    value, env_name = script.resolve_token(["TUSHARE_TOKEN"])

    assert value == "unit-test-token"
    assert env_name == "TUSHARE_TOKEN"


def test_resolve_token_rejects_missing_environment(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    monkeypatch.delenv("TS_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="TUSHARE_TOKEN_NOT_FOUND"):
        script.resolve_token(["TUSHARE_TOKEN", "TS_TOKEN"])


def test_cli_has_no_plaintext_token_argument():
    args = script.parse_args(["--start-date", "20200101", "--end-date", "20200131"])

    assert not hasattr(args, "token")
    assert args.token_env is None
    assert args.start_date == "20200101"
    assert args.end_date == "20200131"


def test_core13_assets_include_requested_adj_nav_universe():
    codes = {asset.ts_code for asset in script.CORE13_ASSETS}

    assert {"513500.SH", "160216.SZ", "160416.SZ"}.issubset(codes)
    assert len(codes) == 13
