from copy import deepcopy

import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.data.loader import DataContractError
from src.data.splits import SplitSpec, create_split, split_to_dict, write_data_split_json


def test_fixed_split_default_ratios():
    dates = pd.bdate_range("2024-01-01", periods=100)
    split = create_split(dates, DEFAULT_CONFIG)

    assert isinstance(split, SplitSpec)
    assert split.fold_id == "fixed"
    assert len(split.train_dates) == 70
    assert len(split.validation_dates) == 15
    assert len(split.test_dates) == 15
    assert split.train_dates.max() < split.validation_dates.min()
    assert split.validation_dates.max() < split.test_dates.min()


def test_purged_split_uses_auxiliary_horizon():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    config = deepcopy(DEFAULT_CONFIG)
    config["split"]["mode"] = "purged"
    config["split"]["train_ratio"] = 0.50
    config["split"]["validation_ratio"] = 0.25
    config["split"]["test_ratio"] = 0.25

    split = create_split(dates, config)

    assert split.train_dates.tolist() == list(dates[:45])
    assert split.validation_dates.tolist() == list(dates[50:70])
    assert split.test_dates.tolist() == list(dates[75:])


def test_embargo_split_removes_boundary_dates():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    config = deepcopy(DEFAULT_CONFIG)
    config["split"]["mode"] = "embargo"
    config["split"]["train_ratio"] = 0.50
    config["split"]["validation_ratio"] = 0.25
    config["split"]["test_ratio"] = 0.25
    config["split"]["embargo_days"] = 3

    split = create_split(dates, config)

    assert split.train_dates.tolist() == list(dates[:47])
    assert split.validation_dates.tolist() == list(dates[53:72])
    assert split.test_dates.tolist() == list(dates[78:])


def test_walk_forward_split_windows():
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    config = deepcopy(DEFAULT_CONFIG)
    config["split"]["mode"] = "walk_forward"

    splits = create_split(dates, config)

    assert isinstance(splits, list)
    assert len(splits) >= 2
    assert [split.fold_id for split in splits[:2]] == [0, 1]
    assert splits[0].train_dates.min() == dates[0]
    assert splits[0].train_dates.max() < splits[0].validation_dates.min()
    assert splits[0].validation_dates.max() < splits[0].test_dates.min()
    assert splits[0].train_dates.max() < pd.Timestamp("2023-01-01")
    assert splits[0].validation_dates.min() >= pd.Timestamp("2023-01-01")
    assert splits[0].validation_dates.max() < pd.Timestamp("2023-07-01")
    assert splits[0].test_dates.min() >= pd.Timestamp("2023-07-01")
    assert splits[0].test_dates.max() < pd.Timestamp("2024-01-01")
    assert splits[1].train_dates.min() == pd.Timestamp("2020-07-01")


def test_walk_forward_drops_incomplete_tail_fold():
    dates = pd.bdate_range("2020-01-01", "2024-08-01")
    config = deepcopy(DEFAULT_CONFIG)
    config["split"]["mode"] = "walk_forward"

    splits = create_split(dates, config)

    assert [split.fold_id for split in splits] == [0, 1]
    assert splits[-1].test_dates.max() < pd.Timestamp("2024-07-01")


def test_strict_no_lookahead_records_last_decision_dates():
    dates = pd.bdate_range("2024-01-01", periods=100)
    split = create_split(dates, DEFAULT_CONFIG)

    assert split.train_last_decision_date == split.train_dates[-3]
    assert split.validation_last_decision_date == split.validation_dates[-3]
    assert split.test_last_decision_date == split.test_dates[-3]


def test_empty_split_raises():
    with pytest.raises(DataContractError) as error:
        create_split(pd.bdate_range("2024-01-01", periods=2), DEFAULT_CONFIG)

    assert error.value.code == "ERR_SPLIT_EMPTY"

    invalid_config = deepcopy(DEFAULT_CONFIG)
    invalid_config["split"]["mode"] = "unsupported"
    with pytest.raises(DataContractError) as invalid_error:
        create_split(pd.bdate_range("2024-01-01", periods=20), invalid_config)
    assert invalid_error.value.code == "ERR_SPLIT_EMPTY"

    with pytest.raises(DataContractError) as unordered_error:
        create_split(pd.DatetimeIndex(["2024-01-02", "2024-01-02"]), DEFAULT_CONFIG)
    assert unordered_error.value.code == "ERR_SPLIT_EMPTY"


def test_data_split_json_serialization(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=20)
    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    output_path = tmp_path / "data_split.json"
    split = create_split(dates, config)

    write_data_split_json(split, output_path, config)

    payload = split_to_dict(split)
    assert output_path.read_text(encoding="utf-8")
    assert payload["fold_id"] == "fixed"
    assert payload["train_dates"][0] == "2024-01-01"
    assert payload["test_last_decision_date"] == "2024-01-24"
