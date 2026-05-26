from copy import deepcopy

import pytest
import yaml

from src.config import (
    ConfigError,
    ConfigLoader,
    DEFAULT_CONFIG,
    PROJECT_ROOT,
    VALID_EXPERIMENT_TYPES,
    assert_path_allowed,
    canonical_json,
    config_hash,
    validate_config_paths,
)
from src.agents.dqn_agent import DQNAgentConfig
from src.agents.ppo_agent import PPOAgentConfig
from src.experiments.registry import (
    HYBRID_DQN_OPTIMIZER_ALIAS,
    HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    _baseline_factories,
    _expand_baseline_aliases,
)
from src.utils.logger import save_yaml_atomic


EXPECTED_TOP_LEVEL_KEYS = {
    "long_running",
    "data",
    "data_governance",
    "portfolio",
    "split",
    "env",
    "execution_model",
    "cost_model",
    "reward",
    "reward_ablation",
    "asset_universe_sensitivity",
    "constraints",
    "constraint_priority",
    "rebalance",
    "model",
    "feature_matrix",
    "feature_audit",
    "feature_reduction",
    "ppo",
    "dqn",
    "dqn_template",
    "hybrid_dqn_optimizer",
    "auxiliary",
    "preference",
    "uncertainty",
    "distributional_cvar",
    "partial_rebalance",
    "new_model_protocol",
    "cage_eiie",
    "gt_rcpo_lite",
    "ra_gt_rcpo",
    "training",
    "optimizer",
    "scheduler",
    "baselines",
    "evaluation",
    "statistics",
    "hpo",
    "protocol",
    "rankability",
    "paper_run_guard",
    "experiment",
    "full_reproduction",
    "device",
    "logging",
    "output",
    "registry",
    "reproducibility",
    "security",
}
GENERATED_TOP_LEVEL_KEYS = {"config_hash"}


def write_yaml(tmp_path, payload, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_unknown_key_fails_fast(tmp_path):
    top_level_path = write_yaml(tmp_path, {"unknown": True})
    with pytest.raises(ConfigError) as top_level_error:
        ConfigLoader.load(top_level_path)
    assert top_level_error.value.code == "ERR_CONFIG_UNKNOWN_KEY"
    assert "unknown" in str(top_level_error.value)

    nested_path = write_yaml(tmp_path, {"data": {"bad_path": "x"}})
    with pytest.raises(ConfigError) as nested_error:
        ConfigLoader.load(nested_path)
    assert nested_error.value.code == "ERR_CONFIG_UNKNOWN_KEY"
    assert "data.bad_path" in str(nested_error.value)


def test_defaults_are_filled(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            "data": {"root": "data"},
            "portfolio": {"currency": "CNY"},
            "execution_model": {"execution_price": "next_open"},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )

    config = ConfigLoader.load(path)

    assert set(DEFAULT_CONFIG) == EXPECTED_TOP_LEVEL_KEYS
    assert set(config) == EXPECTED_TOP_LEVEL_KEYS | GENERATED_TOP_LEVEL_KEYS
    assert len(config["config_hash"]) == 64
    int(config["config_hash"], 16)
    assert config["data"]["root"] == "data"
    assert config["data"]["asset_universe_path"] == "data/processed/core13_asset_universe.csv"
    assert config["data"]["metrics_factory"]["enabled"] is True
    assert config["data"]["start_date"] == "2014-01-01"
    assert config["data"]["strict_common_history_mode"] is False
    assert config["execution_model"]["execution_price"] == "next_open"
    assert config["execution_model"]["strict_no_lookahead_execution"] is True
    assert config["execution_model"]["delayed_action_execution"] is False
    assert config["execution_model"]["same_close_idealized_execution_enabled"] is False
    assert config["execution_model"]["idealized_execution"] is False
    assert config["execution_model"]["fixed_cost_unit"] == "nav_fraction"
    assert config["cost_model"]["mode"] == "empirical_default"
    assert config["cost_model"]["proportional_cost"] == 0.001
    assert config["cost_model"]["slippage"] == 0.0005
    assert config["cost_model"]["market_impact_enabled"] is True
    assert config["cost_model"]["adv_eps"] == 1000000.0
    assert config["cost_model"]["calibration"]["min_bucket_samples"] == 30
    assert config["model"]["default_encoder"] == "cnn"
    assert config["model"]["encoder"]["type"] == "cnn"
    assert config["model"]["encoder"]["kernel_size_time"] == 3
    assert config["model"]["encoder"]["kernel_size_asset"] == 3
    assert config["model"]["encoder"]["cross_asset_attention"]["enabled"] is False
    assert config["preference"]["omega"] == [0.20, 0.20, 0.20, 0.20, 0.20]
    assert config["ppo"]["entropy_coef"] == 0.01
    assert config["ppo"]["clip_range"] == config["ppo"]["clip_ratio"]
    assert config["dqn"]["target_update_interval"] == 500
    assert config["dqn"]["epsilon_decay_steps"] == 20000
    assert config["dqn"]["use_double_dqn"] == config["dqn"]["double_dqn"]
    assert config["dqn"]["n_steps"] == config["dqn"]["n_step"]
    assert config["dqn"]["use_prioritized_replay"] == config["dqn"]["per_enabled"]
    assert config["hybrid_dqn_optimizer"]["lookback_window"] == 252
    assert config["hybrid_dqn_optimizer"]["risk_parity_optimizer_maxiter"] == 300
    assert config["optimizer"]["weight_decay"] == 1.0e-4
    assert config["experiment"]["type"] == "main_model"
    assert config["training"]["checkpoint_load_path"] is None
    assert config["training"]["checkpoint_include_replay_buffer"] is True
    assert config["evaluation"]["benchmark_csv_path"] is None
    assert config["feature_audit"]["warning_policy"] == "keep"
    assert config["constraints"]["constraint_method"] == "hard_projection"
    assert config["constraints"]["asset_class_mapping"] == {}
    assert config["constraints"]["asset_class_required"] is False
    assert config["constraints"]["partial_rebalance_post_check_policy"] == "report_only"
    assert config["portfolio"]["initial_nav"] == 1.0
    assert config["portfolio"]["initial_capital_currency"] == 100000000.0
    assert config["portfolio"]["currency"] == "CNY"
    assert config["security"]["offline_mode"] is True
    assert DEFAULT_CONFIG["security"]["path_whitelist"] == ["/Users/chenjunming/Desktop/DRL_PM"]


def test_paper_configs_use_disk_bounded_checkpoints():
    for path in (PROJECT_ROOT / "configs" / "paper").glob("*.yaml"):
        config = ConfigLoader.load(path)
        assert config["training"]["checkpoint_include_replay_buffer"] is False, path.name


def test_m6_t4_paper_baseline_smoke_config_expands_related_work_alias():
    config = ConfigLoader.load(PROJECT_ROOT / "configs/paper/p0_native_baseline_smoke.yaml")
    native_models = config["baselines"]["native_rl"]["enabled_models"]
    expanded_models = _expand_baseline_aliases(native_models)
    expected_models = ["ppo_dqn_hierarchical_reimplementation", *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES]
    factories = _baseline_factories(config)

    assert native_models == ["ppo_dqn_hierarchical_reimplementation", HYBRID_DQN_OPTIMIZER_ALIAS]
    assert expanded_models == expected_models
    assert HYBRID_DQN_OPTIMIZER_ALIAS not in factories
    assert all(model_name in factories for model_name in expected_models)
    assert config["baselines"]["native_rl"]["epochs"] == 1
    assert config["baselines"]["native_rl"]["max_train_steps"] == 128
    assert config["baselines"]["native_rl"]["max_validation_steps"] == 128
    assert config["baselines"]["native_rl"]["max_gradient_updates_per_epoch"] == 16
    assert "smoke" in config["output"]["run_name"].lower()


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/paper/baseline_comparison_native.yaml",
        "configs/experiments/baseline_comparison.yaml",
        "configs/baselines.yaml",
    ],
)
def test_m6_t4_related_work_baseline_entrypoints_expand_alias_without_family_row(relative_path):
    config = ConfigLoader.load(PROJECT_ROOT / relative_path)
    native_models = config["baselines"]["native_rl"]["enabled_models"]
    expanded_models = _expand_baseline_aliases(native_models)
    expected_models = ("ppo_dqn_hierarchical_reimplementation", *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES)
    factories = _baseline_factories(config)

    assert "ppo_dqn_hierarchical_reimplementation" in native_models
    assert HYBRID_DQN_OPTIMIZER_ALIAS in native_models
    assert HYBRID_DQN_OPTIMIZER_ALIAS not in expanded_models
    assert HYBRID_DQN_OPTIMIZER_ALIAS not in factories
    assert all(model_name in expanded_models for model_name in expected_models)
    assert all(model_name in factories for model_name in expected_models)


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/paper/hpo_equal_budget_related_work.yaml",
        "configs/paper/hpo_equal_budget_native_pilot.yaml",
        "configs/experiments/hyperparameter_sweep.yaml",
    ],
)
def test_m6_t5_related_work_hpo_configs_use_six_budget_entries_and_pins(relative_path):
    config = ConfigLoader.load(PROJECT_ROOT / relative_path)
    expected_models = ["ppo_dqn_hierarchical_reimplementation", *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES]

    assert config["hpo"]["trainable_models"] == expected_models
    assert HYBRID_DQN_OPTIMIZER_ALIAS not in config["hpo"]["trainable_models"]
    assert config["hpo"]["metric"] == "validation_sharpe_minus_drawdown_turnover_penalty"
    assert config["hpo"]["objective"] == "validation_sharpe_minus_drawdown_turnover_penalty"
    assert config["hybrid_dqn_optimizer"] == {
        "lookback_window": 252,
        "min_observations": 60,
        "covariance_shrinkage": 0.1,
        "risk_free_rate": 0.0,
        "lambda_risk": 1.0,
        "markowitz_optimizer_maxiter": 200,
        "risk_parity_optimizer_maxiter": 300,
    }
    if relative_path == "configs/paper/hpo_equal_budget_related_work.yaml":
        assert config["long_running"] is True
    if relative_path == "configs/paper/hpo_equal_budget_native_pilot.yaml":
        assert "pilot" in config["output"]["run_name"].lower()


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/paper/hpo_equal_budget_native.yaml",
        "configs/paper/hpo_equal_budget_main_native.yaml",
    ],
)
def test_main_hpo_configs_use_seven_budget_entries(relative_path):
    config = ConfigLoader.load(PROJECT_ROOT / relative_path)

    assert config["hpo"]["trainable_models"] == [
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_native",
        "cnn_ppo_native",
        "bernoulli_gated_ppo_native",
        "dqn_template_native",
        "eiie_native",
        "pgportfolio_eiie_native",
    ]
    assert config["baselines"]["native_rl"]["enabled_models"] == [
        "ppo_native",
        "cnn_ppo_native",
        "bernoulli_gated_ppo_native",
        "dqn_template_native",
        "eiie_native",
        "pgportfolio_eiie_native",
    ]
    assert config["hpo"]["n_trials_per_model"] == 50
    assert config["long_running"] is True


def test_hpo_pin_validation_uses_native_rl_enabled_models_when_trainable_models_empty(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            "experiment": {"type": "hyperparameter_sweep"},
            "hpo": {
                "enabled": True,
                "trainable_models": [],
            },
            "baselines": {
                "native_rl": {
                    "enabled_models": [HYBRID_DQN_OPTIMIZER_ALIAS],
                },
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
        name="native_rl_default_hpo.yaml",
    )

    with pytest.raises(ConfigError) as error:
        ConfigLoader.load(path)
    assert error.value.code == "ERR_CONFIG_MISSING_HYBRID_DQN_OPTIMIZER_PIN"


def test_m6_t5_hybrid_hpo_pins_fail_only_for_consumed_branch_params(tmp_path):
    missing_risk_parity_pin = write_yaml(
        tmp_path,
        {
            "experiment": {"type": "hyperparameter_sweep"},
            "hpo": {
                "enabled": True,
                "native_only": True,
                "trainable_models": ["hybrid_dqn_optimizer_risk_parity"],
            },
            "hybrid_dqn_optimizer": {
                "lookback_window": 252,
                "min_observations": 60,
                "covariance_shrinkage": 0.1,
                "risk_free_rate": 0.0,
                "lambda_risk": 1.0,
                "markowitz_optimizer_maxiter": 200,
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    with pytest.raises(ConfigError) as error:
        ConfigLoader.load(missing_risk_parity_pin)
    assert error.value.code == "ERR_CONFIG_MISSING_HYBRID_DQN_OPTIMIZER_PIN"
    assert error.value.key_path == "hybrid_dqn_optimizer.risk_parity_optimizer_maxiter"

    equal_weight_only = write_yaml(
        tmp_path,
        {
            "experiment": {"type": "hyperparameter_sweep"},
            "hpo": {
                "enabled": True,
                "native_only": True,
                "trainable_models": ["hybrid_dqn_optimizer_equal_weight"],
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
        name="equal_weight_only.yaml",
    )
    config = ConfigLoader.load(equal_weight_only)
    assert config["hpo"]["trainable_models"] == ["hybrid_dqn_optimizer_equal_weight"]


def test_m3_t5_constraint_keys_are_schema_valid(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            "constraints": {
                "turnover_limit": 0.10,
                "hhi_limit": 0.50,
                "asset_class_exposure": {"equity": {"max_exposure": 0.70}},
                "asset_class_mapping": {"510300.SH": "equity"},
                "asset_class_required": True,
                "constraint_method": "soft_penalty",
                "soft_penalty_enabled": True,
                "ppo_lagrangian_enabled": False,
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )

    config = ConfigLoader.load(path)

    assert config["constraints"]["turnover_limit"] == 0.10
    assert config["constraints"]["hhi_limit"] == 0.50
    assert config["constraints"]["asset_class_exposure"] == {"equity": {"max_exposure": 0.70}}
    assert config["constraints"]["asset_class_mapping"] == {"510300.SH": "equity"}
    assert config["constraints"]["asset_class_required"] is True
    assert config["constraints"]["constraint_method"] == "soft_penalty"

    typo_path = write_yaml(
        tmp_path,
        {
            "constraints": {
                "asset_class_exposure": {"equity": {"max_expsoure": 0.70}},
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
        name="asset_class_typo.yaml",
    )
    with pytest.raises(ConfigError) as error:
        ConfigLoader.load(typo_path)
    assert error.value.code == "ERR_CONFIG_UNKNOWN_KEY"
    assert "constraints.asset_class_exposure.equity.max_expsoure" in str(error.value)


def test_path_whitelist_denies_parent_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    allowed_file = allowed / "data.csv"
    assert assert_path_allowed(allowed_file, [allowed], "data.root") == allowed_file.resolve()

    denied_paths = [
        ("../outside.yaml", "config_path"),
        ("https://example.com/data.csv", "data.root"),
        (tmp_path / "outside.csv", "data.root"),
    ]
    for raw_path, key_path in denied_paths:
        with pytest.raises(ConfigError) as error:
            assert_path_allowed(raw_path, [allowed], key_path)
        assert error.value.code == "ERR_SECURITY_PATH_DENIED"
        assert key_path in str(error.value)


def test_validate_config_paths_covers_config_data_output_checkpoint_benchmark(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(allowed)]
    config["data"]["root"] = str(allowed / "data")
    config["data"]["asset_universe_path"] = str(allowed / "data/asset_universe.csv")
    config["data"]["metrics_factory"]["all_metrics_features_path"] = str(allowed / "metrics/all.parquet")
    config["output"]["root"] = str(allowed / "results")
    config["registry"]["path"] = str(allowed / "results/run_registry.sqlite")
    config["training"]["checkpoint_load_path"] = str(allowed / "checkpoints/model.pt")
    config["evaluation"]["benchmark_csv_path"] = str(allowed / "benchmarks/cnn.csv")

    validate_config_paths(config, allowed / "configs/base.yaml")

    config["output"]["root"] = str(tmp_path / "outside_results")
    with pytest.raises(ConfigError) as error:
        validate_config_paths(config, allowed / "configs/base.yaml")
    assert error.value.code == "ERR_SECURITY_PATH_DENIED"
    assert "output.root" in str(error.value)


def test_next_close_requires_delayed_execution(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            "execution_model": {
                "execution_price": "next_close",
                "delayed_action_execution": False,
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )

    with pytest.raises(ConfigError) as error:
        ConfigLoader.load(path)
    assert error.value.code == "ERR_CONFIG_INVALID_EXECUTION_MODEL"
    assert "execution_model.delayed_action_execution" in str(error.value)

    valid_path = write_yaml(
        tmp_path,
        {
            "execution_model": {
                "execution_price": "next_close",
                "delayed_action_execution": True,
                "same_close_idealized_execution_enabled": True,
                "idealized_execution": False,
            },
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    config = ConfigLoader.load(valid_path)
    assert config["execution_model"]["idealized_execution"] is True


def test_market_impact_requires_capital(tmp_path):
    missing_capital_path = write_yaml(
        tmp_path,
        {
            "portfolio": {"initial_capital_currency": None},
            "cost_model": {"market_impact_enabled": True},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    with pytest.raises(ConfigError) as capital_error:
        ConfigLoader.load(missing_capital_path)
    assert capital_error.value.code == "ERR_CONFIG_PORTFOLIO_VALUE_REQUIRED"
    assert "portfolio.initial_capital_currency" in str(capital_error.value)

    zero_capital_path = write_yaml(
        tmp_path,
        {
            "portfolio": {"initial_capital_currency": 0},
            "cost_model": {"market_impact_enabled": True},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    with pytest.raises(ConfigError) as zero_error:
        ConfigLoader.load(zero_capital_path)
    assert zero_error.value.code == "ERR_CONFIG_PORTFOLIO_VALUE_REQUIRED"

    invalid_mode_path = write_yaml(
        tmp_path,
        {
            "cost_model": {"mode": "turnover_square_proxy"},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    with pytest.raises(ConfigError) as mode_error:
        ConfigLoader.load(invalid_mode_path)
    assert mode_error.value.code == "ERR_CONFIG_INVALID_COST_MODE"
    assert "cost_model.mode" in str(mode_error.value)

    invalid_fixed_cost_unit_path = write_yaml(
        tmp_path,
        {
            "execution_model": {"fixed_cost_unit": "bps"},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    with pytest.raises(ConfigError) as fixed_cost_unit_error:
        ConfigLoader.load(invalid_fixed_cost_unit_path)
    assert fixed_cost_unit_error.value.code == "ERR_CONFIG_INVALID_FIXED_COST_UNIT"
    assert "execution_model.fixed_cost_unit" in str(fixed_cost_unit_error.value)

    valid_path = write_yaml(
        tmp_path,
        {
            "portfolio": {"initial_capital_currency": 1000000.0},
            "cost_model": {
                "market_impact_enabled": True,
                "mode": "calibrated",
            },
            "execution_model": {"fixed_cost_unit": "currency"},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )
    config = ConfigLoader.load(valid_path)
    assert config["cost_model"]["mode"] == "calibrated"
    assert config["execution_model"]["fixed_cost_unit"] == "currency"


def test_agent_config_aliases_are_normalized(tmp_path):
    path = write_yaml(
        tmp_path,
        {
            "ppo": {"clip_range": 0.17},
            "dqn": {
                "use_double_dqn": False,
                "use_n_step": False,
                "n_steps": 5,
                "use_prioritized_replay": False,
            },
            "optimizer": {"learning_rate": 0.0012, "ppo_lr": 0.0007, "dqn_lr": 0.0002},
            "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        },
    )

    config = ConfigLoader.load(path)
    assert config["ppo"]["clip_ratio"] == 0.17
    assert config["ppo"]["clip_range"] == 0.17
    assert config["dqn"]["double_dqn"] is False
    assert config["dqn"]["use_double_dqn"] is False
    assert config["dqn"]["n_step"] == 5
    assert config["dqn"]["n_steps"] == 5
    assert config["dqn"]["per_enabled"] is False
    assert config["dqn"]["use_prioritized_replay"] is False

    ppo_config = PPOAgentConfig.from_mapping(config)
    dqn_config = DQNAgentConfig.from_mapping(config)
    assert ppo_config.clip_range == 0.17
    assert ppo_config.lr == 0.0007
    assert dqn_config.use_double_dqn is False
    assert dqn_config.use_prioritized_replay is False
    assert dqn_config.n_steps == 1
    assert dqn_config.lr == 0.0002


def test_config_hash_is_stable(tmp_path):
    whitelist = [str(PROJECT_ROOT), str(tmp_path)]
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(
        "\n".join(
            [
                "portfolio:",
                "  currency: CNY",
                "data:",
                "  root: data",
                "security:",
                "  path_whitelist:",
                f"    - {whitelist[0]}",
                f"    - {whitelist[1]}",
            ]
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        "\n".join(
            [
                "security:",
                "  path_whitelist:",
                f"    - {whitelist[0]}",
                f"    - {whitelist[1]}",
                "data:",
                "  root: data",
                "portfolio:",
                "  currency: CNY",
            ]
        ),
        encoding="utf-8",
    )

    first_config = ConfigLoader.load(first_path)
    second_config = ConfigLoader.load(second_path)

    assert first_config["config_hash"] == second_config["config_hash"]
    assert len(first_config["config_hash"]) == 64
    int(first_config["config_hash"], 16)
    assert config_hash(first_config) == first_config["config_hash"]
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})

    snapshot_path = tmp_path / "logs" / "config_snapshot.yaml"
    assert save_yaml_atomic(first_config, snapshot_path) == snapshot_path
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["config_hash"] == first_config["config_hash"]
    assert list(snapshot_path.parent.glob(f".{snapshot_path.name}.*.tmp")) == []


def test_all_config_files_load():
    config_paths = sorted(
        path
        for path in (PROJECT_ROOT / "configs").rglob("*.yaml")
        if "configs/data" not in path.as_posix()
    )
    assert config_paths

    loaded_types = set()
    for path in config_paths:
        config = ConfigLoader.load(path)
        assert config["security"]["path_whitelist"] == [str(PROJECT_ROOT)]
        assert config["experiment"]["type"] in VALID_EXPERIMENT_TYPES
        if path.is_relative_to(PROJECT_ROOT / "configs/experiments"):
            loaded_types.add(config["experiment"]["type"])

    assert loaded_types == VALID_EXPERIMENT_TYPES
