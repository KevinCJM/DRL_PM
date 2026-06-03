from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALID_COST_MODES = {"empirical_default", "calibrated"}
VALID_FIXED_COST_UNITS = {"nav_fraction", "currency"}
CANONICAL_VALIDATION_METRIC = "validation_sharpe_minus_drawdown_turnover_penalty"
VALIDATION_METRIC_ALIASES = {
    "validation_penalized_sharpe": CANONICAL_VALIDATION_METRIC,
}
HYBRID_DQN_OPTIMIZER_ALIAS = "hybrid_dqn_optimizer_reimplementation"
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES = (
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
)
HYBRID_DQN_OPTIMIZER_PIN_VALUES = {
    "lookback_window": 252,
    "min_observations": 60,
    "covariance_shrinkage": 0.1,
    "risk_free_rate": 0.0,
    "lambda_risk": 1.0,
    "markowitz_optimizer_maxiter": 200,
    "risk_parity_optimizer_maxiter": 300,
}
HYBRID_DQN_OPTIMIZER_REQUIRED_PINS_BY_MODEL = {
    "hybrid_dqn_optimizer_equal_weight": (),
    "hybrid_dqn_optimizer_markowitz_mean_variance": (
        "lookback_window",
        "min_observations",
        "covariance_shrinkage",
        "risk_free_rate",
        "lambda_risk",
        "markowitz_optimizer_maxiter",
    ),
    "hybrid_dqn_optimizer_minimum_variance": (
        "lookback_window",
        "min_observations",
        "covariance_shrinkage",
        "markowitz_optimizer_maxiter",
    ),
    "hybrid_dqn_optimizer_sharpe_maximization": (
        "lookback_window",
        "min_observations",
        "covariance_shrinkage",
        "risk_free_rate",
        "markowitz_optimizer_maxiter",
    ),
    "hybrid_dqn_optimizer_risk_parity": (
        "lookback_window",
        "min_observations",
        "covariance_shrinkage",
        "risk_parity_optimizer_maxiter",
    ),
}
VALID_EXPERIMENT_TYPES = {
    "main_model",
    "baseline_comparison",
    "ablation",
    "input_matrix_ablation",
    "pca_ablation",
    "kernel_size_ablation",
    "reward_ablation",
    "walk_forward",
    "transaction_cost_sensitivity",
    "asset_universe_sensitivity",
    "market_regime",
    "seed_stability",
    "hyperparameter_sweep",
    "auxiliary_task_sensitivity",
    "rebalance_frequency_analysis",
    "preference_conditioned_analysis",
    "uncertainty_analysis",
    "distributional_cvar_analysis",
    "partial_rebalance_analysis",
    "full_reproduction",
}
OPEN_MAPPING_PATHS = {
    "constraints.asset_class_mapping",
    "hpo.search_space",
}
ASSET_CLASS_EXPOSURE_RULE_KEYS = {"min_exposure", "max_exposure", "min", "max"}


DEFAULT_CONFIG: dict[str, Any] = {
    "long_running": False,
    "data": {
        "root": "data",
        "asset_universe_path": "data/processed/core13_asset_universe.csv",
        "panel_path": "data/processed/core13_etf_lof_daily_panel.parquet",
        "wide_open_path": "data/processed/core13_wide_open.parquet",
        "wide_high_path": "data/processed/core13_wide_high.parquet",
        "wide_low_path": "data/processed/core13_wide_low.parquet",
        "wide_close_path": "data/processed/core13_wide_close.parquet",
        "wide_adj_nav_path": "data/processed/core13_wide_adj_nav_tushare.parquet",
        "wide_pre_close_path": "data/processed/core13_wide_pre_close.parquet",
        "wide_pct_chg_path": "data/processed/core13_wide_pct_chg.parquet",
        "wide_log_return_path": "data/processed/core13_wide_log_return.parquet",
        "wide_amount_path": "data/processed/core13_wide_amount.parquet",
        "wide_vol_path": "data/processed/core13_wide_vol.parquet",
        "wide_turnover_rate_path": "data/processed/core13_wide_turnover_rate.parquet",
        "all_metrics_features_path": "data/metrics_factory/core13_all_metrics_features.parquet",
        "download_manifest_path": "data/reports/core13_data_download_manifest.json",
        "metrics_manifest_path": "data/reports/core13_metrics_factory_manifest.json",
        "metrics_factory": {
            "enabled": True,
            "all_metrics_features_path": "data/metrics_factory/core13_all_metrics_features.parquet",
        },
        "start_date": "2014-01-01",
        "end_date": None,
        "data_mode": "availability_mask",
        "strict_common_history_mode": False,
        "asset_universe_pools": [],
        "asset_universe_assets": [],
    },
    "data_governance": {
        "execution_return_mode": "derived_from_execution_prices",
        "return_source": None,
        "valuation_source": None,
        "reward_return_source": None,
        "metrics_return_source": None,
        "execution_price_source": None,
        "valuation_table": None,
        "execution_price_table": None,
        "valuation_execution_split": False,
        "reward_valuation_split": False,
        "execution_price_fields": {
            "open": "wide_open",
            "close": "wide_close",
        },
        "turnover_rate_required": False,
        "amount_is_proxy": True,
        "amount_proxy_formula": "close * vol * 100",
        "same_close_idealized_execution_enabled": False,
    },
    "portfolio": {
        "initial_nav": 1.0,
        "initial_capital_currency": 100000000.0,
        "currency": "CNY",
    },
    "split": {
        "mode": "fixed",
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "walk_forward": {
            "train_years": 3,
            "validation_months": 6,
            "test_months": 6,
            "step_months": 6,
        },
        "purge_days": 0,
        "embargo_days": 0,
    },
    "env": {
        "window_size": 60,
        "reward_mode": "A2_net_log_return_after_cost",
        "observation_dtype": "float32",
    },
    "execution_model": {
        "execution_price": "next_open",
        "strict_no_lookahead_execution": True,
        "signal_timestamp": "close_t_after_market",
        "delayed_action_execution": False,
        "same_close_idealized_execution_enabled": False,
        "idealized_execution": False,
        "t_plus_one": False,
        "t_plus_one_position_tracking": "disabled",
        "cash_enabled": False,
        "partial_fill_enabled": False,
        "suspend_policy": "freeze_and_force_exit_when_available",
        "initial_build_cost": True,
        "fixed_cost_unit": "nav_fraction",
    },
    "cost_model": {
        "mode": "empirical_default",
        "proportional_cost": 0.001,
        "fixed_cost": 0.0,
        "slippage": 0.0005,
        "market_impact_enabled": True,
        "market_impact_coef": 0.10,
        "adv_eps": 1000000.0,
        "volatility_eps": 1.0e-8,
        "calibration": {
            "min_bucket_samples": 30,
            "fallback_mode": "empirical_default",
        },
    },
    "reward": {
        "mode": "A2_net_log_return_after_cost",
        "risk_penalty_enabled": True,
        "turnover_penalty_enabled": True,
        "cvar_confidence": 0.95,
        "differential_sharpe_window": 60,
    },
    "reward_ablation": {
        "enabled": False,
        "variant": "A2_net_log_return_after_cost",
    },
    "constraints": {
        "long_only": True,
        "simplex": True,
        "max_weight": 1.0,
        "min_weight": 0.0,
        "turnover_limit": None,
        "hhi_limit": None,
        "asset_class_exposure": {},
        "asset_class_mapping": {},
        "asset_class_required": False,
        "constraint_method": "hard_projection",
        "soft_penalty_enabled": False,
        "ppo_lagrangian_enabled": False,
        "partial_rebalance_post_check_policy": "report_only",
    },
    "constraint_priority": {
        "order": ["availability", "long_only", "simplex", "max_weight", "min_weight", "turnover"],
    },
    "rebalance": {
        "mode": "monthly",
        "every_n_days": 20,
        "threshold_weight_drift": 0.05,
        "threshold_turnover": 0.10,
        "calendar_rule": "last_trading_day",
        "calendar_position": "last_trading_day",
        "calendar_dates": [],
        "volatility_threshold_annual": 0.25,
        "drawdown_threshold": 0.10,
        "risk_budget_tolerance": 0.05,
    },
    "execution_activity": {
        "protocol": "monthly_gate",
        "scheduler_blocks_model_actions": True,
        "activity_gate_enforced": False,
        "turnover_optimization_protocol_id": "legacy_monthly_v1",
        "model_rebalance_turnover_threshold": 0.01,
        "min_model_rebalance_hit_rate": 0.0,
        "max_model_rebalance_hit_rate": 1.0,
        "min_non_initial_turnover_per_opportunity": 0.0,
    },
    "model": {
        "name": "full_dqn_gated_multitask_cnn_ppo",
        "default_encoder": "cnn",
        "latent_dim": 256,
        "dropout": 0.10,
        "activation": "GELU",
        "normalization": "LayerNorm",
        "encoder": {
            "type": "cnn",
            "dropout": 0.10,
            "use_layer_norm": True,
            "kernel_size_time": 3,
            "kernel_size_asset": 3,
            "stride": 1,
            "cross_asset_attention": {
                "enabled": False,
                "n_heads": 4,
                "n_layers": 1,
                "dropout": 0.10,
            },
        },
    },
    "feature_matrix": {
        "input_matrix_id": "M6",
        "window_size": 60,
        "technical_windows": [5, 10, 20, 60, 120],
        "risk_windows": [20, 60, 120],
        "materialize_market_images": False,
    },
    "feature_audit": {
        "warning_policy": "keep",
        "include_warning_features": True,
        "audit_abs_error_tolerance": 1.0e-8,
        "blacklist_patterns": ["future", "forward", "label", "target", "next", "lead", "t+1"],
    },
    "feature_reduction": {
        "imputer": {
            "strategy": "median",
            "fit_scope": "train_only",
        },
        "winsorize": {
            "enabled": True,
            "lower_quantile": 0.005,
            "upper_quantile": 0.995,
        },
        "feature_selection": {
            "enabled": False,
            "variance_threshold": 1.0e-8,
            "correlation_threshold": 0.98,
            "max_features": 512,
        },
        "pca": {
            "enabled": True,
            "explained_variance": 0.95,
            "fixed_components": None,
            "fit_scope": "train_only",
        },
    },
    "ppo": {
        "enabled": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.20,
        "clip_range": 0.20,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "advantage_normalization": True,
    },
        "dqn": {
            "enabled": True,
            "batch_size": 128,
            "warmup_steps": 1000,
            "double_dqn": True,
            "use_double_dqn": True,
            "dueling": True,
        "n_step": 3,
        "use_n_step": True,
        "n_steps": 3,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 20000,
        "target_update_interval": 500,
        "per_enabled": True,
        "use_prioritized_replay": True,
    },
    "dqn_template": {
        "momentum_top_k": 3,
        "invalid_action_penalty": 1.0,
    },
    "hybrid_dqn_optimizer": {
        "lookback_window": 252,
        "min_observations": 60,
        "covariance_shrinkage": 0.1,
        "risk_free_rate": 0.0,
        "lambda_risk": 1.0,
        "markowitz_optimizer_maxiter": 200,
        "risk_parity_optimizer_maxiter": 300,
    },
    "auxiliary": {
        "enabled": True,
        "future_return_horizons": [5, 20],
        "future_volatility_horizons": [20],
        "purge_horizon_days": 5,
        "loss_weight": 1.0,
    },
    "preference": {
        "enabled": False,
        "omega": [0.20, 0.20, 0.20, 0.20, 0.20],
        "evaluation_omegas": [],
    },
    "uncertainty": {
        "enabled": False,
        "method": "dropout",
        "n_samples": 20,
    },
    "distributional_cvar": {
        "enabled": False,
        "n_quantiles": 51,
        "cvar_alpha": 0.05,
    },
    "partial_rebalance": {
        "enabled": False,
        "mode": "hybrid_dqn_beta",
        "discrete_rho_values": [0.0, 0.25, 0.5, 0.75, 1.0],
        "beta_min_concentration": 1.0e-4,
    },
    "new_model_protocol": {
        "phase": "diagnostic",
        "base_protocol_id": "core13_v2_full_reset_20260522",
        "model_extension_id": "core13_v2_p12_p13_20260524",
        "post_hoc_development_disclosure": True,
        "data_mode": "availability_mask",
        "selection_split": "validation",
        "test_used_for_model_selection": False,
        "validation_only_promotion_gate": True,
    },
    "cage_eiie": {
        "enabled": False,
        "variant": "distributional",
        "rho_actions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "fixed_rho": None,
        "gate_type": "cost_aware_multilevel",
        "initial_build_full_rho": True,
        "lambda_turnover": 2.0,
        "lambda_cost": 10.0,
        "lambda_cvar": 0.25,
        "lambda_dd": 0.25,
        "cvar_loss_budget": 0.02,
        "drawdown_budget": 0.10,
        "n_quantiles": 51,
        "gate_scoring": {
            "mode": "legacy_raw",
            "alpha_scale": 0.001,
            "turnover_scale": 0.05,
            "cost_scale": 0.001,
            "cvar_scale": 0.01,
            "drawdown_scale": 0.05,
            "alpha_activation_threshold": 0.25,
            "hold_opportunity_penalty": -0.20,
            "turnover_budget_per_trade": 0.05,
            "cost_budget_per_trade": 0.001,
            "lambda_turnover": 0.20,
            "lambda_cost": 0.20,
            "lambda_cvar": 0.20,
            "lambda_drawdown": 0.20,
        },
    },
    "gt_rcpo_lite": {
        "enabled": False,
        "rho_actions": [0.0, 0.25, 0.5, 1.0],
        "temporal_encoder": "lite_attention",
        "graph_feature_mode": "decision_visible_rolling_correlation",
        "correlation_lookback": 60,
        "initial_build_full_rho": True,
        "turnover_budget": 0.20,
        "cost_budget": 0.002,
        "cvar_loss_budget": 0.02,
        "drawdown_budget": 0.10,
        "lambda_turnover": 2.0,
        "lambda_cost": 10.0,
        "lambda_cvar": 0.35,
        "lambda_dd": 0.25,
        "lambda_lr": 0.01,
        "gate_scoring": {
            "mode": "normalized",
            "alpha_scale": 0.001,
            "turnover_scale": 0.05,
            "cost_scale": 0.001,
            "cvar_scale": 0.01,
            "drawdown_scale": 0.05,
            "alpha_activation_threshold": 0.25,
            "hold_opportunity_penalty": -0.20,
            "turnover_budget_per_trade": 0.05,
            "cost_budget_per_trade": 0.001,
            "lambda_turnover": 0.20,
            "lambda_cost": 0.20,
            "lambda_cvar": 0.20,
            "lambda_drawdown": 0.20,
        },
    },
    "ra_gt_rcpo": {
        "enabled": False,
        "rho_policy": "score_rho_normalized",
        "rho_actions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "rho_temperature": {
            "tau_start": 1.0,
            "tau_end": 0.20,
            "tau_decay_steps": 2048,
            "eval_mode": "argmax",
            "min_entropy_threshold": 0.05,
            "eval_high_entropy_threshold": 0.80,
        },
        "model_dim": 64,
        "transformer_layers": 1,
        "attention_heads": 2,
        "dropout": 0.05,
        "use_graph": True,
        "use_transformer": True,
        "mlp_actor_critic": False,
        "graph_feature_mode": "decision_visible_rolling_correlation",
        "graph_edge_threshold": 0.10,
        "initial_build_full_rho": True,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "batch_size": 32,
        "entropy_coef": 1.0e-3,
        "critic_coef": 0.20,
        "average_turnover_per_step_budget": 0.20,
        "average_cost_per_step_budget": 0.001,
        "cvar_loss_budget": 0.02,
        "drawdown_budget": 0.10,
        "lambda_turnover": 2.0,
        "lambda_cost": 10.0,
        "lambda_cvar": 0.35,
        "lambda_drawdown": 0.25,
        "min_gradient_updates_for_formal": 128,
        "min_env_steps_for_formal": 2048,
        "gate_scoring": {
            "mode": "normalized",
            "alpha_scale": 0.001,
            "turnover_scale": 0.05,
            "cost_scale": 0.001,
            "cvar_scale": 0.01,
            "drawdown_scale": 0.05,
            "alpha_activation_threshold": 0.25,
            "hold_opportunity_penalty": -0.20,
            "turnover_budget_per_trade": 0.05,
            "cost_budget_per_trade": 0.001,
            "lambda_turnover": 0.20,
            "lambda_cost": 0.20,
            "lambda_cvar": 0.20,
            "lambda_drawdown": 0.20,
        },
    },
    "training": {
        "epochs": 1,
        "batch_size": 64,
        "max_grad_norm": 0.5,
        "max_train_steps": None,
        "max_validation_steps": None,
        "validation_interval": 1,
        "deterministic_evaluation": True,
        "checkpoint_load_path": None,
        "checkpoint_include_replay_buffer": True,
    },
    "optimizer": {
        "name": "adamw",
        "learning_rate": 0.0003,
        "ppo_lr": None,
        "dqn_lr": None,
        "auxiliary_lr": None,
        "max_grad_norm": None,
        "weight_decay": 1.0e-4,
    },
    "scheduler": {
        "name": "none",
        "warmup_steps": 0,
    },
    "asset_universe_sensitivity": {
        "pools": [],
    },
    "baselines": {
        "enabled": True,
        "traditional": [
            "fixed_ratio",
            "equal_weight",
            "buy_and_hold",
            "traditional_markowitz_mean_variance",
            "markowitz_min_variance",
            "markowitz_max_sharpe",
            "risk_parity",
            "inverse_volatility",
            "minimum_drawdown",
            "risk_evaluation",
            "hrp",
            "momentum",
        ],
        "deep": ["ppo_proxy", "cnn_ppo_proxy", "bernoulli_gated_ppo_proxy", "dqn_template_proxy", "eiie_proxy"],
        "native": [],
        "native_rl": {
            "enabled_models": [],
            "epochs": 1,
            "max_train_steps": None,
            "max_validation_steps": None,
            "max_gradient_updates_per_epoch": None,
        },
        "external": [],
        "external_pgportfolio": {
            "enabled": False,
            "repo_path": None,
            "repo_whitelist": ["external/PGPortfolio"],
            "import_results_csv": None,
            "import_whitelist": [str(PROJECT_ROOT)],
            "python_executable": None,
            "command": [],
            "config_template_path": None,
            "timeout_seconds": 86400,
            "docker_image": None,
        },
        "deep_training": {
            "enabled": True,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "max_samples": 512,
            "turnover_penalty": 0.0,
            "prior_blend_weight": 0.5,
            "current_weight_mode": "rolling_equal_weight",
        },
    },
    "evaluation": {
        "annualization_factor": 252,
        "risk_free_rate_annual": 0.015,
        "benchmark_csv_path": None,
        "metrics": [
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "omega",
            "max_drawdown",
            "var",
            "cvar",
            "turnover",
            "total_transaction_cost",
        ],
    },
    "statistics": {
        "primary_metric": "net_sharpe",
        "primary_benchmark": "CNN-PPO",
        "secondary_benchmark": "best_validation_classical_baseline",
        "annualization_factor": 252,
        "risk_free_rate_annual": 0.015,
        "var_confidence": 0.95,
        "cvar_confidence": 0.95,
        "bootstrap": {
            "method": "stationary",
            "n_bootstrap": 5000,
            "block_length": 20,
            "confidence_level": 0.95,
            "seed": 42,
        },
        "hac": {
            "lag_rule": "floor(4 * (n / 100) ** (2 / 9))",
        },
        "multiple_testing": {
            "method": "Holm-Bonferroni",
            "alpha": 0.05,
        },
    },
    "hpo": {
        "enabled": False,
        "study_name": None,
        "storage": None,
        "sampler": "optuna_tpe",
        "pruner": "median_pruner",
        "pruner_warmup_trials": 0,
        "pruner_warmup_steps": 0,
        "n_trials": None,
        "n_trials_per_model": 50,
        "timeout": None,
        "timeout_per_model_seconds": None,
        "metric": CANONICAL_VALIDATION_METRIC,
        "direction": "maximize",
        "run_mode": None,
        "seed": None,
        "objective": CANONICAL_VALIDATION_METRIC,
        "selection_split": "validation",
        "final_report_split": "test",
        "equal_budget_across_models": True,
        "native_only": False,
        "trainable_models": [],
        "search_space": {},
        "activity_constraints": {
            "enabled": False,
            "scope_baseline_families": ["new_model_extension", "platform_native_rl"],
            "scope_activity_protocols": ["weekly_gate", "daily_gate_with_cost_constraint"],
            "min_model_rebalance_hit_rate": 0.05,
            "max_model_rebalance_hit_rate": 0.60,
            "min_non_initial_turnover_per_opportunity": 0.002,
            "max_average_turnover": 0.030,
            "hit_rate_underuse_penalty": 5.0,
            "turnover_underuse_penalty": 5.0,
            "turnover_overuse_penalty": 2.0,
            "cost_budget": 0.010,
            "cost_over_budget_penalty": 10.0,
        },
    },
    "protocol": {
        "protocol_id": "core13_v2_full_reset_20260522",
        "asset_universe_id": "core13_v2",
        "data_cutoff_date": "2026-05-20",
    },
    "rankability": {
        "rankable_in_unified_table": False,
        "diagnostic_status": "diagnostic",
        "discard_reason": None,
    },
    "paper_run_guard": {
        "require_core13_paths": False,
        "require_core13_data_contract": False,
        "require_valuation_execution_split": False,
        "require_calendar_loss_report": False,
        "require_calendar_loss_summary": False,
        "allowed_data_modes": [],
        "min_calendar_loss_retention_ratio": 0.90,
        "require_availability_mask_contract_if_mask_mode": False,
        "forbid_legacy_17_asset_paths": False,
        "forbid_nav_only_proxy_in_main_table": False,
    },
    "experiment": {
        "type": "main_model",
    },
    "full_reproduction": {
        "resume_completed_children": False,
    },
    "device": {
        "mode": "auto",
        "amp": True,
    },
    "logging": {
        "level": "INFO",
        "save_config_snapshot": True,
        "log_dir": "logs",
    },
    "output": {
        "root": "results",
        "run_name": "default",
        "overwrite": False,
    },
    "registry": {
        "enabled": True,
        "path": "results/run_registry.sqlite",
    },
    "reproducibility": {
        "seed": 42,
        "seeds": [42, 123, 2024, 3407, 9999],
        "deterministic_torch": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "record_env_vars": True,
        "require_uv_lock_if_present": True,
        "docker_supported": True,
    },
    "security": {
        "offline_mode": True,
        "path_whitelist": [str(PROJECT_ROOT)],
        "safe_torch_load": True,
        "forbid_pickle_untrusted": True,
    },
}


class ConfigError(ValueError):
    def __init__(self, code: str, key_path: str, message: str | None = None) -> None:
        self.code = code
        self.key_path = key_path
        detail = message or f"{code}: {key_path}"
        super().__init__(detail)


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(config: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(config))
    payload.pop("config_hash", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _reject_unsafe_path_syntax(path: str | Path, key_path: str) -> None:
    raw_path = str(path)
    parsed = urlparse(raw_path)
    if parsed.scheme or parsed.netloc or raw_path.startswith(("//", "\\\\")):
        raise ConfigError("ERR_SECURITY_PATH_DENIED", key_path, f"ERR_SECURITY_PATH_DENIED: {key_path}")
    if ".." in Path(raw_path).parts:
        raise ConfigError("ERR_SECURITY_PATH_DENIED", key_path, f"ERR_SECURITY_PATH_DENIED: {key_path}")


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_path_allowed(
    path: str | Path,
    path_whitelist: Sequence[str | Path],
    key_path: str = "path",
) -> Path:
    _reject_unsafe_path_syntax(path, key_path)
    resolved = _resolve_project_path(path)
    allowed_roots: list[Path] = []
    for item in path_whitelist:
        _reject_unsafe_path_syntax(item, "security.path_whitelist")
        allowed_roots.append(_resolve_project_path(item))

    if not allowed_roots:
        raise ConfigError("ERR_SECURITY_PATH_DENIED", "security.path_whitelist")

    if any(resolved == root or _is_relative_to(resolved, root) for root in allowed_roots):
        return resolved

    raise ConfigError("ERR_SECURITY_PATH_DENIED", key_path, f"ERR_SECURITY_PATH_DENIED: {key_path}")


def validate_config_paths(config: Mapping[str, Any], config_path: str | Path | None = None) -> None:
    whitelist = config["security"]["path_whitelist"]
    if config_path is not None:
        assert_path_allowed(config_path, whitelist, "config_path")

    for key, value in config["data"].items():
        if value is None:
            continue
        if key == "metrics_factory":
            assert_path_allowed(
                value["all_metrics_features_path"],
                whitelist,
                "data.metrics_factory.all_metrics_features_path",
            )
        elif key == "root" or key.endswith("_path"):
            assert_path_allowed(value, whitelist, f"data.{key}")

    assert_path_allowed(config["output"]["root"], whitelist, "output.root")
    assert_path_allowed(config["registry"]["path"], whitelist, "registry.path")

    checkpoint_path = config["training"].get("checkpoint_load_path")
    if checkpoint_path is not None:
        assert_path_allowed(checkpoint_path, whitelist, "training.checkpoint_load_path")

    benchmark_path = config["evaluation"].get("benchmark_csv_path")
    if benchmark_path is not None:
        assert_path_allowed(benchmark_path, whitelist, "evaluation.benchmark_csv_path")


def _validate_asset_class_exposure(value: Any, key_path: str) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ConfigError("ERR_CONFIG_INVALID_TYPE", key_path, f"ERR_CONFIG_INVALID_TYPE: {key_path}")

    result: dict[Any, Any] = {}
    for asset_class, rule in value.items():
        rule_path = f"{key_path}.{asset_class}"
        if isinstance(rule, dict):
            for rule_key in rule:
                if rule_key not in ASSET_CLASS_EXPOSURE_RULE_KEYS:
                    raise ConfigError("ERR_CONFIG_UNKNOWN_KEY", f"{rule_path}.{rule_key}")
        result[asset_class] = deepcopy(rule)
    return result


def _move_alias_value(mapping: dict[str, Any], *, canonical: str, alias: str, key_path: str) -> None:
    if alias not in mapping:
        return
    alias_value = mapping.pop(alias)
    if canonical in mapping and mapping[canonical] != alias_value:
        raise ConfigError("ERR_CONFIG_ALIAS_CONFLICT", f"{key_path}.{alias}")
    mapping[canonical] = alias_value


def _expand_hybrid_dqn_optimizer_aliases(model_names: Sequence[Any]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for model_name in model_names:
        name = str(model_name)
        candidates = HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES if name == HYBRID_DQN_OPTIMIZER_ALIAS else (name,)
        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return expanded


def _hpo_configured_models(config: Mapping[str, Any]) -> list[Any]:
    hpo_config = config.get("hpo")
    if not isinstance(hpo_config, Mapping):
        return []
    explicit = hpo_config.get("trainable_models")
    if explicit:
        return list(explicit)
    model_config = config.get("model")
    baseline_config = config.get("baselines")
    model_name = (
        model_config.get("name", "full_dqn_gated_multitask_cnn_ppo")
        if isinstance(model_config, Mapping)
        else "full_dqn_gated_multitask_cnn_ppo"
    )
    deep_baselines = baseline_config.get("deep", ()) if isinstance(baseline_config, Mapping) else ()
    native_config = baseline_config.get("native_rl") if isinstance(baseline_config, Mapping) else None
    native_models = list(native_config.get("enabled_models", ())) if isinstance(native_config, Mapping) else []
    native_models.extend(list(baseline_config.get("native", ()) or ()) if isinstance(baseline_config, Mapping) else [])
    return [model_name, *list(deep_baselines or ()), *native_models]


class ConfigLoader:
    @classmethod
    def load(cls, config_path: str | Path, cli_overrides: Mapping[str, Any] | object | None = None) -> dict[str, Any]:
        _reject_unsafe_path_syntax(config_path, "config_path")
        path = Path(config_path)
        if path.suffix not in {".yaml", ".yml"}:
            raise ConfigError("ERR_CONFIG_INVALID_PATH", "config_path", "config_path must be .yaml or .yml")

        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ConfigError("ERR_CONFIG_INVALID_TYPE", "config", "config root must be a mapping")

        loaded = cls._normalize_loaded_aliases(loaded)
        resolved = cls._merge_defaults(DEFAULT_CONFIG, loaded, "")
        cls._apply_cli_overrides(resolved, cli_overrides)
        cls._normalize_runtime_aliases(resolved)
        cls._normalize_hpo_metric_aliases(resolved)
        cls._normalize_security_whitelist(resolved)
        cls.validate_execution_activity(resolved, loaded)
        cls.validate_execution_model(resolved)
        cls.validate_cost_model(resolved)
        cls.validate_experiment(resolved)
        cls.validate_hybrid_dqn_optimizer_pins(resolved, loaded)
        validate_config_paths(resolved, path)
        resolved["config_hash"] = config_hash(resolved)
        return resolved

    @classmethod
    def _normalize_security_whitelist(cls, config: dict[str, Any]) -> None:
        security = config.setdefault("security", {})
        raw_whitelist = security.get("path_whitelist", [])
        current_root = str(PROJECT_ROOT)
        legacy_roots = {"/Users/chenjunming/Desktop/DRL_PM"}
        raw_items = (
            list(raw_whitelist)
            if isinstance(raw_whitelist, Sequence) and not isinstance(raw_whitelist, (str, bytes))
            else []
        )
        whitelist: list[str] = []
        for item in raw_items:
            normalized = current_root if str(item) in legacy_roots else str(item)
            if normalized not in whitelist:
                whitelist.append(normalized)
        if current_root not in whitelist:
            whitelist.append(current_root)
        security["path_whitelist"] = whitelist

    @classmethod
    def validate_execution_model(cls, config: dict[str, Any]) -> None:
        execution_model = config["execution_model"]
        if (
            execution_model["execution_price"] == "next_close"
            and execution_model["delayed_action_execution"] is not True
        ):
            raise ConfigError(
                "ERR_CONFIG_INVALID_EXECUTION_MODEL",
                "execution_model.delayed_action_execution",
                "ERR_CONFIG_INVALID_EXECUTION_MODEL: execution_model.delayed_action_execution",
            )

        if execution_model["same_close_idealized_execution_enabled"] is True:
            execution_model["idealized_execution"] = True

    @classmethod
    def validate_execution_activity(cls, config: dict[str, Any], loaded: Mapping[str, Any]) -> None:
        activity = config["execution_activity"]
        protocol = str(activity.get("protocol", "monthly_gate"))
        if protocol not in {"monthly_gate", "weekly_gate", "daily_gate_with_cost_constraint"}:
            raise ConfigError(
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
                "execution_activity.protocol",
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: execution_activity.protocol",
            )
        scheduler_blocks = bool(activity.get("scheduler_blocks_model_actions", True))
        if protocol in {"monthly_gate", "weekly_gate"} and scheduler_blocks is not True:
            raise ConfigError(
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
                "execution_activity.scheduler_blocks_model_actions",
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: execution_activity.scheduler_blocks_model_actions",
            )
        if protocol == "daily_gate_with_cost_constraint" and scheduler_blocks is not False:
            raise ConfigError(
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
                "execution_activity.scheduler_blocks_model_actions",
                "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: daily_gate_with_cost_constraint requires scheduler_blocks_model_actions=false",
            )
        loaded_activity = loaded.get("execution_activity") if isinstance(loaded.get("execution_activity"), Mapping) else {}
        if isinstance(loaded_activity, Mapping) and loaded_activity.get("activity_gate_enforced") is True:
            for required_key in ("protocol", "scheduler_blocks_model_actions"):
                if required_key not in loaded_activity:
                    raise ConfigError(
                        "ERR_CONFIG_MISSING_EXECUTION_ACTIVITY_FIELD",
                        f"execution_activity.{required_key}",
                        f"ERR_CONFIG_MISSING_EXECUTION_ACTIVITY_FIELD: execution_activity.{required_key}",
                    )

    @classmethod
    def validate_cost_model(cls, config: dict[str, Any]) -> None:
        cost_model = config["cost_model"]
        execution_model = config["execution_model"]

        if cost_model["mode"] not in VALID_COST_MODES:
            raise ConfigError(
                "ERR_CONFIG_INVALID_COST_MODE",
                "cost_model.mode",
                "ERR_CONFIG_INVALID_COST_MODE: cost_model.mode",
            )

        if execution_model["fixed_cost_unit"] not in VALID_FIXED_COST_UNITS:
            raise ConfigError(
                "ERR_CONFIG_INVALID_FIXED_COST_UNIT",
                "execution_model.fixed_cost_unit",
                "ERR_CONFIG_INVALID_FIXED_COST_UNIT: execution_model.fixed_cost_unit",
            )

        initial_capital = config["portfolio"]["initial_capital_currency"]
        if cost_model["market_impact_enabled"] is True and (
            not isinstance(initial_capital, (int, float)) or initial_capital <= 0
        ):
            raise ConfigError(
                "ERR_CONFIG_PORTFOLIO_VALUE_REQUIRED",
                "portfolio.initial_capital_currency",
                "ERR_CONFIG_PORTFOLIO_VALUE_REQUIRED: portfolio.initial_capital_currency",
            )

    @classmethod
    def validate_experiment(cls, config: dict[str, Any]) -> None:
        experiment_type = config["experiment"]["type"]
        if experiment_type not in VALID_EXPERIMENT_TYPES:
            raise ConfigError(
                "ERR_CONFIG_INVALID_EXPERIMENT_TYPE",
                "experiment.type",
                "ERR_CONFIG_INVALID_EXPERIMENT_TYPE: experiment.type",
            )

    @classmethod
    def validate_hybrid_dqn_optimizer_pins(cls, config: dict[str, Any], loaded: Mapping[str, Any]) -> None:
        hpo_config = config["hpo"]
        if hpo_config.get("enabled") is not True:
            return
        trainable_models = _expand_hybrid_dqn_optimizer_aliases(_hpo_configured_models(config))
        hybrid_models = [model for model in trainable_models if model in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES]
        if not hybrid_models:
            return
        loaded_pins = loaded.get("hybrid_dqn_optimizer")
        if not isinstance(loaded_pins, Mapping):
            loaded_pins = {}
        for model_name in hybrid_models:
            for pin_key in HYBRID_DQN_OPTIMIZER_REQUIRED_PINS_BY_MODEL[model_name]:
                key_path = f"hybrid_dqn_optimizer.{pin_key}"
                if pin_key not in loaded_pins or loaded_pins.get(pin_key) is None:
                    raise ConfigError(
                        "ERR_CONFIG_MISSING_HYBRID_DQN_OPTIMIZER_PIN",
                        key_path,
                        f"ERR_CONFIG_MISSING_HYBRID_DQN_OPTIMIZER_PIN: {key_path}",
                    )
                expected = HYBRID_DQN_OPTIMIZER_PIN_VALUES[pin_key]
                actual = loaded_pins.get(pin_key)
                if isinstance(expected, float):
                    matched = abs(float(actual) - expected) <= 1.0e-12
                else:
                    matched = int(actual) == int(expected)
                if not matched:
                    raise ConfigError(
                        "ERR_CONFIG_INVALID_HYBRID_DQN_OPTIMIZER_PIN",
                        key_path,
                        f"ERR_CONFIG_INVALID_HYBRID_DQN_OPTIMIZER_PIN: {key_path}",
                    )

    @classmethod
    def _normalize_loaded_aliases(cls, loaded: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(loaded))
        ppo = result.get("ppo")
        if isinstance(ppo, dict):
            _move_alias_value(ppo, canonical="clip_ratio", alias="clip_range", key_path="ppo")
        dqn = result.get("dqn")
        if isinstance(dqn, dict):
            _move_alias_value(dqn, canonical="double_dqn", alias="use_double_dqn", key_path="dqn")
            _move_alias_value(dqn, canonical="n_step", alias="n_steps", key_path="dqn")
            _move_alias_value(dqn, canonical="per_enabled", alias="use_prioritized_replay", key_path="dqn")
        return result

    @classmethod
    def _normalize_runtime_aliases(cls, config: dict[str, Any]) -> None:
        ppo = config["ppo"]
        ppo["clip_range"] = ppo["clip_ratio"]

        dqn = config["dqn"]
        dqn["use_double_dqn"] = dqn["double_dqn"]
        dqn["n_steps"] = dqn["n_step"]
        dqn["use_prioritized_replay"] = dqn["per_enabled"]

    @classmethod
    def _normalize_hpo_metric_aliases(cls, config: dict[str, Any]) -> None:
        hpo_config = config["hpo"]
        for key in ("metric", "objective"):
            value = hpo_config.get(key)
            if isinstance(value, str):
                hpo_config[key] = VALIDATION_METRIC_ALIASES.get(value, value)

    @classmethod
    def _merge_defaults(cls, defaults: Mapping[str, Any], overrides: Mapping[str, Any], path: str) -> dict[str, Any]:
        result = deepcopy(dict(defaults))
        for key, value in overrides.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key not in defaults:
                raise ConfigError("ERR_CONFIG_UNKNOWN_KEY", key_path)

            default_value = defaults[key]
            if key_path == "constraints.asset_class_exposure":
                result[key] = _validate_asset_class_exposure(value, key_path)
                continue
            if key_path in OPEN_MAPPING_PATHS:
                if not isinstance(value, dict):
                    raise ConfigError("ERR_CONFIG_INVALID_TYPE", key_path, f"ERR_CONFIG_INVALID_TYPE: {key_path}")
                result[key] = deepcopy(value)
                continue
            if isinstance(default_value, dict):
                if not isinstance(value, dict):
                    raise ConfigError("ERR_CONFIG_INVALID_TYPE", key_path, f"ERR_CONFIG_INVALID_TYPE: {key_path}")
                result[key] = cls._merge_defaults(default_value, value, key_path)
            else:
                result[key] = deepcopy(value)
        return result

    @staticmethod
    def _apply_cli_overrides(config: dict[str, Any], cli_overrides: Mapping[str, Any] | object | None) -> None:
        if cli_overrides is None:
            return

        def read(name: str) -> Any:
            if isinstance(cli_overrides, Mapping):
                return cli_overrides.get(name)
            return getattr(cli_overrides, name, None)

        seed = read("seed")
        if seed is not None:
            config["reproducibility"]["seed"] = seed

        device = read("device")
        if device is not None:
            config["device"]["mode"] = device

        output = read("output")
        if output is not None:
            config["output"]["root"] = output

        run_name = read("run_name")
        if run_name is not None:
            config["output"]["run_name"] = run_name
