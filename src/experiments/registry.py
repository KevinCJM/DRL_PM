from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.baselines import (
    BuyAndHoldStrategy,
    CageEIIEDistributionalStrategy,
    CageEIIEDistributionalNoCvarStrategy,
    CageEIIEFixedRho25Strategy,
    CageEIIEFixedRho50Strategy,
    CageEIIEFixedRho75Strategy,
    CageEIIEFrozenGateStrategy,
    CageEIIEJointLightStrategy,
    CageEIIEMultilevelGateStrategy,
    CageEIIENoCvarStrategy,
    CNNPPOBaselineStrategy,
    EqualWeightStrategy,
    FixedRatioStrategy,
    GTRCPOLiteStrategy,
    HRPStrategy,
    HybridDQNOptimizerEqualWeightStrategy,
    HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
    HybridDQNOptimizerMinimumVarianceStrategy,
    HybridDQNOptimizerRiskParityStrategy,
    HybridDQNOptimizerSharpeMaximizationStrategy,
    InverseVolatilityStrategy,
    MarkowitzMaxSharpeStrategy,
    MarkowitzMeanVarianceStrategy,
    MarkowitzMinVarianceStrategy,
    MinimumDrawdownStrategy,
    MomentumStrategy,
    NativeBernoulliGatedPPOBaselineStrategy,
    NativeDQNTemplateStrategy,
    NativeEIIEStrategy,
    NativeCNNPPOBaselineStrategy,
    NativePPOBaselineStrategy,
    PGPortfolioEIIEStrategy,
    PPODQNHierarchicalReimplementationStrategy,
    PPOBaselineStrategy,
    RiskEvaluationStrategy,
    RiskAwareGTRCPOStrategy,
    RiskParityStrategy,
)
from src.baselines.bernoulli_gated_ppo import BernoulliGatedPPOStrategy
from src.baselines.dqn_only import DQNOnlyStrategy
from src.baselines.eiie import EIIEStrategy
from src.config import DEFAULT_CONFIG, PROJECT_ROOT, VALID_EXPERIMENT_TYPES, assert_path_allowed
from src.envs.constraint_manager import ConstraintManager
from src.envs.cost_model import CostModel
from src.envs.portfolio_execution_core import PortfolioExecutionCore
from src.experiments.pipeline import (
    build_pipeline_artifacts,
    objective_metric,
    run_strategy_backtest,
    run_seed_stability_training,
    run_strategy_comparison,
    run_trained_model_experiment,
    run_trained_variant_matrix,
    run_trained_walk_forward_experiment,
)


OUTPUT_SCHEMA: dict[str, tuple[str, ...]] = {
    "daily_returns": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "pre_execution_return",
        "post_execution_return",
        "gross_return",
        "transaction_cost",
        "transaction_cost_on_initial_nav",
        "net_return",
        "portfolio_log_return",
        "nav",
    ),
    "daily_weights": ("date", "split", "seed", "fold_id", "model_name", "asset_id", "weight"),
    "daily_turnover": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "turnover",
        "rebalance_action",
        "rebalance_intensity",
        "average_holding_period",
    ),
    "daily_rebalance": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "rebalance_action",
        "rebalance_intensity",
        "estimated_turnover",
        "realized_turnover",
        "turnover",
        "estimated_cost",
        "realized_cost",
        "q_hold",
        "q_rebalance",
        "q_gap",
    ),
    "daily_costs": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "proportional_cost",
        "fixed_cost",
        "slippage_cost",
        "market_impact_cost",
        "total_transaction_cost",
        "estimated_cost",
        "realized_cost",
        "turnover",
    ),
}


TRADITIONAL_BASELINE_CLASSES = {
    "fixed_ratio": FixedRatioStrategy,
    "equal_weight": EqualWeightStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "traditional_markowitz_mean_variance": MarkowitzMeanVarianceStrategy,
    "markowitz": MarkowitzMeanVarianceStrategy,
    "markowitz_min_variance": MarkowitzMinVarianceStrategy,
    "markowitz_max_sharpe": MarkowitzMaxSharpeStrategy,
    "risk_parity": RiskParityStrategy,
    "inverse_volatility": InverseVolatilityStrategy,
    "minimum_drawdown": MinimumDrawdownStrategy,
    "risk_evaluation": RiskEvaluationStrategy,
    "hrp": HRPStrategy,
    "momentum": MomentumStrategy,
}
DEEP_BASELINE_CLASSES = {
    "ppo_proxy": PPOBaselineStrategy,
    "ppo_baseline": PPOBaselineStrategy,
    "cnn_ppo_proxy": CNNPPOBaselineStrategy,
    "cnn_ppo_baseline": CNNPPOBaselineStrategy,
    "bernoulli_gated_ppo_proxy": BernoulliGatedPPOStrategy,
    "bernoulli_gated_ppo": BernoulliGatedPPOStrategy,
    "dqn_template_proxy": DQNOnlyStrategy,
    "dqn_only": DQNOnlyStrategy,
    "eiie_proxy": EIIEStrategy,
    "eiie": EIIEStrategy,
    "ppo_native": NativePPOBaselineStrategy,
    "cnn_ppo_native": NativeCNNPPOBaselineStrategy,
    "bernoulli_gated_ppo_native": NativeBernoulliGatedPPOBaselineStrategy,
    "dqn_template_native": NativeDQNTemplateStrategy,
    "eiie_native": NativeEIIEStrategy,
    "pgportfolio_eiie_native": PGPortfolioEIIEStrategy,
    "ppo_dqn_hierarchical_reimplementation": PPODQNHierarchicalReimplementationStrategy,
    "cage_eiie_frozen_gate": CageEIIEFrozenGateStrategy,
    "cage_eiie_multilevel_gate": CageEIIEMultilevelGateStrategy,
    "cage_eiie_distributional": CageEIIEDistributionalStrategy,
    "cage_eiie_no_cvar": CageEIIENoCvarStrategy,
    "cage_eiie_distributional_no_cvar": CageEIIEDistributionalNoCvarStrategy,
    "cage_eiie_joint_light": CageEIIEJointLightStrategy,
    "cage_eiie_fixed_rho_25": CageEIIEFixedRho25Strategy,
    "cage_eiie_fixed_rho_50": CageEIIEFixedRho50Strategy,
    "cage_eiie_fixed_rho_75": CageEIIEFixedRho75Strategy,
    "graph_transformer_risk_constrained_actor_critic_lite": GTRCPOLiteStrategy,
    "gt_rcpo_lite": GTRCPOLiteStrategy,
    "risk_aware_graph_transformer_constrained_actor_critic": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_no_graph": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_no_transformer": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_no_cvar_constraint": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_no_cost_constraint": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_no_turnover_constraint": RiskAwareGTRCPOStrategy,
    "ra_gt_rcpo_mlp_actor_critic": RiskAwareGTRCPOStrategy,
    "hybrid_dqn_optimizer_equal_weight": HybridDQNOptimizerEqualWeightStrategy,
    "hybrid_dqn_optimizer_markowitz_mean_variance": HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
    "hybrid_dqn_optimizer_minimum_variance": HybridDQNOptimizerMinimumVarianceStrategy,
    "hybrid_dqn_optimizer_sharpe_maximization": HybridDQNOptimizerSharpeMaximizationStrategy,
    "hybrid_dqn_optimizer_risk_parity": HybridDQNOptimizerRiskParityStrategy,
}
HYBRID_DQN_OPTIMIZER_ALIAS = "hybrid_dqn_optimizer_reimplementation"
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES = (
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
)
ABLATION_TYPES = {"ablation", "input_matrix_ablation", "pca_ablation", "kernel_size_ablation", "reward_ablation"}
ABLATION_GUARD_TYPES = ABLATION_TYPES | {"auxiliary_task_sensitivity"}
SENSITIVITY_TYPES = {
    "transaction_cost_sensitivity",
    "asset_universe_sensitivity",
    "market_regime",
    "seed_stability",
    "auxiliary_task_sensitivity",
    "rebalance_frequency_analysis",
}
MODULE_ANALYSIS_TYPES = {
    "preference_conditioned_analysis",
    "uncertainty_analysis",
    "distributional_cvar_analysis",
    "partial_rebalance_analysis",
}
ABLATION_IGNORED_PATH_PREFIXES = (
    "config_hash",
    "device.",
    "experiment.",
    "full_reproduction.",
    "hpo.",
    "logging.",
    "output.",
    "registry.",
    "reproducibility.",
    "security.",
    "data_governance.",
    "training.checkpoint_include_replay_buffer",
    "execution_activity.",
    "rebalance.",
)
MATRIX_OUTPUT_NAMES = {
    "transaction_cost_sensitivity": "transaction_cost_sensitivity",
    "asset_universe_sensitivity": "asset_universe_sensitivity",
    "market_regime": "market_regime_results",
    "seed_stability": "seed_stability",
    "auxiliary_task_sensitivity": "auxiliary_task_sensitivity",
    "rebalance_frequency_analysis": "rebalance_frequency_analysis",
    "preference_conditioned_analysis": "preference_conditioned_results",
    "uncertainty_analysis": "uncertainty_results",
    "distributional_cvar_analysis": "distributional_cvar_results",
    "partial_rebalance_analysis": "partial_rebalance_results",
}


@dataclass(frozen=True)
class ExperimentContext:
    config: dict[str, Any]
    execution_core: PortfolioExecutionCore
    cost_model: CostModel
    constraint_manager: ConstraintManager
    output_schema: dict[str, tuple[str, ...]]
    device: Any | None = None
    run_dir: Path | None = None


@dataclass
class BaseExperiment:
    context: ExperimentContext
    experiment_type: str
    output_name: str

    @property
    def config(self) -> dict[str, Any]:
        return self.context.config

    @property
    def execution_core(self) -> PortfolioExecutionCore:
        return self.context.execution_core

    @property
    def cost_model(self) -> CostModel:
        return self.context.cost_model

    @property
    def constraint_manager(self) -> ConstraintManager:
        return self.context.constraint_manager

    @property
    def output_schema(self) -> dict[str, tuple[str, ...]]:
        return self.context.output_schema

    def run(self) -> dict[str, Any]:
        raise NotImplementedError(f"ERR_EXPERIMENT_NOT_IMPLEMENTED: {self.experiment_type}")


@dataclass
class MainModelExperiment(BaseExperiment):
    model_name: str

    def run(self) -> dict[str, Any]:
        result = run_trained_model_experiment(
            self.config,
            model_name=self.model_name,
            run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
        )
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        return result


@dataclass
class BaselineComparisonExperiment(BaseExperiment):
    baselines: dict[str, Any]

    def run(self) -> dict[str, Any]:
        result = run_strategy_comparison(
            self.config,
            self.baselines,
            segment="test",
            run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
        )
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        return result


@dataclass
class AblationExperiment(BaseExperiment):
    ablation_type: str
    ablation_id: str = ""
    changed_key_path: str = ""

    def run(self) -> dict[str, Any]:
        model_name = str(self.config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"))
        result = run_trained_variant_matrix(
            self.config,
            model_name=model_name,
            matrix_name=self.output_name,
            variants=_ablation_variants(self.config, self.experiment_type),
            run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
        )
        result["ablation_id"] = self.ablation_id
        result["changed_key_path"] = self.changed_key_path
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        return result


@dataclass
class WalkForwardExperiment(BaseExperiment):
    fold_id: str = "walk_forward"

    def run(self) -> dict[str, Any]:
        result = run_trained_walk_forward_experiment(
            self.config,
            model_name=str(self.config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo")),
            run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
        )
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        return result


@dataclass
class HPOExperiment(BaseExperiment):
    hpo_enabled: bool = True

    def run(self) -> dict[str, Any]:
        from src.experiments.run_experiment import run_hpo

        result = dict(run_hpo(self))
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        return result

    def run_trial(self, trial: Any, train_split: str, validation_split: str) -> dict[str, Any]:
        trial_config = _config_with_trial_params(self.config, trial)
        split_override = getattr(self, "active_split", None)
        model_name = _active_hpo_model_name(self, trial_config)
        if model_name in DEEP_BASELINE_CLASSES:
            artifacts = _cached_hpo_pipeline_artifacts(
                self,
                trial_config,
                split_override=split_override,
                params=getattr(trial, "params", {}),
            )
            artifact_kwargs = {"artifacts": artifacts} if artifacts is not None else {}
            result = run_strategy_backtest(
                trial_config,
                DEEP_BASELINE_CLASSES[model_name],
                model_name=model_name,
                segment=validation_split,
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir / f"trial_{trial.number}"),
                split_override=split_override,
                **artifact_kwargs,
            )
        else:
            result = run_trained_model_experiment(
                trial_config,
                model_name=model_name,
                train_split=train_split,
                validation_split=validation_split,
                test_split=validation_split,
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir / f"trial_{trial.number}"),
                split_override=split_override,
            )
        metric = str(trial_config.get("hpo", {}).get("metric") or "validation_metric")
        result["validation_metric"] = objective_metric(result, metric, config=trial_config)
        result["objective_value"] = result["validation_metric"]
        result["train_split"] = train_split
        result["validation_split"] = validation_split
        result["hpo_model_name"] = model_name
        return result

    def run_final_test(self, best_trial: Any, split: str) -> dict[str, Any]:
        final_config = _config_with_params(self.config, getattr(best_trial, "params", {}))
        split_override = getattr(self, "active_split", None)
        final_label = str(getattr(self, "final_test_label", "final_test"))
        model_name = _active_hpo_model_name(self, final_config)
        if model_name in DEEP_BASELINE_CLASSES:
            artifacts = _cached_hpo_pipeline_artifacts(
                self,
                final_config,
                split_override=split_override,
                params=getattr(best_trial, "params", {}),
            )
            artifact_kwargs = {"artifacts": artifacts} if artifacts is not None else {}
            result = run_strategy_backtest(
                final_config,
                DEEP_BASELINE_CLASSES[model_name],
                model_name=model_name,
                segment=split,
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir / final_label),
                split_override=split_override,
                **artifact_kwargs,
            )
        else:
            result = run_trained_model_experiment(
                final_config,
                model_name=model_name,
                test_split=split,
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir / final_label),
                split_override=split_override,
            )
        result["final_split"] = split
        result["best_trial_number"] = getattr(best_trial, "number", None)
        result["hpo_model_name"] = model_name
        return result


@dataclass
class SensitivityExperiment(BaseExperiment):
    sensitivity_type: str = ""

    def run(self) -> dict[str, Any]:
        model_name = str(self.config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"))
        if self.sensitivity_type == "seed_stability":
            result = run_seed_stability_training(
                self.config,
                model_name=model_name,
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
            )
        else:
            result = run_trained_variant_matrix(
                self.config,
                model_name=model_name,
                matrix_name=self.output_name,
                variants=_sensitivity_variants(self.config, self.sensitivity_type),
                run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
            )
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        result["sensitivity_type"] = self.sensitivity_type
        return result


@dataclass
class ModuleAnalysisExperiment(BaseExperiment):
    module_name: str = ""

    def run(self) -> dict[str, Any]:
        analysis_config = _config_for_module_model(self.config, self.experiment_type, self.module_name)
        model_name = str(analysis_config.get("model", {}).get("name", self.module_name or "module_analysis"))
        result = run_trained_model_experiment(
            analysis_config,
            model_name=model_name,
            run_dir=None if self.context.run_dir is None else str(self.context.run_dir),
        )
        result["experiment_type"] = self.experiment_type
        result["output_name"] = self.output_name
        result["module_name"] = self.module_name
        if self.output_name not in result:
            result[self.output_name] = _module_analysis_results(result, model_name, self.module_name, self.output_name)
        return result


@dataclass
class FullReproductionExperiment(BaseExperiment):
    sequence: tuple[str, ...] = (
        "main_model",
        "baseline_comparison",
        "ablation",
        "input_matrix_ablation",
        "pca_ablation",
        "kernel_size_ablation",
        "reward_ablation",
        "transaction_cost_sensitivity",
        "asset_universe_sensitivity",
        "auxiliary_task_sensitivity",
        "rebalance_frequency_analysis",
        "seed_stability",
        "hyperparameter_sweep",
        "market_regime",
        "preference_conditioned_analysis",
        "uncertainty_analysis",
        "distributional_cvar_analysis",
        "partial_rebalance_analysis",
        "walk_forward",
    )

    def run(self) -> dict[str, Any]:
        from src.experiments.run_all import run_experiment_matrix

        return run_experiment_matrix(
            self.config,
            registry=ExperimentRegistry(self.output_schema),
            device=self.context.device,
            run_dir=self.context.run_dir,
            experiment_sequence=self.sequence,
        )


class ExperimentRegistry:
    def __init__(self, output_schema: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self.output_schema = dict(output_schema or OUTPUT_SCHEMA)

    def create_experiment(
        self,
        config: Mapping[str, Any],
        device: Any | None = None,
        run_dir: str | Path | None = None,
    ) -> BaseExperiment:
        resolved_config = _config_copy(config)
        experiment_type = _experiment_type(resolved_config)
        ablation_meta = _validate_ablation_single_switch(resolved_config, experiment_type)
        context = self._context(resolved_config, device=device, run_dir=run_dir)
        if _hpo_enabled(resolved_config) or experiment_type == "hyperparameter_sweep":
            return HPOExperiment(context, experiment_type, "hpo_trials", hpo_enabled=True)
        if experiment_type == "main_model":
            return MainModelExperiment(
                context,
                experiment_type,
                "main_comparison",
                model_name=str(resolved_config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo")),
            )
        if experiment_type == "baseline_comparison":
            return BaselineComparisonExperiment(
                context,
                experiment_type,
                "baseline_comparison",
                baselines=_baseline_factories(resolved_config),
            )
        if experiment_type in ABLATION_TYPES:
            return AblationExperiment(
                context,
                experiment_type,
                _ablation_output_name(experiment_type),
                experiment_type,
                ablation_id=ablation_meta["ablation_id"],
                changed_key_path=ablation_meta["changed_key_path"],
            )
        if experiment_type == "walk_forward":
            return WalkForwardExperiment(context, experiment_type, "walk_forward_results")
        if experiment_type in SENSITIVITY_TYPES:
            return SensitivityExperiment(
                context,
                experiment_type,
                _matrix_output_name(experiment_type),
                sensitivity_type=experiment_type,
            )
        if experiment_type in MODULE_ANALYSIS_TYPES:
            return ModuleAnalysisExperiment(
                context,
                experiment_type,
                _matrix_output_name(experiment_type),
                module_name=experiment_type.removesuffix("_analysis"),
            )
        if experiment_type == "full_reproduction":
            return FullReproductionExperiment(context, experiment_type, "full_reproduction_summary")
        raise ValueError(f"ERR_EXPERIMENT_UNKNOWN_TYPE: experiment.type={experiment_type}")

    def _context(
        self,
        config: dict[str, Any],
        device: Any | None = None,
        run_dir: str | Path | None = None,
    ) -> ExperimentContext:
        cost_model = CostModel(config)
        constraint_manager = ConstraintManager(config)
        execution_core = PortfolioExecutionCore(config, cost_model=cost_model)
        execution_core.constraint_manager = constraint_manager
        return ExperimentContext(
            config=config,
            execution_core=execution_core,
            cost_model=cost_model,
            constraint_manager=constraint_manager,
            output_schema=self.output_schema,
            device=device,
            run_dir=None if run_dir is None else Path(run_dir),
        )


def create_experiment(
    config: Mapping[str, Any],
    device: Any | None = None,
    run_dir: str | Path | None = None,
) -> BaseExperiment:
    return ExperimentRegistry().create_experiment(config, device=device, run_dir=run_dir)


def _config_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("ERR_EXPERIMENT_CONFIG_TYPE")
    return deepcopy(dict(config))


def _experiment_type(config: Mapping[str, Any]) -> str:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("ERR_EXPERIMENT_UNKNOWN_TYPE: experiment.type")
    experiment_type = str(experiment.get("type", ""))
    if experiment_type not in VALID_EXPERIMENT_TYPES:
        raise ValueError(f"ERR_EXPERIMENT_UNKNOWN_TYPE: experiment.type={experiment_type}")
    return experiment_type


def _hpo_enabled(config: Mapping[str, Any]) -> bool:
    hpo = config.get("hpo")
    return bool(isinstance(hpo, Mapping) and hpo.get("enabled") is True)


def _active_hpo_model_name(experiment: HPOExperiment, config: Mapping[str, Any]) -> str:
    active = getattr(experiment, "active_model_name", None)
    if active is not None:
        return str(active)
    return str(config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"))


def _expand_baseline_aliases(model_names: Sequence[Any]) -> list[str]:
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


def _baseline_factories(config: Mapping[str, Any]) -> dict[str, Any]:
    baseline_config = config.get("baselines")
    if not isinstance(baseline_config, Mapping):
        baseline_config = DEFAULT_CONFIG["baselines"]
    result: dict[str, Any] = {}
    for name in baseline_config.get("traditional", ()):
        result[str(name)] = TRADITIONAL_BASELINE_CLASSES[str(name)]
    for name in _expand_baseline_aliases(baseline_config.get("deep", ())):
        result[str(name)] = DEEP_BASELINE_CLASSES[str(name)]
    native_config = baseline_config.get("native_rl")
    native_models = []
    if isinstance(native_config, Mapping):
        native_models = list(native_config.get("enabled_models", ()))
    native_models.extend(list(baseline_config.get("native", ())))
    for name in _expand_baseline_aliases(native_models):
        result[str(name)] = DEEP_BASELINE_CLASSES[str(name)]
    for name in baseline_config.get("external", ()):
        if str(name) != "pgportfolio_original_external":
            raise KeyError(str(name))
        result[str(name)] = None
    external_pgportfolio = baseline_config.get("external_pgportfolio")
    if isinstance(external_pgportfolio, Mapping) and external_pgportfolio.get("enabled") is True:
        result["pgportfolio_original_external"] = None
    return result


def _main_strategy_result(config: Mapping[str, Any], output_name: str, experiment_type: str) -> dict[str, Any]:
    result = run_trained_model_experiment(
        config,
        model_name=str(config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo")),
    )
    result["experiment_type"] = experiment_type
    result["output_name"] = output_name
    return result


def _module_analysis_results(
    result: Mapping[str, Any],
    model_name: str,
    module_name: str,
    output_name: str,
) -> pd.DataFrame:
    comparison = result.get("main_comparison")
    if isinstance(comparison, pd.DataFrame) and not comparison.empty:
        frame = comparison.copy()
    else:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        frame = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "status": result.get("status", "completed"),
                    **{str(key): value for key, value in dict(metrics).items()},
                }
            ]
        )
    if "module_name" not in frame.columns:
        frame["module_name"] = module_name
    if "analysis_name" not in frame.columns:
        frame["analysis_name"] = output_name
    if "status" not in frame.columns:
        frame["status"] = result.get("status", "completed")
    return frame


def _ablation_variants(config: Mapping[str, Any], experiment_type: str) -> list[dict[str, Any]]:
    if experiment_type == "input_matrix_ablation":
        return [_input_matrix_variant(config, matrix_id) for matrix_id in ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")]
    if experiment_type == "pca_ablation":
        return [
            _variant(config, "no_pca", "feature_reduction.pca.enabled", False, {"feature_reduction.pca.enabled": False}),
            _variant(
                config,
                "pca_80pct_explained_variance",
                "feature_reduction.pca.explained_variance",
                0.80,
                {"feature_reduction.pca.enabled": True, "feature_reduction.pca.explained_variance": 0.80, "feature_reduction.pca.fixed_components": None},
            ),
            _variant(
                config,
                "pca_90pct_explained_variance",
                "feature_reduction.pca.explained_variance",
                0.90,
                {"feature_reduction.pca.enabled": True, "feature_reduction.pca.explained_variance": 0.90, "feature_reduction.pca.fixed_components": None},
            ),
            _variant(
                config,
                "pca_95pct_explained_variance",
                "feature_reduction.pca.explained_variance",
                0.95,
                {"feature_reduction.pca.enabled": True, "feature_reduction.pca.explained_variance": 0.95, "feature_reduction.pca.fixed_components": None},
            ),
            _variant(
                config,
                "pca_99pct_explained_variance",
                "feature_reduction.pca.explained_variance",
                0.99,
                {"feature_reduction.pca.enabled": True, "feature_reduction.pca.explained_variance": 0.99, "feature_reduction.pca.fixed_components": None},
            ),
            *[
                _variant(
                    config,
                    f"pca_fixed_{components}",
                    "feature_reduction.pca.fixed_components",
                    components,
                    {
                        "feature_reduction.pca.enabled": True,
                        "feature_reduction.pca.fixed_components": components,
                    },
                )
                for components in (16, 32, 64, 128)
            ],
        ]
    if experiment_type == "kernel_size_ablation":
        return [
            _kernel_size_variant(config, "kernel_single_day_1x1", 1, 1),
            _kernel_size_variant(config, "kernel_single_day_cross_asset_1x3", 1, 3),
            _kernel_size_variant(config, "kernel_short_3x3", 3, 3),
            _kernel_size_variant(config, "kernel_week_5x3", 5, 3),
            _kernel_size_variant(config, "kernel_long_11x3", 11, 3),
            _kernel_size_variant(config, "kernel_long_21x3", 21, 3),
        ]
    if experiment_type == "reward_ablation":
        reward_modes = (
            "A0_raw_simple_return",
            "A1_log_return",
            "A2_net_log_return_after_cost",
            "A3_net_log_return_plus_turnover",
            "A4_net_log_return_plus_turnover_downside",
            "A5_net_log_return_plus_turnover_drawdown",
            "A6_net_log_return_plus_turnover_downside_drawdown",
            "A7_differential_sharpe",
            "A8_cvar_sensitive",
            "A9_benchmark_relative",
            "A10_ppo_lagrangian",
            "A11_regime_aware",
            "A12_multi_objective_preference_conditioned",
        )
        return [_variant(config, f"reward_{mode.split('_', 1)[0].lower()}", "reward.mode", mode, {"reward.mode": mode}) for mode in reward_modes]
    if experiment_type == "ablation":
        return _generic_component_ablation_variants(config)
    return [
        _variant(config, "base", "experiment.type", "main_model", {}),
        _variant(config, "current", "experiment.type", experiment_type, {}),
    ]


def _generic_component_ablation_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_config = _mapping(config.get("model"))
    encoder_config = _mapping(model_config.get("encoder"))
    attention_config = _mapping(encoder_config.get("cross_asset_attention"))
    dqn_config = _mapping(config.get("dqn"))
    auxiliary_config = _mapping(config.get("auxiliary"))

    if dqn_config.get("enabled") is False:
        return [
            _variant(config, "full_model", "dqn.enabled", True, {"dqn.enabled": True}),
            _variant(config, "without_dqn_gate", "dqn.enabled", False, {"dqn.enabled": False}),
        ]
    if auxiliary_config.get("enabled") is False:
        return [
            _variant(config, "full_model", "auxiliary.enabled", True, {"auxiliary.enabled": True}),
            _variant(config, "without_auxiliary", "auxiliary.enabled", False, {"auxiliary.enabled": False}),
        ]
    if str(model_config.get("default_encoder", "")).lower() == "mlp" or str(encoder_config.get("type", "")).lower() == "mlp":
        return [
            _variant(config, "full_model", "model.encoder.type", "cnn", {"model.default_encoder": "cnn", "model.encoder.type": "cnn"}),
            _variant(config, "mlp_encoder", "model.encoder.type", "mlp", {"model.default_encoder": "mlp", "model.encoder.type": "mlp"}),
        ]
    if attention_config.get("enabled") is True:
        return [
            _variant(config, "full_model", "model.encoder.cross_asset_attention.enabled", False, {"model.encoder.cross_asset_attention.enabled": False}),
            _variant(config, "attention_enabled", "model.encoder.cross_asset_attention.enabled", True, {"model.encoder.cross_asset_attention.enabled": True}),
        ]
    return [
        _variant(config, "full_model", "experiment.type", "main_model", {}),
        _variant(config, "current", "experiment.type", "ablation", {}),
    ]


def _kernel_size_variant(config: Mapping[str, Any], variant_id: str, time_kernel: int, asset_kernel: int) -> dict[str, Any]:
    return _variant(
        config,
        variant_id,
        "model.encoder.kernel_size",
        f"{time_kernel}x{asset_kernel}",
        {
            "model.default_encoder": "cnn",
            "model.encoder.type": "cnn",
            "model.encoder.kernel_size_time": int(time_kernel),
            "model.encoder.kernel_size_asset": int(asset_kernel),
        },
    )


def _sensitivity_variants(config: Mapping[str, Any], experiment_type: str) -> list[dict[str, Any]]:
    if experiment_type == "transaction_cost_sensitivity":
        values = [0.0, 0.0005, 0.0010, 0.0020, 0.0050]
        return [
            _variant(config, f"cost_{value:g}", "cost_model.proportional_cost", value, {"cost_model.proportional_cost": value})
            for value in values
        ]
    if experiment_type == "asset_universe_sensitivity":
        return _asset_universe_variants(config)
    if experiment_type == "auxiliary_task_sensitivity":
        return [
            _variant(config, "auxiliary_off", "auxiliary.enabled", False, {"auxiliary.enabled": False}),
            _variant(config, "auxiliary_on", "auxiliary.enabled", True, {"auxiliary.enabled": True}),
        ]
    if experiment_type == "rebalance_frequency_analysis":
        return [
            _variant(config, "rebalance_daily", "rebalance.mode", "daily", {"rebalance.mode": "daily", "rebalance.every_n_days": 1}),
            _variant(config, "rebalance_weekly", "rebalance.mode", "weekly", {"rebalance.mode": "weekly", "rebalance.every_n_days": 5}),
            _variant(config, "rebalance_monthly", "rebalance.mode", "monthly", {"rebalance.mode": "monthly", "rebalance.every_n_days": 20}),
            _variant(config, "rebalance_quarterly", "rebalance.mode", "quarterly", {"rebalance.mode": "quarterly", "rebalance.every_n_days": 60}),
            _variant(
                config,
                "rebalance_threshold_weight_drift",
                "rebalance.mode",
                "threshold_weight_drift",
                {"rebalance.mode": "threshold_weight_drift", "rebalance.threshold_weight_drift": 0.05},
            ),
            _variant(
                config,
                "rebalance_threshold_turnover",
                "rebalance.mode",
                "threshold_turnover",
                {"rebalance.mode": "threshold_turnover", "rebalance.threshold_turnover": 0.10},
            ),
        ]
    if experiment_type == "market_regime":
        return [
            _variant(config, "regime_bull", "market_regime.segment", "bull", {"market_regime.segment": "bull"}),
            _variant(config, "regime_bear", "market_regime.segment", "bear", {"market_regime.segment": "bear"}),
            _variant(config, "regime_sideways", "market_regime.segment", "sideways", {"market_regime.segment": "sideways"}),
            _variant(config, "regime_high_volatility", "market_regime.segment", "high_volatility", {"market_regime.segment": "high_volatility"}),
            _variant(config, "regime_low_volatility", "market_regime.segment", "low_volatility", {"market_regime.segment": "low_volatility"}),
        ]
    return [_variant(config, "current", "experiment.type", experiment_type, {})]


def _input_matrix_variant(config: Mapping[str, Any], matrix_id: str) -> dict[str, Any]:
    updates: dict[str, Any] = {"feature_matrix.input_matrix_id": matrix_id}
    if matrix_id in {"M6", "M7"}:
        updates["feature_reduction.pca.enabled"] = True
    else:
        updates["feature_reduction.pca.enabled"] = False
    updates["feature_reduction.feature_selection.enabled"] = matrix_id == "M7"
    return _variant(config, f"input_matrix_{matrix_id}", "feature_matrix.input_matrix_id", matrix_id, updates)


def _asset_universe_variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    pools = _configured_asset_pools(config) or _asset_pools_from_universe(config)
    variants = [
        _variant(
            config,
            "asset_pool_all",
            "data.asset_universe_pools",
            "all",
            {"data.asset_universe_pools": [], "data.asset_universe_assets": []},
        )
    ]
    for pool in pools:
        variants.append(
            _variant(
                config,
                f"asset_pool_{_variant_key(pool)}",
                "data.asset_universe_pools",
                pool,
                {"data.asset_universe_pools": [pool], "data.asset_universe_assets": []},
            )
        )
    if len(variants) == 1:
        variants.extend(
            [
                _variant(
                    config,
                    "common_history_off",
                    "data.strict_common_history_mode",
                    False,
                    {"data.strict_common_history_mode": False},
                ),
                _variant(
                    config,
                    "common_history_on",
                    "data.strict_common_history_mode",
                    True,
                    {"data.strict_common_history_mode": True},
                ),
            ]
        )
    return variants


def _configured_asset_pools(config: Mapping[str, Any]) -> list[str]:
    sensitivity = config.get("asset_universe_sensitivity")
    if isinstance(sensitivity, Mapping) and sensitivity.get("pools"):
        return _string_list(sensitivity.get("pools"))
    data_config = config.get("data")
    if isinstance(data_config, Mapping) and data_config.get("asset_universe_pools"):
        return _string_list(data_config.get("asset_universe_pools"))
    return []


def _asset_pools_from_universe(config: Mapping[str, Any]) -> list[str]:
    data_config = config.get("data")
    if not isinstance(data_config, Mapping):
        return []
    path = data_config.get("asset_universe_path")
    if path is None:
        return []
    whitelist = _path_whitelist(config)
    path_obj = _asset_universe_path(path, whitelist)
    try:
        frame = pd.read_csv(path_obj)
    except Exception:
        return []
    if "pool" not in frame.columns:
        return []
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).eq("ok")]
    return sorted({str(value) for value in frame["pool"].dropna().tolist() if str(value)})


def _asset_universe_path(path: Any, whitelist: Sequence[str | Path]) -> Path:
    raw_path = Path(str(path))
    if not raw_path.is_absolute():
        cwd_path = (Path.cwd() / raw_path).resolve()
        if cwd_path.exists():
            return assert_path_allowed(cwd_path, whitelist, "data.asset_universe_path")
    return assert_path_allowed(path, whitelist, "data.asset_universe_path")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)]


def _path_whitelist(config: Mapping[str, Any]) -> list[str | Path]:
    security = config.get("security")
    if isinstance(security, Mapping):
        whitelist = security.get("path_whitelist")
        if whitelist:
            return list(whitelist)
    return [PROJECT_ROOT]


def _variant_key(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value)).strip("_").lower() or "pool"


def _variant(
    config: Mapping[str, Any],
    variant_id: str,
    changed_key_path: str,
    variant_value: Any,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    for path, value in updates.items():
        _set_path(resolved, path, value)
    return {
        "variant_id": variant_id,
        "changed_key_path": changed_key_path,
        "variant_value": variant_value,
        "config": resolved,
    }


def _set_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    target = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


MODULE_MODEL_NAMES = {
    "preference_conditioned_analysis": "preference_conditioned_gated_ppo",
    "uncertainty_analysis": "uncertainty_aware_gated_ppo",
    "distributional_cvar_analysis": "distributional_cvar_gated_ppo",
    "partial_rebalance_analysis": "partial_rebalance_gated_ppo",
}
PIPELINE_ARTIFACT_HPO_PARAM_PREFIXES = (
    "data.",
    "data_governance.",
    "feature_matrix.",
    "feature_reduction.",
    "split.",
    "splits.",
    "env.window_size",
)
PIPELINE_ARTIFACT_CONFIG_KEYS = (
    "data",
    "data_governance",
    "env",
    "feature_matrix",
    "feature_reduction",
    "security",
    "split",
    "splits",
)


def _cached_hpo_pipeline_artifacts(
    experiment: HPOExperiment,
    config: Mapping[str, Any],
    *,
    split_override: Any | None,
    params: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if split_override is not None or _hpo_params_affect_pipeline_artifacts(params):
        return None
    cache = getattr(experiment, "_hpo_pipeline_artifact_cache", None)
    if cache is None:
        cache = {}
        setattr(experiment, "_hpo_pipeline_artifact_cache", cache)
    key = _pipeline_artifact_cache_key(config)
    if key not in cache:
        cache[key] = build_pipeline_artifacts(config)
    return cache[key]


def _hpo_params_affect_pipeline_artifacts(params: Mapping[str, Any]) -> bool:
    for name in params:
        path = str(name)
        if "." not in path:
            path = _default_hpo_param_path(path)
        if path.startswith(PIPELINE_ARTIFACT_HPO_PARAM_PREFIXES):
            return True
    return False


def _pipeline_artifact_cache_key(config: Mapping[str, Any]) -> str:
    relevant = {
        key: deepcopy(config.get(key))
        for key in PIPELINE_ARTIFACT_CONFIG_KEYS
        if key in config
    }
    return json.dumps(relevant, sort_keys=True, default=str, ensure_ascii=False)


def _config_for_module_model(config: Mapping[str, Any], experiment_type: str, module_name: str) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    model_config = dict(resolved.get("model", {}))
    model_config["name"] = MODULE_MODEL_NAMES.get(experiment_type, module_name or "module_analysis")
    resolved["model"] = model_config
    return resolved


def _config_with_trial_params(config: Mapping[str, Any], trial: Any) -> dict[str, Any]:
    hpo_config = config.get("hpo")
    search_space = hpo_config.get("search_space") if isinstance(hpo_config, Mapping) else None
    params: dict[str, Any] = {}
    if isinstance(search_space, Mapping):
        for name, spec in search_space.items():
            params[str(name)] = _suggest_param(trial, str(name), spec)
    return _config_with_params(config, params)


def _config_with_params(config: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    for key, value in params.items():
        path = str(key)
        if "." not in path:
            path = _default_hpo_param_path(path)
        _set_nested_value(resolved, path.split("."), value)
    return resolved


def _suggest_param(trial: Any, name: str, spec: Any) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    param_type = str(spec.get("type", "float"))
    if "choices" in spec:
        return trial.suggest_categorical(name, list(spec["choices"]))
    if param_type == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    low = float(spec["low"])
    high = float(spec["high"])
    return trial.suggest_float(name, low, high, log=bool(spec.get("log", False)))


def _default_hpo_param_path(name: str) -> str:
    mapping = {
        "learning_rate": "optimizer.learning_rate",
        "weight_decay": "optimizer.weight_decay",
        "ppo_lr": "optimizer.ppo_lr",
        "dqn_lr": "optimizer.dqn_lr",
        "rebalance_intensity": "rebalance_intensity",
    }
    return mapping.get(name, name)


def _set_nested_value(config: dict[str, Any], path: Sequence[str], value: Any) -> None:
    cursor: dict[str, Any] = config
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value


def _validate_ablation_single_switch(config: Mapping[str, Any], experiment_type: str) -> dict[str, str]:
    if experiment_type not in ABLATION_GUARD_TYPES:
        return {"ablation_id": "", "changed_key_path": ""}

    _assert_real_costs_enabled(config)
    changed_paths = _ablation_changed_paths(DEFAULT_CONFIG, config)
    families = {_ablation_family(experiment_type, path) for path in changed_paths}
    if "invalid" in families or len(families) > 1:
        joined = ",".join(changed_paths)
        raise ValueError(f"ERR_EXPERIMENT_ABLATION_NOT_SINGLE_SWITCH: {joined}")

    if not changed_paths:
        changed_key_path = ""
    elif len(changed_paths) == 1:
        changed_key_path = changed_paths[0]
    else:
        changed_key_path = next(iter(families))
    ablation_id = f"{experiment_type}.{changed_key_path or 'base'}"
    return {"ablation_id": ablation_id, "changed_key_path": changed_key_path}


def _ablation_changed_paths(base: Any, current: Any, path: str = "") -> list[str]:
    if _ignored_ablation_path(path):
        return []
    if isinstance(base, Mapping) and isinstance(current, Mapping):
        paths: list[str] = []
        for key in base:
            key_path = f"{path}.{key}" if path else str(key)
            if key in current:
                paths.extend(_ablation_changed_paths(base[key], current[key], key_path))
        for key in current:
            if key not in base:
                key_path = f"{path}.{key}" if path else str(key)
                if not _ignored_ablation_path(key_path):
                    paths.append(key_path)
        return paths
    return [] if base == current else [path]


def _ignored_ablation_path(path: str) -> bool:
    return any(path == prefix.removesuffix(".") or path.startswith(prefix) for prefix in ABLATION_IGNORED_PATH_PREFIXES)


def _ablation_family(experiment_type: str, path: str) -> str:
    if experiment_type == "input_matrix_ablation":
        return "feature_matrix.input_matrix_id" if path == "feature_matrix.input_matrix_id" else "invalid"
    if experiment_type == "pca_ablation":
        return "feature_reduction.pca" if path.startswith("feature_reduction.pca.") else "invalid"
    if experiment_type == "kernel_size_ablation":
        if path in {"model.default_encoder", "model.encoder.type"}:
            return "model.encoder.kernel_size"
        if path in {"model.encoder.kernel_size_time", "model.encoder.kernel_size_asset"}:
            return "model.encoder.kernel_size"
        return "invalid"
    if experiment_type == "reward_ablation":
        if path.startswith("reward.") or path.startswith("reward_ablation."):
            return "reward"
        return "invalid"
    if experiment_type == "auxiliary_task_sensitivity":
        return "auxiliary" if path.startswith("auxiliary.") else "invalid"
    if path.startswith("feature_reduction.pca."):
        return "feature_reduction.pca"
    if path.startswith("reward_ablation.") or path.startswith("reward."):
        return "reward"
    return path.split(".", 1)[0]


def _assert_real_costs_enabled(config: Mapping[str, Any]) -> None:
    cost_model = config.get("cost_model")
    if not isinstance(cost_model, Mapping):
        return
    proportional_cost = float(cost_model.get("proportional_cost", 0.0) or 0.0)
    fixed_cost = float(cost_model.get("fixed_cost", 0.0) or 0.0)
    slippage = float(cost_model.get("slippage", 0.0) or 0.0)
    market_impact_enabled = bool(cost_model.get("market_impact_enabled", False))
    market_impact_coef = float(cost_model.get("market_impact_coef", 0.0) or 0.0)
    if (
        proportional_cost <= 0.0
        and fixed_cost <= 0.0
        and slippage <= 0.0
        and (not market_impact_enabled or market_impact_coef <= 0.0)
    ):
        raise ValueError("ERR_EXPERIMENT_ABLATION_NOT_SINGLE_SWITCH: cost_model.transaction_cost_removed")


def _ablation_output_name(experiment_type: str) -> str:
    if experiment_type == "input_matrix_ablation":
        return "input_matrix_ablation_results"
    if experiment_type == "pca_ablation":
        return "PCA_ablation_results"
    if experiment_type == "kernel_size_ablation":
        return "kernel_size_ablation_results"
    if experiment_type == "reward_ablation":
        return "reward_ablation_results"
    return "ablation_results"


def _matrix_output_name(experiment_type: str) -> str:
    return MATRIX_OUTPUT_NAMES[experiment_type]


__all__ = [
    "AblationExperiment",
    "BaseExperiment",
    "BaselineComparisonExperiment",
    "ExperimentContext",
    "ExperimentRegistry",
    "FullReproductionExperiment",
    "HPOExperiment",
    "MainModelExperiment",
    "ModuleAnalysisExperiment",
    "OUTPUT_SCHEMA",
    "SensitivityExperiment",
    "WalkForwardExperiment",
    "create_experiment",
]
