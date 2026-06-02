from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.dqn_agent import DQNAgentConfig
from src.baselines.base_strategy import BaseStrategy
from src.baselines.eiie import _continuous_weight_rebalance_decision
from src.baselines.risk_parity import _optimize_risk_parity
from src.data.loader import DataContractError
from src.data.leakage_checks import assert_decision_visibility_contract
from src.data.splits import SplitSpec
from src.envs.backtest_engine import BacktestEngine, _build_decision_market_state
from src.envs.constraint_manager import ConstraintManager
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.encoders import EncoderFactory
from src.utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint, save_checkpoint
from src.utils.optimization import PortfolioOptimizationResult, optimize_long_only_portfolio, shrink_covariance


HYBRID_DQN_OPTIMIZER_ALIAS = "hybrid_dqn_optimizer_reimplementation"
HYBRID_DQN_SIGNAL_ACTION_EXCLUDE = 0
HYBRID_DQN_SIGNAL_ACTION_NEUTRAL = 1
HYBRID_DQN_SIGNAL_ACTION_INCLUDE = 2
HYBRID_DQN_SIGNAL_ACTION_DIM = 3
HYBRID_DQN_SIGNAL_INVALID_Q = -1.0e9
HYBRID_DQN_OPTIMIZER_LOOKBACK_WINDOW = 252
HYBRID_DQN_OPTIMIZER_MIN_OBSERVATIONS = 60
HYBRID_DQN_OPTIMIZER_COVARIANCE_SHRINKAGE = 0.1
HYBRID_DQN_OPTIMIZER_RISK_FREE_RATE = 0.0
HYBRID_DQN_OPTIMIZER_LAMBDA_RISK = 1.0
HYBRID_DQN_OPTIMIZER_MARKOWITZ_MAXITER = 200
HYBRID_DQN_OPTIMIZER_RISK_PARITY_MAXITER = 300


@dataclass(frozen=True)
class HybridDQNOptimizerChildSpec:
    paper_model_id: str
    child_model_name: str
    optimizer_name: str


HYBRID_DQN_OPTIMIZER_CHILD_SPECS: Mapping[str, HybridDQNOptimizerChildSpec] = MappingProxyType(
    {
        "hybrid_dqn_optimizer_equal_weight": HybridDQNOptimizerChildSpec(
            paper_model_id="hybrid_dqn_optimizer_equal_weight",
            child_model_name="hybrid_dqn_optimizer_equal_weight",
            optimizer_name="equal_weight",
        ),
        "hybrid_dqn_optimizer_markowitz_mean_variance": HybridDQNOptimizerChildSpec(
            paper_model_id="hybrid_dqn_optimizer_markowitz_mean_variance",
            child_model_name="hybrid_dqn_optimizer_markowitz_mean_variance",
            optimizer_name="markowitz_mean_variance",
        ),
        "hybrid_dqn_optimizer_minimum_variance": HybridDQNOptimizerChildSpec(
            paper_model_id="hybrid_dqn_optimizer_minimum_variance",
            child_model_name="hybrid_dqn_optimizer_minimum_variance",
            optimizer_name="minimum_variance",
        ),
        "hybrid_dqn_optimizer_sharpe_maximization": HybridDQNOptimizerChildSpec(
            paper_model_id="hybrid_dqn_optimizer_sharpe_maximization",
            child_model_name="hybrid_dqn_optimizer_sharpe_maximization",
            optimizer_name="sharpe_maximization",
        ),
        "hybrid_dqn_optimizer_risk_parity": HybridDQNOptimizerChildSpec(
            paper_model_id="hybrid_dqn_optimizer_risk_parity",
            child_model_name="hybrid_dqn_optimizer_risk_parity",
            optimizer_name="risk_parity",
        ),
    }
)
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES = tuple(HYBRID_DQN_OPTIMIZER_CHILD_SPECS.keys())


class HybridDQNAssetSignalQNetwork(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        n_features: int,
        window_size: int,
        n_assets: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.n_features = _positive_int("n_features", n_features)
        self.window_size = _positive_int("window_size", window_size)
        self.n_assets = _positive_int("n_assets", n_assets)
        self.action_dim = HYBRID_DQN_SIGNAL_ACTION_DIM
        layers: list[nn.Module] = []
        in_dim = self.latent_dim + self.n_features * self.window_size
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout))])
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, market_image: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: latent must be [batch,latent_dim]")
        if market_image.ndim != 4:
            raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: market_image must be [batch,n_features,window_size,n_assets]")
        batch_size, n_features, window_size, n_assets = market_image.shape
        if n_features != self.n_features or window_size != self.window_size or n_assets != self.n_assets:
            raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: market_image dimensions do not match config")
        if latent.shape[0] != batch_size:
            raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: latent batch must match market_image batch")
        asset_features = market_image.permute(0, 3, 1, 2).reshape(batch_size, n_assets, -1)
        latent_features = latent.unsqueeze(1).expand(-1, n_assets, -1)
        q_values = self.net(torch.cat([latent_features, asset_features], dim=-1))
        return q_values.reshape(batch_size, n_assets, self.action_dim)


class HybridDQNOptimizerReimplementationStrategy(BaseStrategy):
    strategy_name = HYBRID_DQN_OPTIMIZER_ALIAS
    optimizer_name = "hybrid_dqn_optimizer"
    fit_required = True

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.device = _device(self.config)
        self.model_config = _resolved_model_config(self.config)
        self.n_assets = int(self.model_config["n_assets"])
        self.n_features = int(self.model_config["n_features"])
        self.window_size = int(self.model_config["window_size"])
        self.latent_dim = int(self.model_config["latent_dim"])
        signal_config = _mapping(self.model_config.get("dqn") or _mapping(self.model_config.get("model")).get("dqn"))
        self.dqn_config = DQNAgentConfig.from_mapping(self.model_config)
        self.encoder = EncoderFactory.create(self.model_config).to(self.device)
        self.asset_signal_q_network = HybridDQNAssetSignalQNetwork(
            latent_dim=self.latent_dim,
            n_features=self.n_features,
            window_size=self.window_size,
            n_assets=self.n_assets,
            hidden_dims=_hidden_dims(signal_config.get("hidden_dims"), default=(128, 64)),
            dropout=float(signal_config.get("dropout", _mapping(self.model_config.get("model")).get("dropout", 0.10))),
        ).to(self.device)
        self.target_asset_signal_q_network = HybridDQNAssetSignalQNetwork(
            latent_dim=self.latent_dim,
            n_features=self.n_features,
            window_size=self.window_size,
            n_assets=self.n_assets,
            hidden_dims=_hidden_dims(signal_config.get("hidden_dims"), default=(128, 64)),
            dropout=float(signal_config.get("dropout", _mapping(self.model_config.get("model")).get("dropout", 0.10))),
        ).to(self.device)
        self.target_asset_signal_q_network.load_state_dict(self.asset_signal_q_network.state_dict())
        self.signal_optimizer = torch.optim.AdamW(
            _dqn_signal_parameters(self.asset_signal_q_network, self.encoder, self.dqn_config),
            lr=self.dqn_config.lr,
        )
        child_spec = HYBRID_DQN_OPTIMIZER_CHILD_SPECS.get(self.strategy_name)
        if child_spec is None:
            self.paper_model_id = self.strategy_name
            self.child_model_name = None
            self.optimizer_name = "hybrid_dqn_optimizer"
        else:
            self.paper_model_id = child_spec.paper_model_id
            self.child_model_name = child_spec.child_model_name
            self.optimizer_name = child_spec.optimizer_name
        self.optimizer_parameters = _optimizer_parameters(self.optimizer_name, self.config)
        self.training_history: pd.DataFrame = pd.DataFrame()
        self.training_result: dict[str, Any] = _training_result(
            self.paper_model_id,
            self.child_model_name,
            self.optimizer_name,
            self.training_history,
        )
        self._checkpoint_loaded_path: str | None = None
        self._gradient_update_step = 0

    def fit(
        self,
        train_data: Any | None = None,
        validation_data: Any | None = None,
    ) -> HybridDQNOptimizerReimplementationStrategy:
        self.training_history = pd.DataFrame()
        self.is_fitted = False
        if self.strategy_name not in HYBRID_DQN_OPTIMIZER_CHILD_SPECS:
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="deferred_variant",
                reason="orchestration_alias_only",
            )
            return self
        if not isinstance(train_data, Mapping):
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="missing_train_data",
            )
            return self

        train_dates = _dates(train_data.get("dates"))
        if train_dates.empty:
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="empty_train_dates",
            )
            return self
        validation_dates = _dates(_mapping(validation_data).get("dates"))
        if len(validation_dates) < 2:
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_no_finite_validation_metric",
                reason="missing_validation_data",
            )
            return self
        try:
            dataset = train_data["dataset"]
        except KeyError:
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="missing_dataset",
            )
            return self

        split = SplitSpec(
            train_dates=train_dates,
            validation_dates=validation_dates,
            test_dates=validation_dates,
            fold_id=str(_mapping(train_data.get("config")).get("fold_id", self.paper_model_id)),
        )
        market_image_dataset = train_data.get("market_image_dataset")
        try:
            train_env = PortfolioRebalanceEnv(
                dataset,
                split,
                config=self.config,
                segment="train",
                market_image_dataset=market_image_dataset,
            )
        except DataContractError as exc:
            if exc.code != "ERR_SPLIT_EMPTY":
                raise
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="empty_train_env",
            )
            return self

        native_cfg = _native_rl_config(self.config)
        training_cfg = _mapping(self.config.get("training"))
        epochs = max(1, int(native_cfg.get("epochs", training_cfg.get("epochs", 1))))
        max_train_steps = _optional_positive_int(native_cfg.get("max_train_steps", training_cfg.get("max_train_steps")))
        max_validation_steps = _optional_positive_int(
            native_cfg.get("max_validation_steps", training_cfg.get("max_validation_steps"))
        )
        max_gradient_updates_per_epoch = _optional_positive_int(
            native_cfg.get(
                "max_gradient_updates_per_epoch",
                training_cfg.get("max_gradient_updates_per_epoch"),
            )
        )

        history_rows: list[dict[str, Any]] = []
        env_steps = 0
        gradient_updates = 0
        best_validation_metric = -np.inf
        failure_status: str | None = None
        failure_reason: str | None = None
        checkpoint_paths = self._checkpoint_paths()
        self._gradient_update_step = 0

        for epoch in range(epochs):
            epoch_result = self._train_epoch(
                train_env,
                max_steps=max_train_steps,
                max_gradient_updates=max_gradient_updates_per_epoch,
            )
            env_steps += int(epoch_result["env_steps"])
            gradient_updates += int(epoch_result["gradient_updates"])
            epoch_status = str(epoch_result["status"])
            if epoch_result["status"] != "completed":
                failure_status = epoch_status
                failure_reason = str(epoch_result.get("reason") or failure_status)
                validation_metric = np.nan
            else:
                validation_metric = _hybrid_validation_metric_from_backtest(
                    self,
                    dataset,
                    split,
                    market_image_dataset=market_image_dataset,
                    max_steps=max_validation_steps,
                )
                if not np.isfinite(validation_metric):
                    failure_status = "failed_no_finite_validation_metric"
                    failure_reason = "failed_no_finite_validation_metric"
                    epoch_status = failure_status
                elif validation_metric > best_validation_metric:
                    best_validation_metric = float(validation_metric)
                    self._save_checkpoint_state(checkpoint_paths["best"], epoch, gradient_updates, best_validation_metric)

            history_rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(epoch_result.get("train_reward", np.nan)),
                    "validation_metric": float(validation_metric),
                    "loss": float(epoch_result.get("loss", np.nan)),
                    "max_train_steps": max_train_steps,
                    "max_validation_steps": max_validation_steps,
                    "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
                    "include_count": int(epoch_result.get("include_count", 0)),
                    "neutral_count": int(epoch_result.get("neutral_count", 0)),
                    "exclude_count": int(epoch_result.get("exclude_count", 0)),
                    "selected_asset_count": int(epoch_result.get("selected_asset_count", 0)),
                    "optimizer_asset_count": int(epoch_result.get("optimizer_asset_count", 0)),
                    "optimizer_fallback_count": int(epoch_result.get("optimizer_fallback_count", 0)),
                    "status": epoch_status,
                }
            )
            if failure_status is not None:
                break

        self.training_history = pd.DataFrame(history_rows)
        if failure_status is not None:
            if failure_status == "failed_no_finite_validation_metric" and checkpoint_paths["best"] is not None:
                try:
                    checkpoint_paths["best"].unlink(missing_ok=True)
                except OSError:
                    pass
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status=failure_status,
                reason=failure_reason,
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            )
            return self

        self._save_checkpoint_state(
            checkpoint_paths["last"],
            epochs - 1,
            gradient_updates,
            None if not np.isfinite(best_validation_metric) else best_validation_metric,
        )
        if not _has_finite_validation(self.training_history):
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_no_finite_validation_metric",
                reason="failed_no_finite_validation_metric",
                checkpoint_best_path=None,
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            )
            return self
        if checkpoint_paths["best"] is None or not checkpoint_paths["best"].exists():
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_missing_best_checkpoint",
                reason="failed_missing_best_checkpoint",
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                best_validation_metric=float(best_validation_metric),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            )
            return self
        try:
            self._load_checkpoint_state(checkpoint_paths["best"])
        except Exception:
            self.training_result = _training_result(
                self.paper_model_id,
                self.child_model_name,
                self.optimizer_name,
                self.training_history,
                status="failed_checkpoint_load",
                reason="failed_checkpoint_load",
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                best_validation_metric=float(best_validation_metric),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            )
            return self

        self.training_result = _training_result(
            self.paper_model_id,
            self.child_model_name,
            self.optimizer_name,
            self.training_history,
            status="completed",
            reason=None,
            checkpoint_best_path=_path_string(checkpoint_paths["best"]),
            checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            evaluated_checkpoint_path=_path_string(checkpoint_paths["best"]),
            best_validation_metric=float(best_validation_metric),
            env_steps=env_steps,
            gradient_updates=gradient_updates,
            max_train_steps=max_train_steps,
            max_validation_steps=max_validation_steps,
            max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            rankable_in_unified_table=True,
        )
        self.is_fitted = True
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        if self.is_fitted is not True:
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                f"ERR_STRATEGY_ACTION_CONTRACT: {self.strategy_name} not fitted",
            )
        return self.validate_portfolio_action(self._policy_action(state, portfolio))

    def _asset_action_q_values(self, decision_input: DecisionMarketState | Mapping[str, Any]) -> np.ndarray:
        market_image = _market_image(decision_input, self.n_features, self.window_size, self.n_assets)
        market_image_tensor = torch.as_tensor(
            np.nan_to_num(market_image, nan=0.0, posinf=0.0, neginf=0.0)[None, ...],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            latent = self.encoder(market_image_tensor)
            q_values = self.asset_signal_q_network(latent, market_image_tensor)
        return q_values.detach().cpu().numpy()[0].astype(np.float32, copy=False)

    def _mask_asset_action_q_values(self, asset_action_q_values: Any, available_mask: Any) -> np.ndarray:
        q_values = _asset_action_q_array(asset_action_q_values, self.n_assets)
        available = _available_mask(available_mask, self.n_assets)
        valid_rows = np.isfinite(q_values).all(axis=1) & available
        masked = np.asarray(q_values, dtype=np.float32).copy()
        masked[~valid_rows, :] = HYBRID_DQN_SIGNAL_INVALID_Q
        masked[~valid_rows, HYBRID_DQN_SIGNAL_ACTION_EXCLUDE] = 0.0
        return masked

    def _select_asset_signal_actions(self, asset_action_q_values: Any, available_mask: Any) -> np.ndarray:
        masked = self._mask_asset_action_q_values(asset_action_q_values, available_mask)
        actions = np.argmax(masked, axis=1).astype(np.int64)
        available = _available_mask(available_mask, self.n_assets)
        finite_rows = np.isfinite(_asset_action_q_array(asset_action_q_values, self.n_assets)).all(axis=1)
        actions[~available | ~finite_rows] = HYBRID_DQN_SIGNAL_ACTION_EXCLUDE
        return actions

    def _select_candidate_assets(self, asset_action_q_values: Any, available_mask: Any) -> dict[str, Any]:
        q_values = _asset_action_q_array(asset_action_q_values, self.n_assets)
        available = _available_mask(available_mask, self.n_assets)
        finite_rows = np.isfinite(q_values).all(axis=1)
        actions = self._select_asset_signal_actions(q_values, available)
        min_selected_assets = 2
        raw_include = (actions == HYBRID_DQN_SIGNAL_ACTION_INCLUDE) & available & finite_rows
        candidate_mask = raw_include.copy()
        selected_asset_count = int(raw_include.sum())
        available_asset_count = int(available.sum())

        if selected_asset_count < min_selected_assets and available_asset_count >= min_selected_assets:
            fill_scores = q_values[:, HYBRID_DQN_SIGNAL_ACTION_INCLUDE] - np.maximum(
                q_values[:, HYBRID_DQN_SIGNAL_ACTION_NEUTRAL],
                q_values[:, HYBRID_DQN_SIGNAL_ACTION_EXCLUDE],
            )
            fill_scores = np.where(np.isfinite(fill_scores), fill_scores, -np.inf)
            fill_order = np.argsort(-fill_scores, kind="stable")
            eligible = available & finite_rows & ~raw_include
            for asset_idx in fill_order:
                if int(candidate_mask.sum()) >= min_selected_assets:
                    break
                if bool(eligible[asset_idx]):
                    candidate_mask[asset_idx] = True

        return {
            "candidate_asset_indices": np.flatnonzero(candidate_mask).astype(np.int64),
            "asset_signal_actions": actions,
            "selected_asset_count": selected_asset_count,
            "optimizer_asset_count": int(candidate_mask.sum()),
            "include_count": int((actions == HYBRID_DQN_SIGNAL_ACTION_INCLUDE).sum()),
            "neutral_count": int((actions == HYBRID_DQN_SIGNAL_ACTION_NEUTRAL).sum()),
            "exclude_count": int((actions == HYBRID_DQN_SIGNAL_ACTION_EXCLUDE).sum()),
            "available_asset_count": available_asset_count,
            "min_selected_assets": min_selected_assets,
        }

    def _compute_optimizer_weights(
        self,
        decision_market_state: DecisionMarketState | Any,
        candidate_asset_indices: Any | None = None,
    ) -> dict[str, Any]:
        if isinstance(candidate_asset_indices, DecisionMarketState):
            decision_market_state, candidate_asset_indices = candidate_asset_indices, decision_market_state
        assert_decision_visibility_contract(strategy_state=["log_return_window"])
        candidates = _candidate_asset_indices(candidate_asset_indices, self.n_assets)
        optimizer_name = str(self.optimizer_name)
        parameters = dict(self.optimizer_parameters)
        available = (
            _available_mask(decision_market_state.available_mask_at_decision, self.n_assets)
            if isinstance(decision_market_state, DecisionMarketState)
            else np.ones(self.n_assets, dtype=bool)
        )
        available_indices = np.flatnonzero(available).astype(np.int64)
        candidates = candidates[available[candidates]]

        if available_indices.size == 0:
            return _terminal_optimizer_failure(
                optimizer_name,
                candidates,
                available,
                "failed_no_valid_action",
                "failed_no_valid_action",
                "no_available_asset",
                parameters,
            )
        if available_indices.size == 1:
            raw_weights = _full_asset_weights(available_indices, np.array([1.0], dtype=np.float32), self.n_assets)
            return _finalize_optimizer_weights(
                raw_weights,
                available,
                available_indices,
                candidates,
                optimizer_name,
                parameters,
                self.config,
                optimizer_success=False,
                optimizer_status="fallback_single_available_asset",
                fallback_reason="single_available_asset",
                fallback_source="single_available_asset",
            )

        if optimizer_name == "equal_weight":
            if candidates.size == 0:
                return _fallback_optimizer_weights(
                    available,
                    candidates,
                    optimizer_name,
                    "failed_no_candidate_asset",
                    "no_candidate_asset",
                    parameters,
                    self.config,
                    self.n_assets,
                )
            raw_weights = _full_asset_weights(candidates, _equal_candidate_weights(candidates.size), self.n_assets)
            return _finalize_optimizer_weights(
                raw_weights,
                available,
                candidates,
                candidates,
                optimizer_name,
                parameters,
                self.config,
                optimizer_success=True,
                optimizer_status="success",
                fallback_reason=None,
                fallback_source=None,
            )

        if not isinstance(decision_market_state, DecisionMarketState):
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: decision_market_state must be DecisionMarketState",
            )
        returns = _optimizer_return_window(
            decision_market_state.log_return_window,
            candidates,
            int(parameters["lookback_window"]),
        )
        if candidates.size == 0:
            return _fallback_optimizer_weights(
                available,
                candidates,
                optimizer_name,
                "failed_no_candidate_asset",
                "no_candidate_asset",
                parameters,
                self.config,
                self.n_assets,
            )
        if returns.shape[0] < int(parameters["min_observations"]):
            return _fallback_optimizer_weights(
                available,
                candidates,
                optimizer_name,
                "failed_insufficient_history",
                "insufficient_history",
                parameters,
                self.config,
                self.n_assets,
            )
        if _has_singular_covariance(returns):
            return _fallback_optimizer_weights(
                available,
                candidates,
                optimizer_name,
                "failed_singular_covariance",
                "singular_covariance",
                parameters,
                self.config,
                self.n_assets,
            )

        covariance = shrink_covariance(returns, float(parameters["covariance_shrinkage"]))
        if optimizer_name == "risk_parity":
            risk_result = _optimize_risk_parity(covariance, int(parameters["optimizer_maxiter"]))
            result = PortfolioOptimizationResult(
                weights=risk_result.weights,
                success=risk_result.success,
                fallback_reason=risk_result.fallback_reason,
            )
        else:
            objective_by_optimizer = {
                "markowitz_mean_variance": "mean_variance",
                "minimum_variance": "min_variance",
                "sharpe_maximization": "max_sharpe",
            }
            objective = objective_by_optimizer.get(optimizer_name)
            if objective is None:
                return _fallback_optimizer_weights(
                    available,
                    candidates,
                    optimizer_name,
                    "failed_invalid_optimizer_name",
                    "invalid_optimizer_name",
                    parameters,
                    self.config,
                    self.n_assets,
                )
            result = optimize_long_only_portfolio(
                returns.mean(axis=0),
                covariance,
                objective,
                lambda_risk=float(parameters.get("lambda_risk", HYBRID_DQN_OPTIMIZER_LAMBDA_RISK)),
                risk_free_rate=float(parameters.get("risk_free_rate", HYBRID_DQN_OPTIMIZER_RISK_FREE_RATE)),
                maxiter=int(parameters["optimizer_maxiter"]),
            )
        if not result.success:
            fallback_reason = _optimizer_fallback_reason(result.fallback_reason or "optimizer_failed")
            return _optimizer_failure_result(
                available,
                optimizer_name,
                candidates,
                fallback_reason,
                parameters,
                self.config,
                self.n_assets,
            )
        weights = _finite_normalized_weights(result.weights)
        if weights is None:
            return _fallback_optimizer_weights(
                available,
                candidates,
                optimizer_name,
                "failed_non_finite_weights",
                "non_finite_optimizer_weights",
                parameters,
                self.config,
                self.n_assets,
            )
        raw_weights = _full_asset_weights(candidates, weights, self.n_assets)
        return _finalize_optimizer_weights(
            raw_weights,
            available,
            candidates,
            candidates,
            optimizer_name,
            parameters,
            self.config,
            optimizer_success=True,
            optimizer_status="success",
            fallback_reason=None,
            fallback_source=None,
        )

    def _train_epoch(
        self,
        env: PortfolioRebalanceEnv,
        *,
        max_steps: int | None = None,
        max_gradient_updates: int | None = None,
    ) -> dict[str, Any]:
        observation, _ = env.reset()
        terminated = False
        truncated = False
        env_steps = 0
        gradient_updates = 0
        reward_total = 0.0
        losses: list[float] = []
        counters = {
            "include_count": 0,
            "neutral_count": 0,
            "exclude_count": 0,
            "selected_asset_count": 0,
            "optimizer_asset_count": 0,
            "optimizer_fallback_count": 0,
        }

        while not (terminated or truncated):
            if max_steps is not None and env_steps >= int(max_steps):
                break
            decision_date = pd.Timestamp(env.decision_dates[env._step_pos])
            decision_state = _build_decision_market_state(
                env.dataset,
                decision_date,
                self.config,
                market_image_dataset=env.market_image_dataset,
            )
            decision = self._policy_decision(decision_state, env._state)
            optimizer_result = decision["optimizer_result"]
            if optimizer_result.get("success") is not True:
                return {
                    "status": str(optimizer_result.get("status", "failed_no_valid_optimizer_result")),
                    "reason": str(optimizer_result.get("reason", optimizer_result.get("fallback_reason", "optimizer_failed"))),
                    "env_steps": env_steps,
                    "gradient_updates": gradient_updates,
                    "train_reward": reward_total,
                    "loss": float(np.nanmean(losses)) if losses else np.nan,
                    **counters,
                }
            action_info = dict(decision["action_info"])
            target_weights = np.asarray(optimizer_result["target_weights"], dtype=np.float32)
            env_action = self._gated_env_action(
                observation,
                target_weights,
                {
                    **action_info,
                    "estimated_turnover": _turnover(target_weights, observation["current_weights"]),
                    "estimated_cost": 0.0,
                },
            )
            next_observation, reward, terminated, truncated, _ = env.step(env_action)
            if max_gradient_updates is None or gradient_updates < int(max_gradient_updates):
                losses.append(float(self._update_signal(observation, decision, float(reward))["loss"]))
                gradient_updates += 1
            for key in ("include_count", "neutral_count", "exclude_count", "selected_asset_count", "optimizer_asset_count"):
                counters[key] += int(decision["candidate_info"].get(key, 0))
            if optimizer_result.get("fallback_reason") is not None or optimizer_result.get("optimizer_status") != "success":
                counters["optimizer_fallback_count"] += 1
            observation = next_observation
            reward_total += float(reward)
            env_steps += 1

        return {
            "status": "completed",
            "env_steps": env_steps,
            "gradient_updates": gradient_updates,
            "train_reward": reward_total,
            "loss": float(np.nanmean(losses)) if losses else np.nan,
            **counters,
        }

    def _policy_decision(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> dict[str, Any]:
        q_values = self._asset_action_q_values(decision_market_state)
        candidate_info = self._select_candidate_assets(q_values, decision_market_state.available_mask_at_decision)
        optimizer_result = self._compute_optimizer_weights(decision_market_state, candidate_info["candidate_asset_indices"])
        target_weights = np.asarray(optimizer_result.get("target_weights", np.zeros(self.n_assets)), dtype=np.float32)
        action_info = {
            "paper_model_id": self.paper_model_id,
            "child_model_name": self.child_model_name,
            "baseline_family": "native_rl_reimplementation",
            "optimizer_name": self.optimizer_name,
            "include_count": int(candidate_info["include_count"]),
            "exclude_count": int(candidate_info["exclude_count"]),
            "neutral_count": int(candidate_info["neutral_count"]),
            "selected_asset_count": int(candidate_info["selected_asset_count"]),
            "optimizer_asset_count": int(candidate_info["optimizer_asset_count"]),
            "optimizer_status": optimizer_result.get("optimizer_status"),
            "fallback_reason": optimizer_result.get("fallback_reason"),
            "factorized_q": True,
            "portfolio_level_reward_shared": True,
            "counterfactual_asset_reward": False,
            "platform_adapted_approximation": True,
            "estimated_turnover": _turnover(target_weights, portfolio_state.current_weights),
            "estimated_cost": 0.0,
        }
        return {
            "q_values": q_values,
            "asset_signal_actions": candidate_info["asset_signal_actions"],
            "candidate_info": candidate_info,
            "optimizer_result": optimizer_result,
            "action_info": action_info,
        }

    def _update_signal(
        self,
        observation: Mapping[str, Any],
        decision: Mapping[str, Any],
        reward: float,
    ) -> dict[str, float]:
        assert_decision_visibility_contract(observation=observation)
        market_image = torch.as_tensor(
            _market_image(observation, self.n_features, self.window_size, self.n_assets)[None, ...],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.asarray(decision["asset_signal_actions"], dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        ).view(1, self.n_assets, 1)
        latent = self.encoder(market_image)
        q_values = self.asset_signal_q_network(latent, market_image)
        selected_q = q_values.gather(2, actions).view(-1)
        target = torch.full_like(selected_q, float(reward))
        loss = F.smooth_l1_loss(selected_q, target)
        if not torch.isfinite(loss):
            raise DataContractError("ERR_TRAINING_NON_FINITE_LOSS", "ERR_TRAINING_NON_FINITE_LOSS: hybrid_dqn_optimizer")
        self.signal_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            _dqn_signal_parameters(self.asset_signal_q_network, self.encoder, self.dqn_config),
            self.dqn_config.max_grad_norm,
        )
        self.signal_optimizer.step()
        self._gradient_update_step += 1
        if self._gradient_update_step % int(self.dqn_config.target_update_interval) == 0:
            self.target_asset_signal_q_network.load_state_dict(self.asset_signal_q_network.state_dict())
        return {"loss": float(loss.detach().cpu())}

    def _policy_action(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        decision = self._policy_decision(decision_market_state, portfolio_state)
        optimizer_result = decision["optimizer_result"]
        if optimizer_result.get("success") is not True:
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                f"ERR_STRATEGY_ACTION_CONTRACT: {optimizer_result.get('status', 'failed_no_valid_optimizer_result')}",
            )
        return PortfolioAction(
            np.asarray(optimizer_result["target_weights"], dtype=np.float32),
            *self._rebalance_action_tuple(
                portfolio_state,
                np.asarray(optimizer_result["target_weights"], dtype=np.float32),
                decision["action_info"],
            ),
        )

    def _gated_env_action(
        self,
        observation: Mapping[str, Any],
        target_weights: np.ndarray,
        action_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        portfolio_state = _portfolio_state_from_observation(observation)
        rebalance_action, rebalance_intensity, merged_info = self._rebalance_action_tuple(
            portfolio_state,
            target_weights,
            action_info,
        )
        return {
            "weights": np.asarray(target_weights, dtype=np.float32),
            "rebalance": rebalance_action,
            "rebalance_action": rebalance_action,
            "rebalance_intensity": rebalance_intensity,
            **merged_info,
        }

    def _rebalance_action_tuple(
        self,
        portfolio_state: PortfolioState,
        target_weights: np.ndarray,
        action_info: Mapping[str, Any],
    ) -> tuple[int, float, dict[str, Any]]:
        rebalance_decision = _continuous_weight_rebalance_decision(
            self.config,
            self.strategy_name,
            portfolio_state,
            np.asarray(target_weights, dtype=float),
            getattr(self, "decision_context", {}),
        )
        merged_info = {**dict(action_info), **rebalance_decision["action_info"]}
        return (
            int(rebalance_decision["rebalance_action"]),
            float(rebalance_decision["rebalance_intensity"]),
            merged_info,
        )

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}

    def _save_checkpoint_state(self, path: Path | None, epoch: int, global_step: int, best_metric: float | None) -> None:
        if path is None:
            return
        save_checkpoint(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "policy_model_state": None,
                "target_policy_model_state": None,
                "encoder_state": self.encoder.state_dict(),
                "ppo_actor_state": None,
                "ppo_critic_state": None,
                "dqn_gate_state": self.asset_signal_q_network.state_dict(),
                "dqn_target_network_state": self.target_asset_signal_q_network.state_dict(),
                "auxiliary_head_states": None,
                "optimizer_states": {
                    "ppo": None,
                    "dqn": self.signal_optimizer.state_dict(),
                    "auxiliary": None,
                },
                "scheduler_states": {},
                "amp_grad_scaler_state": None,
                "replay_buffer_state": None,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_metric": best_metric,
                "rng_states": {},
                "resolved_config": dict(self.config),
            },
            path,
        )

    def _load_checkpoint_state(self, path: Path) -> None:
        payload = load_checkpoint(path, device=self.device, restore_rng_state=False)
        self.encoder.load_state_dict(payload["encoder_state"])
        self.asset_signal_q_network.load_state_dict(payload["dqn_gate_state"])
        self.target_asset_signal_q_network.load_state_dict(payload["dqn_target_network_state"])
        optimizer_states = payload["optimizer_states"]
        if optimizer_states.get("dqn") is not None:
            self.signal_optimizer.load_state_dict(optimizer_states["dqn"])
        self._checkpoint_loaded_path = str(path)


class HybridDQNOptimizerEqualWeightStrategy(HybridDQNOptimizerReimplementationStrategy):
    strategy_name = "hybrid_dqn_optimizer_equal_weight"
    optimizer_name = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_name].optimizer_name


class HybridDQNOptimizerMarkowitzMeanVarianceStrategy(HybridDQNOptimizerReimplementationStrategy):
    strategy_name = "hybrid_dqn_optimizer_markowitz_mean_variance"
    optimizer_name = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_name].optimizer_name


class HybridDQNOptimizerMinimumVarianceStrategy(HybridDQNOptimizerReimplementationStrategy):
    strategy_name = "hybrid_dqn_optimizer_minimum_variance"
    optimizer_name = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_name].optimizer_name


class HybridDQNOptimizerSharpeMaximizationStrategy(HybridDQNOptimizerReimplementationStrategy):
    strategy_name = "hybrid_dqn_optimizer_sharpe_maximization"
    optimizer_name = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_name].optimizer_name


class HybridDQNOptimizerRiskParityStrategy(HybridDQNOptimizerReimplementationStrategy):
    strategy_name = "hybrid_dqn_optimizer_risk_parity"
    optimizer_name = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_name].optimizer_name


class _HybridDQNValidationStrategy(BaseStrategy):
    fit_required = False

    def __init__(self, strategy: HybridDQNOptimizerReimplementationStrategy) -> None:
        super().__init__(strategy.config)
        self.strategy = strategy
        self.strategy_name = strategy.strategy_name

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        return self.validate_portfolio_action(
            self.strategy._policy_action(
                self.validate_decision_market_state(decision_market_state),
                self.validate_portfolio_state(portfolio_state),
            )
        )


def _resolved_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    model_config = _mapping(config.get("model"))
    feature_matrix = _mapping(config.get("feature_matrix"))
    env_config = _mapping(config.get("env"))
    result["n_assets"] = _positive_int("n_assets", config.get("n_assets", model_config.get("n_assets", 1)))
    result["n_features"] = _positive_int("n_features", config.get("n_features", model_config.get("n_features", 1)))
    result["window_size"] = _positive_int(
        "window_size",
        config.get(
            "window_size",
            feature_matrix.get("window_size", env_config.get("window_size", model_config.get("window_size", 1))),
        ),
    )
    result["latent_dim"] = _positive_int("latent_dim", config.get("latent_dim", model_config.get("latent_dim", 256)))
    return result


def _market_image(
    decision_input: DecisionMarketState | Mapping[str, Any],
    n_features: int,
    window_size: int,
    n_assets: int,
) -> np.ndarray:
    if isinstance(decision_input, DecisionMarketState):
        value = decision_input.market_image
    elif isinstance(decision_input, Mapping):
        value = decision_input.get("market_image")
    else:
        raise TypeError("ERR_HYBRID_DQN_SIGNAL_INPUT: decision_input must be DecisionMarketState or mapping")
    image = np.asarray(value, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == int(n_features) and image.shape[2] == int(n_assets):
        current = image.shape[1]
        if current > int(window_size):
            image = image[:, current - int(window_size) :, :]
        elif current < int(window_size):
            image = np.pad(
                image,
                ((0, 0), (int(window_size) - current, 0), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )
    if image.shape != (int(n_features), int(window_size), int(n_assets)):
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: market_image must match [n_features,window_size,n_assets]")
    return image


def _asset_action_q_array(asset_action_q_values: Any, n_assets: int) -> np.ndarray:
    q_values = np.asarray(asset_action_q_values, dtype=np.float32)
    if q_values.shape != (int(n_assets), HYBRID_DQN_SIGNAL_ACTION_DIM):
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: asset_action_q_values must be [asset_count,3]")
    return q_values


def _available_mask(value: Any, n_assets: int) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (int(n_assets),):
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_SHAPE: available_mask must match asset_count")
    return mask


def _candidate_asset_indices(value: Any, n_assets: int) -> np.ndarray:
    indices = np.asarray([] if value is None else value, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("ERR_HYBRID_DQN_OPTIMIZER_SHAPE: candidate_asset_indices must be 1d")
    if indices.size and ((indices < 0).any() or (indices >= int(n_assets)).any()):
        raise ValueError("ERR_HYBRID_DQN_OPTIMIZER_SHAPE: candidate_asset_indices out of range")
    return indices


def _optimizer_return_window(log_return_window: Any, candidate_asset_indices: np.ndarray, lookback_window: int) -> np.ndarray:
    returns = np.asarray(log_return_window, dtype=np.float32)
    if returns.ndim != 2:
        raise ValueError("ERR_HYBRID_DQN_OPTIMIZER_SHAPE: log_return_window must be 2d")
    if candidate_asset_indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    active_returns = returns[-int(lookback_window) :, candidate_asset_indices]
    finite_rows = np.isfinite(active_returns).all(axis=1)
    return active_returns[finite_rows].astype(np.float32, copy=False)


def _equal_candidate_weights(n_assets: int) -> np.ndarray:
    if int(n_assets) <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.full(int(n_assets), 1.0 / int(n_assets), dtype=np.float32)


def _finite_normalized_weights(value: Any) -> np.ndarray | None:
    weights = np.asarray(value, dtype=np.float32)
    if weights.ndim != 1 or not np.isfinite(weights).all():
        return None
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return None
    return (weights / weight_sum).astype(np.float32, copy=False)


def _full_asset_weights(asset_indices: np.ndarray, weights: np.ndarray, n_assets: int) -> np.ndarray:
    target = np.zeros(int(n_assets), dtype=np.float32)
    target[np.asarray(asset_indices, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    return target


def _project_target_weights(raw_weights: np.ndarray, available: np.ndarray, config: Mapping[str, Any]) -> np.ndarray | None:
    try:
        constraint_result = ConstraintManager(config).project(raw_weights, available)
    except DataContractError:
        return None
    projected = np.asarray(constraint_result.projected_weights, dtype=np.float32)
    projected[np.abs(projected) <= 1.0e-10] = 0.0
    projected[~available] = 0.0
    return _finite_normalized_weights(projected)


def _has_singular_covariance(returns: np.ndarray) -> bool:
    matrix = np.asarray(returns, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] <= 1:
        return False
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    if covariance.shape != (matrix.shape[1], matrix.shape[1]) or not np.isfinite(covariance).all():
        return True
    return int(np.linalg.matrix_rank(covariance)) < int(matrix.shape[1])


def _optimizer_fallback_reason(reason: str) -> str:
    if reason in {"non_finite_weights", "non_finite_moments"}:
        return "non_finite_optimizer_weights"
    return str(reason)


def _optimizer_status_for_reason(reason: str) -> str:
    return {
        "insufficient_history": "failed_insufficient_history",
        "singular_covariance": "failed_singular_covariance",
        "non_finite_optimizer_weights": "failed_non_finite_weights",
        "constraint_projection_failed": "failed_constraint_projection",
        "no_candidate_asset": "failed_no_candidate_asset",
        "invalid_optimizer_name": "failed_invalid_optimizer_name",
    }.get(reason, f"failed_{reason}")


def _finalize_optimizer_weights(
    raw_weights: np.ndarray,
    available: np.ndarray,
    allocation_asset_indices: np.ndarray,
    candidate_asset_indices: np.ndarray,
    optimizer_name: str,
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    optimizer_success: bool,
    optimizer_status: str,
    fallback_reason: str | None,
    fallback_source: str | None,
) -> dict[str, Any]:
    target_weights = _project_target_weights(raw_weights, available, config)
    if target_weights is None:
        if optimizer_success and fallback_source is None:
            return _fallback_optimizer_weights(
                available,
                candidate_asset_indices,
                optimizer_name,
                "failed_constraint_projection",
                "constraint_projection_failed",
                parameters,
                config,
                raw_weights.size,
            )
        return _terminal_optimizer_failure(
            optimizer_name,
            candidate_asset_indices,
            available,
            "failed_no_valid_optimizer_result",
            "failed_constraint_projection",
            "constraint_projection_failed",
            parameters,
            fallback_source=fallback_source,
        )
    return {
        "weights": target_weights[np.asarray(allocation_asset_indices, dtype=np.int64)].astype(np.float32, copy=False),
        "target_weights": target_weights.astype(np.float32, copy=False),
        "success": True,
        "optimizer_success": bool(optimizer_success),
        "optimizer_name": optimizer_name,
        "optimizer_status": optimizer_status,
        "fallback_reason": fallback_reason,
        "fallback_source": fallback_source,
        "candidate_asset_indices": candidate_asset_indices,
        "allocation_asset_indices": np.asarray(allocation_asset_indices, dtype=np.int64),
        "optimizer_parameters": dict(parameters),
    }


def _fallback_optimizer_weights(
    available: np.ndarray,
    candidate_asset_indices: np.ndarray,
    optimizer_name: str,
    optimizer_status: str,
    fallback_reason: str,
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
    n_assets: int,
) -> dict[str, Any]:
    fallback_indices = candidate_asset_indices if candidate_asset_indices.size else np.flatnonzero(available).astype(np.int64)
    fallback_source = "candidate_pool_equal_weight" if candidate_asset_indices.size else "all_available_equal_weight"
    fallback_weights = _equal_candidate_weights(fallback_indices.size)
    if fallback_weights.size == 0:
        return _terminal_optimizer_failure(
            optimizer_name,
            candidate_asset_indices,
            available,
            "failed_no_valid_optimizer_result",
            optimizer_status,
            fallback_reason,
            parameters,
            fallback_source=fallback_source,
        )
    return _finalize_optimizer_weights(
        _full_asset_weights(fallback_indices, fallback_weights, n_assets),
        available,
        fallback_indices,
        candidate_asset_indices,
        optimizer_name,
        parameters,
        config,
        optimizer_success=False,
        optimizer_status=optimizer_status,
        fallback_reason=fallback_reason,
        fallback_source=fallback_source,
    )


def _optimizer_parameters(optimizer_name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if optimizer_name == "equal_weight":
        return {"optimizer_name": optimizer_name}
    if optimizer_name not in {"markowitz_mean_variance", "minimum_variance", "sharpe_maximization", "risk_parity"}:
        return {"optimizer_name": optimizer_name}
    parameter_config = _mapping(_mapping(config).get("hybrid_dqn_optimizer"))
    parameters: dict[str, Any] = {
        "optimizer_name": optimizer_name,
        "lookback_window": int(parameter_config.get("lookback_window", HYBRID_DQN_OPTIMIZER_LOOKBACK_WINDOW)),
        "min_observations": int(parameter_config.get("min_observations", HYBRID_DQN_OPTIMIZER_MIN_OBSERVATIONS)),
        "covariance_shrinkage": float(
            parameter_config.get("covariance_shrinkage", HYBRID_DQN_OPTIMIZER_COVARIANCE_SHRINKAGE)
        ),
        "optimizer_maxiter": int(
            parameter_config.get(
                "risk_parity_optimizer_maxiter" if optimizer_name == "risk_parity" else "markowitz_optimizer_maxiter",
                HYBRID_DQN_OPTIMIZER_RISK_PARITY_MAXITER
                if optimizer_name == "risk_parity"
                else HYBRID_DQN_OPTIMIZER_MARKOWITZ_MAXITER,
            )
        ),
    }
    if optimizer_name in {"markowitz_mean_variance", "sharpe_maximization"}:
        parameters["risk_free_rate"] = float(parameter_config.get("risk_free_rate", HYBRID_DQN_OPTIMIZER_RISK_FREE_RATE))
    if optimizer_name == "markowitz_mean_variance":
        parameters["lambda_risk"] = float(parameter_config.get("lambda_risk", HYBRID_DQN_OPTIMIZER_LAMBDA_RISK))
    return parameters


def _optimizer_failure_result(
    available: np.ndarray,
    optimizer_name: str,
    candidate_asset_indices: np.ndarray,
    fallback_reason: str,
    parameters: Mapping[str, Any],
    config: Mapping[str, Any],
    n_assets: int,
) -> dict[str, Any]:
    return _fallback_optimizer_weights(
        available,
        candidate_asset_indices,
        optimizer_name,
        _optimizer_status_for_reason(fallback_reason),
        fallback_reason,
        parameters,
        config,
        n_assets,
    )


def _terminal_optimizer_failure(
    optimizer_name: str,
    candidate_asset_indices: np.ndarray,
    available: np.ndarray,
    status: str,
    optimizer_status: str,
    fallback_reason: str,
    parameters: Mapping[str, Any],
    *,
    fallback_source: str | None = None,
) -> dict[str, Any]:
    return {
        "weights": np.zeros(0, dtype=np.float32),
        "target_weights": np.zeros(available.shape, dtype=np.float32),
        "success": False,
        "optimizer_success": False,
        "status": status,
        "reason": fallback_reason,
        "optimizer_name": optimizer_name,
        "optimizer_status": optimizer_status,
        "fallback_reason": fallback_reason,
        "fallback_source": fallback_source,
        "candidate_asset_indices": candidate_asset_indices,
        "allocation_asset_indices": np.zeros(0, dtype=np.int64),
        "optimizer_parameters": dict(parameters),
    }


def _device(config: Mapping[str, Any]) -> torch.device:
    value = config.get("device")
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        return torch.device(value)
    if isinstance(value, Mapping):
        mode = str(value.get("mode", "cpu")).lower()
        if mode in {"cuda", "auto"} and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dqn_signal_parameters(
    signal_network: nn.Module,
    encoder: nn.Module,
    dqn_config: DQNAgentConfig,
) -> list[nn.Parameter]:
    modules = (signal_network,) if dqn_config.detach_encoder else (signal_network, encoder)
    return _unique_parameters(*modules)


def _unique_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            parameters.append(parameter)
            seen.add(id(parameter))
    return parameters


def _hidden_dims(value: Any, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(dim) for dim in default)
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_CONFIG: hidden_dims must be a sequence")
    result = tuple(_positive_int("hidden_dims", dim) for dim in value)
    if not result:
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_CONFIG: hidden_dims must be non-empty")
    return result


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_HYBRID_DQN_SIGNAL_CONFIG: {name} must be > 0")
    return result


def _dates(value: Any) -> pd.DatetimeIndex:
    if value is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(value))).sort_values()


def _native_rl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = _mapping(config.get("baselines"))
    return _mapping(baselines.get("native_rl") or baselines.get("native_training"))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ERR_HYBRID_DQN_SIGNAL_CONFIG: max step limits must be > 0")
    return result


def _has_finite_validation(history: pd.DataFrame) -> bool:
    if history.empty or "validation_metric" not in history.columns:
        return False
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    return bool(np.isfinite(values).any())


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _turnover(weights: Any, current_weights: Any) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(weights, dtype=np.float32) - np.asarray(current_weights, dtype=np.float32))))


def _portfolio_state_from_observation(observation: Mapping[str, Any]) -> PortfolioState:
    current_weights = np.asarray(observation.get("current_weights"), dtype=float)
    portfolio_value = float(np.asarray(observation.get("portfolio_value", 0.0), dtype=float))
    return PortfolioState(
        date=pd.Timestamp("1970-01-01"),
        nav=1.0,
        portfolio_value=portfolio_value,
        current_weights=current_weights,
        step_index=0 if float(current_weights.sum()) <= 0.0 else 1,
    )


def _hybrid_validation_metric_from_backtest(
    strategy: HybridDQNOptimizerReimplementationStrategy,
    dataset: Any,
    split: SplitSpec,
    *,
    market_image_dataset: Any | None,
    max_steps: int | None,
) -> float:
    validation_split = _validation_split(split, max_steps=max_steps)
    try:
        result = BacktestEngine(strategy.config, market_image_dataset=market_image_dataset).run(
            dataset,
            validation_split,
            _HybridDQNValidationStrategy(strategy),
            segment="validation",
        )
        from src.experiments.pipeline import objective_metric

        value = objective_metric(
            {"daily_returns": result.daily_returns, "metrics": result.metrics},
            "validation_sharpe_minus_drawdown_turnover_penalty",
        )
    except (DataContractError, ValueError, KeyError):
        return float("-inf")
    return float(value) if np.isfinite(value) else float("-inf")


def _validation_split(split: SplitSpec, *, max_steps: int | None) -> SplitSpec:
    validation_dates = pd.DatetimeIndex(split.validation_dates)
    validation_last_decision_date = split.validation_last_decision_date
    if max_steps is not None and int(max_steps) > 0:
        limit = int(max_steps) + 1 if len(validation_dates) > 1 else int(max_steps)
        validation_dates = validation_dates[:limit]
        if len(validation_dates) > 0:
            decision_pos = min(int(max_steps) - 1, len(validation_dates) - 1)
            validation_last_decision_date = pd.Timestamp(validation_dates[decision_pos])
    return SplitSpec(
        train_dates=split.train_dates,
        validation_dates=validation_dates,
        test_dates=split.test_dates,
        fold_id=split.fold_id,
        train_last_decision_date=split.train_last_decision_date,
        validation_last_decision_date=validation_last_decision_date,
        test_last_decision_date=split.test_last_decision_date,
    )


def _training_result(
    paper_model_id: str,
    child_model_name: str | None,
    optimizer_name: str,
    training_history: pd.DataFrame,
    *,
    status: str = "not_started",
    reason: str | None = "training_logic_not_implemented",
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    rankable_in_unified_table: bool = False,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
    max_gradient_updates_per_epoch: int | None = None,
) -> dict[str, Any]:
    is_alias = paper_model_id == HYBRID_DQN_OPTIMIZER_ALIAS
    return {
        "model_name": HYBRID_DQN_OPTIMIZER_ALIAS,
        "paper_model_id": paper_model_id,
        "child_model_name": None if is_alias else child_model_name,
        "baseline_family": "native_rl_reimplementation",
        "status": status,
        "reason": reason,
        "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
        "optimizer_name": optimizer_name,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "clean_room_reimplementation": True,
        "algorithm_fidelity": "platform_adapted",
        "factorized_q": True,
        "portfolio_level_reward_shared": True,
        "counterfactual_asset_reward": False,
        "platform_adapted_approximation": True,
        "rankable_in_unified_table": bool(rankable_in_unified_table),
        "training_history": training_history,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_last_path": checkpoint_last_path,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "env_steps": int(env_steps),
        "gradient_updates": int(gradient_updates),
        "max_train_steps": max_train_steps,
        "max_validation_steps": max_validation_steps,
        "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
    }


__all__ = [
    "HYBRID_DQN_OPTIMIZER_ALIAS",
    "HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES",
    "HYBRID_DQN_OPTIMIZER_CHILD_SPECS",
    "HYBRID_DQN_SIGNAL_ACTION_DIM",
    "HYBRID_DQN_SIGNAL_ACTION_EXCLUDE",
    "HYBRID_DQN_SIGNAL_ACTION_INCLUDE",
    "HYBRID_DQN_SIGNAL_ACTION_NEUTRAL",
    "HybridDQNAssetSignalQNetwork",
    "HybridDQNOptimizerChildSpec",
    "HybridDQNOptimizerReimplementationStrategy",
    "HybridDQNOptimizerEqualWeightStrategy",
    "HybridDQNOptimizerMarkowitzMeanVarianceStrategy",
    "HybridDQNOptimizerMinimumVarianceStrategy",
    "HybridDQNOptimizerSharpeMaximizationStrategy",
    "HybridDQNOptimizerRiskParityStrategy",
]
