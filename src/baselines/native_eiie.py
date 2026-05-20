from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.baselines.base_strategy import BaseStrategy
from src.baselines.deep_training import execution_aligned_return_component_frames
from src.baselines.eiie import MASKED_SCORE_VALUE, _eiie_asset_tensor, _previous_weights
from src.data.splits import SplitSpec
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


EIIE_NATIVE_ALGORITHM = "eiie_policy_gradient_pvm"


class NativeEIIEStrategy(BaseStrategy):
    strategy_name = "eiie_native"
    fit_required = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.n_assets = int(config["n_assets"])
        self.n_features = int(config["n_features"])
        self.window_size = int(config["window_size"])
        self.device = _device(self.config)
        self.evaluator = nn.Sequential(
            nn.Conv1d(self.n_features + 1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        ).to(self.device)
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> NativeEIIEStrategy:
        if not isinstance(train_data, Mapping):
            self.training_result = _training_result(self.strategy_name, "failed_missing_train_data", pd.DataFrame())
            self.is_fitted = False
            return self
        native_cfg = _native_rl_config(self.config)
        epochs = max(1, int(native_cfg.get("epochs", _mapping(self.config.get("training")).get("epochs", 1))))
        max_train_steps = _optional_positive_int(native_cfg.get("max_train_steps"))
        max_validation_steps = _optional_positive_int(native_cfg.get("max_validation_steps"))
        lr = _training_lr(self.config)
        optimizer = torch.optim.AdamW(self.evaluator.parameters(), lr=lr)
        checkpoint_paths = self._checkpoint_paths()
        history_rows: list[dict[str, Any]] = []
        best_metric = -np.inf
        gradient_updates = 0
        env_steps = 0

        for epoch in range(epochs):
            train_stats = self._train_epoch(train_data, optimizer, max_steps=max_train_steps)
            gradient_updates += int(train_stats["gradient_updates"])
            env_steps += int(train_stats["env_steps"])
            validation_metric = self._evaluate(validation_data or train_data, max_steps=max_validation_steps)
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(train_stats["train_reward"]),
                    "validation_metric": float(validation_metric),
                    "loss": float(train_stats["loss"]),
                    "portfolio_vector_memory": True,
                    "pre_execution_return_in_actor_loss": False,
                    "pre_execution_return_used_for_drift": True,
                    "max_train_steps": max_train_steps,
                    "max_validation_steps": max_validation_steps,
                    "status": "completed",
                }
            )
            if np.isfinite(validation_metric) and validation_metric > best_metric:
                best_metric = float(validation_metric)
                self._save_state(checkpoint_paths["best"], epoch, gradient_updates, best_metric)

        self._save_state(
            checkpoint_paths["last"],
            epochs - 1,
            gradient_updates,
            None if not np.isfinite(best_metric) else best_metric,
        )
        history = pd.DataFrame(history_rows)
        if not _has_finite_validation(history):
            self.training_history = history
            self.training_result = _training_result(
                self.strategy_name,
                "failed_no_finite_validation_metric",
                history,
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            )
            self.is_fitted = False
            return self
        if checkpoint_paths["best"] is None or not checkpoint_paths["best"].exists():
            self.training_history = history
            self.training_result = _training_result(
                self.strategy_name,
                "failed_missing_best_checkpoint",
                history,
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            )
            self.is_fitted = False
            return self

        self._load_state(checkpoint_paths["best"])
        self.training_history = history
        self.training_result = _training_result(
            self.strategy_name,
            "completed",
            history,
            checkpoint_best_path=_path_string(checkpoint_paths["best"]),
            checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            evaluated_checkpoint_path=_path_string(checkpoint_paths["best"]),
            best_validation_metric=float(best_metric),
            env_steps=env_steps,
            gradient_updates=gradient_updates,
            max_train_steps=max_train_steps,
            max_validation_steps=max_validation_steps,
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
        previous = _previous_weights(portfolio)
        x = _eiie_asset_tensor(
            state.market_image,
            previous,
            self.device,
            self.n_features,
            self.window_size,
            self.n_assets,
        )
        with torch.no_grad():
            raw_scores = self.evaluator(x).squeeze(-1)
            mask = torch.as_tensor(state.available_mask_at_decision, dtype=torch.bool, device=self.device)
            scores = raw_scores.masked_fill(~mask, MASKED_SCORE_VALUE)
            weights = torch.softmax(scores, dim=0).detach().cpu().numpy()
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=weights,
                rebalance_action=1,
                rebalance_intensity=1.0,
                action_info={
                    "strategy": self.strategy_name,
                    "scores": scores.detach().cpu().numpy(),
                    "raw_scores": raw_scores.detach().cpu().numpy(),
                    "previous_weights": previous,
                    "training_algorithm": EIIE_NATIVE_ALGORITHM,
                    "rl_training": True,
                    "platform_native_rl_training": True,
                    "portfolio_vector_memory": True,
                    "score_input_fields": ("market_image", "available_mask_at_decision", "previous_weights"),
                },
            )
        )

    def _train_epoch(
        self,
        train_data: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        samples = _sequential_samples(
            train_data,
            self.n_features,
            self.window_size,
            self.n_assets,
            max_samples=max_steps,
        )
        if not samples:
            return {"loss": np.nan, "train_reward": np.nan, "env_steps": 0, "gradient_updates": 0}
        self.evaluator.train()
        previous_weights = _initial_weights(samples[0]["mask"])
        losses: list[float] = []
        rewards: list[float] = []
        updates = 0
        turnover_penalty = float(_mapping(self.config.get("eiie_native")).get("turnover_penalty", 0.0))
        eps = float(_mapping(self.config.get("eiie_native")).get("log_growth_eps", 1.0e-6))
        for sample in samples:
            mask_np = np.asarray(sample["mask"], dtype=bool)
            previous_weights = _normalize_previous(previous_weights, mask_np)
            x = _eiie_asset_tensor(
                sample["market_image"],
                previous_weights,
                self.device,
                self.n_features,
                self.window_size,
                self.n_assets,
            )
            pre_trade_weights = _drift_weights(previous_weights, sample["pre_execution_returns"])
            returns = torch.as_tensor(sample["holding_returns"], dtype=torch.float32, device=self.device)
            previous_tensor = torch.as_tensor(pre_trade_weights, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
            scores = self.evaluator(x).squeeze(-1).masked_fill(~mask, MASKED_SCORE_VALUE)
            weights = torch.softmax(scores, dim=0)
            turnover = 0.5 * torch.sum(torch.abs(weights - previous_tensor))
            portfolio_growth = torch.sum(weights * returns) - float(turnover_penalty) * turnover
            loss = -torch.log(torch.clamp(1.0 + portfolio_growth, min=eps))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.evaluator.parameters(), max_norm=1.0)
            optimizer.step()
            detached_weights = weights.detach().cpu().numpy()
            previous_weights = _drift_weights(detached_weights, sample["holding_returns"])
            losses.append(float(loss.detach().cpu()))
            rewards.append(float(portfolio_growth.detach().cpu()))
            updates += 1
        self.evaluator.eval()
        return {
            "loss": float(np.mean(losses)),
            "train_reward": float(np.mean(rewards)),
            "env_steps": float(len(samples)),
            "gradient_updates": float(updates),
        }

    def _evaluate(self, payload: Any | None, max_steps: int | None = None) -> float:
        if not isinstance(payload, Mapping):
            return float("-inf")
        samples = _sequential_samples(
            payload,
            self.n_features,
            self.window_size,
            self.n_assets,
            max_samples=max_steps,
        )
        if not samples:
            return float("-inf")
        self.evaluator.eval()
        previous_weights = _initial_weights(samples[0]["mask"])
        rewards: list[float] = []
        with torch.no_grad():
            for sample in samples:
                mask_np = np.asarray(sample["mask"], dtype=bool)
                previous_weights = _normalize_previous(previous_weights, mask_np)
                x = _eiie_asset_tensor(
                    sample["market_image"],
                    previous_weights,
                    self.device,
                    self.n_features,
                    self.window_size,
                    self.n_assets,
                )
                scores = self.evaluator(x).squeeze(-1)
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                weights_tensor = torch.softmax(scores.masked_fill(~mask, MASKED_SCORE_VALUE), dim=0)
                weights = weights_tensor.cpu().numpy()
                pre_trade_weights = _drift_weights(previous_weights, sample["pre_execution_returns"])
                turnover = 0.5 * np.sum(np.abs(weights - pre_trade_weights))
                turnover_penalty = float(_mapping(self.config.get("eiie_native")).get("turnover_penalty", 0.0))
                reward = float(
                    np.sum(weights * np.asarray(sample["holding_returns"], dtype=np.float32))
                    - turnover_penalty * turnover
                )
                previous_weights = _drift_weights(weights, sample["holding_returns"])
                rewards.append(reward)
        return float(np.sum(rewards))

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}

    def _save_state(self, path: Path | None, epoch: int, global_step: int, best_metric: float | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "evaluator_state": self.evaluator.state_dict(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_metric": best_metric,
            },
            path,
        )

    def _load_state(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.evaluator.load_state_dict(payload["evaluator_state"])


def _sequential_samples(
    payload: Mapping[str, Any],
    n_features: int,
    window_size: int,
    n_assets: int,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    dataset = payload.get("dataset")
    if dataset is None:
        return []
    dates = pd.DatetimeIndex(pd.to_datetime(list(payload.get("dates", []))))
    component_frames = execution_aligned_return_component_frames(payload, n_assets)
    if component_frames is None:
        return []
    pre_execution_frame = component_frames["pre_execution_returns"]
    holding_frame = component_frames["holding_returns"]
    if pre_execution_frame.empty or holding_frame.empty:
        return []
    market_image_dataset = payload.get("market_image_dataset")
    asset_order = _asset_order(dataset, n_assets)
    samples: list[dict[str, Any]] = []
    for date in dates:
        if max_samples is not None and len(samples) >= int(max_samples):
            break
        timestamp = pd.Timestamp(date)
        if timestamp not in holding_frame.index or timestamp not in pre_execution_frame.index:
            continue
        image = _market_image(dataset, market_image_dataset, asset_order, timestamp, n_features, window_size)
        if image is None:
            continue
        mask = _availability_mask(dataset, asset_order, timestamp)
        if not mask.any():
            continue
        samples.append(
            {
                "date": timestamp,
                "market_image": image,
                "mask": mask,
                "pre_execution_returns": pre_execution_frame.loc[timestamp].to_numpy(dtype=np.float32, copy=True),
                "holding_returns": holding_frame.loc[timestamp].to_numpy(dtype=np.float32, copy=True),
                "future_returns": holding_frame.loc[timestamp].to_numpy(dtype=np.float32, copy=True),
            }
        )
    return samples


def _market_image(
    dataset: Any,
    market_image_dataset: Any | None,
    asset_order: list[str],
    date: pd.Timestamp,
    n_features: int,
    window_size: int,
) -> np.ndarray | None:
    if market_image_dataset is not None:
        try:
            image = np.asarray(market_image_dataset[date], dtype=np.float32)
            if image.shape == (n_features, window_size, len(asset_order)):
                return image
        except Exception:
            pass
    wide = getattr(dataset, "wide", {})
    if not isinstance(wide, Mapping):
        return None
    close = wide.get("close")
    if close is None:
        return None
    date_index = pd.DatetimeIndex(pd.to_datetime(close.index))
    try:
        position = int(np.flatnonzero(date_index == pd.Timestamp(date))[0])
    except IndexError:
        return None
    if position < window_size - 1:
        return None
    feature_cols = [str(item) for item in getattr(dataset, "feature_cols", [])]
    frames = []
    for feature in feature_cols:
        frame = wide.get(feature)
        if frame is None:
            continue
        window = frame.reindex(columns=asset_order).iloc[position - window_size + 1 : position + 1]
        frames.append(np.nan_to_num(window.to_numpy(dtype=np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0))
    if len(frames) == n_features:
        return np.stack(frames, axis=0)
    log_return = wide.get("log_return")
    if log_return is not None and n_features == 1:
        window = log_return.reindex(columns=asset_order).iloc[position - window_size + 1 : position + 1]
        return np.nan_to_num(window.to_numpy(dtype=np.float32, copy=True)[np.newaxis, :, :], nan=0.0)
    return None


def _training_result(
    model_name: str,
    status: str,
    training_history: pd.DataFrame,
    *,
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "baseline_family": "native_rl",
        "status": status,
        "training_algorithm": EIIE_NATIVE_ALGORITHM,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "rankable_in_unified_table": True,
        "portfolio_vector_memory": True,
        "pre_execution_return_in_actor_loss": False,
        "pre_execution_return_used_for_drift": True,
        "pre_execution_return_in_observation": False,
        "training_history": training_history,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_last_path": checkpoint_last_path,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "env_steps": int(env_steps),
        "gradient_updates": int(gradient_updates),
        "max_train_steps": max_train_steps,
        "max_validation_steps": max_validation_steps,
    }


def _availability_mask(dataset: Any, asset_order: list[str], decision_date: pd.Timestamp) -> np.ndarray:
    availability = getattr(dataset, "availability_mask", None)
    if availability is None:
        return np.ones(len(asset_order), dtype=bool)
    return availability.reindex(columns=asset_order).loc[pd.Timestamp(decision_date)].to_numpy(dtype=bool, copy=True)


def _asset_order(dataset: Any, n_assets: int) -> list[str]:
    manifest = getattr(dataset, "data_manifest", {})
    order = manifest.get("canonical_asset_order") if isinstance(manifest, Mapping) else None
    if isinstance(order, list) and order:
        return [str(item) for item in order[: int(n_assets)]]
    return [str(index) for index in range(int(n_assets))]


def _initial_weights(mask: np.ndarray) -> np.ndarray:
    return _normalize_previous(np.zeros(mask.shape, dtype=np.float32), mask)


def _normalize_previous(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float32).copy()
    result[~np.asarray(mask, dtype=bool)] = 0.0
    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        result = np.zeros(mask.shape, dtype=np.float32)
        if mask.any():
            result[mask] = 1.0 / float(mask.sum())
        return result
    return (result / total).astype(np.float32, copy=False)


def _drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    gross = np.asarray(weights, dtype=np.float32) * (1.0 + np.asarray(returns, dtype=np.float32))
    total = float(np.nansum(gross))
    if not np.isfinite(total) or total <= 0.0:
        return np.asarray(weights, dtype=np.float32).copy()
    return (gross / total).astype(np.float32, copy=False)


def _has_finite_validation(history: pd.DataFrame) -> bool:
    if history.empty or "validation_metric" not in history.columns:
        return False
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    return bool(np.isfinite(values).any())


def _native_rl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = _mapping(config.get("baselines"))
    return _mapping(baselines.get("native_rl") or baselines.get("native_training"))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ERR_EIIE_NATIVE_CONFIG_INVALID: max step limits must be > 0")
    return result


def _training_lr(config: Mapping[str, Any]) -> float:
    optimizer = _mapping(config.get("optimizer"))
    native = _mapping(config.get("eiie_native"))
    for value in (native.get("learning_rate"), optimizer.get("learning_rate"), 3.0e-4):
        if value is not None:
            return float(value)
    return 3.0e-4


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


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["EIIE_NATIVE_ALGORITHM", "NativeEIIEStrategy"]
