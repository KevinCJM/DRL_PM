from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.baselines.eiie import MASKED_SCORE_VALUE
from src.baselines.native_eiie import (
    NativeEIIEStrategy,
    _drift_weights,
    _eiie_asset_tensor,
    _initial_weights,
    _mapping,
    _normalize_previous,
    _sequential_samples,
)
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


PGPORTFOLIO_EIIE_ALGORITHM = "pgportfolio_eiie_osbl"


class PGPortfolioEIIEStrategy(NativeEIIEStrategy):
    strategy_name = "pgportfolio_eiie_native"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.osbl_sampled_dates: list[pd.Timestamp] = []
        self.pvm_update_trace: list[dict[str, Any]] = []
        self._osbl_epoch = 0
        self._osbl_epoch_stats: list[dict[str, Any]] = []
        self._pvm: np.ndarray | None = None
        self._pvm_dates: tuple[pd.Timestamp, ...] = ()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> PGPortfolioEIIEStrategy:
        self.osbl_sampled_dates = []
        self.pvm_update_trace = []
        self._osbl_epoch = 0
        self._osbl_epoch_stats = []
        self._pvm = None
        self._pvm_dates = ()
        super().fit(train_data, validation_data)
        self._mark_pgportfolio_result()
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        action.action_info.update(
            {
                "strategy": self.strategy_name,
                "training_algorithm": PGPORTFOLIO_EIIE_ALGORITHM,
                "online_stochastic_batch_learning": True,
                "clean_room_reimplementation": True,
                "source_code_vendored": False,
            }
        )
        return self.validate_portfolio_action(action)

    def _train_epoch(
        self,
        train_data: Mapping[str, Any],
        optimizer: torch.optim.Optimizer,
        max_steps: int | None = None,
    ) -> dict[str, float]:
        samples = sorted(
            _sequential_samples(
                train_data,
                self.n_features,
                self.window_size,
                self.n_assets,
                max_samples=max_steps,
            ),
            key=lambda item: pd.Timestamp(item["date"]),
        )
        if not samples:
            self._osbl_epoch_stats.append({"osbl_sample_count": 0, "osbl_batch_count": 0})
            return {"loss": np.nan, "train_reward": np.nan, "env_steps": 0, "gradient_updates": 0}

        cfg = _pgportfolio_config(self.config)
        batch_size = max(1, int(cfg.get("osbl_batch_size", cfg.get("batch_size", min(16, len(samples))))))
        batches_per_epoch = max(1, int(cfg.get("osbl_batches_per_epoch", cfg.get("batches_per_epoch", 1))))
        sample_bias = float(cfg.get("buffer_biased", cfg.get("sample_bias", 5.0e-5)))
        is_permed = bool(cfg.get("osbl_permuted", cfg.get("is_permed", True)))
        turnover_penalty = float(cfg.get("turnover_penalty", _mapping(self.config.get("eiie_native")).get("turnover_penalty", 0.0)))
        eps = float(cfg.get("log_growth_eps", _mapping(self.config.get("eiie_native")).get("log_growth_eps", 1.0e-6)))
        seed = int(cfg.get("seed", _mapping(self.config.get("reproducibility")).get("seed", 0)))
        rng = np.random.default_rng(seed + int(self._osbl_epoch))
        index_batches = osbl_sample_indices(
            len(samples),
            batch_size,
            batches_per_epoch,
            rng,
            sample_bias=sample_bias,
            is_permed=is_permed,
        )
        pvm = self._pvm_for_samples(samples)

        self.evaluator.train()
        losses: list[float] = []
        rewards: list[float] = []
        updates = 0
        selected_count = 0
        for indices in index_batches:
            optimizer.zero_grad(set_to_none=True)
            loss_terms: list[torch.Tensor] = []
            reward_terms: list[torch.Tensor] = []
            pvm_updates: list[tuple[int, np.ndarray, pd.Timestamp]] = []
            for index in indices:
                sample = samples[int(index)]
                mask_np = np.asarray(sample["mask"], dtype=bool)
                previous_weights = _normalize_previous(pvm[int(index)], mask_np)
                pvm[int(index)] = previous_weights
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
                portfolio_growth = torch.sum(weights * returns) - turnover_penalty * turnover
                loss_terms.append(-torch.log(torch.clamp(1.0 + portfolio_growth, min=eps)))
                reward_terms.append(portfolio_growth.detach())
                pvm_updates.append(
                    (
                        int(index),
                        weights.detach().cpu().numpy().astype(np.float32, copy=True),
                        pd.Timestamp(sample["date"]),
                    )
                )
                self.osbl_sampled_dates.append(pd.Timestamp(sample["date"]))
                selected_count += 1
            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.evaluator.parameters(), max_norm=1.0)
            optimizer.step()
            _apply_pvm_updates(samples, pvm, pvm_updates, self.pvm_update_trace)
            losses.append(float(loss.detach().cpu()))
            rewards.extend(float(item.cpu()) for item in reward_terms)
            updates += 1

        self.evaluator.eval()
        self._osbl_epoch += 1
        self._osbl_epoch_stats.append(
            {
                "osbl_sample_count": int(selected_count),
                "osbl_batch_count": int(updates),
            }
        )
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "train_reward": float(np.mean(rewards)) if rewards else float("nan"),
            "env_steps": float(selected_count),
            "gradient_updates": float(updates),
        }

    def _mark_pgportfolio_result(self) -> None:
        if self.training_history is not None:
            history = self.training_history.copy()
            history["training_algorithm"] = PGPORTFOLIO_EIIE_ALGORITHM
            history["online_stochastic_batch_learning"] = True
            history["clean_room_reimplementation"] = True
            history["source_code_vendored"] = False
            history["platform_adapted_surrogate"] = True
            stats = self._osbl_epoch_stats
            if stats and len(stats) == len(history):
                history["osbl_sample_count"] = [int(row.get("osbl_sample_count", 0)) for row in stats]
                history["osbl_batch_count"] = [int(row.get("osbl_batch_count", 0)) for row in stats]
            elif stats:
                history["osbl_sample_count"] = int(sum(int(row.get("osbl_sample_count", 0)) for row in stats))
                history["osbl_batch_count"] = int(sum(int(row.get("osbl_batch_count", 0)) for row in stats))
            self.training_history = history

        if isinstance(self.training_result, dict):
            self.training_result.update(
                {
                    "model_name": self.strategy_name,
                    "training_algorithm": PGPORTFOLIO_EIIE_ALGORITHM,
                    "rl_training": True,
                    "platform_native_rl_training": True,
                    "proxy_training": False,
                    "external_original_implementation": False,
                    "rankable_in_unified_table": True,
                    "portfolio_vector_memory": True,
                    "online_stochastic_batch_learning": True,
                    "clean_room_reimplementation": True,
                    "source_code_vendored": False,
                    "platform_adapted_surrogate": True,
                    "training_history": self.training_history,
                }
            )
            if self._osbl_epoch_stats:
                self.training_result["osbl_sample_count"] = int(
                    sum(int(row.get("osbl_sample_count", 0)) for row in self._osbl_epoch_stats)
                )
                self.training_result["osbl_batch_count"] = int(
                    sum(int(row.get("osbl_batch_count", 0)) for row in self._osbl_epoch_stats)
                )

    def _pvm_for_samples(self, samples: list[dict[str, Any]]) -> np.ndarray:
        dates = tuple(pd.Timestamp(sample["date"]) for sample in samples)
        if self._pvm is None or self._pvm.shape[0] != len(samples) or self._pvm_dates != dates:
            self._pvm = _initial_pvm(samples)
            self._pvm_dates = dates
        return self._pvm


def osbl_sample_indices(
    n_samples: int,
    batch_size: int,
    n_batches: int,
    rng: np.random.Generator,
    *,
    sample_bias: float = 5.0e-5,
    is_permed: bool = True,
) -> list[np.ndarray]:
    if int(n_samples) <= 0 or int(batch_size) <= 0 or int(n_batches) <= 0:
        return []
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    n_batches = int(n_batches)
    batches: list[np.ndarray] = []
    for _ in range(n_batches):
        if is_permed:
            batches.append(
                np.asarray(
                    [_geometric_recent_index(0, n_samples, sample_bias, rng) for _ in range(batch_size)],
                    dtype=np.int64,
                )
            )
        else:
            start_end = max(1, n_samples - batch_size + 1)
            start = _geometric_recent_index(0, start_end, sample_bias, rng)
            batches.append(np.arange(start, min(start + batch_size, n_samples), dtype=np.int64))
    return batches


def _initial_pvm(samples: list[dict[str, Any]]) -> np.ndarray:
    pvm = np.zeros((len(samples), len(samples[0]["mask"])), dtype=np.float32)
    for index, sample in enumerate(samples):
        mask = np.asarray(sample["mask"], dtype=bool)
        pvm[index] = _initial_weights(mask)
    return pvm


def _apply_pvm_updates(
    samples: list[dict[str, Any]],
    pvm: np.ndarray,
    updates: list[tuple[int, np.ndarray, pd.Timestamp]],
    trace: list[dict[str, Any]],
) -> None:
    for index, weights, date in updates:
        mask = np.asarray(samples[int(index)]["mask"], dtype=bool)
        pvm[int(index)] = _normalize_previous(weights, mask)
        trace.append(
            {
                "date": pd.Timestamp(date),
                "sample_index": int(index),
                "updated_sample_state": True,
            }
        )


def _geometric_recent_index(
    start: int,
    end_exclusive: int,
    sample_bias: float,
    rng: np.random.Generator,
) -> int:
    start = int(start)
    end_exclusive = int(end_exclusive)
    width = max(1, end_exclusive - start)
    bias = float(sample_bias)
    if not np.isfinite(bias) or bias <= 0.0 or bias > 1.0:
        raise ValueError("ERR_PGPORTFOLIO_OSBL_SAMPLE_BIAS_INVALID")
    if bias == 1.0:
        return int(end_exclusive - 1)
    ranks = np.arange(1, width + 1, dtype=np.float64)
    log_prob = np.log(bias) + (ranks - 1.0) * np.log1p(-bias)
    probabilities = np.exp(log_prob - np.max(log_prob))
    probabilities = probabilities / probabilities.sum()
    rank = int(rng.choice(ranks.astype(np.int64), p=probabilities))
    return int(end_exclusive - rank)


def _pgportfolio_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("pgportfolio_eiie_native") or config.get("pgportfolio_eiie"))


__all__ = ["PGPORTFOLIO_EIIE_ALGORITHM", "PGPortfolioEIIEStrategy", "osbl_sample_indices"]
