from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_CALIBRATION_LABELS = {
    "realized_gate_utility_t",
    "realized_tail_loss_proxy_t",
    "net_log_return_t",
    "ppo_shaped_reward_t",
}

_DEFAULT_CONFIG = {
    "calibration_horizon_H": 5,
    "min_executed_action_count_for_calibration": {"rebalance": 50, "hold": 50},
    "tail_coverage_target": 0.05,
    "tail_coverage_tolerance": 0.03,
    "block_length": 20,
    "n_bootstrap": 1000,
    "seed": 42,
    "ci_level": 0.95,
}


class TailCalibrator:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self.H = int(cfg["calibration_horizon_H"])
        self.min_counts = dict(cfg["min_executed_action_count_for_calibration"])
        self.tail_coverage_target = float(cfg["tail_coverage_target"])
        self.tail_coverage_tolerance = float(cfg["tail_coverage_tolerance"])
        self.block_length = int(cfg["block_length"])
        self.n_bootstrap = int(cfg["n_bootstrap"])
        self.seed = int(cfg["seed"])
        self.ci_level = float(cfg["ci_level"])

    def calibrate(self, daily_diagnostics_df: pd.DataFrame, calibration_label: str = "realized_gross_simple_return_t") -> dict[str, Any]:
        if daily_diagnostics_df.empty:
            return self._invalid_result("empty_diagnostics")

        if calibration_label in FORBIDDEN_CALIBRATION_LABELS:
            return self._invalid_result(f"forbidden_calibration_label: {calibration_label}")

        required_cols = {"episode_id", "timestep", "done_t", "split",
                         "executed_gate_action", "predicted_5pct_quantile_executed",
                         calibration_label}
        missing = required_cols - set(daily_diagnostics_df.columns)
        if missing:
            return self._invalid_result(f"missing_columns: {sorted(missing)}")

        df = daily_diagnostics_df.sort_values(["episode_id", "timestep"]).reset_index(drop=True)

        if not self._validate_boundary_fields(df):
            return self._invalid_result("boundary_validation_failed")

        df = df[df["split"].isin(["validation", "test"])].copy()
        if df.empty:
            return self._invalid_result("no_validation_test_data")

        df["G_gross_t_H"] = self._compute_return_to_go(df, calibration_label)

        rebalance_mask = df["executed_gate_action"].fillna(0).astype(int) == 1
        hold_mask = df["executed_gate_action"].fillna(0).astype(int) == 0

        rebalance_df = df[rebalance_mask].copy()
        hold_df = df[hold_mask].copy()

        rebalance_count = len(rebalance_df)
        hold_count = len(hold_df)
        min_rebalance = self.min_counts.get("rebalance", 50)
        min_hold = self.min_counts.get("hold", 50)

        if rebalance_count < min_rebalance or hold_count < min_hold:
            return {
                "calibration_status": "low_action_count",
                "rebalance_count": rebalance_count,
                "hold_count": hold_count,
                "tail_coverage_error": None,
                "realized_below_quantile_frequency": None,
                "quantile_pinball_loss": None,
                "ci": self._bootstrap_ci(df) if len(df) > 0 else None,
            }

        freq_rebalance = self._below_quantile_frequency(rebalance_df)
        freq_hold = self._below_quantile_frequency(hold_df)
        combined_freq = self._below_quantile_frequency(df)

        error_rebalance = abs(freq_rebalance - self.tail_coverage_target)
        error_hold = abs(freq_hold - self.tail_coverage_target)
        error_combined = abs(combined_freq - self.tail_coverage_target)

        pinball_rebalance = self._pinball_loss(rebalance_df)
        pinball_hold = self._pinball_loss(hold_df)
        pinball_combined = self._pinball_loss(df)

        status = "passed" if error_combined <= self.tail_coverage_tolerance else "failed"

        return {
            "calibration_status": status,
            "rebalance_count": rebalance_count,
            "hold_count": hold_count,
            "tail_coverage_error": float(error_combined),
            "realized_below_quantile_frequency": float(combined_freq),
            "quantile_pinball_loss": float(pinball_combined),
            "rebalance": {
                "tail_coverage_error": float(error_rebalance),
                "realized_below_quantile_frequency": float(freq_rebalance),
                "quantile_pinball_loss": float(pinball_rebalance),
            },
            "hold": {
                "tail_coverage_error": float(error_hold),
                "realized_below_quantile_frequency": float(freq_hold),
                "quantile_pinball_loss": float(pinball_hold),
            },
        }

    def ex_post_counterfactual(
        self,
        daily_diagnostics_df: pd.DataFrame,
        daily_asset_returns_df: pd.DataFrame,
    ) -> dict[str, Any]:
        if daily_diagnostics_df.empty or daily_asset_returns_df.empty:
            return {"status": "empty_data"}

        required_diag = {"episode_id", "timestep", "executed_gate_action",
                         "candidate_weights_json", "pre_trade_drifted_weights_json",
                         "actual_pre_execution_return_t"}
        missing_diag = required_diag - set(daily_diagnostics_df.columns)
        if missing_diag:
            return {"status": f"missing_diag_columns: {sorted(missing_diag)}"}

        required_ret = {"episode_id", "timestep", "asset_index", "post_execution_simple_return"}
        missing_ret = required_ret - set(daily_asset_returns_df.columns)
        if missing_ret:
            return {"status": f"missing_ret_columns: {sorted(missing_ret)}"}

        diag_df = daily_diagnostics_df.sort_values(["episode_id", "timestep"]).reset_index(drop=True)
        ret_df = daily_asset_returns_df.sort_values(["episode_id", "timestep", "asset_index"]).reset_index(drop=True)

        records: list[dict[str, Any]] = []
        for (ep_id, ts), group in ret_df.groupby(["episode_id", "timestep"]):
            diag_rows = diag_df[(diag_df["episode_id"] == ep_id) & (diag_df["timestep"] == ts)]
            if diag_rows.empty:
                continue
            diag_row = diag_rows.iloc[0]

            post_returns = group.sort_values("asset_index")["post_execution_simple_return"].values
            pre_ret = float(diag_row.get("actual_pre_execution_return_t", 0.0))
            gate_action = int(diag_row.get("executed_gate_action", 0))

            candidate_weights = self._parse_weights(diag_row.get("candidate_weights_json", "[]"))
            hold_weights = self._parse_weights(diag_row.get("pre_trade_drifted_weights_json", "[]"))

            if len(candidate_weights) != len(post_returns) or len(hold_weights) != len(post_returns):
                continue

            candidate_gross = (1.0 + pre_ret) * (1.0 + float(np.dot(candidate_weights, post_returns))) - 1.0
            hold_gross = (1.0 + pre_ret) * (1.0 + float(np.dot(hold_weights, post_returns))) - 1.0

            records.append({
                "episode_id": ep_id,
                "timestep": ts,
                "gate_action": gate_action,
                "candidate_gross_return": float(candidate_gross),
                "hold_gross_return": float(hold_gross),
                "actual_gross_return": float(candidate_gross) if gate_action == 1 else float(hold_gross),
            })

        return {
            "status": "completed",
            "records": records,
            "counterfactual_df": pd.DataFrame(records) if records else pd.DataFrame(),
        }

    def _validate_boundary_fields(self, df: pd.DataFrame) -> bool:
        for ep_id, ep_group in df.groupby("episode_id"):
            timesteps = ep_group["timestep"].values
            expected = np.arange(len(timesteps))
            if not np.array_equal(timesteps, expected):
                return False
            done_count = int(ep_group["done_t"].sum())
            if done_count != 1:
                return False
        return True

    def _compute_return_to_go(self, df: pd.DataFrame, calibration_label: str) -> pd.Series:
        gamma = 0.99
        result = pd.Series(np.nan, index=df.index)
        for ep_id, ep_group in df.groupby("episode_id"):
            ep_group = ep_group.sort_values("timestep")
            returns = ep_group[calibration_label].fillna(0.0).values
            n = len(returns)
            gtg = np.zeros(n)
            for t in range(n):
                for k in range(self.H):
                    if t + k < n:
                        gtg[t] += (gamma ** k) * returns[t + k]
            result.loc[ep_group.index] = gtg
        return result

    def _below_quantile_frequency(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        predicted = df["predicted_5pct_quantile_executed"].fillna(0.0).values
        realized = df["G_gross_t_H"].fillna(0.0).values
        return float(np.mean(predicted > realized))

    def _pinball_loss(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        tau = self.tail_coverage_target
        predicted = df["predicted_5pct_quantile_executed"].fillna(0.0).values
        realized = df["G_gross_t_H"].fillna(0.0).values
        diff = realized - predicted
        loss = np.where(diff < 0, (tau - 1.0) * diff, tau * diff)
        return float(np.mean(loss))

    def _bootstrap_ci(self, df: pd.DataFrame) -> dict[str, float]:
        rng = np.random.RandomState(self.seed)
        n = len(df)
        if n < self.block_length:
            return {"lower": 0.0, "upper": 1.0}

        freqs: list[float] = []
        for _ in range(self.n_bootstrap):
            n_blocks = max(1, n // self.block_length)
            block_starts = rng.randint(0, n - self.block_length + 1, size=n_blocks)
            indices = np.concatenate([np.arange(s, s + self.block_length) for s in block_starts])
            sample = df.iloc[indices[:n]]
            freqs.append(self._below_quantile_frequency(sample))

        alpha = 1.0 - self.ci_level
        return {
            "lower": float(np.percentile(freqs, 100 * alpha / 2)),
            "upper": float(np.percentile(freqs, 100 * (1 - alpha / 2))),
        }

    @staticmethod
    def _parse_weights(json_str: str) -> np.ndarray:
        try:
            weights = json.loads(str(json_str))
            return np.asarray(weights, dtype=float)
        except (json.JSONDecodeError, TypeError, ValueError):
            return np.array([], dtype=float)

    def _invalid_result(self, reason: str) -> dict[str, Any]:
        return {
            "calibration_status": "invalid",
            "reason": reason,
            "rebalance_count": 0,
            "hold_count": 0,
            "tail_coverage_error": None,
            "realized_below_quantile_frequency": None,
            "quantile_pinball_loss": None,
        }


__all__ = ["TailCalibrator", "FORBIDDEN_CALIBRATION_LABELS"]
