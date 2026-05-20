from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


OPT_EPS = 1.0e-12


@dataclass
class PortfolioOptimizationResult:
    weights: np.ndarray
    success: bool
    fallback_reason: str | None = None
    message: str = ""


def shrink_covariance(returns: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("returns must be 2d")
    n_assets = matrix.shape[1]
    if n_assets == 0:
        raise ValueError("returns must include assets")
    covariance = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
    if covariance.shape != (n_assets, n_assets):
        covariance = covariance.reshape((n_assets, n_assets))
    covariance = np.where(np.isfinite(covariance), covariance, 0.0)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal_target = np.diag(np.diag(covariance))
    shrink = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = (1.0 - shrink) * covariance + shrink * diagonal_target
    return 0.5 * (shrunk + shrunk.T) + np.eye(n_assets, dtype=float) * OPT_EPS


def optimize_long_only_portfolio(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    objective: str,
    *,
    lambda_risk: float = 1.0,
    risk_free_rate: float = 0.0,
    maxiter: int = 200,
) -> PortfolioOptimizationResult:
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or sigma.shape != (mu.shape[0], mu.shape[0]):
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "invalid_moment_shape")
    if mu.shape[0] == 0:
        return PortfolioOptimizationResult(mu.astype(float), False, "no_available_asset")
    if mu.shape[0] == 1:
        return PortfolioOptimizationResult(np.array([1.0], dtype=float), True)
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "non_finite_moments")

    risk_aversion = max(float(lambda_risk), 0.0)
    rf = float(risk_free_rate)
    x0 = np.full(mu.shape[0], 1.0 / mu.shape[0], dtype=float)
    bounds = [(0.0, 1.0)] * mu.shape[0]
    constraints = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}

    def variance(weights: np.ndarray) -> float:
        return float(weights @ sigma @ weights)

    if objective == "mean_variance":
        objective_func = lambda weights: risk_aversion * variance(weights) - float(mu @ weights)
    elif objective == "min_variance":
        objective_func = variance
    elif objective == "max_sharpe":
        objective_func = lambda weights: -(
            (float(mu @ weights) - rf) / np.sqrt(max(variance(weights), OPT_EPS))
        )
    else:
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "invalid_objective")

    try:
        result = minimize(
            objective_func,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": 1.0e-12, "disp": False},
        )
    except Exception as exc:  # pragma: no cover - scipy exception types are not stable across versions
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "optimizer_exception", str(exc))

    if not result.success:
        return PortfolioOptimizationResult(
            np.zeros_like(mu, dtype=float),
            False,
            "optimizer_failed",
            str(result.message),
        )

    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "non_finite_weights")
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(weights.sum())
    if weight_sum <= OPT_EPS:
        return PortfolioOptimizationResult(np.zeros_like(mu, dtype=float), False, "zero_weight_sum")
    return PortfolioOptimizationResult(weights / weight_sum, True, message=str(result.message))


__all__ = [
    "OPT_EPS",
    "PortfolioOptimizationResult",
    "optimize_long_only_portfolio",
    "shrink_covariance",
]
