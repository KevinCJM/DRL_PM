from __future__ import annotations

import os
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_FIGURE_FILES: tuple[str, ...] = (
    "equity_curve.png",
    "drawdown_curve_abs.png",
    "drawdown_curve_signed.png",
    "rolling_return_curve.png",
    "rolling_volatility_curve.png",
    "rolling_Sharpe_curve.png",
    "rolling_Sortino_curve.png",
    "return_distribution_histogram.png",
    "return_qq_plot.png",
    "VaR_CVaR_tail_curve.png",
    "portfolio_weight_heatmap.png",
    "portfolio_weight_stack_area.png",
    "asset_class_exposure_stack_area.png",
    "rebalance_timeline.png",
    "turnover_curve.png",
    "transaction_cost_curve.png",
    "cost_breakdown_stack_area.png",
    "risk_contribution_bar.png",
    "HHI_concentration_curve.png",
    "correlation_heatmap.png",
    "covariance_heatmap.png",
    "train_reward_curve.png",
    "validation_reward_curve.png",
    "episodic_return_curve.png",
    "PPO_actor_loss_curve.png",
    "PPO_critic_loss_curve.png",
    "PPO_entropy_curve.png",
    "PPO_approx_kl_curve.png",
    "PPO_clip_fraction_curve.png",
    "PPO_advantage_distribution.png",
    "DQN_Q_value_curve.png",
    "DQN_Q_gap_curve.png",
    "DQN_TD_error_curve.png",
    "DQN_epsilon_curve.png",
    "DQN_gate_action_ratio_curve.png",
    "auxiliary_prediction_loss_curve.png",
    "gradient_norm_curve.png",
    "learning_rate_curve.png",
    "policy_concentration_curve.png",
    "constraint_violation_curve.png",
    "input_matrix_validation_score_bar.png",
    "PCA_explained_variance_curve.png",
    "PCA_component_sensitivity_curve.png",
    "preference_frontier.png",
    "uncertainty_action_heatmap.png",
    "cvar_tail_risk_curve.png",
    "rebalance_intensity_histogram.png",
)
CONDITIONAL_FIGURE_FILES: tuple[str, ...] = (
    "DQN_replay_priority_distribution.png",
    "attention_heatmap.png",
)
ALL_FIGURE_FILES: tuple[str, ...] = REQUIRED_FIGURE_FILES + CONDITIONAL_FIGURE_FILES

_HISTORY_COLUMNS: dict[str, tuple[str, ...]] = {
    "train_reward_curve.png": ("train_reward", "reward", "train_rewards"),
    "validation_reward_curve.png": ("validation_reward", "validation_rewards", "val_reward"),
    "episodic_return_curve.png": ("episodic_return", "episode_return", "episodic_returns"),
    "PPO_actor_loss_curve.png": ("PPO_actor_loss", "ppo_actor_loss", "actor_loss"),
    "PPO_critic_loss_curve.png": ("PPO_critic_loss", "ppo_critic_loss", "critic_loss", "value_loss"),
    "PPO_entropy_curve.png": ("PPO_entropy", "ppo_entropy", "entropy"),
    "PPO_approx_kl_curve.png": ("PPO_approx_kl", "ppo_approx_kl", "approx_kl"),
    "PPO_clip_fraction_curve.png": ("PPO_clip_fraction", "ppo_clip_fraction", "clip_fraction"),
    "DQN_Q_value_curve.png": ("DQN_Q_value", "dqn_q_value", "q_value", "q_values"),
    "DQN_Q_gap_curve.png": ("DQN_Q_gap", "dqn_q_gap", "q_gap"),
    "DQN_TD_error_curve.png": ("DQN_TD_error", "dqn_td_error", "td_error"),
    "DQN_epsilon_curve.png": ("DQN_epsilon", "dqn_epsilon", "epsilon"),
    "DQN_gate_action_ratio_curve.png": ("DQN_gate_action_ratio", "dqn_gate_action_ratio", "gate_action_ratio"),
    "auxiliary_prediction_loss_curve.png": ("auxiliary_prediction_loss", "auxiliary_loss", "aux_loss"),
    "gradient_norm_curve.png": ("gradient_norm", "grad_norm"),
    "learning_rate_curve.png": ("learning_rate", "lr"),
    "constraint_violation_curve.png": ("constraint_violation", "constraint_violation_count"),
    "PCA_explained_variance_curve.png": ("PCA_explained_variance", "pca_explained_variance"),
    "PCA_component_sensitivity_curve.png": ("PCA_component_sensitivity", "pca_component_sensitivity"),
}


def generate_figures(
    result: Any,
    run_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    figures_dir = Path(run_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    context = _PlotContext(result=result, config=config or {})
    paths: dict[str, Path] = {}
    for filename in ALL_FIGURE_FILES:
        path = figures_dir / filename
        _save_figure(path, lambda fig, ax, name=filename: _render_figure(name, fig, ax, context))
        paths[filename] = path
    return paths


class _PlotContext:
    def __init__(self, *, result: Any, config: Mapping[str, Any]) -> None:
        self.result = result
        self.config = config

    def frame(self, key: str) -> pd.DataFrame:
        return _frame(_value(self.result, key))

    def value(self, key: str) -> Any:
        return _value(self.result, key)


def _render_figure(filename: str, fig: Any, ax: Any, context: _PlotContext) -> bool:
    if filename == "equity_curve.png":
        return _plot_equity(ax, context)
    if filename == "drawdown_curve_abs.png":
        return _plot_drawdown(ax, context, signed=False)
    if filename == "drawdown_curve_signed.png":
        return _plot_drawdown(ax, context, signed=True)
    if filename == "rolling_return_curve.png":
        return _plot_rolling(ax, context, "mean", annualize=True)
    if filename == "rolling_volatility_curve.png":
        return _plot_rolling(ax, context, "volatility", annualize=True)
    if filename == "rolling_Sharpe_curve.png":
        return _plot_rolling(ax, context, "sharpe", annualize=True)
    if filename == "rolling_Sortino_curve.png":
        return _plot_rolling(ax, context, "sortino", annualize=True)
    if filename == "return_distribution_histogram.png":
        return _plot_return_histogram(ax, context)
    if filename == "return_qq_plot.png":
        return _plot_return_qq(ax, context)
    if filename in {"VaR_CVaR_tail_curve.png", "cvar_tail_risk_curve.png"}:
        return _plot_tail_risk(ax, context)
    if filename == "portfolio_weight_heatmap.png":
        return _plot_weight_heatmap(fig, ax, context)
    if filename == "portfolio_weight_stack_area.png":
        return _plot_weight_stack(ax, context)
    if filename == "asset_class_exposure_stack_area.png":
        return _plot_asset_class_stack(ax, context)
    if filename == "rebalance_timeline.png":
        return _plot_rebalance_timeline(ax, context)
    if filename == "turnover_curve.png":
        return _plot_column_curve(ax, context.frame("daily_turnover"), "turnover", "turnover")
    if filename == "transaction_cost_curve.png":
        return _plot_column_curve(ax, context.frame("daily_costs"), "total_transaction_cost", "transaction_cost")
    if filename == "cost_breakdown_stack_area.png":
        return _plot_cost_breakdown(ax, context)
    if filename == "risk_contribution_bar.png":
        return _plot_risk_contribution(ax, context)
    if filename in {"HHI_concentration_curve.png", "policy_concentration_curve.png"}:
        return _plot_hhi(ax, context, filename.removesuffix(".png"))
    if filename == "correlation_heatmap.png":
        return _plot_return_matrix(fig, ax, context, cov=False)
    if filename == "covariance_heatmap.png":
        return _plot_return_matrix(fig, ax, context, cov=True)
    if filename == "PPO_advantage_distribution.png":
        return _plot_distribution(ax, context, ("PPO_advantage", "ppo_advantage", "advantage"), "PPO_advantage")
    if filename == "input_matrix_validation_score_bar.png":
        return _plot_bar_from_result(ax, context, ("input_matrix_validation_score", "validation_scores"), "input_matrix")
    if filename == "preference_frontier.png":
        return _plot_preference_frontier(ax, context)
    if filename == "uncertainty_action_heatmap.png":
        return _plot_heatmap_from_result(fig, ax, context, ("uncertainty_action_heatmap", "uncertainty_actions"), "uncertainty")
    if filename == "rebalance_intensity_histogram.png":
        return _plot_distribution_from_frame(ax, context.frame("daily_rebalance"), "rebalance_intensity", "rebalance_intensity")
    if filename == "DQN_replay_priority_distribution.png":
        if not _enabled(context.config, ("dqn", "per_enabled"), default=False):
            _placeholder(ax, filename, "not_applicable")
            return True
        return _plot_distribution(ax, context, ("DQN_replay_priority", "dqn_replay_priority", "replay_priority"), "replay_priority")
    if filename == "attention_heatmap.png":
        if not _enabled(context.config, ("model", "encoder", "cross_asset_attention", "enabled"), default=False):
            _placeholder(ax, filename, "not_applicable")
            return True
        return _plot_heatmap_from_result(fig, ax, context, ("attention_heatmap", "attention_weights"), "attention")
    if filename in _HISTORY_COLUMNS:
        return _plot_history(ax, context, _HISTORY_COLUMNS[filename], filename.removesuffix(".png"))
    return False


def _save_figure(path: Path, render: Callable[[Any, Any], bool]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    temp_path: Path | None = None
    try:
        if not render(fig, ax):
            _placeholder(ax, path.name, "not_applicable")
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
        fig.savefig(temp_path, format="png", dpi=100, bbox_inches="tight")
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        plt.close(fig)


def _plot_equity(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("daily_returns")
    nav = _nav(frame)
    if nav.empty:
        return False
    ax.plot(_x_values(frame, len(nav)), nav.to_numpy(dtype=float), linewidth=1.6)
    ax.set_title("equity_curve")
    ax.set_ylabel("nav")
    return True


def _plot_drawdown(ax: Any, context: _PlotContext, *, signed: bool) -> bool:
    frame = context.frame("daily_returns")
    nav = _nav(frame)
    if nav.empty:
        return False
    drawdown = nav / nav.cummax() - 1.0
    values = drawdown if signed else drawdown.abs()
    ax.plot(_x_values(frame, len(values)), values.to_numpy(dtype=float), linewidth=1.6)
    ax.set_title("drawdown_signed" if signed else "drawdown_abs")
    return True


def _plot_rolling(ax: Any, context: _PlotContext, mode: str, *, annualize: bool) -> bool:
    frame = context.frame("daily_returns")
    returns = _returns(frame)
    if returns.empty:
        return False
    window = _rolling_window(len(returns))
    rolling = returns.rolling(window=window, min_periods=1)
    if mode == "mean":
        values = rolling.mean()
        if annualize:
            values = values * 252.0
    elif mode == "volatility":
        values = rolling.std(ddof=0)
        if annualize:
            values = values * np.sqrt(252.0)
    elif mode == "sharpe":
        mean = rolling.mean()
        std = rolling.std(ddof=0).replace(0.0, np.nan)
        values = mean / std
        if annualize:
            values = values * np.sqrt(252.0)
    else:
        values = rolling.apply(_sortino_value, raw=True)
        if annualize:
            values = values * np.sqrt(252.0)
    ax.plot(_x_values(frame, len(values)), values.to_numpy(dtype=float), linewidth=1.4)
    ax.set_title(f"rolling_{mode}")
    return True


def _plot_return_histogram(ax: Any, context: _PlotContext) -> bool:
    returns = _returns(context.frame("daily_returns"))
    if returns.empty:
        return False
    ax.hist(returns.to_numpy(dtype=float), bins=min(20, max(5, len(returns))), color="#4472C4", alpha=0.85)
    ax.set_title("return_distribution")
    return True


def _plot_return_qq(ax: Any, context: _PlotContext) -> bool:
    returns = np.sort(_returns(context.frame("daily_returns")).to_numpy(dtype=float))
    if returns.size == 0:
        return False
    probs = (np.arange(1, returns.size + 1, dtype=float) - 0.5) / float(returns.size)
    normal = NormalDist()
    theoretical = np.array([normal.inv_cdf(float(prob)) for prob in probs], dtype=float)
    ax.scatter(theoretical, returns, s=14)
    ax.set_title("return_qq")
    return True


def _plot_tail_risk(ax: Any, context: _PlotContext) -> bool:
    returns = np.sort(_returns(context.frame("daily_returns")).to_numpy(dtype=float))
    if returns.size == 0:
        return False
    alpha = 0.05
    cutoff = max(1, int(np.ceil(returns.size * alpha)))
    tail = returns[:cutoff]
    losses = -tail
    ax.plot(np.arange(1, losses.size + 1), losses, marker="o", linewidth=1.2)
    ax.axhline(max(0.0, -float(np.quantile(returns, alpha))), color="#C00000", linestyle="--", linewidth=1.0)
    ax.set_title("tail_risk")
    return True


def _plot_weight_heatmap(fig: Any, ax: Any, context: _PlotContext) -> bool:
    pivot = _weight_pivot(context)
    if pivot.empty:
        return False
    image = ax.imshow(pivot.to_numpy(dtype=float).T, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.columns)))
    ax.set_yticklabels([str(col) for col in pivot.columns], fontsize=7)
    ax.set_title("portfolio_weight_heatmap")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    return True


def _plot_weight_stack(ax: Any, context: _PlotContext) -> bool:
    pivot = _weight_pivot(context)
    if pivot.empty:
        return False
    x = np.arange(len(pivot), dtype=float)
    ax.stackplot(x, pivot.T.to_numpy(dtype=float), labels=[str(col) for col in pivot.columns])
    _small_legend(ax)
    ax.set_title("portfolio_weight_stack")
    return True


def _plot_asset_class_stack(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("daily_weights")
    if frame.empty or "weight" not in frame.columns:
        return False
    group_col = "asset_class" if "asset_class" in frame.columns else "asset_id"
    if group_col not in frame.columns:
        frame["asset_id"] = "all_assets"
        group_col = "asset_id"
    frame = frame.copy()
    frame["date"] = _date_column(frame, len(frame)).to_numpy()
    grouped = frame.pivot_table(index="date", columns=group_col, values="weight", aggfunc="sum").fillna(0.0)
    if grouped.empty:
        return False
    x = np.arange(len(grouped), dtype=float)
    ax.stackplot(x, grouped.T.to_numpy(dtype=float), labels=[str(col) for col in grouped.columns])
    _small_legend(ax)
    ax.set_title("asset_class_exposure")
    return True


def _plot_rebalance_timeline(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("daily_rebalance")
    if frame.empty:
        return False
    values = _numeric_series(frame, "rebalance_intensity")
    if values.empty:
        values = _numeric_series(frame, "rebalance_action")
    if values.empty:
        return False
    ax.vlines(_x_values(frame, len(values)), 0.0, values.to_numpy(dtype=float), linewidth=1.5)
    ax.set_title("rebalance_timeline")
    return True


def _plot_column_curve(ax: Any, frame: pd.DataFrame, column: str, title: str) -> bool:
    values = _numeric_series(frame, column)
    if values.empty:
        return False
    ax.plot(_x_values(frame, len(values)), values.to_numpy(dtype=float), linewidth=1.5)
    ax.set_title(title)
    return True


def _plot_cost_breakdown(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("daily_costs")
    columns = ("proportional_cost", "fixed_cost", "slippage_cost", "market_impact_cost")
    available = [column for column in columns if column in frame.columns]
    if frame.empty or not available:
        return False
    values = frame.loc[:, available].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if values.empty:
        return False
    ax.stackplot(np.arange(len(values), dtype=float), values.T.to_numpy(dtype=float), labels=available)
    _small_legend(ax)
    ax.set_title("cost_breakdown")
    return True


def _plot_risk_contribution(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("risk_contribution")
    if not frame.empty and {"asset_id", "risk_contribution"}.issubset(frame.columns):
        values = frame.set_index("asset_id")["risk_contribution"].pipe(pd.to_numeric, errors="coerce").dropna()
    else:
        pivot = _weight_pivot(context)
        values = pd.Series(dtype=float) if pivot.empty else pivot.tail(1).T.iloc[:, 0].abs()
        total = float(values.sum())
        if total > 0.0:
            values = values / total
    if values.empty:
        return False
    ax.bar([str(index) for index in values.index], values.to_numpy(dtype=float))
    ax.set_title("risk_contribution")
    ax.tick_params(axis="x", labelrotation=45)
    return True


def _plot_hhi(ax: Any, context: _PlotContext, title: str) -> bool:
    pivot = _weight_pivot(context)
    if pivot.empty:
        return False
    hhi = (pivot.astype(float) ** 2.0).sum(axis=1)
    ax.plot(np.arange(len(hhi), dtype=float), hhi.to_numpy(dtype=float), linewidth=1.4)
    ax.set_title(title)
    return True


def _plot_return_matrix(fig: Any, ax: Any, context: _PlotContext, *, cov: bool) -> bool:
    matrix_value = context.value("covariance_matrix" if cov else "correlation_matrix")
    matrix = _matrix(matrix_value)
    if matrix.size == 0:
        returns = _returns(context.frame("daily_returns"))
        if returns.empty:
            return False
        matrix = np.array([[float(returns.var(ddof=0) if cov else 1.0)]], dtype=float)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title("covariance_heatmap" if cov else "correlation_heatmap")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    return True


def _plot_history(ax: Any, context: _PlotContext, candidates: Sequence[str], title: str) -> bool:
    values = _series_from_candidates(context, candidates)
    if values.empty:
        return False
    ax.plot(np.arange(len(values), dtype=float), values.to_numpy(dtype=float), linewidth=1.5)
    ax.set_title(title)
    return True


def _plot_distribution(ax: Any, context: _PlotContext, candidates: Sequence[str], title: str) -> bool:
    values = _series_from_candidates(context, candidates)
    if values.empty:
        return False
    ax.hist(values.to_numpy(dtype=float), bins=min(20, max(5, len(values))), color="#70AD47", alpha=0.85)
    ax.set_title(title)
    return True


def _plot_distribution_from_frame(ax: Any, frame: pd.DataFrame, column: str, title: str) -> bool:
    values = _numeric_series(frame, column)
    if values.empty:
        return False
    ax.hist(values.to_numpy(dtype=float), bins=min(20, max(5, len(values))), color="#70AD47", alpha=0.85)
    ax.set_title(title)
    return True


def _plot_bar_from_result(ax: Any, context: _PlotContext, candidates: Sequence[str], title: str) -> bool:
    value = next((context.value(candidate) for candidate in candidates if context.value(candidate) is not None), None)
    frame = _frame(value)
    if frame.empty:
        series = _series(value)
        if series.empty:
            return False
        ax.bar([str(i) for i in range(len(series))], series.to_numpy(dtype=float))
    elif len(frame.columns) >= 2:
        labels = frame.iloc[:, 0].astype(str).tolist()
        values = pd.to_numeric(frame.iloc[:, 1], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.bar(labels, values)
    else:
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
        if values.empty:
            return False
        ax.bar([str(i) for i in range(len(values))], values.to_numpy(dtype=float))
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=45)
    return True


def _plot_preference_frontier(ax: Any, context: _PlotContext) -> bool:
    frame = context.frame("preference_frontier")
    if not frame.empty and {"risk", "return"}.issubset(frame.columns):
        risk = pd.to_numeric(frame["risk"], errors="coerce")
        ret = pd.to_numeric(frame["return"], errors="coerce")
        valid = risk.notna() & ret.notna()
        if valid.any():
            ax.scatter(risk[valid], ret[valid], s=20)
            ax.set_title("preference_frontier")
            return True
    metrics = _mapping(context.value("metrics"))
    risk = metrics.get("cvar", metrics.get("max_drawdown"))
    ret = metrics.get("annualized_return", metrics.get("cumulative_return"))
    if risk is None or ret is None:
        return False
    ax.scatter([float(risk)], [float(ret)], s=24)
    ax.set_title("preference_frontier")
    return True


def _plot_heatmap_from_result(fig: Any, ax: Any, context: _PlotContext, candidates: Sequence[str], title: str) -> bool:
    matrix = np.array([], dtype=float)
    for candidate in candidates:
        matrix = _matrix(context.value(candidate))
        if matrix.size > 0:
            break
    if matrix.size == 0:
        return False
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    return True


def _weight_pivot(context: _PlotContext) -> pd.DataFrame:
    frame = context.frame("daily_weights")
    if frame.empty or not {"asset_id", "weight"}.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = _date_column(frame, len(frame)).to_numpy()
    pivot = frame.pivot_table(index="date", columns="asset_id", values="weight", aggfunc="sum").fillna(0.0)
    return pivot.sort_index()


def _nav(frame: pd.DataFrame) -> pd.Series:
    nav = _numeric_series(frame, "nav")
    if not nav.empty:
        return nav.reset_index(drop=True)
    returns = _returns(frame)
    if returns.empty:
        return pd.Series(dtype=float)
    return pd.Series(np.cumprod(1.0 + returns.to_numpy(dtype=float)))


def _returns(frame: pd.DataFrame) -> pd.Series:
    return _numeric_series(frame, "net_return").reset_index(drop=True)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _series_from_candidates(context: _PlotContext, candidates: Sequence[str]) -> pd.Series:
    history = context.frame("training_history")
    for candidate in candidates:
        value = context.value(candidate)
        series = _series(value)
        if not series.empty:
            return series
        if not history.empty and candidate in history.columns:
            series = _series(history[candidate])
            if not series.empty:
                return series
    return pd.Series(dtype=float)


def _series(value: Any) -> pd.Series:
    if value is None:
        return pd.Series(dtype=float)
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(value, Mapping):
        frame = _frame(value)
        if frame.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame.iloc[:, -1], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(value, (str, Path)):
        try:
            frame = pd.read_csv(value)
        except Exception:
            return pd.Series(dtype=float)
        if frame.empty:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame.iloc[:, -1], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if np.isscalar(value):
        return pd.Series([float(value)], dtype=float)
    return pd.to_numeric(pd.Series(value), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _matrix(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    frame = _frame(value)
    if not frame.empty:
        data = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        return data if data.size > 0 else np.array([], dtype=float)
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.array([], dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix if matrix.size > 0 else np.array([], dtype=float)


def _x_values(frame: pd.DataFrame, length: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    if not frame.empty and "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.notna().sum() == length:
            return np.arange(length, dtype=float)
    return np.arange(length, dtype=float)


def _date_column(frame: pd.DataFrame, length: int) -> pd.Series:
    for column in ("date", "next_valuation_date", "execution_date", "decision_date"):
        if column in frame.columns:
            return pd.Series(frame[column]).astype(str)
    return pd.Series(np.arange(length, dtype=int)).astype(str)


def _rolling_window(length: int) -> int:
    return max(1, min(20, int(length)))


def _sortino_value(values: np.ndarray) -> float:
    downside = values[values < 0.0]
    downside_std = float(np.std(downside)) if downside.size else np.nan
    return np.nan if downside_std == 0.0 or np.isnan(downside_std) else float(np.mean(values) / downside_std)


def _frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(dict(value))
        except ValueError:
            return pd.DataFrame([dict(value)])
    return pd.DataFrame(value)


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _enabled(config: Mapping[str, Any], path: Sequence[str], *, default: bool) -> bool:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return bool(current)


def _small_legend(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels and len(labels) <= 12:
        ax.legend(loc="best", fontsize=6, frameon=False)


def _placeholder(ax: Any, title: str, status: str) -> None:
    ax.text(0.5, 0.5, status, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title.removesuffix(".png"))
    ax.set_xticks([])
    ax.set_yticks([])


__all__ = [
    "ALL_FIGURE_FILES",
    "CONDITIONAL_FIGURE_FILES",
    "REQUIRED_FIGURE_FILES",
    "generate_figures",
]
