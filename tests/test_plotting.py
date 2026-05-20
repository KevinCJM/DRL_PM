import pandas as pd

from src.config import DEFAULT_CONFIG
from src.utils.logger import write_run_outputs
from src.utils.plotting import ALL_FIGURE_FILES, CONDITIONAL_FIGURE_FILES, REQUIRED_FIGURE_FILES, generate_figures


def test_generate_figures_writes_frozen_png_names(tmp_path):
    result = _plot_result()
    config = {
        "dqn": {"per_enabled": False},
        "model": {"encoder": {"cross_asset_attention": {"enabled": False}}},
    }

    artifacts = generate_figures(result, tmp_path, config=config)

    assert tuple(REQUIRED_FIGURE_FILES) == REQUIRED_FIGURE_FILES
    assert set(artifacts) == set(ALL_FIGURE_FILES)
    assert set(CONDITIONAL_FIGURE_FILES).issubset(artifacts)
    assert {path.name for path in (tmp_path / "figures").glob("*.png")} == set(ALL_FIGURE_FILES)
    for path in artifacts.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 0


def test_write_run_outputs_persists_figure_artifacts(tmp_path):
    config = DEFAULT_CONFIG.copy()
    config["dqn"] = dict(DEFAULT_CONFIG["dqn"])
    config["model"] = {
        "encoder": {
            "cross_asset_attention": {
                "enabled": True,
            }
        }
    }
    config["output"] = {"run_name": "plot_test"}

    artifacts = write_run_outputs(_plot_result(), tmp_path, config=config)

    for filename in ALL_FIGURE_FILES:
        key = f"figure_{filename.removesuffix('.png')}"
        assert artifacts[key] == tmp_path / "figures" / filename
        assert artifacts[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _plot_result():
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    daily_returns = pd.DataFrame(
        {
            "date": dates,
            "decision_date": dates - pd.Timedelta(days=1),
            "execution_date": dates,
            "execution_price_type": ["open"] * len(dates),
            "next_valuation_date": dates,
            "split": ["test"] * len(dates),
            "seed": [42] * len(dates),
            "fold_id": ["fixed"] * len(dates),
            "model_name": ["model"] * len(dates),
            "pre_execution_return": [0.001, 0.002, -0.001, 0.0, 0.003, 0.001],
            "post_execution_return": [0.004, -0.002, 0.001, 0.003, -0.001, 0.002],
            "gross_return": [0.005, 0.0, 0.0, 0.003, 0.002, 0.003],
            "transaction_cost": [0.0005] * len(dates),
            "transaction_cost_on_initial_nav": [0.0005] * len(dates),
            "net_return": [0.0045, -0.0005, -0.0005, 0.0025, 0.0015, 0.0025],
            "portfolio_log_return": [0.00449, -0.00050, -0.00050, 0.00249, 0.00149, 0.00249],
            "nav": [1.0045, 1.0040, 1.0035, 1.0060, 1.0075, 1.0100],
        }
    )
    daily_weights = pd.DataFrame(
        [
            {
                "date": date,
                "split": "test",
                "seed": 42,
                "fold_id": "fixed",
                "model_name": "model",
                "asset_id": asset,
                "asset_class": asset_class,
                "weight": weight,
            }
            for date, weights in zip(dates, [(0.6, 0.4), (0.55, 0.45), (0.50, 0.50), (0.52, 0.48), (0.58, 0.42), (0.57, 0.43)])
            for asset, asset_class, weight in (("510300.SH", "equity", weights[0]), ("511010.SH", "bond", weights[1]))
        ]
    )
    daily_turnover = pd.DataFrame(
        {
            "date": dates,
            "decision_date": dates - pd.Timedelta(days=1),
            "execution_date": dates,
            "execution_price_type": ["open"] * len(dates),
            "next_valuation_date": dates,
            "split": ["test"] * len(dates),
            "seed": [42] * len(dates),
            "fold_id": ["fixed"] * len(dates),
            "model_name": ["model"] * len(dates),
            "turnover": [0.0, 0.05, 0.04, 0.02, 0.06, 0.01],
            "rebalance_action": [0, 1, 1, 0, 1, 0],
            "rebalance_intensity": [0.0, 0.6, 0.5, 0.0, 0.7, 0.0],
            "average_holding_period": [1, 2, 3, 4, 5, 6],
        }
    )
    daily_rebalance = daily_turnover.assign(
        estimated_turnover=[0.0, 0.04, 0.03, 0.02, 0.05, 0.01],
        realized_turnover=[0.0, 0.05, 0.04, 0.02, 0.06, 0.01],
        estimated_cost=[0.0, 0.0004, 0.0003, 0.0002, 0.0005, 0.0001],
        realized_cost=[0.0, 0.0005, 0.0004, 0.0002, 0.0006, 0.0001],
        q_hold=[0.1, 0.2, 0.15, 0.13, 0.18, 0.2],
        q_rebalance=[0.12, 0.25, 0.18, 0.12, 0.24, 0.19],
        q_gap=[0.02, 0.05, 0.03, -0.01, 0.06, -0.01],
    )
    daily_costs = daily_turnover.assign(
        proportional_cost=[0.0002] * len(dates),
        fixed_cost=[0.0] * len(dates),
        slippage_cost=[0.0001] * len(dates),
        market_impact_cost=[0.0002] * len(dates),
        total_transaction_cost=[0.0005] * len(dates),
        estimated_cost=[0.0004] * len(dates),
        realized_cost=[0.0005] * len(dates),
    )
    training_history = pd.DataFrame(
        {
            "train_reward": [1.0, 1.2, 1.1],
            "validation_reward": [0.9, 1.0, 1.05],
            "episodic_return": [0.01, 0.02, 0.015],
            "PPO_actor_loss": [0.3, 0.2, 0.1],
            "PPO_critic_loss": [0.4, 0.3, 0.2],
            "PPO_entropy": [0.7, 0.6, 0.5],
            "PPO_approx_kl": [0.01, 0.02, 0.015],
            "PPO_clip_fraction": [0.2, 0.1, 0.05],
            "DQN_Q_value": [1.0, 1.1, 1.2],
            "DQN_Q_gap": [0.1, 0.2, 0.15],
            "DQN_TD_error": [0.3, 0.2, 0.1],
            "DQN_epsilon": [1.0, 0.8, 0.6],
            "DQN_gate_action_ratio": [0.4, 0.5, 0.6],
            "auxiliary_prediction_loss": [0.5, 0.4, 0.3],
            "gradient_norm": [0.8, 0.7, 0.6],
            "learning_rate": [0.001, 0.001, 0.0005],
            "constraint_violation": [0, 1, 0],
            "PCA_explained_variance": [0.5, 0.7, 0.9],
            "PCA_component_sensitivity": [0.1, 0.2, 0.15],
        }
    )
    return {
        "daily_returns": daily_returns,
        "daily_weights": daily_weights,
        "daily_turnover": daily_turnover,
        "daily_rebalance": daily_rebalance,
        "daily_costs": daily_costs,
        "training_history": training_history,
        "PPO_advantage": [0.1, -0.1, 0.2, 0.0],
        "input_matrix_validation_score": pd.DataFrame({"matrix": ["M6", "M7"], "score": [1.1, 1.2]}),
        "preference_frontier": pd.DataFrame({"risk": [0.1, 0.2], "return": [0.05, 0.08]}),
        "uncertainty_action_heatmap": [[0.1, 0.2], [0.3, 0.4]],
        "DQN_replay_priority": [0.3, 0.2, 0.5],
        "attention_heatmap": [[0.6, 0.4], [0.2, 0.8]],
        "risk_contribution": pd.DataFrame({"asset_id": ["510300.SH", "511010.SH"], "risk_contribution": [0.65, 0.35]}),
    }
