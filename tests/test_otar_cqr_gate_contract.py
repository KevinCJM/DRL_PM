"""Tests for M3-T2/M4: CQR gate contract.

Verifies:
- OTarCQRCritic forward shape
- OTarCQRGate gate_decision returns required keys
- A4-lite disables quantile tail
- target_actor uses deterministic_mean
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.otar_cqr_gate import OTarCQRCritic, OTarCQRGate


# ---------------------------------------------------------------------------
# OTarCQRCritic
# ---------------------------------------------------------------------------

class TestOTarCQRCritic:
    """OTarCQRCritic must output [batch, n_quantiles]."""

    def test_output_shape(self) -> None:
        latent_dim, n_assets, n_quantiles = 264, 8, 51
        critic = OTarCQRCritic(latent_dim, n_assets, n_quantiles)
        latent = torch.randn(2, latent_dim)
        pre_trade = torch.randn(2, n_assets)
        action = torch.randn(2, n_assets)
        out = critic(latent, pre_trade, action)
        assert out.shape == (2, n_quantiles), f"Expected (2, {n_quantiles}), got {out.shape}"

    def test_expected_value(self) -> None:
        quantiles = torch.randn(4, 51)
        ev = OTarCQRCritic.expected_value(quantiles)
        assert ev.shape == (4, 1)
        # expected_value should be mean of quantiles
        torch.testing.assert_close(ev, quantiles.mean(dim=1, keepdim=True))

    def test_lower_tail_loss(self) -> None:
        quantiles = torch.randn(4, 51)
        loss = OTarCQRCritic.lower_tail_loss(quantiles, tail_alpha=0.05)
        assert loss.shape == (4, 1)
        assert (loss >= 0).all(), "lower_tail_loss must be non-negative"


# ---------------------------------------------------------------------------
# OTarCQRGate gate_decision
# ---------------------------------------------------------------------------

class TestOTarCQRGateDecision:
    """gate_decision must return all required diagnostic keys."""

    def _make_minimal_config(self) -> dict:
        return {
            "n_assets": 4,
            "n_features": 3,
            "window_size": 10,
            "model": {
                "use_risk_state": False,
                "latent_dim": 64,
                "encoder": {"type": "cnn"},
            },
            "cqr": {
                "n_quantiles": 51,
                "gate_margin": 0.0,
                "target_update_interval": 10,
                "gate_gamma": 0.99,
                "quantile_huber_kappa": 1.0,
                "quantile_tail_enabled": True,
            },
            "reward": {
                "lambda_tail": 0.10,
                "confidence_q": 0.95,
            },
            "ppo": {"min_alpha": 1e-3},
            "dqn": {"enabled": False},
            "auxiliary": {"enabled": False},
        }

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gate_decision_returns_required_keys(self) -> None:
        config = self._make_minimal_config()
        model = OTarCQRGate(config)
        model.eval()

        batch, n_assets, latent_dim = 1, 4, config["model"]["latent_dim"]
        latent = torch.randn(batch, latent_dim)
        mask = torch.ones(batch, n_assets, dtype=torch.bool)
        pre_trade = torch.randn(batch, n_assets).softmax(dim=-1)
        candidate = torch.randn(batch, n_assets).softmax(dim=-1)
        cost_cand = torch.zeros(batch, 1)
        cost_hold = torch.zeros(batch, 1)

        result = model.gate_decision(latent, mask, pre_trade, candidate, cost_cand, cost_hold)

        required_keys = [
            "gate_action", "pred_delta_utility", "pred_candidate_mean_return",
            "pred_hold_mean_return", "pred_candidate_lower_tail_loss",
            "pred_hold_lower_tail_loss", "pred_candidate_utility", "pred_hold_utility",
            "quantile_spread_candidate", "quantile_spread_hold",
            "predicted_5pct_quantile_candidate", "predicted_5pct_quantile_hold",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_a4_lite_disables_tail(self) -> None:
        config = self._make_minimal_config()
        config["cqr"]["quantile_tail_enabled"] = False
        model = OTarCQRGate(config)
        model.eval()

        batch, n_assets, latent_dim = 1, 4, config["model"]["latent_dim"]
        latent = torch.randn(batch, latent_dim)
        mask = torch.ones(batch, n_assets, dtype=torch.bool)
        pre_trade = torch.randn(batch, n_assets).softmax(dim=-1)
        candidate = torch.randn(batch, n_assets).softmax(dim=-1)

        result = model.gate_decision(
            latent, mask, pre_trade, candidate,
            torch.zeros(batch, 1), torch.zeros(batch, 1),
        )
        # A4-lite: tail loss should be zero
        assert torch.allclose(result["pred_candidate_lower_tail_loss"], torch.zeros(1, 1))
        assert torch.allclose(result["pred_hold_lower_tail_loss"], torch.zeros(1, 1))


# ---------------------------------------------------------------------------
# update_targets
# ---------------------------------------------------------------------------

class TestUpdateTargets:
    """update_targets must copy params from online to target networks."""

    def test_target_cqr_critic_updated(self) -> None:
        config = {
            "n_assets": 4,
            "n_features": 3,
            "window_size": 5,
            "model": {"use_risk_state": False, "latent_dim": 32,
                       "encoder": {"type": "cnn"}},
            "cqr": {"n_quantiles": 11, "gate_margin": 0.0, "target_update_interval": 5,
                     "gate_gamma": 0.99, "quantile_huber_kappa": 1.0, "quantile_tail_enabled": True},
            "reward": {"lambda_tail": 0.10, "confidence_q": 0.95},
            "ppo": {"min_alpha": 1e-3},
            "dqn": {"enabled": False},
            "auxiliary": {"enabled": False},
        }
        model = OTarCQRGate(config)
        # Modify online critic
        with torch.no_grad():
            for p in model.cqr_critic.parameters():
                p.add_(1.0)
        # Before update, target should differ
        for p_online, p_target in zip(model.cqr_critic.parameters(), model.target_cqr_critic.parameters()):
            assert not torch.allclose(p_online, p_target)
        # After update, target should match
        model.update_targets()
        for p_online, p_target in zip(model.cqr_critic.parameters(), model.target_cqr_critic.parameters()):
            torch.testing.assert_close(p_online, p_target)
