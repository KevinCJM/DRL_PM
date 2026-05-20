import torch
import pytest
from src.models.ppo_critic import PPOCritic
from src.models.distributional_critic import DistributionalCritic

def test_ppo_critic_shapes():
    batch_size = 8
    latent_dim = 256
    
    critic = PPOCritic(latent_dim=latent_dim)
    latent = torch.randn(batch_size, latent_dim)
    
    value = critic(latent)
    assert value.shape == (batch_size, 1)

    old_value = torch.zeros(batch_size, 1)
    returns = torch.ones(batch_size, 1)
    loss = critic.clipped_value_loss(value, old_value, returns, clip_range=0.2)
    assert loss.shape == ()
    assert torch.isfinite(loss)

def test_quantile_cvar_shapes():
    batch_size = 8
    latent_dim = 256
    n_quantiles = 51
    
    critic = DistributionalCritic(latent_dim=latent_dim, n_quantiles=n_quantiles)
    latent = torch.randn(batch_size, latent_dim)
    
    quantiles = critic(latent)
    assert quantiles.shape == (batch_size, n_quantiles)
    
    # Test CVaR calculation
    cvar = critic.get_cvar(quantiles, alpha=0.05)
    assert cvar.shape == (batch_size, 1)
    assert critic.expected_value(quantiles).shape == (batch_size, 1)
    tail_loss = critic.get_tail_loss(quantiles, alpha=0.05)
    assert tail_loss.shape == (batch_size, 1)
    assert torch.all(tail_loss >= 0.0)
    
    # Test tail loss (Quantile Regression loss)
    target = torch.randn(batch_size, 1)
    loss = critic.quantile_huber_loss(quantiles, target)
    assert loss.shape == () # Reduced to scalar
    assert torch.isfinite(loss)

def test_distributional_cvar_gate_features():
    from src.models.distributional_cvar_gated_ppo import DistributionalCVaRGatedPPO
    batch_size = 2
    n_features = 10
    window_size = 60
    n_assets = 12
    n_quantiles = 51

    config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": 256,
        "encoder": {"type": "cnn"},
        "distributional_cvar": {"enabled": True, "n_quantiles": n_quantiles, "cvar_alpha": 0.05}
    }

    model = DistributionalCVaRGatedPPO(config)

    x = torch.randn(batch_size, n_features, window_size, n_assets)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)
    current_weights = torch.randn(batch_size, n_assets)
    estimated_turnover = torch.randn(batch_size, 1)
    estimated_cost = torch.randn(batch_size, 1)

    outputs = model(
        x,
        mask,
        current_weights,
        estimated_turnover,
        estimated_cost
    )

    assert "candidate_weights" in outputs
    assert "gate_q" in outputs
    assert "expected_value" in outputs
    assert "cvar" in outputs
    assert "candidate_expected_value" in outputs
    assert "hold_expected_value" in outputs
    assert "candidate_cvar" in outputs
    assert "hold_cvar" in outputs
    assert "candidate_tail_loss" in outputs
    assert "hold_tail_loss" in outputs
    assert "delta_U" in outputs
    assert "gate_risk_features" in outputs
    assert "distributional_features" in outputs

    assert outputs["cvar"].shape == (batch_size, 1)
    assert outputs["hold_cvar"].shape == (batch_size, 1)
    assert outputs["delta_U"].shape == (batch_size, 1)
    assert outputs["gate_risk_features"].shape == (batch_size, 7)
    assert torch.equal(outputs["gate_risk_features"], outputs["distributional_features"])
    assert outputs["cvar_alpha"] == pytest.approx(0.05)
    assert outputs["gate_input_extensions"] == (
        "candidate_expected_value",
        "hold_expected_value",
        "candidate_cvar",
        "hold_cvar",
        "candidate_tail_loss",
        "hold_tail_loss",
        "delta_U",
    )

    target = torch.randn(batch_size, 1)
    loss = model.quantile_huber_loss(outputs["candidate_quantiles"], target)
    assert loss.shape == ()
    assert torch.isfinite(loss)
