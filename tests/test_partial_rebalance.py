import torch
import pytest
from src.models.partial_rebalance_gate import (
    BetaIntensityActor,
    DEFAULT_DISCRETE_RHO_VALUES,
    DiscretePartialGate,
    RHO_EPS,
)

def test_beta_rebalance_intensity_distribution():
    batch_size = 8
    latent_dim = 128
    
    actor = BetaIntensityActor(latent_dim=latent_dim)
    with torch.no_grad():
        for param in actor.parameters():
            param.zero_()
    latent = torch.randn(batch_size, latent_dim)
    
    dist = actor.get_distribution(latent)
    sample = dist.rsample() # Beta distribution is reparameterizable
    
    assert sample.shape == (batch_size, 1)
    assert torch.all(sample >= 0) and torch.all(sample <= 1)
    
    log_prob = dist.log_prob(sample)
    assert log_prob.shape == (batch_size, 1)
    assert torch.allclose(actor.log_prob(latent, sample), log_prob)

    boundary_rho = torch.cat(
        [
            torch.zeros(batch_size // 2, 1),
            torch.ones(batch_size - batch_size // 2, 1),
        ],
        dim=0,
    )
    boundary_log_prob = actor.log_prob(latent, boundary_rho)
    expected_boundary_log_prob = dist.log_prob(boundary_rho.clamp(RHO_EPS, 1.0 - RHO_EPS))
    assert torch.isfinite(boundary_log_prob).all()
    assert torch.allclose(boundary_log_prob, expected_boundary_log_prob)

    rho, sample_log_prob = actor.sample_with_log_prob(latent)
    assert rho.shape == (batch_size, 1)
    assert torch.all(rho >= 0) and torch.all(rho <= 1)
    assert sample_log_prob.shape == (batch_size, 1)
    assert torch.isfinite(sample_log_prob).all()
    
    # Test deterministic
    mean = actor(latent, deterministic=True)
    assert mean.shape == (batch_size, 1)
    assert torch.all(mean >= 0) and torch.all(mean <= 1)

def test_beta_concentration_limits():
    actor = BetaIntensityActor(latent_dim=64, min_concentration=1e-2)
    latent = torch.randn(1, 64)
    dist = actor.get_distribution(latent)
    
    assert torch.all(dist.concentration0 >= 1e-2)
    assert torch.all(dist.concentration1 >= 1e-2)

    default_actor = BetaIntensityActor(latent_dim=64)
    default_dist = default_actor.get_distribution(latent)
    assert torch.all(default_dist.concentration0 >= 1e-3)
    assert torch.all(default_dist.concentration1 >= 1e-3)

    invalid_latent = torch.full((1, 64), float("nan"))
    with pytest.raises(ValueError, match="ERR_BETA_CONCENTRATION_NON_FINITE"):
        actor.get_distribution(invalid_latent)

    with pytest.raises(ValueError, match="ERR_BETA_MIN_CONCENTRATION_INVALID"):
        BetaIntensityActor(latent_dim=64, min_concentration=0.0)

def test_partial_gate_modes():
    from src.models.partial_rebalance_gated_ppo import PartialRebalanceGatedPPO
    batch_size = 2
    n_features = 10
    window_size = 60
    n_assets = 12

    x = torch.randn(batch_size, n_features, window_size, n_assets)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)
    current_weights = torch.randn(batch_size, n_assets)
    estimated_turnover = torch.randn(batch_size, 1)
    estimated_cost = torch.randn(batch_size, 1)

    for mode in ("discrete_dqn", "continuous_beta", "hybrid_dqn_beta"):
        config = {
            "n_features": n_features,
            "window_size": window_size,
            "n_assets": n_assets,
            "latent_dim": 256,
            "encoder": {"type": "cnn"},
            "dqn": {"output_dim": 7},
            "partial_rebalance": {
                "enabled": True,
                "mode": mode,
                "discrete_rho_values": list(DEFAULT_DISCRETE_RHO_VALUES),
            },
        }

        model = PartialRebalanceGatedPPO(config)
        outputs = model(x, mask, current_weights, estimated_turnover, estimated_cost)

        assert outputs["rebalance_intensity"].shape == (batch_size, 1)
        assert outputs["intensity_log_prob"].shape == (batch_size, 1)
        assert outputs["joint_log_prob"].shape == outputs["log_prob"].shape
        assert torch.allclose(
            outputs["joint_log_prob"],
            outputs["log_prob"] + outputs["intensity_log_prob"].squeeze(1),
        )
        assert torch.all(outputs["rebalance_intensity"] >= 0.0)
        assert torch.all(outputs["rebalance_intensity"] <= 1.0)

        if mode == "discrete_dqn":
            assert model.gate.output_dim == len(DEFAULT_DISCRETE_RHO_VALUES)
            assert model.discrete_gate.gate.output_dim == len(DEFAULT_DISCRETE_RHO_VALUES)
            assert outputs["gate_q"].shape == (batch_size, len(DEFAULT_DISCRETE_RHO_VALUES))
            assert outputs["gate_action"].shape == (batch_size,)
        elif mode == "continuous_beta":
            assert model.gate.output_dim == 2
            assert torch.equal(outputs["gate_action"], torch.ones(batch_size, dtype=torch.long))
        else:
            assert model.gate.output_dim == 2

    with pytest.raises(ValueError, match="ERR_PARTIAL_REBALANCE_MODE_INVALID"):
        PartialRebalanceGatedPPO(
            {
                "n_features": n_features,
                "window_size": window_size,
                "n_assets": n_assets,
                "partial_rebalance": {"mode": "invalid"},
            }
        )

    with pytest.raises(ValueError, match="ERR_PARTIAL_RHO_VALUES_INVALID"):
        DiscretePartialGate(latent_dim=64, n_assets=4, rho_values=[])

    nested_dqn_config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": 256,
        "model": {"dqn": {"output_dim": 9}},
        "partial_rebalance": {"enabled": True, "mode": "hybrid_dqn_beta"},
    }
    assert PartialRebalanceGatedPPO(nested_dqn_config).gate.output_dim == 2
