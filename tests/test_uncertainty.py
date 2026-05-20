import torch
import pytest
from src.models.uncertainty_aware_gated_ppo import UncertaintyAwareGatedPPO

def test_uncertainty_estimation_shapes():
    batch_size = 2
    n_features = 10
    window_size = 60
    n_assets = 12
    
    config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": 256,
        "encoder": {"type": "cnn"},
        "uncertainty": {"enabled": True, "method": "dropout", "n_samples": 5}
    }
    
    model = UncertaintyAwareGatedPPO(config)
    
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
    assert "mean_candidate_weights" in outputs
    assert "candidate_weight_variance" in outputs
    assert "weight_uncertainty" in outputs
    assert "q_uncertainty" in outputs
    assert "uncertainty_features" in outputs
    
    assert outputs["mean_candidate_weights"].shape == (batch_size, n_assets)
    assert outputs["candidate_weight_variance"].shape == (batch_size, n_assets)
    assert outputs["weight_uncertainty"].shape == (batch_size, n_assets)
    assert outputs["q_uncertainty"].shape == (batch_size, 2)
    assert outputs["uncertainty_features"].shape == (batch_size, n_assets + 2)
    assert torch.allclose(outputs["candidate_weights"], outputs["mean_candidate_weights"])
    assert torch.allclose(outputs["candidate_weights"].sum(dim=1), torch.ones(batch_size))
    assert torch.all(outputs["weight_uncertainty"] >= 0.0)
    assert torch.all(outputs["q_uncertainty"] >= 0.0)
    assert "holding_simple_return" not in outputs
    assert "return_from_decision_to_execution" not in outputs

def test_multi_head_uncertainty_shapes():
    batch_size = 2
    n_features = 10
    window_size = 60
    n_assets = 12

    config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": 256,
        "encoder": {"type": "cnn"},
        "uncertainty": {"enabled": True, "method": "multi_head", "n_heads": 3}
    }

    model = UncertaintyAwareGatedPPO(config)

    outputs = model(
        torch.randn(batch_size, n_features, window_size, n_assets),
        torch.ones(batch_size, n_assets, dtype=torch.bool),
        torch.randn(batch_size, n_assets),
        torch.randn(batch_size, 1),
        torch.randn(batch_size, 1),
        deterministic=True,
    )

    assert outputs["mean_candidate_weights"].shape == (batch_size, n_assets)
    assert outputs["candidate_weight_variance"].shape == (batch_size, n_assets)
    assert outputs["q_uncertainty"].shape == (batch_size, 2)
    assert outputs["uncertainty_features"].shape == (batch_size, n_assets + 2)
