import torch
import pytest
from src.models.preference_conditioner import PreferenceConditioner

def test_preference_conditioning():
    batch_size = 4
    latent_dim = 256
    omega_dim = 5
    
    conditioner = PreferenceConditioner(latent_dim=latent_dim, omega_dim=omega_dim)
    
    latent = torch.randn(batch_size, latent_dim)
    omega = torch.tensor([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0]
    ])
    
    conditioned_latent = conditioner(latent, omega)
    assert conditioned_latent.shape == (batch_size, latent_dim)
    assert not torch.allclose(conditioned_latent, latent)

    with pytest.raises(ValueError, match="ERR_PREFERENCE_OMEGA_INVALID"):
        conditioner(latent, torch.tensor([[0.5, 0.5, 0.0, 0.0, -0.1]]).repeat(batch_size, 1))

    with pytest.raises(ValueError, match="ERR_PREFERENCE_OMEGA_INVALID"):
        conditioner(latent, torch.ones(batch_size, omega_dim))

    reward_vector = torch.randn(batch_size, omega_dim)
    preference_reward = conditioner.preference_reward(reward_vector, omega)
    assert preference_reward.shape == (batch_size, 1)

    evaluation_omegas = conditioner.evaluation_omegas()
    assert evaluation_omegas.shape[1] == omega_dim
    assert torch.all(evaluation_omegas >= 0.0)
    assert torch.allclose(evaluation_omegas.sum(dim=1), torch.ones(evaluation_omegas.shape[0]))

def test_preference_conditioned_gated_ppo():
    from src.models.preference_conditioned_gated_ppo import PreferenceConditionedGatedPPO
    
    batch_size = 2
    n_features = 10
    window_size = 60
    n_assets = 12
    omega_dim = 5
    
    config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": 256,
        "encoder": {"type": "cnn"},
        "preference": {"enabled": True, "omega_dim": omega_dim}
    }
    
    model = PreferenceConditionedGatedPPO(config)
    
    x = torch.randn(batch_size, n_features, window_size, n_assets)
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)
    current_weights = torch.randn(batch_size, n_assets)
    estimated_turnover = torch.randn(batch_size, 1)
    estimated_cost = torch.randn(batch_size, 1)
    omega = torch.tensor([[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.2, 0.2, 0.1]])
    reward_vector = torch.randn(batch_size, omega_dim)
    model.eval()
    
    outputs = model(
        x, 
        mask, 
        current_weights, 
        estimated_turnover, 
        estimated_cost,
        omega=omega,
        reward_vector=reward_vector,
        deterministic=True,
    )
    
    assert "candidate_weights" in outputs
    assert "value" in outputs
    assert "gate_q" in outputs
    assert "omega" in outputs
    assert "preference_reward" in outputs
    assert outputs["omega"].shape == (batch_size, omega_dim)
    assert torch.all(outputs["omega"] >= 0.0)
    assert torch.allclose(outputs["omega"].sum(dim=1), torch.ones(batch_size))
    assert outputs["preference_reward"].shape == (batch_size, 1)
    assert torch.allclose(outputs["preference_reward"], (reward_vector * omega).sum(dim=1, keepdim=True))
    assert not torch.allclose(outputs["latent"], outputs["raw_latent"])

    eval_omegas = model.evaluation_omegas()
    assert eval_omegas.shape == (6, omega_dim)
    assert torch.allclose(eval_omegas.sum(dim=1), torch.ones(eval_omegas.shape[0]))
