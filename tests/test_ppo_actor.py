import torch
import numpy as np
import pytest
from src.models.ppo_actor import PPOActor

def test_dirichlet_mask_semantics():
    batch_size = 4
    latent_dim = 256
    n_assets = 10
    
    actor = PPOActor(latent_dim=latent_dim, n_assets=n_assets)
    latent = torch.randn(batch_size, latent_dim)
    
    # Test with all assets available
    mask = torch.ones(batch_size, n_assets, dtype=torch.bool)
    dist = actor.get_distribution(latent, mask)
    sample = dist.sample()
    
    assert sample.shape == (batch_size, n_assets)
    assert torch.allclose(sample.sum(dim=1), torch.ones(batch_size))
    
    # Test with some assets unavailable
    mask = torch.tensor([
        [True, True, True, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, False, False, False, False],
        [True, False, True, False, True, False, True, False, True, False],
        [True] * 10
    ])
    
    dist = actor.get_distribution(latent, mask)
    sample = dist.sample()
    
    assert sample.shape == (batch_size, n_assets)
    assert torch.allclose(sample.sum(dim=1), torch.ones(batch_size))
    assert [indices.numel() for indices in dist.indices] == [3, 3, 5, 10]
    
    # Unavailable assets must have 0 weight
    assert torch.all(sample[~mask] == 0)
    
    # Test log_prob
    log_prob = dist.log_prob(sample)
    assert log_prob.shape == (batch_size,)

    deterministic = actor(latent, mask, deterministic=True)
    assert deterministic.shape == (batch_size, n_assets)
    assert torch.allclose(deterministic.sum(dim=1), torch.ones(batch_size))
    assert torch.all(deterministic[~mask] == 0)
    assert torch.isfinite(dist.log_prob(deterministic)).all()
    
    # Test single available asset
    single_mask = torch.tensor([[True, False, False, False, False, False, False, False, False, False]])
    single_latent = torch.randn(1, latent_dim)
    dist = actor.get_distribution(single_latent, single_mask)
    sample = dist.sample()
    assert sample[0, 0] == 1.0
    assert dist.log_prob(sample).item() == 0.0

    empty_mask = torch.zeros((1, n_assets), dtype=torch.bool)
    with pytest.raises(ValueError, match="ERR_CONSTRAINT_NO_AVAILABLE_ASSET"):
        actor.get_distribution(single_latent, empty_mask)

def test_ppo_actor_invalid_alpha():
    actor = PPOActor(latent_dim=128, n_assets=5)
    latent = torch.full((1, 128), float('nan'))
    mask = torch.ones((1, 5), dtype=torch.bool)
    
    with pytest.raises(ValueError, match="ERR_ALPHA_NON_FINITE"):
        actor.get_distribution(latent, mask)
