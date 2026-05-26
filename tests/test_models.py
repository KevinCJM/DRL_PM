import torch
import pytest
from src.models.encoders import CNNEncoder, CNNAttentionEncoder, TemporalTransformerEncoder, TCNEncoder, MLPEncoder, EncoderFactory

def test_encoder_family_shapes():
    batch_size = 8
    n_features = 10
    window_size = 60
    n_assets = 12
    latent_dim = 256
    
    # Input shape: [batch, n_features, window_size, n_assets]
    x = torch.randn(batch_size, n_features, window_size, n_assets)
    
    # CNN Encoder
    cnn = CNNEncoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    cnn_out = cnn(x)
    assert cnn_out.shape == (batch_size, latent_dim)

    # CNN + cross-asset attention encoder
    cnn_attention = CNNAttentionEncoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    cnn_attention_out = cnn_attention(x)
    assert cnn_attention_out.shape == (batch_size, latent_dim)
    
    # Transformer Encoder
    transformer = TemporalTransformerEncoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    transformer_out = transformer(x)
    assert transformer_out.shape == (batch_size, latent_dim)
    
    # TCN Encoder
    tcn = TCNEncoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    tcn_out = tcn(x)
    assert tcn_out.shape == (batch_size, latent_dim)
    
    # MLP Encoder
    mlp = MLPEncoder(n_features=n_features, window_size=window_size, latent_dim=latent_dim)
    mlp_out = mlp(x)
    assert mlp_out.shape == (batch_size, latent_dim)

def test_encoder_factory():
    config = {
        "n_features": 10,
        "window_size": 60,
        "latent_dim": 256,
        "encoder": {"type": "cnn", "dropout": 0.1, "use_layer_norm": True}
    }
    encoder = EncoderFactory.create(config)
    assert isinstance(encoder, CNNEncoder)
    
    config["encoder"]["type"] = "transformer"
    encoder = EncoderFactory.create(config)
    assert isinstance(encoder, TemporalTransformerEncoder)

    config["encoder"] = {
        "type": "cnn",
        "cross_asset_attention": {"enabled": True, "n_heads": 4, "n_layers": 1},
    }
    encoder = EncoderFactory.create(config)
    assert isinstance(encoder, CNNAttentionEncoder)

def test_cnn_encoder_kernel_size_variants():
    x = torch.randn(2, 6, 60, 13)
    for time_kernel, asset_kernel in ((1, 1), (1, 3), (3, 3), (5, 3), (11, 3), (21, 3)):
        encoder = CNNEncoder(
            n_features=6,
            window_size=60,
            latent_dim=32,
            kernel_size_time=time_kernel,
            kernel_size_asset=asset_kernel,
        )
        assert encoder(x).shape == (2, 32)

    with pytest.raises(ValueError, match="ERR_MODEL_CONFIG_INVALID"):
        CNNEncoder(n_features=6, window_size=60, kernel_size_time=0)

def test_auxiliary_loss_contract():
    batch_size = 4
    latent_dim = 256
    n_assets = 12
    n_features = 10
    window_size = 60
    
    from src.models.auxiliary_heads import AuxiliaryHeads
    
    aux_heads = AuxiliaryHeads(latent_dim=latent_dim, n_assets=n_assets, n_features=n_features, window_size=window_size, config={
        "tasks": ["return", "volatility", "trend", "rank", "downside", "max_drawdown", "cvar", "covariance", "reconstruction"],
        "future_return_horizons": [5, 20],
        "future_volatility_horizons": [20],
        "future_trend_horizons": [10],
    })
    
    latent_out = torch.randn(batch_size, latent_dim, requires_grad=True)
    aux_outputs = aux_heads(latent_out)
    
    assert "return_5" in aux_outputs
    assert aux_outputs["return_5"].shape == (batch_size, n_assets)
    assert "return_20" in aux_outputs
    assert "volatility_20" in aux_outputs
    assert aux_outputs["volatility_20"].shape == (batch_size, n_assets)
    assert "trend_10" in aux_outputs
    assert aux_outputs["trend_10"].shape == (batch_size, n_assets)
    assert "rank" in aux_outputs
    assert "downside_volatility" in aux_outputs
    assert "max_drawdown" in aux_outputs
    assert "cvar" in aux_outputs
    assert "covariance" in aux_outputs
    assert "reconstruction" in aux_outputs
    assert aux_outputs["reconstruction"].shape == (batch_size, n_features, window_size, n_assets)
    
    # Test representation regularization
    reg_loss = aux_heads.get_representation_loss(latent_out)
    assert reg_loss >= 0
    assert reg_loss.requires_grad == True

    targets = {
        "future_log_return_5d": torch.randn(batch_size, n_assets),
        "future_log_return_20d": torch.randn(batch_size, n_assets),
        "future_volatility_20d": torch.rand(batch_size, n_assets),
        "future_trend_10d": torch.randint(0, 2, (batch_size, n_assets)).float(),
        "future_cross_sectional_rank": torch.rand(batch_size, n_assets),
        "future_downside_volatility": torch.rand(batch_size, n_assets),
        "future_max_drawdown": torch.rand(batch_size, n_assets),
        "future_CVaR": torch.rand(batch_size, n_assets),
        "future_correlation_or_covariance": torch.randn(batch_size, n_assets),
        "masked_feature_reconstruction": torch.randn(batch_size, n_features, window_size, n_assets),
    }
    availability_mask = torch.tensor(
        [[True, True, False, True, True, False, True, True, True, True, False, True]] * batch_size
    )
    losses = aux_heads.compute_loss(aux_outputs, targets, latent_out, availability_mask)
    assert losses["total"].shape == ()
    assert losses["total"].requires_grad is True
    assert torch.isfinite(losses["total"])

    per_asset_latent = torch.randn(batch_size, n_assets, latent_dim, requires_grad=True)
    per_asset_outputs = aux_heads(per_asset_latent)
    assert per_asset_outputs["return_5"].shape == (batch_size, n_assets)
    assert per_asset_outputs["reconstruction"].shape == (batch_size, n_features, window_size, n_assets)
    per_asset_losses = aux_heads.compute_loss(per_asset_outputs, targets, per_asset_latent, availability_mask)
    assert per_asset_losses["total"].shape == ()
    assert per_asset_losses["total"].requires_grad is True
    assert torch.isfinite(per_asset_losses["total"])
    per_asset_reg = aux_heads.get_representation_loss(per_asset_latent, availability_mask)
    expected_mask = availability_mask.unsqueeze(-1).expand_as(per_asset_latent)
    expected_reg = (per_asset_latent ** 2).masked_select(expected_mask).mean()
    assert torch.allclose(per_asset_reg, expected_reg)

def test_full_gated_model_forward_contract():
    batch_size = 4
    n_features = 10
    window_size = 60
    n_assets = 12
    latent_dim = 256
    
    from src.models.dqn_gated_multitask_cnn_ppo import FullGatedModel
    
    config = {
        "n_features": n_features,
        "window_size": window_size,
        "n_assets": n_assets,
        "latent_dim": latent_dim,
        "encoder": {"type": "cnn"},
        "auxiliary": {"tasks": ["return"], "future_return_horizons": [5]},
        "dqn": {"dueling": True}
    }
    
    model = FullGatedModel(config)
    
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
    assert "log_prob" in outputs
    assert "value" in outputs
    assert "gate_q" in outputs
    assert "gate_action" in outputs
    assert "estimated_turnover" in outputs
    assert "estimated_cost" in outputs
    assert "aux_outputs" in outputs
    
    assert outputs["candidate_weights"].shape == (batch_size, n_assets)
    assert outputs["value"].shape == (batch_size, 1)
    assert outputs["gate_q"].shape == (batch_size, 2)
    assert outputs["gate_action"].shape == (batch_size,)
    assert outputs["estimated_turnover"].shape == (batch_size, 1)
    assert outputs["estimated_cost"].shape == (batch_size, 1)
