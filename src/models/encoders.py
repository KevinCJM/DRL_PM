from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CNN_CHANNELS = (32, 64, 128)
DEFAULT_LATENT_DIM = 256
DEFAULT_DROPOUT = 0.10


class EncoderBase(nn.Module):
    def __init__(self, n_features: int, window_size: int, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)

    def _validate_input(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError("ERR_MODEL_INPUT_SHAPE: expected [batch,n_features,window_size,n_assets]")
        batch_size, n_features, window_size, n_assets = x.shape
        if n_features != self.n_features or window_size != self.window_size:
            raise ValueError(
                "ERR_MODEL_INPUT_SHAPE: expected "
                f"n_features={self.n_features}, window_size={self.window_size}"
            )
        return int(batch_size), int(n_features), int(window_size), int(n_assets)


class CNNEncoder(EncoderBase):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        use_layer_norm: bool = True,
        cnn_channels: Sequence[int] | None = None,
        kernel_size_time: int = 3,
        kernel_size_asset: int = 3,
        stride: int = 1,
    ):
        super().__init__(n_features, window_size, latent_dim)
        channels = tuple(int(value) for value in (cnn_channels or DEFAULT_CNN_CHANNELS))
        if not channels:
            raise ValueError("ERR_MODEL_CONFIG_INVALID: encoder.cnn_channels")
        kernel_size = (int(kernel_size_time), int(kernel_size_asset))
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError("ERR_MODEL_CONFIG_INVALID: encoder.kernel_size")
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        layers: list[nn.Module] = []
        in_channels = self.n_features
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=int(stride),
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.Dropout2d(float(dropout)),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Linear(channels[-1], self.latent_dim),
            nn.LayerNorm(self.latent_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        features = self.conv(x)
        pooled = self.pooling(features).flatten(1)
        return self.projection(pooled)


class CNNAttentionEncoder(CNNEncoder):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        n_heads: int = 4,
        n_layers: int = 1,
        **kwargs: Any,
    ):
        super().__init__(
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            dropout=dropout,
            **kwargs,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(n_heads),
            dim_feedforward=self.latent_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.asset_attention = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.attention_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = self._validate_input(x)
        features = self.conv(x)
        asset_features = F.adaptive_avg_pool2d(features, (1, x.shape[-1])).squeeze(2)
        asset_tokens = asset_features.transpose(1, 2)
        projected = self.projection(asset_tokens.reshape(-1, asset_tokens.shape[-1]))
        projected = projected.reshape(batch_size, x.shape[-1], self.latent_dim)
        attended = self.asset_attention(projected)
        return self.attention_norm(attended.mean(dim=1))


CNNWithAttentionEncoder = CNNAttentionEncoder


class TemporalTransformerEncoder(EncoderBase):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        n_heads: int = 4,
        n_layers: int = 2,
        model_dim: int = DEFAULT_LATENT_DIM,
        feedforward_dim: int = 512,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__(n_features, window_size, latent_dim)
        self.model_dim = int(model_dim)
        self.embedding = nn.Linear(n_features, self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(n_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.projection = nn.Sequential(
            nn.Linear(self.model_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features, window_size, n_assets = self._validate_input(x)
        sequence = x.permute(0, 3, 2, 1).reshape(batch_size * n_assets, window_size, n_features)
        encoded = self.transformer(self.embedding(sequence)).mean(dim=1)
        asset_latent = self.projection(encoded).reshape(batch_size, n_assets, self.latent_dim)
        return asset_latent.mean(dim=1)

class TCNEncoder(EncoderBase):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        channels: Sequence[int] | None = None,
        kernel_size: int = 3,
        dilation_base: int = 2,
    ):
        super().__init__(n_features, window_size, latent_dim)
        tcn_channels = tuple(int(value) for value in (channels or (64, 128, 128)))
        layers: list[nn.Module] = []
        in_channels = self.n_features
        for layer_idx, out_channels in enumerate(tcn_channels):
            dilation = int(dilation_base) ** layer_idx
            padding = ((int(kernel_size) - 1) * dilation) // 2
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=int(kernel_size),
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(tcn_channels[-1], self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features, window_size, n_assets = self._validate_input(x)
        sequence = x.permute(0, 3, 1, 2).reshape(batch_size * n_assets, n_features, window_size)
        encoded = self.pooling(self.tcn(sequence)).squeeze(-1)
        asset_latent = self.projection(encoded).reshape(batch_size, n_assets, self.latent_dim)
        return asset_latent.mean(dim=1)

class MLPEncoder(EncoderBase):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        n_assets: int | None = None,
    ):
        super().__init__(n_features, window_size, latent_dim)
        input_dim = None if n_assets is None else self.n_features * self.window_size * int(n_assets)
        first_linear: nn.Module
        if input_dim is None:
            first_linear = nn.LazyLinear(512)
        else:
            first_linear = nn.Linear(input_dim, 512)
        self.fc = nn.Sequential(
            first_linear,
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = self._validate_input(x)
        return self.fc(x.reshape(batch_size, -1))

class EncoderFactory:
    @staticmethod
    def create(config: Mapping[str, Any]) -> EncoderBase:
        model_config = _mapping(config.get("model"))
        encoder_config = _mapping(config.get("encoder") or model_config.get("encoder"))
        encoder_type = str(encoder_config.get("type", model_config.get("default_encoder", "cnn"))).lower()
        n_features = int(config["n_features"])
        window_size = int(config.get("window_size", _mapping(config.get("feature_matrix")).get("window_size")))
        latent_dim = int(config.get("latent_dim", model_config.get("latent_dim", DEFAULT_LATENT_DIM)))
        dropout = float(encoder_config.get("dropout", model_config.get("dropout", DEFAULT_DROPOUT)))
        attention_config = _mapping(
            encoder_config.get("cross_asset_attention") or config.get("cross_asset_attention")
        )

        if encoder_type in {"cnn_attention", "cnn+attention", "cnn_with_attention"} or (
            encoder_type == "cnn" and bool(attention_config.get("enabled", False))
        ):
            return CNNAttentionEncoder(
                n_features=n_features,
                window_size=window_size,
                latent_dim=latent_dim,
                dropout=dropout,
                n_heads=int(attention_config.get("n_heads", 4)),
                n_layers=int(attention_config.get("n_layers", 1)),
                **_cnn_kwargs(encoder_config),
            )
        if encoder_type == "cnn":
            return CNNEncoder(
                n_features=n_features,
                window_size=window_size,
                latent_dim=latent_dim,
                dropout=dropout,
                **_cnn_kwargs(encoder_config),
            )
        if encoder_type in {"transformer", "temporal_transformer"}:
            transformer_config = _mapping(config.get("temporal_transformer"))
            return TemporalTransformerEncoder(
                n_features=n_features,
                window_size=window_size,
                latent_dim=latent_dim,
                n_heads=int(transformer_config.get("n_heads", encoder_config.get("n_heads", 4))),
                n_layers=int(transformer_config.get("n_layers", encoder_config.get("n_layers", 2))),
                model_dim=int(transformer_config.get("model_dim", encoder_config.get("model_dim", latent_dim))),
                feedforward_dim=int(
                    transformer_config.get("feedforward_dim", encoder_config.get("feedforward_dim", 512))
                ),
                dropout=float(transformer_config.get("dropout", dropout)),
            )
        if encoder_type == "tcn":
            tcn_config = _mapping(config.get("tcn"))
            return TCNEncoder(
                n_features=n_features,
                window_size=window_size,
                latent_dim=latent_dim,
                dropout=float(tcn_config.get("dropout", dropout)),
                channels=tcn_config.get("channels", encoder_config.get("channels")),
                kernel_size=int(tcn_config.get("kernel_size", encoder_config.get("kernel_size", 3))),
                dilation_base=int(tcn_config.get("dilation_base", encoder_config.get("dilation_base", 2))),
            )
        if encoder_type == "mlp":
            return MLPEncoder(
                n_features=n_features,
                window_size=window_size,
                latent_dim=latent_dim,
                dropout=dropout,
                n_assets=config.get("n_assets"),
            )
        raise ValueError(f"ERR_MODEL_CONFIG_INVALID: unknown encoder type {encoder_type}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cnn_kwargs(encoder_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "use_layer_norm": bool(encoder_config.get("use_layer_norm", True)),
        "cnn_channels": encoder_config.get("cnn_channels"),
        "kernel_size_time": int(encoder_config.get("kernel_size_time", 3)),
        "kernel_size_asset": int(encoder_config.get("kernel_size_asset", 3)),
        "stride": int(encoder_config.get("stride", 1)),
    }


__all__ = [
    "EncoderBase",
    "CNNEncoder",
    "CNNAttentionEncoder",
    "CNNWithAttentionEncoder",
    "TemporalTransformerEncoder",
    "TCNEncoder",
    "MLPEncoder",
    "EncoderFactory",
]
