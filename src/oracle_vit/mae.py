"""Small masked-autoencoder ViT used for local, split-safe pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .models import OracleViT


@dataclass(frozen=True)
class MAEConfig:
    image_size: int
    patch_size: int
    encoder_embed_dim: int = 256
    encoder_depth: int = 6
    encoder_num_heads: int = 8
    encoder_mlp_ratio: float = 4.0
    decoder_embed_dim: int = 128
    decoder_depth: int = 2
    decoder_num_heads: int = 4
    decoder_mlp_ratio: float = 4.0
    dropout: float = 0.0
    mask_ratio: float = 0.75


def _encoder_layer(embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class MaskedAutoencoderViT(nn.Module):
    """MAE whose encoder parameters match :class:`OracleViT` names exactly."""

    def __init__(self, config: MAEConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not 0 < config.mask_ratio < 1:
            raise ValueError("mask_ratio must be between zero and one")
        self.config = config
        grid = config.image_size // config.patch_size
        self.number_of_patches = grid * grid
        self.patch_embedding = nn.Conv2d(
            3, config.encoder_embed_dim, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.encoder_embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, config.encoder_embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.TransformerEncoder(
            _encoder_layer(
                config.encoder_embed_dim,
                config.encoder_num_heads,
                config.encoder_mlp_ratio,
                config.dropout,
            ),
            num_layers=config.encoder_depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.encoder_embed_dim)

        self.decoder_embed = nn.Linear(config.encoder_embed_dim, config.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_position_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches, config.decoder_embed_dim))
        self.decoder = nn.TransformerEncoder(
            _encoder_layer(
                config.decoder_embed_dim,
                config.decoder_num_heads,
                config.decoder_mlp_ratio,
                config.dropout,
            ),
            num_layers=config.decoder_depth,
            enable_nested_tensor=False,
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(config.decoder_embed_dim, config.patch_size * config.patch_size * 3)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patch = self.config.patch_size
        if images.ndim != 4 or images.shape[-1] != self.config.image_size or images.shape[-2] != self.config.image_size:
            raise ValueError("images must be square tensors matching MAEConfig.image_size")
        grid = self.config.image_size // patch
        return images.reshape(images.shape[0], 3, grid, patch, grid, patch).permute(0, 2, 4, 3, 5, 1).reshape(
            images.shape[0], grid * grid, patch * patch * 3
        )

    def _random_masking(
        self, tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = tokens.shape
        kept = int(length * (1.0 - mask_ratio))
        if kept < 1:
            raise ValueError("mask ratio leaves no visible patch")
        shuffle = torch.argsort(torch.rand(batch, length, device=tokens.device), dim=1)
        restore = torch.argsort(shuffle, dim=1)
        visible = torch.gather(tokens, 1, shuffle[:, :kept, None].expand(-1, -1, tokens.shape[-1]))
        mask = torch.ones(batch, length, device=tokens.device)
        mask[:, :kept] = 0
        mask = torch.gather(mask, 1, restore)
        return visible, mask, restore

    def forward_encoder(
        self, images: torch.Tensor, mask_ratio: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = self.config.mask_ratio if mask_ratio is None else float(mask_ratio)
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        patches = patches + self.position_embedding[:, 1:]
        visible, mask, restore = self._random_masking(patches, ratio)
        class_token = self.class_token + self.position_embedding[:, :1]
        hidden = torch.cat((class_token.expand(images.shape[0], -1, -1), visible), dim=1)
        hidden = self.norm(self.encoder(self.dropout(hidden)))
        return hidden, mask, restore

    def forward_decoder(self, latent: torch.Tensor, restore: torch.Tensor) -> torch.Tensor:
        visible = self.decoder_embed(latent[:, 1:])
        masked_count = restore.shape[1] - visible.shape[1]
        masked = self.mask_token.expand(visible.shape[0], masked_count, -1)
        tokens = torch.cat((visible, masked), dim=1)
        tokens = torch.gather(tokens, 1, restore[:, :, None].expand(-1, -1, tokens.shape[-1]))
        tokens = tokens + self.decoder_position_embedding
        return self.decoder_pred(self.decoder_norm(self.decoder(tokens)))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, restore = self.forward_encoder(images)
        prediction = self.forward_decoder(latent, restore)
        target = self.patchify(images)
        # Per-patch normalization makes reconstruction emphasize shape and local
        # contrast rather than the global ink/background intensity of a modality.
        target = (target - target.mean(dim=-1, keepdim=True)) / (target.var(dim=-1, keepdim=True) + 1.0e-6).sqrt()
        loss = (prediction - target).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, prediction, mask

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)

    def load_encoder_into(self, classifier: OracleViT) -> list[str]:
        """Copy the architecture-compatible encoder into a fresh classifier."""
        source = self.state_dict()
        target = classifier.state_dict()
        copied = {
            key: value
            for key, value in source.items()
            if key in target and key != "head.weight" and key != "head.bias" and target[key].shape == value.shape
        }
        required = {"patch_embedding.weight", "class_token", "position_embedding", "norm.weight", "norm.bias"}
        if not required.issubset(copied):
            raise ValueError("MAE encoder does not match the requested OracleViT classifier")
        classifier.load_state_dict(copied, strict=False)
        return sorted(copied)
