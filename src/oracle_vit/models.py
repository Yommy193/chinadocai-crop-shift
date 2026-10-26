from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisionConfig:
    image_size: int
    patch_size: int
    num_classes: int
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass(frozen=True)
class SequenceConfig:
    vocab_size: int
    max_length: int
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


class OracleViT(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        grid = config.image_size // config.patch_size
        number_of_patches = grid * grid
        self.patch_embedding = nn.Conv2d(
            3,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, number_of_patches + 1, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat((class_tokens, patches), dim=1)
        hidden = self.dropout(hidden + self.position_embedding)
        hidden = self.encoder(hidden)
        return self.norm(hidden[:, 0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)


class OracleSequenceModel(nn.Module):
    """Shared token embeddings with a positional MLM and a permutation-equivariant sorter."""

    def __init__(self, config: SequenceConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.max_length, config.embed_dim))
        self.input_dropout = nn.Dropout(config.dropout)

        def encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=int(config.embed_dim * config.mlp_ratio),
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=config.depth, enable_nested_tensor=False)

        self.mlm_encoder = encoder()
        self.sort_encoder = encoder()
        self.mlm_norm = nn.LayerNorm(config.embed_dim)
        self.sort_norm = nn.LayerNorm(config.embed_dim)
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.sort_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, 1),
        )
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_mlm(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.shape[1] > self.config.max_length:
            raise ValueError(f"Sequence length {tokens.shape[1]} exceeds {self.config.max_length}")
        hidden = self.token_embedding(tokens) + self.position_embedding[:, : tokens.shape[1]]
        hidden = self.mlm_encoder(self.input_dropout(hidden), src_key_padding_mask=padding_mask)
        hidden = self.mlm_norm(hidden)
        return torch.nn.functional.linear(hidden, self.token_embedding.weight, self.mlm_bias)

    def forward_sort(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.sort_encoder(self.input_dropout(self.token_embedding(tokens)), src_key_padding_mask=padding_mask)
        scores = self.sort_head(self.sort_norm(hidden)).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, torch.inf)
        return scores

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)

