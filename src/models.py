import torch
import torch.nn as nn
from enum import Enum
from src.layers import FullyConnectedBlock, ResidualBlock, ResidualLinearBlock, PLQPLayer


class HeadType(Enum):
    SHALLOW = "shallow"
    DEEP = "deep"
    RESIDUAL = "residual"
    RESIDUAL_V2 = "residual_v2"
    RESIDUAL_V3 = "residual_v3"
    TWO_TOWER = "two_tower"


class AblationType(Enum):
    META_ONLY = "meta_only"
    TAG_ONLY = "tag_only"
    ALL_FEATURES = "all_features"


class RecipeNet(nn.Module):
    def __init__(
        self,
        meta_in: int,
        tag_in: int,
        hidden_dim: int = 128,
        head_type: HeadType = HeadType.RESIDUAL,
        num_meta: int = 10,
        cat_meta: int = 200,
    ):
        """
        Dual-Encoder architecture.

        - Metadata Encoder: Processes structured recipe metadata
          (continuous + one-hot categorical features).
        - Tag Encoder: Processes semantic review feedback features
          (pred_* and intensity_*).

        Multi-head design:
        - SHALLOW: simple baseline
        - DEEP: deeper MLP stack
        - RESIDUAL: residual backbone
        - RESIDUAL_V2: deeper residual-linear backbone
        - RESIDUAL_V3: heterogeneous metadata encoder with PLQP for numeric features
        - TWO_TOWER: asymmetric late-fusion architecture

        Output:
        - A bounded scalar prediction in the recipe rating range [1, 5].
        """
        super().__init__()

        self.meta_in = meta_in
        self.tag_in = tag_in
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.num_meta = num_meta
        self.cat_meta = cat_meta

        # Metadata encoder
        self.default_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

        # TWO_TOWER metadata encoder (compresses to 32D when hidden_dim=128)
        self.two_tower_meta_encoder = nn.Sequential(
            FullyConnectedBlock(meta_in, hidden_dim // 2),
            FullyConnectedBlock(hidden_dim // 2, hidden_dim // 4),
        )

        # RESIDUAL_V3 heterogeneous metadata path
        self.plqp = PLQPLayer(
            num_features=num_meta,
            num_bins=15,
            embeddings_dim=16,
        )
        self.num_proj = FullyConnectedBlock(num_meta * 16, hidden_dim // 2)
        self.cat_proj = FullyConnectedBlock(cat_meta, hidden_dim // 2)

        # Tag encoder
        self.tag_encoder = nn.Sequential(
            FullyConnectedBlock(tag_in, hidden_dim),
            FullyConnectedBlock(hidden_dim, hidden_dim),
        )

        fusion_dim = hidden_dim * 2

        # Head selection
        if head_type == HeadType.SHALLOW:
            self.head = self.build_shallow_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.DEEP:
            self.head = self.build_deep_head(fusion_dim, hidden_dim)
        elif head_type == HeadType.RESIDUAL:
            self.head = self.build_residual_head(fusion_dim, hidden_dim)
        elif head_type in [HeadType.RESIDUAL_V2, HeadType.RESIDUAL_V3]:
            self.head = self.build_residual_head_v2(fusion_dim, hidden_dim)
        # In TWO_TOWER, the residual head is applied only to the tag stream before late fusion
        elif head_type == HeadType.TWO_TOWER:
            self.head = self.build_residual_head_v2(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported head type: {head_type}")

        # Output head
        if head_type == HeadType.TWO_TOWER:
            self.meta_norm = nn.LayerNorm(hidden_dim // 4)  # 32D
            self.tag_norm = nn.LayerNorm(hidden_dim)        # 128D

            two_tower_fusion_dim = (hidden_dim // 4) + hidden_dim
            self.regressor = nn.Sequential(
                nn.Linear(two_tower_fusion_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            standard_regressor = nn.Linear(hidden_dim, 1)
            nn.init.xavier_uniform_(standard_regressor.weight)
            if standard_regressor.bias is not None:
                nn.init.zeros_(standard_regressor.bias)
            self.regressor = standard_regressor

    def build_shallow_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def build_deep_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        layers = [FullyConnectedBlock(fusion_dim, fusion_dim)]

        for _ in range(8):
            layers.append(FullyConnectedBlock(fusion_dim, fusion_dim))

        layers.append(FullyConnectedBlock(fusion_dim, hidden_dim))
        return nn.Sequential(*layers)

    def build_residual_head(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            ResidualBlock(fusion_dim),
            ResidualBlock(fusion_dim),
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def build_residual_head_v2(self, fusion_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            FullyConnectedBlock(fusion_dim, fusion_dim),
            *[ResidualLinearBlock(fusion_dim, fusion_dim, expansion=2) for _ in range(6)],
            FullyConnectedBlock(fusion_dim, hidden_dim),
        )

    def forward(
        self,
        meta_x: torch.Tensor,
        tag_x: torch.Tensor,
        return_embeddings: bool = False,
        ablation: AblationType = AblationType.ALL_FEATURES,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Input ablation
        if ablation == AblationType.META_ONLY:
            tag_x = torch.zeros_like(tag_x)
        elif ablation == AblationType.TAG_ONLY:
            meta_x = torch.zeros_like(meta_x)

        # Encode metadata
        if self.head_type == HeadType.RESIDUAL_V3:

            num_x = meta_x[:, :self.num_meta]
            cat_x = meta_x[:, self.num_meta:]

            if self.num_meta > 0:
                num_emb = self.plqp(num_x)
                num_out = self.num_proj(num_emb)
            else:
                num_out = torch.zeros(
                    meta_x.shape[0],
                    self.hidden_dim // 2,
                    device=meta_x.device
                )

            if self.cat_meta > 0:
                cat_out = self.cat_proj(cat_x)
            else:
                cat_out = torch.zeros(
                    meta_x.shape[0],
                    self.hidden_dim // 2,
                    device=meta_x.device
                )

            meta_out = torch.cat((num_out, cat_out), dim=1)
        
        elif self.head_type == HeadType.TWO_TOWER:
            meta_out = self.two_tower_meta_encoder(meta_x)
        else:
            meta_out = self.default_meta_encoder(meta_x)

        # Encode semantic review tags
        tag_out = self.tag_encoder(tag_x)

        # Fuse streams
        if self.head_type == HeadType.TWO_TOWER:
            deep_tags = self.head(tag_out)

            meta_normed = self.meta_norm(meta_out)
            tags_normed = self.tag_norm(deep_tags)

            fused = torch.cat((meta_normed, tags_normed), dim=1)

            raw_prediction = self.regressor(fused)
            embeddings = fused
        else:
            fused = torch.cat((meta_out, tag_out), dim=1)
            embeddings = self.head(fused)
            raw_prediction = self.regressor(embeddings)

        # Bound outputs to [1, 5]
        prediction = 1.0 + 4.0 * torch.sigmoid(raw_prediction)

        if return_embeddings:
            return prediction, embeddings
        return prediction