# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.project_kv = nn.Linear(1, embed_dim)  # Project 1D features into embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, query, kv):
        """
        query: shape (B, 2048, 768)
        kv: shape (B, 2048)
        """
        B, seq_len, d_q = query.shape

        # Project kv from (B, 2048) -> (B, 2048, 768)
        kv = kv.unsqueeze(-1) # (B, 2048, 1)
        kv_proj = self.project_kv(kv)  # (B, 2048, 768)

        # Normalize inputs
        query = self.norm_q(query)
        kv_proj = self.norm_kv(kv_proj)

        # Perform multihead attention
        # MHA expects (B, seq, dim) if batch_first=True
        out, attn_weights = self.attn(query, kv_proj, kv_proj)  # Q, K, V all shape (B, 2048, 768)

        # Residual + dropout + norm (Post-LN)
        out = self.dropout(out)
        out = self.norm_out(out + query)
        
        return out, attn_weights

class EfficientCrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(2048, embed_dim)
        self.v_proj = nn.Linear(2048, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

    def reshape_heads(self, x):
        # (B, L, D) -> (B, num_heads, L, head_dim)
        B, L, D = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, num_heads, L, head_dim)

    def forward(self, query, kv):
        """
        query: shape (B, 2048, 768)
        kv: shape (B, 2048)
        """
        B, L, _ = query.shape

        query = self.norm_q(query)
        kv = self.k_proj(kv)  # (B, 2048, 768)
        kv = self.norm_kv(kv)

        # Project Q, K, V
        Q = self.q_proj(query)     # (B, 2048, 768)
        K = kv                     # K already projected and normalized
        V = self.v_proj(kv)        # (B, 2048, 768)

        # Reshape for multi-head
        Q = self.reshape_heads(Q)  # (B, num_heads, 2048, head_dim)
        K = self.reshape_heads(K)
        V = self.reshape_heads(V)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.embed_dim)

        # Final linear projection
        out = self.out_proj(attn_output)  # (B, 2048, 768)

        return out 

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
        use_contour: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.use_contour = use_contour
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        if self.use_contour:
            self.patch_embedding_contour = PatchEmbeddingBlock(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                pos_embed=pos_embed,
                dropout_rate=dropout_rate,
                spatial_dims=spatial_dims,
            )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # if post_activation == "Tanh":
            #     self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            # else:
            #     self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x, contour):
        if self.use_contour:
            print("Using contour")
            x = self.patch_embedding(x) + self.patch_embedding_contour(contour)
        else:
            x = self.patch_embedding(x)
        # print("Contour included")
        # print("x shape", x.shape)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            # print("x shape again", x.shape)
            hidden_states_out.append(x)
        x = self.norm(x)
        # if hasattr(self, "classification_head"):
        #     x = self.classification_head(x[:, 0])
        return x, hidden_states_out


class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.use_contour = config.use_contour

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
            use_contour = self.use_contour,
        )

    def forward(self, images, contours):
        last_feature, hidden_states = self.vision_tower(images, contours)
        # print("Last_feature shape:", last_feature.shape)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size
    
class ViTMerlin3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature
        self.use_contour = config.use_contour

        self.vision_tower = ViT(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
            use_contour=self.use_contour
        )

        self.merlin = Merlin()

        self.cross_attention = CrossAttentionBlock(embed_dim=768, num_heads=8)

    def forward(self, images, contours):
        # Separately encode images
        last_feature, hidden_states = self.vision_tower(images, contours) # (bsz, 2048, 768)
        merlin_features = self.merlin(images) # (bsz, 2048)

        # Determine the layer of the vision transformer to extract features from (Have fusion here later as well)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        # Determine if we're including the classification patch or not
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        # Logic to combine the image_features and merlin (or future) features
        image_features = self.cross_attention(image_features, merlin_features)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size