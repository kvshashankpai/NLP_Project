"""
encoder.py
----------
Transformer Encoder: N stacked EncoderLayers.

Each EncoderLayer contains:
  1. Multi-Head Self-Attention
  2. Add & Norm
  3. Position-wise Feed-Forward Network
  4. Add & Norm
"""

import torch.nn as nn
from utils import (
    MultiHeadAttention,
    LayerNormalization,
    PositionwiseFeedForward,
    SentenceEmbedding,
)


class EncoderLayer(nn.Module):
    """
    Single encoder block:
        x → MHA(x, mask) → Dropout → AddNorm → FFN → Dropout → AddNorm
    """

    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        residual = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        # --- Sub-layer 2: Position-wise FFN ---
        residual = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x


class SequentialEncoder(nn.Sequential):
    """Chains N EncoderLayer modules while threading the mask argument."""

    def forward(self, x, self_attention_mask):
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    """
    Full encoder: SentenceEmbedding + N × EncoderLayer.

    Args:
        d_model             : model dimension (embedding size)
        ffn_hidden          : hidden size of the FFN sub-layer
        num_heads           : number of attention heads
        drop_prob           : dropout probability
        num_layers          : number of stacked EncoderLayer blocks (N)
        max_sequence_length : maximum token length
        language_to_index   : character→index mapping for the source language
        START_TOKEN         : special start token string
        END_TOKEN           : special end token string
        PADDING_TOKEN       : special padding token string
    """

    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        vocab_size: int,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, d_model, vocab_size
        )
        self.layers = SequentialEncoder(
            *[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
              for _ in range(num_layers)]
        )

    def forward(self, x, self_attention_mask):
        """
        x                  : integer tensor (B, S)
        self_attention_mask: additive padding mask
        """
        x = self.sentence_embedding(x)
        x = self.layers(x, self_attention_mask)
        return x
