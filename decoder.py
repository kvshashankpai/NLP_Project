"""
decoder.py
----------
Transformer Decoder: N stacked DecoderLayers.

Each DecoderLayer contains:
  1. Masked Multi-Head Self-Attention  (causal mask)
  2. Add & Norm
  3. Multi-Head Cross-Attention        (encoder output as K, V)
  4. Add & Norm
  5. Position-wise Feed-Forward Network
  6. Add & Norm
"""

import torch.nn as nn
from utils import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
    LayerNormalization,
    PositionwiseFeedForward,
    SentenceEmbedding,
)


class DecoderLayer(nn.Module):
    """
    Single decoder block:
        y → MaskedMHA(y) → Dropout → AddNorm
          → CrossMHA(encoder_out, y) → Dropout → AddNorm
          → FFN → Dropout → AddNorm
    """

    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()

        # --- Sub-layer 1: Masked Self-Attention ---
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        # --- Sub-layer 2: Cross-Attention (encoder-decoder) ---
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        # --- Sub-layer 3: Position-wise FFN ---
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        """
        x : encoder output  (batch, src_len, d_model)
        y : decoder input   (batch, tgt_len, d_model)
        self_attention_mask  : causal + padding mask for decoder self-attention
        cross_attention_mask : padding mask for cross-attention
        """
        # 1. Masked Self-Attention
        residual = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + residual)

        # 2. Cross-Attention
        residual = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + residual)

        # 3. Feed-Forward
        residual = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + residual)

        return y


class SequentialDecoder(nn.Sequential):
    """Chains N DecoderLayer modules while threading all mask arguments."""

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


class Decoder(nn.Module):
    """
    Full decoder: SentenceEmbedding + N × DecoderLayer.

    Args:
        d_model             : model dimension
        ffn_hidden          : FFN hidden size
        num_heads           : number of attention heads
        drop_prob           : dropout probability
        num_layers          : number of stacked DecoderLayer blocks
        max_sequence_length : maximum token length
        language_to_index   : character→index mapping for the target language
        START_TOKEN / END_TOKEN / PADDING_TOKEN : special tokens
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
        self.layers = SequentialDecoder(
            *[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
              for _ in range(num_layers)]
        )

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        """
        x : encoder output        (batch, src_len, d_model)
        y : integer tensor        (batch, tgt_len)
        """
        y = self.sentence_embedding(y)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y
