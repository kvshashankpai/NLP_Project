"""
transformer.py
--------------
Full Transformer model for sequence-to-sequence translation.

Architecture (Vaswani et al., "Attention Is All You Need"):
  - Encoder: N × EncoderLayer  (self-attention + FFN)
  - Decoder: N × DecoderLayer  (masked self-attention + cross-attention + FFN)
  - Final linear projection to target vocabulary size
"""

import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from utils import get_device


class Transformer(nn.Module):
    """
    End-to-end Transformer for English → Hindi translation.

    Args:
        d_model             : embedding / model dimension (512 in original paper)
        ffn_hidden          : FFN inner dimension (2048 in original paper)
        num_heads           : number of attention heads (8 in original paper)
        drop_prob           : dropout probability (0.1 in original paper)
        num_layers          : N encoder + N decoder layers (6 in original paper)
        max_sequence_length : maximum sequence length (character-level)
        hindi_vocab_size    : size of Hindi character vocabulary
        english_to_index    : English character → index mapping
        hindi_to_index      : Hindi character → index mapping
        START_TOKEN         : string used as start-of-sequence marker
        END_TOKEN           : string used as end-of-sequence marker
        PADDING_TOKEN       : string used for padding
    """

    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        hindi_vocab_size: int,
        english_vocab_size: int,
    ):
        super().__init__()
        self.device = get_device()

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
        self.encoder = Encoder(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            drop_prob=drop_prob,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            vocab_size=english_vocab_size,
        )

        # ------------------------------------------------------------------ #
        # Decoder
        # ------------------------------------------------------------------ #
        self.decoder = Decoder(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            num_heads=num_heads,
            drop_prob=drop_prob,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            vocab_size=hindi_vocab_size,
        )

        # ------------------------------------------------------------------ #
        # Output projection → Hindi vocabulary logits
        # ------------------------------------------------------------------ #
        self.linear = nn.Linear(d_model, hindi_vocab_size)

    def forward(
        self,
        x,                              # source integer tensor (B, S)
        y,                              # target integer tensor (B, S)
        encoder_self_attention_mask=None,
        decoder_self_attention_mask=None,
        decoder_cross_attention_mask=None,
    ):
        """
        Returns:
            out : (batch, seq_len, hindi_vocab_size)  — raw logits
        """
        # 1. Encode source
        enc_out = self.encoder(
            x,
            encoder_self_attention_mask,
        )

        # 2. Decode with cross-attention on encoder output
        dec_out = self.decoder(
            enc_out,
            y,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
        )

        # 3. Project to vocabulary
        out = self.linear(dec_out)  # (B, S, hindi_vocab_size)
        return out
