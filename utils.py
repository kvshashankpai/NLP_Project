"""
utils.py
--------
Core building blocks for the Transformer model.
Implements from scratch (no nn.Transformer):
  - scaled_dot_product attention
  - MultiHeadAttention
  - LayerNormalization
  - PositionalEncoding
  - PositionwiseFeedForward
  - SentenceEmbedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def scaled_dot_product(q, k, v, mask=None):
    """
    Args:
        q, k, v : (batch, heads, seq_len, head_dim)
        mask    : additive mask broadcastable to (heads, batch, seq_len, seq_len)
    Returns:
        values  : (batch, heads, seq_len, head_dim)
        attention : (batch, heads, seq_len, seq_len)
    """
    d_k = q.size(-1)
    # (batch, heads, seq, seq)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    if mask is not None:
        # mask shape: (heads, batch, seq, seq) → permute to (batch, heads, seq, seq)
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    """Self-attention with num_heads heads. Used in the Encoder."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Project input to Q, K, V in one shot
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, d_model)
        mask : additive mask (heads, batch, seq, seq)  [optional]
        """
        batch_size, seq_len, d_model = x.size()

        qkv = self.qkv_layer(x)                                             # (B, S, 3*D)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)  # (B, S, H, 3*hd)
        qkv = qkv.permute(0, 2, 1, 3)                                       # (B, H, S, 3*hd)
        q, k, v = qkv.chunk(3, dim=-1)                                      # each (B, H, S, hd)

        values, _ = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        out = self.linear_layer(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention used in the Decoder:
      - keys & values come from encoder output  (x)
      - queries come from decoder               (y)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_layer = nn.Linear(d_model, 2 * d_model)   # for encoder output
        self.q_layer = nn.Linear(d_model, d_model)         # for decoder query
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        """
        x : encoder output (batch, seq_len, d_model)
        y : decoder query  (batch, seq_len, d_model)
        """
        batch_size, src_len, d_model = x.size()
        tgt_len = y.size(1)

        kv = self.kv_layer(x)                                               # (B, src_len, 2*D)
        q  = self.q_layer(y)                                                # (B, tgt_len, D)

        kv = kv.reshape(batch_size, src_len, self.num_heads, 2 * self.head_dim).permute(0, 2, 1, 3)
        q  = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k, v = kv.chunk(2, dim=-1)
        values, _ = scaled_dot_product(q, k, v, mask)

        values = values.permute(0, 2, 1, 3).reshape(batch_size, tgt_len, d_model)
        out = self.linear_layer(values)
        return out


# ---------------------------------------------------------------------------
# Layer Normalisation  (manual — no nn.LayerNorm)
# ---------------------------------------------------------------------------

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps: float = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        return self.gamma * y + self.beta


# ---------------------------------------------------------------------------
# Positional Encoding  (sinusoidal, as in "Attention Is All You Need")
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)

        even_PE = torch.sin(position / denominator)
        odd_PE  = torch.cos(position / denominator)

        stacked = torch.stack([even_PE, odd_PE], dim=2)         # (S, d/2, 2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)     # (S, d_model)
        return PE


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# ---------------------------------------------------------------------------
# Sentence Embedding (token + positional)
# ---------------------------------------------------------------------------

class SentenceEmbedding(nn.Module):
    """
    Converts a batch of integer tensors into
    dense embeddings with positional encoding.
    """

    def __init__(self, max_sequence_length, d_model, vocab_size):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        """
        x: (B, S) integer tensor
        """
        x = self.embedding(x)                                  # (B, S, D)
        pos = self.position_encoder().to(x.device)         # (max_S, D)
        pos = pos[:x.size(1), :]                           # (S, D)
        x = self.dropout(x + pos)
        return x
