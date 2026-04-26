"""
dataset.py
----------
Dataset utilities:
  - Download / cache the OPUS English-Hindi dataset via HuggingFace
  - Build character-level vocabularies for English and Hindi
  - Create attention masks (padding + causal)
  - TranslationDataset (torch.utils.data.Dataset)
"""

import torch
from torch.utils.data import Dataset
from utils import get_device


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
START_TOKEN   = "<START>"
END_TOKEN     = "<END>"
PADDING_TOKEN = "<PAD>"


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(sentences, vocab_size=10000):
    """
    Train a BPE tokenizer using HuggingFace tokenizers.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PADDING_TOKEN, START_TOKEN, END_TOKEN, "<UNK>"]
    )
    tokenizer.train_from_iterator(sentences, trainer)
    return tokenizer


# ---------------------------------------------------------------------------
# Attention mask generators
# ---------------------------------------------------------------------------

def create_padding_mask(token_batch, pad_idx: int = 0):
    """
    Returns an additive mask (0 for real tokens, -inf for padding).
    Shape: (heads=1, batch, 1, seq_len)  → broadcasts correctly.
    """
    # token_batch: (B, S)  integer tensor
    mask = (token_batch == pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
    mask = mask.float().masked_fill(mask, float('-inf'))
    return mask.permute(1, 0, 2, 3)  # (1, B, 1, S)


def create_causal_mask(seq_len: int):
    """
    Upper-triangular causal mask so each position can only attend
    to positions ≤ itself. Returns additive mask (1, 1, S, S).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.float().masked_fill(mask, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, S, S)


def build_masks(en_indices, hi_indices, max_seq_len, pad_idx=0):
    """
    Build all three masks needed for a training step.

    Returns:
        enc_mask   : encoder self-attention padding mask
        dec_mask   : decoder self-attention causal + padding mask
        cross_mask : decoder cross-attention padding mask (based on source)
    """
    device = get_device()

    # Encoder self-attention: just padding
    enc_mask = create_padding_mask(en_indices, pad_idx).to(device)

    # Decoder self-attention: causal + padding
    dec_pad  = create_padding_mask(hi_indices, pad_idx).to(device)
    dec_caus = create_causal_mask(max_seq_len).to(device)
    dec_mask = torch.clamp(dec_pad + dec_caus, min=float('-inf'), max=0)

    # Cross attention: padding over source sequence
    cross_mask = create_padding_mask(en_indices, pad_idx).to(device)

    return enc_mask, dec_mask, cross_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """
    Wraps parallel English–Hindi sentence pairs.

    Args:
        en_sentences : list of English strings
        hi_sentences : list of Hindi strings
    """

    def __init__(self, en_sentences, hi_sentences):
        assert len(en_sentences) == len(hi_sentences)
        self.en = en_sentences
        self.hi = hi_sentences

    def __len__(self):
        return len(self.en)

    def __getitem__(self, idx):
        return self.en[idx], self.hi[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(max_samples: int = 100_000, max_seq_len: int = 200, vocab_size: int = 10000):
    """
    Load English-Hindi pairs from HuggingFace datasets.
    Filters out pairs where either sentence is too long.

    Returns:
        train_en, train_hi : lists of strings
        val_en,   val_hi   : lists of strings
        en_tokenizer, hi_tokenizer : trained Tokenizer objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print(" Loading cfilt/iitb-english-hindi dataset …")
    ds = load_dataset("cfilt/iitb-english-hindi", split="train", trust_remote_code=True)
    
    # Shuffle the dataset so we get a mix of technical strings, news, and conversational text
    ds = ds.shuffle(seed=42)

    en_sentences, hi_sentences = [], []
    for item in ds:
        pair = item["translation"]
        en, hi = pair.get("en", ""), pair.get("hi", "")
        if not en or not hi:
            continue
        # +2 for START / END tokens
        if len(en) + 2 > max_seq_len or len(hi) + 2 > max_seq_len:
            continue
        en_sentences.append(en.strip())
        hi_sentences.append(hi.strip())
        if len(en_sentences) >= max_samples:
            break

    print(f" Loaded {len(en_sentences):,} sentence pairs.")

    # Train / validation split (90 / 10)
    split = int(0.9 * len(en_sentences))
    train_en, val_en = en_sentences[:split], en_sentences[split:]
    train_hi, val_hi = hi_sentences[:split], hi_sentences[split:]

    print("   Training English BPE Tokenizer …")
    en_tokenizer = train_tokenizer(train_en, vocab_size=vocab_size)
    print("   Training Hindi BPE Tokenizer …")
    hi_tokenizer = train_tokenizer(train_hi, vocab_size=vocab_size)

    print(f"   English vocab size : {en_tokenizer.get_vocab_size():,} subwords")
    print(f"   Hindi   vocab size : {hi_tokenizer.get_vocab_size():,} subwords")
    print(f"   Train pairs        : {len(train_en):,}")
    print(f"   Val   pairs        : {len(val_en):,}")

    return train_en, train_hi, val_en, val_hi, en_tokenizer, hi_tokenizer
