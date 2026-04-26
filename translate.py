"""
translate.py
------------
Interactive English → Hindi translation using a trained checkpoint.

Usage:
    python translate.py --checkpoint checkpoints/best_model.pt

Then type English sentences at the prompt. Type 'quit' to exit.
"""

import os
import sys
import json
import argparse

import torch

from transformer import Transformer
from dataset import START_TOKEN, END_TOKEN, PADDING_TOKEN
from utils import get_device


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, checkpoint_dir: str):
    from tokenizers import Tokenizer
    device = get_device()

    # Load tokenizers
    en_tokenizer_path = os.path.join(checkpoint_dir, "en_tokenizer.json")
    hi_tokenizer_path = os.path.join(checkpoint_dir, "hi_tokenizer.json")
    config_path   = os.path.join(checkpoint_dir, "config.json")

    if not os.path.exists(en_tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {en_tokenizer_path}. Train first.")

    en_tokenizer = Tokenizer.from_file(en_tokenizer_path)
    hi_tokenizer = Tokenizer.from_file(hi_tokenizer_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    model = Transformer(
        d_model=config["d_model"],
        ffn_hidden=config["ffn_hidden"],
        num_heads=config["num_heads"],
        drop_prob=0.0,          # no dropout at inference
        num_layers=config["num_layers"],
        max_sequence_length=config["max_seq_len"],
        hindi_vocab_size=hi_tokenizer.get_vocab_size(),
        english_vocab_size=en_tokenizer.get_vocab_size(),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f" Loaded checkpoint: {checkpoint_path}")
    print(f"   Trained for epoch {ckpt.get('epoch','?')}  |  BLEU {ckpt.get('bleu', 0):.4f}\n")
    return model, en_tokenizer, hi_tokenizer, config["max_seq_len"], device


# ---------------------------------------------------------------------------
# Greedy decoder (token-by-token autoregressive)
# ---------------------------------------------------------------------------

@torch.no_grad()
def translate(model, src_sentence: str, en_tokenizer, hi_tokenizer,
              max_seq_len: int, device, beam_size: int = 1) -> str:
    """
    Greedy (beam_size=1) left-to-right decoding.
    Returns the Hindi translation string.
    """
    from dataset import PADDING_TOKEN, START_TOKEN, END_TOKEN

    end_idx  = hi_tokenizer.token_to_id(END_TOKEN)
    pad_idx  = en_tokenizer.token_to_id(PADDING_TOKEN)

    # Build source index tensor for mask
    def tokenize(sent, tokenizer, start=False, end=False):
        pad = tokenizer.token_to_id(PADDING_TOKEN)
        ids = []
        if start: ids.append(tokenizer.token_to_id(START_TOKEN))
        ids.extend(tokenizer.encode(sent).ids)
        if end: ids.append(tokenizer.token_to_id(END_TOKEN))
        ids = ids[:max_seq_len]
        while len(ids) < max_seq_len:
            ids.append(pad)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, S)

    src_ids = tokenize(src_sentence, en_tokenizer)
    enc_mask = (src_ids == pad_idx).unsqueeze(1).unsqueeze(2)
    enc_mask = enc_mask.float().masked_fill(enc_mask, float('-inf')).permute(1, 0, 2, 3).to(device)

    # Encode source once
    enc_out = model.encoder(src_ids, enc_mask)

    # Autoregressive decode
    tgt_indices = [hi_tokenizer.token_to_id(START_TOKEN)]
    hi_pad = hi_tokenizer.token_to_id(PADDING_TOKEN)
    
    for _ in range(max_seq_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Pad to max_seq_len for the mask
        pad_len = max_seq_len - tgt_tensor.size(1)
        padded_tgt = torch.cat([tgt_tensor, torch.full((1, pad_len), hi_pad, dtype=torch.long, device=device)], dim=1)

        dec_pad  = (padded_tgt == hi_pad).unsqueeze(1).unsqueeze(2)
        dec_pad  = dec_pad.float().masked_fill(dec_pad, float('-inf')).permute(1, 0, 2, 3).to(device)
        causal   = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        causal   = causal.float().masked_fill(causal, float('-inf')).unsqueeze(0).unsqueeze(0).to(device)
        dec_mask = torch.clamp(dec_pad + causal, min=float('-inf'), max=0)

        dec_mask_trunc = dec_mask[:, :, :tgt_tensor.size(1), :tgt_tensor.size(1)]

        dec_out = model.decoder(enc_out, tgt_tensor, dec_mask_trunc, enc_mask)
        logits  = model.linear(dec_out)   # (1, S, V)

        cur_pos  = min(len(tgt_indices), max_seq_len - 1)
        next_idx = logits[0, -1, :].argmax(dim=-1).item()

        if next_idx == end_idx:
            break

        tgt_indices.append(next_idx)

    return hi_tokenizer.decode(tgt_indices[1:])


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl(model, en_tokenizer, hi_tokenizer, max_seq_len, device):
    print("=" * 55)
    print("  English → Hindi Translator  (type 'quit' to exit)")
    print("=" * 55)

    while True:
        try:
            src = input("\n  English : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not src:
            continue
        if src.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = translate(model, src, en_tokenizer, hi_tokenizer, max_seq_len, device)
        print(f"  Hindi   : {result if result else '[empty output — try more training]'}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Translate English → Hindi")
    p.add_argument("--checkpoint",      type=str, default="checkpoints/best_model.pt")
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    p.add_argument("--sentence",        type=str, default=None,
                   help="Translate a single sentence (non-interactive mode)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, en_tokenizer, hi_tokenizer, max_seq_len, device = load_model(
        args.checkpoint, args.checkpoint_dir
    )

    if args.sentence:
        result = translate(model, args.sentence, en_tokenizer, hi_tokenizer, max_seq_len, device)
        print(f"EN : {args.sentence}")
        print(f"HI : {result}")
    else:
        repl(model, en_tokenizer, hi_tokenizer, max_seq_len, device)