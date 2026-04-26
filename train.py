"""
train.py
--------
Training script for the English → Hindi Transformer.

Usage:
    python train.py [--epochs N] [--batch_size B] [--d_model D]
                    [--num_layers L] [--num_heads H] [--max_seq_len S]
                    [--lr LR] [--max_samples M] [--checkpoint_dir DIR]

The terminal shows a live progress bar for every batch and a BLEU score
evaluation at the end of every epoch.
"""

import os
import sys
import math
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    TranslationDataset, load_data, build_masks,
    START_TOKEN, END_TOKEN, PADDING_TOKEN,
)
from transformer import Transformer
from metrics import corpus_bleu
from utils import get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total
    return f"[{bar}] {pct:5.1f}%"


def tokenize_batch_indices(sentences, tokenizer, max_seq_len,
                           start_token=True, end_token=True):
    """
    Convert a batch of sentences to padded integer tensors using the BPE tokenizer.
    """
    pad_idx = tokenizer.token_to_id(PADDING_TOKEN)
    start_idx = tokenizer.token_to_id(START_TOKEN)
    end_idx = tokenizer.token_to_id(END_TOKEN)

    batch = []
    for sent in sentences:
        encoded = tokenizer.encode(sent).ids
        indices = []
        if start_token:
            indices.append(start_idx)
        indices.extend(encoded)
        if end_token:
            indices.append(end_idx)
        indices = indices[:max_seq_len]
        while len(indices) < max_seq_len:
            indices.append(pad_idx)
        batch.append(torch.tensor(indices, dtype=torch.long))
    return torch.stack(batch)


# ---------------------------------------------------------------------------
# Greedy decode for evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_translate(model, src_sentence, en_tokenizer, hi_tokenizer, max_seq_len, device):
    """
    Autoregressively generate a Hindi translation for a single English sentence.
    Returns the translated string (without special tokens).
    """
    model.eval()
    end_idx   = hi_tokenizer.token_to_id(END_TOKEN)
    pad_idx   = hi_tokenizer.token_to_id(PADDING_TOKEN)

    # Encode source (batch of 1)
    src_indices = tokenize_batch_indices(
        [src_sentence], en_tokenizer, max_seq_len, start_token=False, end_token=False
    ).to(device)

    enc_mask = (src_indices == en_tokenizer.token_to_id(PADDING_TOKEN)).unsqueeze(1).unsqueeze(2)
    enc_mask = enc_mask.float().masked_fill(enc_mask, float('-inf')).permute(1, 0, 2, 3).to(device)

    # Encode once
    enc_out = model.encoder(src_indices, enc_mask)

    # Decode token-by-token
    tgt_indices_list = [hi_tokenizer.token_to_id(START_TOKEN)]
    for _ in range(max_seq_len):
        tgt_tensor = torch.tensor(tgt_indices_list, dtype=torch.long).unsqueeze(0).to(device) # (1, S)
        
        # Pad to max_seq_len for the mask
        pad_len = max_seq_len - tgt_tensor.size(1)
        padded_tgt = torch.cat([tgt_tensor, torch.full((1, pad_len), pad_idx, dtype=torch.long, device=device)], dim=1)

        dec_pad  = (padded_tgt == pad_idx).unsqueeze(1).unsqueeze(2)
        dec_pad  = dec_pad.float().masked_fill(dec_pad, float('-inf')).permute(1, 0, 2, 3).to(device)
        causal   = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        causal   = causal.float().masked_fill(causal, float('-inf')).unsqueeze(0).unsqueeze(0).to(device)
        dec_mask = torch.clamp(dec_pad + causal, min=float('-inf'), max=0)

        # Truncate dec_mask to current sequence length to avoid passing full max_seq_len to model
        dec_mask_trunc = dec_mask[:, :, :tgt_tensor.size(1), :tgt_tensor.size(1)]

        dec_out = model.decoder(enc_out, tgt_tensor, dec_mask_trunc, enc_mask)
        logits  = model.linear(dec_out)  # (1, S, V)

        # Predict next token (last position)
        next_idx = logits[0, -1, :].argmax(dim=-1).item()

        if next_idx == end_idx:
            break

        tgt_indices_list.append(next_idx)

    # Decode indices to string (ignoring START_TOKEN)
    return hi_tokenizer.decode(tgt_indices_list[1:])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_en, val_hi, en_tokenizer, hi_tokenizer,
             max_seq_len, device, num_samples=200):
    """
    Run greedy decoding on `num_samples` validation pairs and return BLEU.
    """
    model.eval()
    hypotheses, references = [], []
    n = min(num_samples, len(val_en))
    for i in range(n):
        hyp = greedy_translate(model, val_en[i], en_tokenizer, hi_tokenizer, max_seq_len, device)
        hypotheses.append(hyp)
        references.append(val_hi[i])

    bleu = corpus_bleu(hypotheses, references)
    # Print a few examples
    print("\n  Sample translations:")
    for i in range(min(3, n)):
        print(f"    EN : {val_en[i][:60]}")
        print(f"    REF: {val_hi[i][:60]}")
        print(f"    HYP: {hypotheses[i][:60]}")
        print()
    return bleu


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = get_device()
    print(f"\n{'='*65}")
    print(f"  English → Hindi Transformer  |  device: {device}")
    print(f"{'='*65}\n")

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    train_en, train_hi, val_en, val_hi, en_tokenizer, hi_tokenizer = load_data(
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )

    train_dataset = TranslationDataset(train_en, train_hi)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # ------------------------------------------------------------------ #
    # 2. Model
    # ------------------------------------------------------------------ #
    model = Transformer(
        d_model=args.d_model,
        ffn_hidden=args.ffn_hidden,
        num_heads=args.num_heads,
        drop_prob=args.drop_prob,
        num_layers=args.num_layers,
        max_sequence_length=args.max_seq_len,
        hindi_vocab_size=hi_tokenizer.get_vocab_size(),
        english_vocab_size=en_tokenizer.get_vocab_size(),
    ).to(device)

    print(f"  Model parameters : {count_parameters(model):,}")
    print(f"  d_model          : {args.d_model}")
    print(f"  num_layers       : {args.num_layers}")
    print(f"  num_heads        : {args.num_heads}")
    print(f"  ffn_hidden       : {args.ffn_hidden}")
    print(f"  max_seq_len      : {args.max_seq_len}")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  epochs           : {args.epochs}")
    print(f"  learning rate    : {args.lr}")
    print()

    # ------------------------------------------------------------------ #
    # 3. Optimiser & Loss
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.98), eps=1e-9)

    # Label-smoothed cross entropy
    criterion = nn.CrossEntropyLoss(
        ignore_index=hi_tokenizer.token_to_id(PADDING_TOKEN),
        label_smoothing=0.1,
    )

    pad_en = en_tokenizer.token_to_id(PADDING_TOKEN)
    pad_hi = hi_tokenizer.token_to_id(PADDING_TOKEN)

    # ------------------------------------------------------------------ #
    # 4. Save artefacts (tokenizers) for inference
    # ------------------------------------------------------------------ #
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    en_tokenizer.save(os.path.join(args.checkpoint_dir, "en_tokenizer.json"))
    hi_tokenizer.save(os.path.join(args.checkpoint_dir, "hi_tokenizer.json"))
    with open(os.path.join(args.checkpoint_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Tokenizers saved to {args.checkpoint_dir}/\n")

    # ------------------------------------------------------------------ #
    # 5. Training loop
    # ------------------------------------------------------------------ #
    best_bleu = 0.0
    history   = []
    start_epoch = 1

    if getattr(args, "resume", None):
        print(f"  Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint["epoch"] + 1

        hist_path = os.path.join(args.checkpoint_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r") as f:
                history = json.load(f)
            if history:
                best_bleu = max(h.get("bleu", 0.0) for h in history)

        print(f"  Resuming from epoch {start_epoch}, previous best BLEU: {best_bleu:.4f}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_start  = time.time()
        total_batches = len(train_loader)

        print(f"{'─'*65}")
        print(f"  Epoch {epoch}/{args.epochs}  —  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*65}")

        for step, (en_batch, hi_batch) in enumerate(train_loader, start=1):
            batch_start = time.time()

            # ---- Build index tensors for mask generation ----
            en_indices = tokenize_batch_indices(
                list(en_batch), en_tokenizer, args.max_seq_len,
                start_token=False, end_token=False,
            ).to(device)
            hi_indices = tokenize_batch_indices(
                list(hi_batch), hi_tokenizer, args.max_seq_len,
                start_token=True, end_token=False,
            ).to(device)

            enc_mask, dec_mask, cross_mask = build_masks(
                en_indices, hi_indices, args.max_seq_len,
                pad_idx=pad_en,
            )

            # ---- Forward pass ----
            optimizer.zero_grad()
            logits = model(
                en_indices, hi_indices,
                encoder_self_attention_mask=enc_mask,
                decoder_self_attention_mask=dec_mask,
                decoder_cross_attention_mask=cross_mask,
            )  # (B, S, V)

            # ---- Build target (shifted right: model predicts next char) ----
            tgt_indices = tokenize_batch_indices(
                list(hi_batch), hi_tokenizer, args.max_seq_len,
                start_token=False, end_token=True,
            ).to(device)  # (B, S)

            # logits: (B, S, V) → (B*S, V)
            loss = criterion(
                logits.reshape(-1, hi_tokenizer.get_vocab_size()),
                tgt_indices.reshape(-1),
            )

            # ---- Backward + clip ----
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss  = loss.item()
            epoch_loss += batch_loss
            elapsed     = time.time() - batch_start
            eta_batch   = elapsed * (total_batches - step)

            # ---- Live terminal progress ----
            bar = progress_bar(step, total_batches)
            print(
                f"\r  Step {step:5d}/{total_batches}  {bar}  "
                f"loss={batch_loss:.4f}  "
                f"ppl={math.exp(min(batch_loss, 20)):.2f}  "
                f"ETA {format_time(eta_batch)}",
                end="",
                flush=True,
            )

            # Mini-log every 200 steps
            if step % 200 == 0:
                avg_so_far = epoch_loss / step
                print(
                    f"\n  [Step {step:5d}]  avg_loss={avg_so_far:.4f}  "
                    f"ppl={math.exp(min(avg_so_far, 20)):.2f}  "
                    f"elapsed={format_time(time.time() - epoch_start)}"
                )

        avg_loss = epoch_loss / total_batches
        epoch_elapsed = time.time() - epoch_start
        print(f"\n")
        print(f"   Epoch {epoch} complete")
        print(f"     avg_loss : {avg_loss:.4f}")
        print(f"     ppl      : {math.exp(min(avg_loss, 20)):.2f}")
        print(f"     time     : {format_time(epoch_elapsed)}")

        # ---- BLEU evaluation ----
        print(f"\n   Computing BLEU on {min(200, len(val_en))} validation samples …")
        bleu = evaluate(model, val_en, val_hi, en_tokenizer, hi_tokenizer,
                        args.max_seq_len, device, num_samples=200)
        print(f"     BLEU : {bleu:.4f}  ({bleu*100:.2f})")

        history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu})

        # ---- Checkpoint ----
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:02d}.pt")
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "loss":        avg_loss,
            "bleu":        bleu,
            "args":        vars(args),
        }, ckpt_path)
        print(f"      Checkpoint → {ckpt_path}")

        if bleu > best_bleu:
            best_bleu = bleu
            best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "bleu":        bleu,
                "args":        vars(args),
            }, best_path)
            print(f"      New best BLEU={best_bleu:.4f}  → {best_path}")

        print()

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    print(f"{'='*65}")
    print(f"  Training complete!")
    print(f"  Best BLEU : {best_bleu:.4f} ({best_bleu*100:.2f})")
    print(f"{'='*65}\n")

    # Save training history
    hist_path = os.path.join(args.checkpoint_dir, "history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {hist_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train English→Hindi Transformer from scratch")
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--resume",          type=str,   default=None, help="Path to checkpoint to resume from")
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--d_model",         type=int,   default=512)
    p.add_argument("--ffn_hidden",      type=int,   default=2048)
    p.add_argument("--num_heads",       type=int,   default=8)
    p.add_argument("--num_layers",      type=int,   default=6)
    p.add_argument("--drop_prob",       type=float, default=0.1)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--max_seq_len",     type=int,   default=200)
    p.add_argument("--max_samples",     type=int,   default=100_000)
    p.add_argument("--checkpoint_dir",  type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)