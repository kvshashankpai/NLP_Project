"""
ft_translate.py  (finetune_updated version)
--------------------------------------------
Interactive English -> Hindi translation using the FINE-TUNED checkpoint
produced by finetune_updated/finetune.py.

Checkpoints are loaded from  finetune_updated/ft_checkpoints/  by default,
completely separate from the old  finetune/ft_checkpoints/  folder.

Usage:
    python finetune_updated/ft_translate.py
    python finetune_updated/ft_translate.py --sentence "How are you?"
    python finetune_updated/ft_translate.py --checkpoint finetune_updated/ft_checkpoints/ft_epoch_05.pt
    python finetune_updated/ft_translate.py --beam 4        # beam-search decode
"""

import os, sys, json, argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformer import Transformer
from dataset import START_TOKEN, END_TOKEN, PADDING_TOKEN
from utils import get_device


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, ckpt_dir):
    from tokenizers import Tokenizer
    device = get_device()

    en_tok = Tokenizer.from_file(os.path.join(ckpt_dir, "en_tokenizer.json"))
    hi_tok = Tokenizer.from_file(os.path.join(ckpt_dir, "hi_tokenizer.json"))
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    model = Transformer(
        d_model=cfg["d_model"], ffn_hidden=cfg["ffn_hidden"],
        num_heads=cfg["num_heads"], drop_prob=0.0, num_layers=cfg["num_layers"],
        max_sequence_length=cfg["max_seq_len"],
        hindi_vocab_size=hi_tok.get_vocab_size(),
        english_vocab_size=en_tok.get_vocab_size(),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset_tag = cfg.get("ft_dataset", "unknown")
    print(f"\n  Checkpoint  : {ckpt_path}")
    print(f"  Epoch       : {ckpt.get('epoch','?')}")
    print(f"  BLEU        : {ckpt.get('bleu', 0):.4f}")
    print(f"  FT dataset  : {dataset_tag}\n")
    return model, en_tok, hi_tok, cfg["max_seq_len"], device


# ══════════════════════════════════════════════════════════════════════════════
#  Greedy decode
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_translate(model, src, en_tok, hi_tok, max_len, device):
    pad_en  = en_tok.token_to_id(PADDING_TOKEN)
    end_id  = hi_tok.token_to_id(END_TOKEN)
    hi_pad  = hi_tok.token_to_id(PADDING_TOKEN)

    ids = en_tok.encode(src).ids[:max_len]
    ids += [pad_en] * (max_len - len(ids))
    src_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    enc_m = (src_t == pad_en).unsqueeze(1).unsqueeze(2)
    enc_m = enc_m.float().masked_fill(enc_m, float('-inf')).permute(1,0,2,3).to(device)
    enc_out = model.encoder(src_t, enc_m)

    tgt = [hi_tok.token_to_id(START_TOKEN)]
    for _ in range(max_len):
        tt = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).to(device)
        pl = max_len - tt.size(1)
        pt = torch.cat([tt, torch.full((1,pl), hi_pad, dtype=torch.long, device=device)], dim=1)
        dp = (pt == hi_pad).unsqueeze(1).unsqueeze(2)
        dp = dp.float().masked_fill(dp, float('-inf')).permute(1,0,2,3).to(device)
        ca = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        ca = ca.float().masked_fill(ca, float('-inf')).unsqueeze(0).unsqueeze(0).to(device)
        dm = torch.clamp(dp + ca, min=float('-inf'), max=0)
        dm = dm[:, :, :tt.size(1), :tt.size(1)]
        dec_out = model.decoder(enc_out, tt, dm, enc_m)
        logits  = model.linear(dec_out)
        ni = logits[0, -1, :].argmax(dim=-1).item()
        if ni == end_id: break
        tgt.append(ni)
    return hi_tok.decode(tgt[1:])


# ══════════════════════════════════════════════════════════════════════════════
#  Beam-search decode  (bonus — not in original script)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def beam_translate(model, src, en_tok, hi_tok, max_len, device, beam_size=4):
    """Simple beam search — returns the highest-scoring hypothesis."""
    import heapq, math

    pad_en  = en_tok.token_to_id(PADDING_TOKEN)
    end_id  = hi_tok.token_to_id(END_TOKEN)
    hi_pad  = hi_tok.token_to_id(PADDING_TOKEN)
    start_id = hi_tok.token_to_id(START_TOKEN)

    ids = en_tok.encode(src).ids[:max_len]
    ids += [pad_en] * (max_len - len(ids))
    src_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    enc_m = (src_t == pad_en).unsqueeze(1).unsqueeze(2)
    enc_m = enc_m.float().masked_fill(enc_m, float('-inf')).permute(1,0,2,3).to(device)
    enc_out = model.encoder(src_t, enc_m)

    # Each beam: (neg_log_prob, token_list)
    beams = [(0.0, [start_id])]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == end_id:
                completed.append((score, seq)); continue
            tt = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            pl = max_len - tt.size(1)
            pt = torch.cat([tt, torch.full((1,pl), hi_pad, dtype=torch.long, device=device)], dim=1)
            dp = (pt == hi_pad).unsqueeze(1).unsqueeze(2)
            dp = dp.float().masked_fill(dp, float('-inf')).permute(1,0,2,3).to(device)
            ca = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
            ca = ca.float().masked_fill(ca, float('-inf')).unsqueeze(0).unsqueeze(0).to(device)
            dm = torch.clamp(dp + ca, min=float('-inf'), max=0)
            dm = dm[:, :, :tt.size(1), :tt.size(1)]
            dec_out = model.decoder(enc_out, tt, dm, enc_m)
            logits  = model.linear(dec_out)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            topk = log_probs.topk(beam_size)
            for lp, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_beams.append((score - lp, seq + [tok]))

        if not new_beams: break
        beams = sorted(new_beams)[:beam_size]
        if all(s[-1] == end_id for _, s in beams):
            completed.extend(beams); break

    completed = completed or beams
    best_seq = min(completed, key=lambda x: x[0])[1]
    # strip START and END tokens
    out = [t for t in best_seq[1:] if t != end_id]
    return hi_tok.decode(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive REPL
# ══════════════════════════════════════════════════════════════════════════════

def repl(model, en_tok, hi_tok, max_len, device, beam_size=1):
    decode_mode = f"beam={beam_size}" if beam_size > 1 else "greedy"
    print("=" * 60)
    print(f"  English -> Hindi (Fine-Tuned · {decode_mode})")
    print("  Type 'quit' to exit")
    print("=" * 60)
    while True:
        try:
            src = input("\n  English : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not src: continue
        if src.lower() in ("quit", "exit", "q"):
            print("Bye!"); break
        if beam_size > 1:
            result = beam_translate(model, src, en_tok, hi_tok, max_len, device, beam_size)
        else:
            result = greedy_translate(model, src, en_tok, hi_tok, max_len, device)
        print(f"  Hindi   : {result if result else '[empty]'}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    # Default to THIS script's sibling folder ft_checkpoints
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    ft_dir   = os.path.join(THIS_DIR, "ft_checkpoints")

    p = argparse.ArgumentParser(description="Translate with fine-tuned model (updated)")
    p.add_argument("--checkpoint",     default=os.path.join(ft_dir, "ft_best_model.pt"),
                   help="Path to fine-tuned .pt checkpoint")
    p.add_argument("--checkpoint_dir", default=ft_dir,
                   help="Dir containing config.json + tokenizers")
    p.add_argument("--sentence",       default=None,
                   help="Single sentence (skips interactive mode)")
    p.add_argument("--beam",           type=int, default=1,
                   help="Beam size (1 = greedy, ≥2 = beam search)")
    return p.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    model, en_tok, hi_tok, max_len, device = load_model(args.checkpoint, args.checkpoint_dir)

    if args.sentence:
        if args.beam > 1:
            result = beam_translate(model, args.sentence, en_tok, hi_tok,
                                    max_len, device, args.beam)
        else:
            result = greedy_translate(model, args.sentence, en_tok, hi_tok, max_len, device)
        print(f"EN : {args.sentence}")
        print(f"HI : {result}")
    else:
        repl(model, en_tok, hi_tok, max_len, device, beam_size=args.beam)