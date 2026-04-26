"""
ft_translate.py
---------------
Interactive English -> Hindi translation using the FINE-TUNED checkpoint.

Usage:
    python finetune/ft_translate.py
    python finetune/ft_translate.py --sentence "How are you?"
    python finetune/ft_translate.py --checkpoint finetune/ft_checkpoints/ft_epoch_05.pt
"""

import os, sys, json, argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformer import Transformer
from dataset import START_TOKEN, END_TOKEN, PADDING_TOKEN
from utils import get_device


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

    print(f" Loaded fine-tuned checkpoint: {ckpt_path}")
    print(f"   Epoch {ckpt.get('epoch','?')}  |  BLEU {ckpt.get('bleu',0):.4f}\n")
    return model, en_tok, hi_tok, cfg["max_seq_len"], device


@torch.no_grad()
def translate(model, src, en_tok, hi_tok, max_len, device):
    pad = en_tok.token_to_id(PADDING_TOKEN)
    end_id = hi_tok.token_to_id(END_TOKEN)
    hi_pad = hi_tok.token_to_id(PADDING_TOKEN)

    # Tokenize source
    ids = en_tok.encode(src).ids[:max_len]
    ids += [pad] * (max_len - len(ids))
    src_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    enc_m = (src_t == pad).unsqueeze(1).unsqueeze(2)
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
        logits = model.linear(dec_out)
        ni = logits[0, -1, :].argmax(dim=-1).item()
        if ni == end_id: break
        tgt.append(ni)
    return hi_tok.decode(tgt[1:])


def repl(model, en_tok, hi_tok, max_len, device):
    print("=" * 55)
    print("  English -> Hindi (Fine-Tuned)  |  type 'quit' to exit")
    print("=" * 55)
    while True:
        try:
            src = input("\n  English : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if not src: continue
        if src.lower() in ("quit", "exit", "q"):
            print("Bye!"); break
        result = translate(model, src, en_tok, hi_tok, max_len, device)
        print(f"  Hindi   : {result if result else '[empty]'}")


def parse_args():
    ft_dir = os.path.join(os.path.dirname(__file__), "ft_checkpoints")
    p = argparse.ArgumentParser(description="Translate with fine-tuned model")
    p.add_argument("--checkpoint", default=os.path.join(ft_dir, "ft_best_model.pt"))
    p.add_argument("--checkpoint_dir", default=ft_dir)
    p.add_argument("--sentence", default=None, help="Single sentence (non-interactive)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, en_tok, hi_tok, max_len, device = load_model(args.checkpoint, args.checkpoint_dir)
    if args.sentence:
        result = translate(model, args.sentence, en_tok, hi_tok, max_len, device)
        print(f"EN : {args.sentence}")
        print(f"HI : {result}")
    else:
        repl(model, en_tok, hi_tok, max_len, device)
