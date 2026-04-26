"""
finetune.py
-----------
Fine-tune the pretrained English -> Hindi Transformer.

Pretrained checkpoints are READ-ONLY. All outputs go to finetune/ft_checkpoints/.

Usage:
    python finetune/finetune.py
    python finetune/finetune.py --epochs 15 --lr 2e-5
"""

import os, sys, math, time, json, logging, argparse, shutil
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import (
    TranslationDataset, load_data, build_masks,
    START_TOKEN, END_TOKEN, PADDING_TOKEN,
)
from transformer import Transformer
from metrics import corpus_bleu
from utils import get_device


# ---------- Helpers ----------

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def fmt_time(s):
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def pbar(cur, tot, w=30):
    f = int(w * cur / tot)
    return f"[{'█'*f}{'░'*(w-f)}] {100*cur/tot:5.1f}%"

def tok_batch(sents, tokenizer, max_len, start=True, end=True):
    pad = tokenizer.token_to_id(PADDING_TOKEN)
    si = tokenizer.token_to_id(START_TOKEN)
    ei = tokenizer.token_to_id(END_TOKEN)
    batch = []
    for s in sents:
        ids = []
        if start: ids.append(si)
        ids.extend(tokenizer.encode(s).ids)
        if end: ids.append(ei)
        ids = ids[:max_len]
        ids += [pad] * (max_len - len(ids))
        batch.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(batch)


# ---------- Greedy decode for BLEU ----------

@torch.no_grad()
def greedy_translate(model, src, en_tok, hi_tok, max_len, device):
    model.eval()
    end_id = hi_tok.token_to_id(END_TOKEN)
    pad_id = hi_tok.token_to_id(PADDING_TOKEN)
    src_ids = tok_batch([src], en_tok, max_len, start=False, end=False).to(device)
    enc_m = (src_ids == en_tok.token_to_id(PADDING_TOKEN)).unsqueeze(1).unsqueeze(2)
    enc_m = enc_m.float().masked_fill(enc_m, float('-inf')).permute(1,0,2,3).to(device)
    enc_out = model.encoder(src_ids, enc_m)
    tgt = [hi_tok.token_to_id(START_TOKEN)]
    for _ in range(max_len):
        tt = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).to(device)
        pl = max_len - tt.size(1)
        pt = torch.cat([tt, torch.full((1,pl), pad_id, dtype=torch.long, device=device)], dim=1)
        dp = (pt == pad_id).unsqueeze(1).unsqueeze(2)
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


@torch.no_grad()
def evaluate(model, val_en, val_hi, en_tok, hi_tok, max_len, device, logger, n=200):
    model.eval()
    hyps, refs = [], []
    for i in range(min(n, len(val_en))):
        hyps.append(greedy_translate(model, val_en[i], en_tok, hi_tok, max_len, device))
        refs.append(val_hi[i])
    bleu = corpus_bleu(hyps, refs)
    msg = "\n  Sample translations:"
    for i in range(min(3, len(hyps))):
        msg += f"\n    EN : {val_en[i][:80]}\n    REF: {val_hi[i][:80]}\n    HYP: {hyps[i][:80]}\n"
    print(msg); logger.info(msg)
    return bleu


# ---------- Logger ----------

def setup_logger(path):
    lg = logging.getLogger("finetune")
    lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    lg.addHandler(fh)
    return lg


# ---------- Main fine-tuning loop ----------

def finetune(args):
    device = get_device()
    ft_dir = os.path.join(os.path.dirname(__file__), "ft_checkpoints")
    os.makedirs(ft_dir, exist_ok=True)
    logger = setup_logger(os.path.join(ft_dir, "finetune.log"))

    print(f"\n{'='*65}")
    print(f"  Fine-Tune English -> Hindi Transformer  |  device: {device}")
    print(f"{'='*65}\n")
    logger.info(f"Fine-tuning started on {device}")

    # 1. Load pretrained config & tokenizers
    pt_dir = os.path.join(PROJECT_ROOT, args.pretrained_dir)
    pt_ckpt = os.path.join(PROJECT_ROOT, args.pretrained_checkpoint)
    with open(os.path.join(pt_dir, "config.json")) as f:
        pcfg = json.load(f)

    from tokenizers import Tokenizer
    en_tok = Tokenizer.from_file(os.path.join(pt_dir, "en_tokenizer.json"))
    hi_tok = Tokenizer.from_file(os.path.join(pt_dir, "hi_tokenizer.json"))

    d_model = pcfg["d_model"]; ffn_h = pcfg["ffn_hidden"]
    n_heads = pcfg["num_heads"]; n_layers = pcfg["num_layers"]
    max_len = pcfg["max_seq_len"]

    print(f"  Pretrained ckpt : {pt_ckpt}")
    print(f"  Architecture    : d={d_model}, L={n_layers}, H={n_heads}, FFN={ffn_h}")

    # 2. Data (reuse pretrained tokenizers, NOT newly trained ones)
    print(f"\n  Loading dataset (max_samples={args.max_samples:,}) ...")
    train_en, train_hi, val_en, val_hi, _, _ = load_data(
        max_samples=args.max_samples, max_seq_len=max_len,
    )
    loader = DataLoader(
        TranslationDataset(train_en, train_hi),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    print(f"  Train: {len(train_en):,}  |  Val: {len(val_en):,}")

    # 3. Model + load pretrained weights (READ-ONLY)
    model = Transformer(
        d_model=d_model, ffn_hidden=ffn_h, num_heads=n_heads,
        drop_prob=args.drop_prob, num_layers=n_layers,
        max_sequence_length=max_len,
        hindi_vocab_size=hi_tok.get_vocab_size(),
        english_vocab_size=en_tok.get_vocab_size(),
    ).to(device)

    ckpt = torch.load(pt_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pt_epoch = ckpt.get("epoch", "?")
    pt_bleu = ckpt.get("bleu", 0.0)
    print(f"  Loaded weights (epoch {pt_epoch}, BLEU {pt_bleu:.4f})")
    print(f"  Parameters: {count_parameters(model):,}")
    logger.info(f"Loaded pretrained — epoch {pt_epoch}, BLEU {pt_bleu:.4f}")

    # 4. Fresh optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), eps=1e-9,
                                   weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader),
                                   eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(
        ignore_index=hi_tok.token_to_id(PADDING_TOKEN), label_smoothing=0.1,
    )
    pad_en = en_tok.token_to_id(PADDING_TOKEN)

    print(f"\n  FT Config: lr={args.lr}, wd={args.weight_decay}, bs={args.batch_size}, "
          f"accum={args.grad_accum}, eff_bs={args.batch_size*args.grad_accum}, "
          f"epochs={args.epochs}, patience={args.patience}\n")

    # 5. Copy tokenizers for self-contained inference
    shutil.copy2(os.path.join(pt_dir, "en_tokenizer.json"), os.path.join(ft_dir, "en_tokenizer.json"))
    shutil.copy2(os.path.join(pt_dir, "hi_tokenizer.json"), os.path.join(ft_dir, "hi_tokenizer.json"))
    ft_cfg = {
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "pretrained_epoch": pt_epoch, "pretrained_bleu": pt_bleu,
        "d_model": d_model, "ffn_hidden": ffn_h, "num_heads": n_heads,
        "num_layers": n_layers, "max_seq_len": max_len,
        "ft_epochs": args.epochs, "ft_lr": args.lr,
        "ft_batch_size": args.batch_size, "ft_drop_prob": args.drop_prob,
        "ft_weight_decay": args.weight_decay, "ft_grad_accum": args.grad_accum,
        "ft_max_samples": args.max_samples,
    }
    with open(os.path.join(ft_dir, "config.json"), "w") as f:
        json.dump(ft_cfg, f, indent=2)

    # 6. Training loop
    best_bleu = 0.0; patience_ctr = 0; history = []
    total_batches = len(loader)

    for epoch in range(1, args.epochs + 1):
        model.train(); epoch_loss = 0.0; t0 = time.time()
        optimizer.zero_grad()

        print(f"{'─'*65}")
        print(f"  FT Epoch {epoch}/{args.epochs}  —  {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'─'*65}")
        logger.info(f"Epoch {epoch}/{args.epochs} started")

        for step, (en_b, hi_b) in enumerate(loader, 1):
            st = time.time()
            en_ids = tok_batch(list(en_b), en_tok, max_len, start=False, end=False).to(device)
            hi_ids = tok_batch(list(hi_b), hi_tok, max_len, start=True, end=False).to(device)
            em, dm, cm = build_masks(en_ids, hi_ids, max_len, pad_idx=pad_en)

            logits = model(en_ids, hi_ids,
                           encoder_self_attention_mask=em,
                           decoder_self_attention_mask=dm,
                           decoder_cross_attention_mask=cm)

            tgt = tok_batch(list(hi_b), hi_tok, max_len, start=False, end=True).to(device)
            loss = criterion(logits.reshape(-1, hi_tok.get_vocab_size()), tgt.reshape(-1))
            loss = loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0 or step == total_batches:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            bl = loss.item() * args.grad_accum
            epoch_loss += bl

            if step % 200 == 0 or step == total_batches:
                avg = epoch_loss / step
                eta = (time.time() - t0) / step * (total_batches - step)
                lr_now = scheduler.get_last_lr()[0]
                print(f"\r  Step {step:5d}/{total_batches}  {pbar(step, total_batches)}  "
                      f"avg_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.2f}  "
                      f"lr={lr_now:.2e}  ETA {fmt_time(eta)}",
                      flush=True)
                logger.info(f"Step {step} avg_loss={avg:.4f}")

        avg_loss = epoch_loss / total_batches
        print(f"\n\n   FT Epoch {epoch}: avg_loss={avg_loss:.4f}  "
              f"ppl={math.exp(min(avg_loss,20)):.2f}  time={fmt_time(time.time()-t0)}")
        logger.info(f"Epoch {epoch} avg_loss={avg_loss:.4f}")

        # BLEU evaluation
        print(f"\n   Computing BLEU on {min(200, len(val_en))} val samples ...")
        bleu = evaluate(model, val_en, val_hi, en_tok, hi_tok, max_len, device, logger)
        print(f"     BLEU : {bleu:.4f}  ({bleu*100:.2f})")
        logger.info(f"Epoch {epoch} BLEU={bleu:.4f}")

        history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu})

        # Save checkpoint
        cp = os.path.join(ft_dir, f"ft_epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                     "optim_state": optimizer.state_dict(),
                     "scheduler_state": scheduler.state_dict(),
                     "loss": avg_loss, "bleu": bleu, "config": ft_cfg}, cp)
        print(f"      Checkpoint -> {cp}")

        if bleu > best_bleu:
            best_bleu = bleu; patience_ctr = 0
            bp = os.path.join(ft_dir, "ft_best_model.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                         "bleu": bleu, "config": ft_cfg}, bp)
            print(f"      * New best BLEU={best_bleu:.4f} -> {bp}")
            logger.info(f"New best BLEU={best_bleu:.4f}")
        else:
            patience_ctr += 1
            print(f"      No improvement ({patience_ctr}/{args.patience})")
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping (patience={args.patience})")
                logger.info(f"Early stopping at epoch {epoch}"); break
        print()

    # 7. Summary
    print(f"{'='*65}")
    print(f"  Fine-tuning complete!")
    print(f"  Pretrained BLEU : {pt_bleu:.4f}")
    print(f"  Best FT BLEU    : {best_bleu:.4f}  (delta {best_bleu-pt_bleu:+.4f})")
    print(f"{'='*65}\n")
    logger.info(f"Done — best BLEU={best_bleu:.4f}")

    with open(os.path.join(ft_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Translate: python finetune/ft_translate.py --sentence \"Hello\"\n")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune English->Hindi Transformer")
    p.add_argument("--pretrained_checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--pretrained_dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--drop_prob", type=float, default=0.1)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=200_000)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args)
