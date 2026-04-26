"""
finetune.py  (finetune_updated version)
----------------------------------------
Fine-tune the pretrained English -> Hindi Transformer.

Pretrained checkpoints are READ-ONLY.
All outputs go to  finetune_updated/ft_checkpoints/   (never touches finetune/).

Dataset options (--dataset flag):
  iitb      – cfilt/iitb-english-hindi          ~1.6 M pairs  [default]
  opus      – Helsinki-NLP/opus-100  (en-hi)    ~1 M   pairs
  samanantar– ai4bharat/samanantar   (en-hi)    ~8.4 M pairs  (large, slow to download)
  custom    – provide --custom_data_dir with train.en / train.hi / val.en / val.hi

Usage:
    python finetune_updated/finetune.py
    python finetune_updated/finetune.py --epochs 15 --lr 2e-5
    python finetune_updated/finetune.py --dataset opus --max_samples 100000
    python finetune_updated/finetune.py --dataset custom --custom_data_dir ./my_data
"""

import os, sys, math, time, json, logging, argparse, shutil
from datetime import datetime

# ── project root is one level above finetune_updated/ ──────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import (
    TranslationDataset, build_masks,
    START_TOKEN, END_TOKEN, PADDING_TOKEN,
)
from transformer import Transformer
from metrics import corpus_bleu
from utils import get_device


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

def _clip(pairs, max_samples):
    if max_samples and len(pairs) > max_samples:
        pairs = pairs[:max_samples]
    return pairs


def load_iitb(max_samples, max_seq_len, val_size=2000):
    """IIT-Bombay English-Hindi corpus (HuggingFace: cfilt/iitb-english-hindi)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install HuggingFace datasets:  pip install datasets")

    print("  Downloading cfilt/iitb-english-hindi …")
    ds = load_dataset("cfilt/iitb-english-hindi", trust_remote_code=True)
    train_split = ds["train"]
    val_split   = ds.get("validation") or ds.get("test")

    def extract(split, limit=None):
        en, hi = [], []
        for i, ex in enumerate(split):
            if limit and i >= limit: break
            pair = ex.get("translation", ex)
            e = pair.get("en", "").strip()
            h = pair.get("hi", "").strip()
            if e and h:
                en.append(e); hi.append(h)
        return en, hi

    train_en, train_hi = extract(train_split, max_samples)
    val_en,   val_hi   = extract(val_split,   val_size)
    return train_en, train_hi, val_en, val_hi


def load_opus100(max_samples, max_seq_len, val_size=2000):
    """Helsinki-NLP/opus-100 en-hi subset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install HuggingFace datasets:  pip install datasets")

    print("  Downloading Helsinki-NLP/opus-100 (en-hi) …")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", trust_remote_code=True)
    train_split = ds["train"]
    val_split   = ds.get("validation") or ds.get("test")

    def extract(split, limit=None):
        en, hi = [], []
        for i, ex in enumerate(split):
            if limit and i >= limit: break
            pair = ex.get("translation", {})
            e = pair.get("en", "").strip()
            h = pair.get("hi", "").strip()
            if e and h:
                en.append(e); hi.append(h)
        return en, hi

    train_en, train_hi = extract(train_split, max_samples)
    val_en,   val_hi   = extract(val_split,   val_size)
    return train_en, train_hi, val_en, val_hi


def load_samanantar(max_samples, max_seq_len, val_size=2000):
    """ai4bharat/samanantar en-hi (very large – be patient)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install HuggingFace datasets:  pip install datasets")

    print("  Streaming ai4bharat/samanantar (en-hi) …")
    ds = load_dataset("ai4bharat/samanantar", "hi", streaming=True,
                      trust_remote_code=True)
    train_en, train_hi = [], []
    for i, ex in enumerate(ds["train"]):
        if max_samples and i >= max_samples: break
        e = ex.get("src", "").strip()
        h = ex.get("tgt", "").strip()
        if e and h:
            train_en.append(e); train_hi.append(h)

    # split last val_size rows as validation
    split = min(val_size, len(train_en) // 10)
    val_en,   val_hi   = train_en[-split:], train_hi[-split:]
    train_en, train_hi = train_en[:-split], train_hi[:-split]
    return train_en, train_hi, val_en, val_hi


def load_custom(data_dir, max_samples, val_size=2000):
    """
    Expects plain-text files in data_dir:
        train.en  /  train.hi
        val.en    /  val.hi      (optional; last val_size lines of train used if absent)
    """
    def read(path):
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]

    train_en = read(os.path.join(data_dir, "train.en"))
    train_hi = read(os.path.join(data_dir, "train.hi"))
    train_en, train_hi = _clip(list(zip(train_en, train_hi)), max_samples) if max_samples else list(zip(train_en, train_hi))
    if isinstance(train_en[0], tuple):
        train_en, train_hi = zip(*train_en)
        train_en, train_hi = list(train_en), list(train_hi)

    val_en_path = os.path.join(data_dir, "val.en")
    val_hi_path = os.path.join(data_dir, "val.hi")
    if os.path.exists(val_en_path) and os.path.exists(val_hi_path):
        val_en = read(val_en_path)
        val_hi = read(val_hi_path)
    else:
        split = min(val_size, len(train_en) // 10)
        val_en,   val_hi   = train_en[-split:], train_hi[-split:]
        train_en, train_hi = train_en[:-split], train_hi[:-split]

    return train_en, train_hi, val_en, val_hi


def load_dataset_by_name(name, max_samples, max_seq_len, custom_data_dir=None):
    name = name.lower()
    if name == "iitb":
        return load_iitb(max_samples, max_seq_len)
    elif name == "opus":
        return load_opus100(max_samples, max_seq_len)
    elif name == "samanantar":
        return load_samanantar(max_samples, max_seq_len)
    elif name == "custom":
        if not custom_data_dir:
            raise ValueError("--custom_data_dir must be set when --dataset custom")
        return load_custom(custom_data_dir, max_samples)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose: iitb | opus | samanantar | custom")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (identical to original finetune.py)
# ══════════════════════════════════════════════════════════════════════════════

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
    si  = tokenizer.token_to_id(START_TOKEN)
    ei  = tokenizer.token_to_id(END_TOKEN)
    batch = []
    for s in sents:
        ids = []
        if start: ids.append(si)
        ids.extend(tokenizer.encode(s).ids)
        if end:   ids.append(ei)
        ids = ids[:max_len]
        ids += [pad] * (max_len - len(ids))
        batch.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(batch)


# ══════════════════════════════════════════════════════════════════════════════
#  Greedy decode for BLEU evaluation
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Logger
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(path):
    lg = logging.getLogger("finetune_updated")
    lg.setLevel(logging.INFO); lg.handlers.clear()
    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    lg.addHandler(fh)
    return lg


# ══════════════════════════════════════════════════════════════════════════════
#  Main fine-tuning loop
# ══════════════════════════════════════════════════════════════════════════════

def finetune(args):
    device = get_device()

    # ── output dir: finetune_updated/ft_checkpoints  (isolated from finetune/) ──
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    ft_dir   = os.path.join(THIS_DIR, "ft_checkpoints")
    os.makedirs(ft_dir, exist_ok=True)
    logger = setup_logger(os.path.join(ft_dir, "finetune.log"))

    print(f"\n{'='*65}")
    print(f"  Fine-Tune English -> Hindi Transformer  |  device: {device}")
    print(f"  Output dir : {ft_dir}")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"{'='*65}\n")
    logger.info(f"Fine-tuning started — device={device}  dataset={args.dataset}")

    # ── 1. Load pretrained config & tokenizers ──────────────────────────────
    pt_dir  = os.path.join(PROJECT_ROOT, args.pretrained_dir)
    pt_ckpt = os.path.join(PROJECT_ROOT, args.pretrained_checkpoint)
    with open(os.path.join(pt_dir, "config.json")) as f:
        pcfg = json.load(f)

    from tokenizers import Tokenizer
    en_tok = Tokenizer.from_file(os.path.join(pt_dir, "en_tokenizer.json"))
    hi_tok = Tokenizer.from_file(os.path.join(pt_dir, "hi_tokenizer.json"))

    d_model  = pcfg["d_model"];   ffn_h    = pcfg["ffn_hidden"]
    n_heads  = pcfg["num_heads"]; n_layers = pcfg["num_layers"]
    max_len  = pcfg["max_seq_len"]

    print(f"  Pretrained ckpt : {pt_ckpt}")
    print(f"  Architecture    : d={d_model}, L={n_layers}, H={n_heads}, FFN={ffn_h}")

    # ── 2. Load dataset ─────────────────────────────────────────────────────
    print(f"\n  Loading dataset '{args.dataset}' (max_samples={args.max_samples:,}) …")
    train_en, train_hi, val_en, val_hi = load_dataset_by_name(
        args.dataset, args.max_samples, max_len, args.custom_data_dir
    )
    print(f"  Train: {len(train_en):,}  |  Val: {len(val_en):,}")

    loader = DataLoader(
        TranslationDataset(train_en, train_hi),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    # ── 3. Model + load pretrained weights ──────────────────────────────────
    model = Transformer(
        d_model=d_model, ffn_hidden=ffn_h, num_heads=n_heads,
        drop_prob=args.drop_prob, num_layers=n_layers,
        max_sequence_length=max_len,
        hindi_vocab_size=hi_tok.get_vocab_size(),
        english_vocab_size=en_tok.get_vocab_size(),
    ).to(device)

    ckpt     = torch.load(pt_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    pt_epoch = ckpt.get("epoch", "?")
    pt_bleu  = ckpt.get("bleu", 0.0)
    print(f"  Loaded weights (epoch {pt_epoch}, BLEU {pt_bleu:.4f})")
    print(f"  Parameters : {count_parameters(model):,}")
    logger.info(f"Loaded pretrained — epoch {pt_epoch}, BLEU {pt_bleu:.4f}")

    # ── 4. Freeze encoder (optional) ────────────────────────────────────────
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        print(f"  Encoder frozen — only decoder & linear will be updated")
        logger.info("Encoder frozen")

    # ── 5. Optimizer + scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader), eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=hi_tok.token_to_id(PADDING_TOKEN), label_smoothing=0.1,
    )
    pad_en = en_tok.token_to_id(PADDING_TOKEN)

    print(f"\n  FT Config: dataset={args.dataset}, lr={args.lr}, wd={args.weight_decay}, "
          f"bs={args.batch_size}, accum={args.grad_accum}, "
          f"eff_bs={args.batch_size*args.grad_accum}, "
          f"epochs={args.epochs}, patience={args.patience}, "
          f"freeze_encoder={args.freeze_encoder}\n")

    # ── 6. Copy tokenizers → ft_checkpoints (self-contained inference) ──────
    shutil.copy2(os.path.join(pt_dir, "en_tokenizer.json"),
                 os.path.join(ft_dir, "en_tokenizer.json"))
    shutil.copy2(os.path.join(pt_dir, "hi_tokenizer.json"),
                 os.path.join(ft_dir, "hi_tokenizer.json"))

    ft_cfg = {
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "pretrained_epoch": pt_epoch, "pretrained_bleu": pt_bleu,
        "d_model": d_model, "ffn_hidden": ffn_h, "num_heads": n_heads,
        "num_layers": n_layers, "max_seq_len": max_len,
        "ft_dataset": args.dataset,
        "ft_epochs": args.epochs, "ft_lr": args.lr,
        "ft_batch_size": args.batch_size, "ft_drop_prob": args.drop_prob,
        "ft_weight_decay": args.weight_decay, "ft_grad_accum": args.grad_accum,
        "ft_max_samples": args.max_samples, "ft_freeze_encoder": args.freeze_encoder,
    }
    with open(os.path.join(ft_dir, "config.json"), "w") as f:
        json.dump(ft_cfg, f, indent=2)

    # ── 7. Training loop ────────────────────────────────────────────────────
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
            en_ids = tok_batch(list(en_b), en_tok, max_len, start=False, end=False).to(device)
            hi_ids = tok_batch(list(hi_b), hi_tok, max_len, start=True,  end=False).to(device)
            em, dm, cm = build_masks(en_ids, hi_ids, max_len, pad_idx=pad_en)

            logits = model(en_ids, hi_ids,
                           encoder_self_attention_mask=em,
                           decoder_self_attention_mask=dm,
                           decoder_cross_attention_mask=cm)

            tgt  = tok_batch(list(hi_b), hi_tok, max_len, start=False, end=True).to(device)
            loss = criterion(logits.reshape(-1, hi_tok.get_vocab_size()), tgt.reshape(-1))
            loss = loss / args.grad_accum
            loss.backward()

            if step % args.grad_accum == 0 or step == total_batches:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            bl = loss.item() * args.grad_accum
            epoch_loss += bl

            if step % 200 == 0 or step == total_batches:
                avg  = epoch_loss / step
                eta  = (time.time() - t0) / step * (total_batches - step)
                lr_n = scheduler.get_last_lr()[0]
                print(f"\r  Step {step:5d}/{total_batches}  {pbar(step, total_batches)}  "
                      f"avg_loss={avg:.4f}  ppl={math.exp(min(avg,20)):.2f}  "
                      f"lr={lr_n:.2e}  ETA {fmt_time(eta)}",
                      flush=True)
                logger.info(f"Step {step} avg_loss={avg:.4f}")

        avg_loss = epoch_loss / total_batches
        print(f"\n\n   FT Epoch {epoch}: avg_loss={avg_loss:.4f}  "
              f"ppl={math.exp(min(avg_loss,20)):.2f}  time={fmt_time(time.time()-t0)}")
        logger.info(f"Epoch {epoch} avg_loss={avg_loss:.4f}")

        # BLEU evaluation
        print(f"\n   Computing BLEU on {min(200, len(val_en))} val samples …")
        bleu = evaluate(model, val_en, val_hi, en_tok, hi_tok, max_len, device, logger)
        print(f"     BLEU : {bleu:.4f}  ({bleu*100:.2f})")
        logger.info(f"Epoch {epoch} BLEU={bleu:.4f}")
        history.append({"epoch": epoch, "loss": avg_loss, "bleu": bleu})

        # Per-epoch checkpoint
        cp = os.path.join(ft_dir, f"ft_epoch_{epoch:02d}.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": avg_loss, "bleu": bleu, "config": ft_cfg}, cp)
        print(f"      Checkpoint → {cp}")

        # Best-model checkpoint
        if bleu > best_bleu:
            best_bleu = bleu; patience_ctr = 0
            bp = os.path.join(ft_dir, "ft_best_model.pt")
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "bleu": bleu, "config": ft_cfg}, bp)
            print(f"      ★ New best BLEU={best_bleu:.4f} → {bp}")
            logger.info(f"New best BLEU={best_bleu:.4f}")
        else:
            patience_ctr += 1
            print(f"      No improvement ({patience_ctr}/{args.patience})")
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping (patience={args.patience})")
                logger.info(f"Early stopping at epoch {epoch}"); break
        print()

    # ── 8. Summary ──────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  Fine-tuning complete!")
    print(f"  Pretrained BLEU : {pt_bleu:.4f}")
    print(f"  Best FT BLEU    : {best_bleu:.4f}  (delta {best_bleu-pt_bleu:+.4f})")
    print(f"  Checkpoints in  : {ft_dir}")
    print(f"{'='*65}\n")
    logger.info(f"Done — best BLEU={best_bleu:.4f}")

    with open(os.path.join(ft_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Translate: python finetune_updated/ft_translate.py --sentence \"Hello\"\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune English->Hindi Transformer (updated)")
    # Pretrained model
    p.add_argument("--pretrained_checkpoint", default="checkpoints/best_model.pt",
                   help="Path to pretrained checkpoint (relative to project root)")
    p.add_argument("--pretrained_dir", default="checkpoints",
                   help="Dir with config.json + tokenizers (relative to project root)")
    # Dataset
    p.add_argument("--dataset", default="iitb",
                   choices=["iitb", "opus", "samanantar", "custom"],
                   help="Dataset to fine-tune on")
    p.add_argument("--custom_data_dir", default=None,
                   help="Dir with train.en/train.hi/val.en/val.hi (only for --dataset custom)")
    # Training
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-5)
    p.add_argument("--weight_decay",  type=float, default=0.01)
    p.add_argument("--drop_prob",     type=float, default=0.1)
    p.add_argument("--grad_accum",    type=int,   default=2)
    p.add_argument("--max_samples",   type=int,   default=200_000)
    p.add_argument("--patience",      type=int,   default=3)
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder weights — only train decoder + linear projection")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune(args)