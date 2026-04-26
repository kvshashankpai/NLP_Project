"""
evaluation.py
-------------
Evaluate a pretrained / fine-tuned English -> Hindi Transformer on a held-out
test set.  Zero external NLP libraries required (uses only metrics.py that
ships with this repo).

Metrics computed
~~~~~~~~~~~~~~~~
  • BLEU-1/2/3/4      – modified n-gram precision (Papineni et al., 2002)
  • Corpus BLEU       – full 4-gram BLEU with brevity penalty
  • chrF              – character F-score (Popovic, 2015) – implemented here
  • Length statistics – mean hyp / ref length ratio
  • Throughput        – sentences per second

Usage
~~~~~
  # Evaluate the pretrained checkpoint on the IITB test split
  python evaluation.py

  # Use a fine-tuned checkpoint, limit to 500 sentences, beam=1 (greedy)
  python evaluation.py \\
      --checkpoint finetune_updated/ft_checkpoints/ft_best_model.pt \\
      --config_dir finetune_updated/ft_checkpoints \\
      --dataset iitb --split test --max_samples 500

  # Evaluate on custom files
  python evaluation.py \\
      --dataset custom \\
      --custom_src  ./data/test.en \\
      --custom_ref  ./data/test.hi \\
      --max_samples 1000

  # Write per-sentence results to a TSV
  python evaluation.py --output_file results.tsv
"""

import os, sys, math, time, json, argparse
from collections import Counter
from typing import List, Tuple

# ── project root (one level above this script if placed alongside finetune_updated/) ──
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from tokenizers import Tokenizer

from transformer import Transformer
from dataset import START_TOKEN, END_TOKEN, PADDING_TOKEN
from metrics import corpus_bleu
from utils import get_device


# ══════════════════════════════════════════════════════════════════════════════
#  chrF – character n-gram F-score  (no external library)
# ══════════════════════════════════════════════════════════════════════════════

def _char_ngrams(text: str, n: int) -> Counter:
    text = text.replace(" ", "")          # collapse spaces (standard for chrF)
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def _chrf_sentence(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Per-sentence chrF score."""
    prec_sum = rec_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _char_ngrams(hyp, n)
        ref_ng = _char_ngrams(ref, n)
        matched = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
        hyp_total = sum(hyp_ng.values())
        ref_total = sum(ref_ng.values())
        prec_sum += (matched / hyp_total) if hyp_total else 0.0
        rec_sum  += (matched / ref_total) if ref_total else 0.0
    prec = prec_sum / max_n
    rec  = rec_sum  / max_n
    if prec + rec == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def corpus_chrf(hypotheses: List[str], references: List[str],
                max_n: int = 6, beta: float = 2.0) -> float:
    """Macro-average chrF across the corpus."""
    if not hypotheses:
        return 0.0
    return sum(_chrf_sentence(h, r, max_n, beta)
               for h, r in zip(hypotheses, references)) / len(hypotheses)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-n BLEU helpers (reuses metrics.py internals logic inline)
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams_list(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _modified_prec_n(hyps_tok, refs_tok, n: int) -> float:
    num = den = 0
    for h, r in zip(hyps_tok, refs_tok):
        hng = _ngrams_list(h, n)
        rng = _ngrams_list(r, n)
        num += sum(min(c, rng.get(g, 0)) for g, c in hng.items())
        den += max(len(h) - n + 1, 0)
    return num / den if den else 0.0


def bleu_n(hypotheses: List[str], references: List[str], n: int) -> float:
    """Corpus-level BLEU-N (precision only, no BP)."""
    ht = [list(h) for h in hypotheses]
    rt = [list(r) for r in references]
    return _modified_prec_n(ht, rt, n)


# ══════════════════════════════════════════════════════════════════════════════
#  Tokenisation helpers
# ══════════════════════════════════════════════════════════════════════════════

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
#  Greedy decoder
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_translate(model, src: str, en_tok, hi_tok, max_len: int,
                     device) -> str:
    model.eval()
    end_id = hi_tok.token_to_id(END_TOKEN)
    pad_id = hi_tok.token_to_id(PADDING_TOKEN)

    src_ids = tok_batch([src], en_tok, max_len,
                        start=False, end=False).to(device)
    enc_mask = (src_ids == en_tok.token_to_id(PADDING_TOKEN)) \
                   .unsqueeze(1).unsqueeze(2).float() \
                   .masked_fill_(
                       (src_ids == en_tok.token_to_id(PADDING_TOKEN))
                           .unsqueeze(1).unsqueeze(2),
                       float('-inf')
                   ).permute(1, 0, 2, 3).to(device)

    enc_out = model.encoder(src_ids, enc_mask)
    tgt = [hi_tok.token_to_id(START_TOKEN)]

    for _ in range(max_len):
        tt = torch.tensor(tgt, dtype=torch.long).unsqueeze(0).to(device)
        pl = max_len - tt.size(1)
        pt = torch.cat(
            [tt, torch.full((1, pl), pad_id, dtype=torch.long, device=device)],
            dim=1,
        )
        dp = (pt == pad_id).unsqueeze(1).unsqueeze(2).float() \
               .masked_fill_((pt == pad_id).unsqueeze(1).unsqueeze(2),
                             float('-inf')).permute(1, 0, 2, 3).to(device)
        ca = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        ca = ca.float().masked_fill_(ca, float('-inf')) \
               .unsqueeze(0).unsqueeze(0).to(device)
        dm = torch.clamp(dp + ca, min=float('-inf'), max=0)
        dm = dm[:, :, :tt.size(1), :tt.size(1)]

        dec_out = model.decoder(enc_out, tt, dm, enc_mask)
        logits  = model.linear(dec_out)
        ni      = logits[0, -1, :].argmax(dim=-1).item()
        if ni == end_id:
            break
        tgt.append(ni)

    return hi_tok.decode(tgt[1:])


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loaders (test split only)
# ══════════════════════════════════════════════════════════════════════════════

def load_iitb_test(max_samples: int, split: str = "test") -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")
    print(f"  Downloading cfilt/iitb-english-hindi  (split={split}) …")
    ds = load_dataset("cfilt/iitb-english-hindi", trust_remote_code=True)
    # Graceful fallback: test → validation → train
    if split in ds:
        chosen = ds[split]
    elif split == "test" and "validation" in ds:
        print("  WARNING: no 'test' split found — falling back to 'validation'")
        chosen = ds["validation"]
    else:
        chosen = ds[split]          # will raise a clear KeyError if absent
    en, hi = [], []
    for i, ex in enumerate(chosen):
        if max_samples and i >= max_samples:
            break
        pair = ex.get("translation", ex)
        e = pair.get("en", "").strip()
        h = pair.get("hi", "").strip()
        if e and h:
            en.append(e); hi.append(h)
    return en, hi


def load_opus_test(max_samples: int, split: str = "test") -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets")
    print(f"  Downloading Helsinki-NLP/opus-100 (en-hi)  (split={split}) …")
    ds = load_dataset("Helsinki-NLP/opus-100", "en-hi", trust_remote_code=True)
    if split in ds:
        chosen = ds[split]
    elif split == "test" and "validation" in ds:
        print("  WARNING: no 'test' split found — falling back to 'validation'")
        chosen = ds["validation"]
    else:
        chosen = ds[split]
    en, hi = [], []
    for i, ex in enumerate(chosen):
        if max_samples and i >= max_samples:
            break
        pair = ex.get("translation", {})
        e = pair.get("en", "").strip()
        h = pair.get("hi", "").strip()
        if e and h:
            en.append(e); hi.append(h)
    return en, hi


def load_custom_test(src_file: str, ref_file: str,
                     max_samples: int) -> Tuple[List[str], List[str]]:
    def read(p):
        with open(p, encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    en = read(src_file)
    hi = read(ref_file)
    if max_samples:
        en, hi = en[:max_samples], hi[:max_samples]
    return en, hi


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

def pbar(cur, tot, w=35):
    f = int(w * cur / tot)
    return f"[{'█'*f}{'░'*(w-f)}] {cur}/{tot}"


def fmt_time(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def hr(char="═", n=70):
    print(char * n)


# ══════════════════════════════════════════════════════════════════════════════
#  Main evaluation routine
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(args):
    device = get_device()

    # ── 1. Resolve config + checkpoint paths ────────────────────────────────
    ckpt_path   = os.path.join(PROJECT_ROOT, args.checkpoint)
    config_dir  = os.path.join(PROJECT_ROOT, args.config_dir)
    config_path = os.path.join(config_dir, "config.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in: {config_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    d_model  = cfg["d_model"];   ffn_h    = cfg["ffn_hidden"]
    n_heads  = cfg["num_heads"]; n_layers = cfg["num_layers"]
    max_len  = cfg["max_seq_len"]

    # ── 2. Load tokenizers ──────────────────────────────────────────────────
    en_tok = Tokenizer.from_file(os.path.join(config_dir, "en_tokenizer.json"))
    hi_tok = Tokenizer.from_file(os.path.join(config_dir, "hi_tokenizer.json"))

    # ── 3. Build & load model ───────────────────────────────────────────────
    model = Transformer(
        d_model=d_model, ffn_hidden=ffn_h, num_heads=n_heads,
        drop_prob=0.0, num_layers=n_layers,
        max_sequence_length=max_len,
        hindi_vocab_size=hi_tok.get_vocab_size(),
        english_vocab_size=en_tok.get_vocab_size(),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    saved_bleu  = ckpt.get("bleu", None)
    saved_epoch = ckpt.get("epoch", "?")

    # ── 4. Load test data ───────────────────────────────────────────────────
    print()
    hr()
    print(f"  English → Hindi Transformer — Evaluation")
    hr()
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"    └ epoch={saved_epoch}"
          + (f"  saved_BLEU={saved_bleu:.4f}" if saved_bleu else ""))
    print(f"  Architecture: d={d_model}, L={n_layers}, H={n_heads}, FFN={ffn_h}")
    print(f"  Dataset     : {args.dataset.upper()}  (split={args.split})")

    if args.dataset == "iitb":
        src_sents, ref_sents = load_iitb_test(args.max_samples, args.split)
    elif args.dataset == "opus":
        src_sents, ref_sents = load_opus_test(args.max_samples, args.split)
    elif args.dataset == "custom":
        if not args.custom_src or not args.custom_ref:
            raise ValueError("--custom_src and --custom_ref required for --dataset custom")
        src_sents, ref_sents = load_custom_test(
            args.custom_src, args.custom_ref, args.max_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    N = len(src_sents)
    print(f"  Sentences   : {N:,}")
    hr()

    # ── 5. Translate ────────────────────────────────────────────────────────
    hypotheses: List[str] = []
    t_start = time.time()

    for i, src in enumerate(src_sents, 1):
        hyp = greedy_translate(model, src, en_tok, hi_tok, max_len, device)
        hypotheses.append(hyp)

        if i % 50 == 0 or i == N:
            elapsed = time.time() - t_start
            eta     = elapsed / i * (N - i)
            sps     = i / elapsed
            print(f"\r  {pbar(i, N)}  {sps:.1f} sent/s  ETA {fmt_time(eta)}",
                  end="", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n  Done in {fmt_time(elapsed_total)}  "
          f"({N / elapsed_total:.1f} sent/s avg)\n")

    # ── 6. Compute metrics ──────────────────────────────────────────────────
    bleu   = corpus_bleu(hypotheses, ref_sents)
    bleu1  = bleu_n(hypotheses, ref_sents, 1)
    bleu2  = bleu_n(hypotheses, ref_sents, 2)
    bleu3  = bleu_n(hypotheses, ref_sents, 3)
    bleu4  = bleu_n(hypotheses, ref_sents, 4)
    chrf   = corpus_chrf(hypotheses, ref_sents)

    hyp_lens = [len(h) for h in hypotheses]
    ref_lens  = [len(r) for r in ref_sents]
    avg_hyp   = sum(hyp_lens) / N
    avg_ref   = sum(ref_lens) / N
    len_ratio = avg_hyp / avg_ref if avg_ref else 0.0

    empty_hyps = sum(1 for h in hypotheses if not h.strip())

    # ── 7. Print report ─────────────────────────────────────────────────────
    hr()
    print(f"  {'METRIC':<25}  {'VALUE':>10}")
    hr("─")
    print(f"  {'Corpus BLEU (1-4)':<25}  {bleu*100:>9.2f}")
    print(f"  {'BLEU-1 (precision)':<25}  {bleu1*100:>9.2f}")
    print(f"  {'BLEU-2 (precision)':<25}  {bleu2*100:>9.2f}")
    print(f"  {'BLEU-3 (precision)':<25}  {bleu3*100:>9.2f}")
    print(f"  {'BLEU-4 (precision)':<25}  {bleu4*100:>9.2f}")
    print(f"  {'chrF (β=2)':<25}  {chrf*100:>9.2f}")
    hr("─")
    print(f"  {'Avg hyp length (chars)':<25}  {avg_hyp:>9.1f}")
    print(f"  {'Avg ref length (chars)':<25}  {avg_ref:>9.1f}")
    print(f"  {'Length ratio (hyp/ref)':<25}  {len_ratio:>9.3f}")
    print(f"  {'Empty hypotheses':<25}  {empty_hyps:>9d}")
    print(f"  {'Throughput (sent/s)':<25}  {N/elapsed_total:>9.1f}")
    hr()

    # ── 8. Qualitative examples ─────────────────────────────────────────────
    n_show = min(args.show_examples, N)
    if n_show:
        print(f"\n  Sample translations ({n_show} shown):\n")
        step = max(1, N // n_show)
        for idx in range(0, N, step)[:n_show]:
            per_bleu = corpus_bleu([hypotheses[idx]], [ref_sents[idx]])
            per_chrf = _chrf_sentence(hypotheses[idx], ref_sents[idx])
            print(f"  [{idx+1:>4}]  BLEU={per_bleu*100:.1f}  chrF={per_chrf*100:.1f}")
            print(f"    EN  : {src_sents[idx][:100]}")
            print(f"    REF : {ref_sents[idx][:100]}")
            print(f"    HYP : {hypotheses[idx][:100]}")
            print()

    # ── 9. Optional TSV output ──────────────────────────────────────────────
    if args.output_file:
        out_path = os.path.join(PROJECT_ROOT, args.output_file)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("id\tsrc\tref\thyp\tbleu\tchrf\n")
            for i, (src, ref, hyp) in enumerate(
                    zip(src_sents, ref_sents, hypotheses), 1):
                pb = corpus_bleu([hyp], [ref]) * 100
                pc = _chrf_sentence(hyp, ref) * 100
                f.write(f"{i}\t{src}\t{ref}\t{hyp}\t{pb:.2f}\t{pc:.2f}\n")
        print(f"  Per-sentence results → {out_path}")
        hr()

    # ── 10. Summary dict (for programmatic use) ─────────────────────────────
    results = dict(
        corpus_bleu=round(bleu * 100, 4),
        bleu1=round(bleu1 * 100, 4),
        bleu2=round(bleu2 * 100, 4),
        bleu3=round(bleu3 * 100, 4),
        bleu4=round(bleu4 * 100, 4),
        chrf=round(chrf * 100, 4),
        length_ratio=round(len_ratio, 4),
        avg_hyp_len=round(avg_hyp, 2),
        avg_ref_len=round(avg_ref, 2),
        empty_hypotheses=empty_hyps,
        throughput_sent_per_sec=round(N / elapsed_total, 2),
        n_sentences=N,
    )

    # Optionally save JSON summary alongside checkpoint
    summary_path = os.path.join(
        os.path.dirname(ckpt_path), "eval_results.json"
    )
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON summary → {summary_path}\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate English→Hindi Transformer — no external NLP libs"
    )
    # Model
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Path to .pt checkpoint (relative to project root)")
    p.add_argument("--config_dir", default="checkpoints",
                   help="Dir containing config.json + *_tokenizer.json")
    # Dataset
    p.add_argument("--dataset", default="iitb",
                   choices=["iitb", "opus", "custom"],
                   help="Test corpus to evaluate on")
    p.add_argument("--split", default="test",
                   choices=["test", "validation", "train"],
                   help="Which HuggingFace split to evaluate on (default: test)")
    p.add_argument("--custom_src", default=None,
                   help="Plain-text source file  (one English sentence per line)")
    p.add_argument("--custom_ref", default=None,
                   help="Plain-text reference file (one Hindi sentence per line)")
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Maximum number of test sentences (0 = all)")
    # Output
    p.add_argument("--output_file", default=None,
                   help="Write per-sentence TSV to this path (optional)")
    p.add_argument("--show_examples", type=int, default=8,
                   help="Number of qualitative examples to print (0 to suppress)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())