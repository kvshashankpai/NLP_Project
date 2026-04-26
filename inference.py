"""
inference.py
------------
Translate English → Hindi using a pretrained (or fine-tuned) Transformer.

Sample sentences are hard-coded at the top and run automatically when the
script is executed without --sentence / --file flags, so you can verify the
model instantly after training.

Usage
~~~~~
  # Run the built-in sample sentences
  python inference.py

  # Translate a single sentence (prints to stdout)
  python inference.py --sentence "The train arrives at nine in the morning."

  # Translate every line in a file, write Hindi output to another file
  python inference.py --file input.en --output output.hi

  # Use a fine-tuned checkpoint
  python inference.py \\
      --checkpoint finetune_updated/ft_checkpoints/ft_best_model.pt \\
      --config_dir finetune_updated/ft_checkpoints \\
      --sentence "Good morning, how are you?"

  # Interactive REPL
  python inference.py --interactive
"""

import os, sys, json, argparse, time
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
from tokenizers import Tokenizer

from transformer import Transformer
from dataset import START_TOKEN, END_TOKEN, PADDING_TOKEN
from utils import get_device


# ══════════════════════════════════════════════════════════════════════════════
#  Built-in sample sentences (diverse domains + difficulty levels)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_SENTENCES = [
    # ── WHY these sentences work well ──────────────────────────────────────
    # 1. Short to medium length  (under ~15 words)
    # 2. Subject-Verb-Object structure — maps cleanly to Hindi SOV
    # 3. Common IITB-domain vocabulary: news, government, daily life, nature
    # 4. No idioms, slang, or culturally untranslatable phrases
    # 5. Named entities that appear frequently in Indian news corpora
    # 6. Present / simple past tense  (future & conditionals are harder)

    # ── Greetings & identity (very high confidence) ───────────────────────
    "How are you?",
    "Thank you",

    # ── Daily life (high confidence) ───────────────────────────────────────
    "The water is clean.",
    "She went to the market.",
    "Please sit down.",


    # ── News & government (high confidence — dominant IITB domain) ─────────
    "The Prime Minister met the President today.",
    "The government will build new schools.",
    "The police arrested the accused.",
    "The court gave its decision.",

    # ── Health & education (medium-high confidence) ────────────────────────
    "The doctor examined the patient.",

    # ── Simple descriptions (medium-high confidence) ───────────────────────
    "This road is very long.",

]


# ══════════════════════════════════════════════════════════════════════════════
#  Tokenisation helper
# ══════════════════════════════════════════════════════════════════════════════

def _tok_batch(sents, tokenizer, max_len, start=True, end=True):
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
def greedy_translate(model, src: str, en_tok, hi_tok,
                     max_len: int, device) -> str:
    """Greedy (argmax) decoding — fast, deterministic."""
    model.eval()
    end_id = hi_tok.token_to_id(END_TOKEN)
    pad_id = hi_tok.token_to_id(PADDING_TOKEN)
    pad_en = en_tok.token_to_id(PADDING_TOKEN)

    # Encode source
    src_ids = _tok_batch([src], en_tok, max_len,
                         start=False, end=False).to(device)
    enc_pad = src_ids == pad_en
    enc_m   = enc_pad.unsqueeze(1).unsqueeze(2).float() \
                      .masked_fill_(enc_pad.unsqueeze(1).unsqueeze(2),
                                    float('-inf')) \
                      .permute(1, 0, 2, 3).to(device)
    enc_out = model.encoder(src_ids, enc_m)

    # Decode token-by-token
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

        dec_out = model.decoder(enc_out, tt, dm, enc_m)
        logits  = model.linear(dec_out)
        ni      = logits[0, -1, :].argmax(dim=-1).item()
        if ni == end_id:
            break
        tgt.append(ni)

    return hi_tok.decode(tgt[1:])


# ══════════════════════════════════════════════════════════════════════════════
#  Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, config_dir: str, device):
    """Load model + tokenizers from disk.  Returns (model, en_tok, hi_tok, max_len)."""
    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {config_dir}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    d_model  = cfg["d_model"];   ffn_h    = cfg["ffn_hidden"]
    n_heads  = cfg["num_heads"]; n_layers = cfg["num_layers"]
    max_len  = cfg["max_seq_len"]

    en_tok = Tokenizer.from_file(os.path.join(config_dir, "en_tokenizer.json"))
    hi_tok = Tokenizer.from_file(os.path.join(config_dir, "hi_tokenizer.json"))

    model = Transformer(
        d_model=d_model, ffn_hidden=ffn_h, num_heads=n_heads,
        drop_prob=0.0, num_layers=n_layers,
        max_sequence_length=max_len,
        hindi_vocab_size=hi_tok.get_vocab_size(),
        english_vocab_size=en_tok.get_vocab_size(),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch      = ckpt.get("epoch", "?")
    saved_bleu = ckpt.get("bleu", None)

    print(f"  Model loaded  (epoch={epoch}"
          + (f"  BLEU={saved_bleu:.4f}" if saved_bleu else "") + ")")
    print(f"  Architecture : d={d_model}, L={n_layers}, H={n_heads}, FFN={ffn_h}")
    print(f"  Max seq len  : {max_len}")
    print(f"  EN vocab     : {en_tok.get_vocab_size():,}")
    print(f"  HI vocab     : {hi_tok.get_vocab_size():,}")

    return model, en_tok, hi_tok, max_len


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════════════

def hr(char="═", n=72):
    print(char * n)


def print_translation(en: str, hi: str, idx: int | None = None,
                      elapsed: float | None = None):
    prefix = f"[{idx}] " if idx is not None else ""
    timing = f"  ({elapsed*1000:.0f} ms)" if elapsed is not None else ""
    print(f"\n  {prefix}EN : {en}")
    print(f"  {prefix}HI : {hi}{timing}")


# ══════════════════════════════════════════════════════════════════════════════
#  Mode handlers
# ══════════════════════════════════════════════════════════════════════════════

def run_samples(model, en_tok, hi_tok, max_len, device):
    """Translate the hard-coded SAMPLE_SENTENCES and print results."""
    hr()
    print(f"  Sample Translations  ({len(SAMPLE_SENTENCES)} sentences)")
    hr()

    total_start = time.time()
    for i, src in enumerate(SAMPLE_SENTENCES, 1):
        t0  = time.time()
        hi  = greedy_translate(model, src, en_tok, hi_tok, max_len, device)
        dt  = time.time() - t0
        print_translation(src, hi, idx=i, elapsed=dt)

    elapsed = time.time() - total_start
    print(f"\n  Translated {len(SAMPLE_SENTENCES)} sentences in "
          f"{elapsed:.2f}s  "
          f"({len(SAMPLE_SENTENCES)/elapsed:.1f} sent/s)\n")
    hr()


def run_single(sentence: str, model, en_tok, hi_tok, max_len, device):
    """Translate one sentence and print it."""
    t0 = time.time()
    hi = greedy_translate(model, sentence, en_tok, hi_tok, max_len, device)
    dt = time.time() - t0
    print_translation(sentence, hi, elapsed=dt)
    print()


def run_file(src_file: str, out_file: str | None,
             model, en_tok, hi_tok, max_len, device):
    """Translate every line of src_file; write to out_file or stdout."""
    with open(src_file, encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    print(f"\n  Translating {len(sentences):,} sentences from '{src_file}' …\n")
    translations = []
    t0 = time.time()

    for i, src in enumerate(sentences, 1):
        hi = greedy_translate(model, src, en_tok, hi_tok, max_len, device)
        translations.append(hi)
        if i % 50 == 0 or i == len(sentences):
            elapsed = time.time() - t0
            sps     = i / elapsed
            eta     = elapsed / i * (len(sentences) - i)
            m, s    = divmod(int(eta), 60)
            print(f"\r  {i}/{len(sentences)}  {sps:.1f} sent/s  ETA {m:02d}:{s:02d}",
                  end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  ({len(sentences)/elapsed:.1f} sent/s)\n")

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(translations) + "\n")
        print(f"  Output written → {out_file}\n")
    else:
        print("  Translations:\n")
        for src, hi in zip(sentences, translations):
            print(f"  EN : {src}")
            print(f"  HI : {hi}\n")


def run_interactive(model, en_tok, hi_tok, max_len, device):
    """Simple REPL — type a sentence, get the Hindi translation."""
    hr()
    print("  Interactive mode  (type 'quit' or press Ctrl-C to exit)")
    hr()
    while True:
        try:
            src = input("\n  EN > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!\n"); break
        if not src:
            continue
        if src.lower() in {"quit", "exit", "q"}:
            print("  Bye!\n"); break
        t0 = time.time()
        hi = greedy_translate(model, src, en_tok, hi_tok, max_len, device)
        dt = time.time() - t0
        print(f"  HI > {hi}  ({dt*1000:.0f} ms)")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="English → Hindi inference (pretrained Transformer)"
    )
    # Model
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                   help="Path to .pt checkpoint (relative to project root)")
    p.add_argument("--config_dir", default="checkpoints",
                   help="Dir containing config.json + *_tokenizer.json")
    # Mode (mutually exclusive; default = run sample sentences)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sentence", default=None,
                   help="Translate a single sentence and exit")
    g.add_argument("--file", default=None,
                   help="Translate every line of this file")
    g.add_argument("--interactive", action="store_true",
                   help="Start an interactive translation REPL")
    # File-mode output
    p.add_argument("--output", default=None,
                   help="Output file for --file mode (default: stdout)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    ckpt_path  = os.path.join(PROJECT_ROOT, args.checkpoint)
    config_dir = os.path.join(PROJECT_ROOT, args.config_dir)

    hr()
    print("  English → Hindi Transformer — Inference")
    hr()
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {ckpt_path}\n")

    model, en_tok, hi_tok, max_len = load_model(ckpt_path, config_dir, device)

    if args.sentence:
        hr()
        print("  Single-sentence translation\n")
        run_single(args.sentence, model, en_tok, hi_tok, max_len, device)

    elif args.file:
        run_file(args.file, args.output, model, en_tok, hi_tok, max_len, device)

    elif args.interactive:
        run_interactive(model, en_tok, hi_tok, max_len, device)

    else:
        # Default: run the built-in sample sentences
        run_samples(model, en_tok, hi_tok, max_len, device)


if __name__ == "__main__":
    main()