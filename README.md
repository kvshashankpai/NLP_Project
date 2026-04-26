# English → Hindi Transformer (from scratch)

A complete PyTorch implementation of the original **"Attention Is All You Need"**
Transformer architecture for English-to-Hindi neural machine translation.
Every component is built manually — no `nn.Transformer`, no HuggingFace models.

---

## File structure

```
en_hi_transformer/
├── utils.py          ← Building blocks: MHA, LayerNorm, PositionalEncoding, FFN, SentenceEmbedding
├── encoder.py        ← EncoderLayer + SequentialEncoder + Encoder
├── decoder.py        ← DecoderLayer (masked self-attn + cross-attn) + Decoder
├── transformer.py    ← Full Transformer (Encoder + Decoder + linear head)
├── dataset.py        ← Data loading, vocab building, mask generation
├── metrics.py        ← Corpus BLEU from scratch
├── train.py          ← Training loop with step-level progress bars + BLEU eval
├── translate.py      ← Interactive inference (REPL or single sentence)
├── requirements.txt
└── checkpoints/      ← Saved during training
    ├── en_vocab.json
    ├── hi_vocab.json
    ├── config.json
    ├── best_model.pt
    └── epoch_XX.pt
```

---

## Architecture

Follows Vaswani et al. (2017) exactly:

| Component | Detail |
|---|---|
| Embedding | Character-level + sinusoidal positional encoding |
| Encoder | N × (Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm) |
| Decoder | N × (Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm) |
| Attention | Scaled dot-product; Q,K,V projected from one linear layer |
| Layer Norm | Manual (learnable γ, β) |
| FFN | Linear → ReLU → Dropout → Linear |
| Output | Linear → Hindi vocabulary logits |

Default hyperparameters (matching the original paper):

```
d_model    = 512
ffn_hidden = 2048
num_heads  = 8
num_layers = 6
drop_prob  = 0.1
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py \
  --epochs 20 \
  --batch_size 32 \
  --max_samples 100000 \
  --max_seq_len 200 \
  --d_model 512 \
  --num_heads 8 \
  --num_layers 6 \
  --checkpoint_dir checkpoints
```

For a fast smoke-test on CPU (smaller model):

```bash
python train.py --epochs 2 --max_samples 5000 --d_model 128 --num_heads 4 --num_layers 2 --batch_size 16
```

### 3. Terminal output during training

```
─────────────────────────────────────────────────────────────────
  Epoch 1/20  —  14:32:01
─────────────────────────────────────────────────────────────────
  Step   200/3125  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   6.4%  loss=5.1234  ppl=168.12  ETA 12:34
  ...
  ✅ Epoch 1 complete
     avg_loss : 4.8321
     ppl      : 125.44
     time     : 15:22

  📊 Computing BLEU on 200 validation samples …
     BLEU : 0.0312 (3.12)

     💾 Checkpoint → checkpoints/epoch_01.pt
     🏆 New best BLEU=0.0312  → checkpoints/best_model.pt
```

### 4. Translate

Interactive REPL:

```bash
python translate.py --checkpoint checkpoints/best_model.pt
```

Single sentence:

```bash
python translate.py --sentence "Hello, how are you?"
```

---

## Dataset

The training script downloads the **OPUS-100 English-Hindi** parallel corpus automatically
via HuggingFace `datasets`. No manual download needed.

---

## BLEU metric

`metrics.py` implements corpus-level BLEU from scratch:
- Modified n-gram precision for n = 1 … 4
- Brevity penalty
- Geometric mean with uniform weights

Character-level tokenisation is used (matching the model's vocabulary).

---

## Tips for better results

- Train with `--max_samples 500000` if you have the data and GPU time
- Use `--max_seq_len 150` to reduce memory pressure
- The model needs at least 10–15 epochs to produce coherent Hindi
- BLEU > 5 is achievable after ~20 epochs on 100k pairs with the default hyperparameters
