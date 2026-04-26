# English → Hindi Neural Machine Translation

A from-scratch PyTorch implementation of the *Transformer* architecture (Vaswani et al., "Attention Is All You Need", 2017) for English-to-Hindi sequence-to-sequence translation. Every component — multi-head attention, cross-attention, positional encoding, layer normalisation, and feed-forward networks — is implemented manually without `torch.nn.Transformer`.

---

## Features

- Full Transformer encoder-decoder (6 layers, 8 heads, d_model=512) matching the original paper's base config
- Separate BPE tokenisers (10K vocab each) trained on the corpus via HuggingFace `tokenizers`
- Three attention mask types: encoder padding, decoder causal, decoder cross-attention
- Label-smoothed cross-entropy loss (ε=0.1) with Adam optimisation and gradient clipping
- Live terminal progress bar with per-batch loss and perplexity
- BLEU score evaluation on 200 validation samples after every epoch
- Full checkpoint management: per-epoch saves, best-BLEU tracking, and `--resume` support
- Interactive translation REPL and single-sentence inference mode

---

## Project Structure

```text
.
├── utils.py          # Core building blocks: attention, FFN, LayerNorm, positional encoding, embeddings
├── encoder.py        # EncoderLayer, SequentialEncoder, Encoder
├── decoder.py        # DecoderLayer, SequentialDecoder, Decoder
├── transformer.py    # Full Transformer model (encoder + decoder + output projection)
├── dataset.py        # BPE tokeniser training, mask construction, TranslationDataset, data loading
├── train.py          # Training script with progress bar, BLEU eval, and checkpointing
├── translate.py      # Inference: greedy autoregressive decoding, interactive REPL
└── checkpoints/      # Created automatically during training
    ├── en_tokenizer.json
    ├── hi_tokenizer.json
    ├── config.json
    ├── history.json
    ├── epoch_01.pt
    ├── ...
    └── best_model.pt
```

---

## Installation

*Python 3.8+ required.*

```bash
pip install torch torchvision
pip install datasets
pip install tokenizers
pip install sacrebleu
```

Or install everything at once:

```bash
pip install torch datasets tokenizers sacrebleu
```

---

## Dataset

Training uses the [cfilt/iitb-english-hindi](https://huggingface.co/datasets/cfilt/iitb-english-hindi) parallel corpus from IIT Bombay, loaded automatically from HuggingFace Datasets on first run. Up to 100,000 sentence pairs are used (shuffled, seed=42), split 90/10 into train and validation sets. Sentence pairs where either sentence exceeds `max_seq_len - 2` tokens are filtered out.

---

## Training

```bash
python train.py
```

All arguments are optional — defaults match the original Transformer base config:

```bash
python train.py \
  --epochs 20 \
  --batch_size 32 \
  --d_model 512 \
  --ffn_hidden 2048 \
  --num_heads 8 \
  --num_layers 6 \
  --drop_prob 0.1 \
  --lr 1e-4 \
  --max_seq_len 200 \
  --max_samples 100000 \
  --checkpoint_dir checkpoints
```

*Resume from a checkpoint:*

```bash
python train.py --resume checkpoints/epoch_10.pt --epochs 20
```

Training output looks like this:

```text
=================================================================
  English → Hindi Transformer  |  device: cuda
=================================================================

  Epoch 1/20  —  14:03:22
  Step   250/2812  [███████░░░░░░░░░░░░░░░░░░░░░░░]  8.9%  loss=4.8821  ppl=132.01  ETA 18:42
```

At the end of each epoch, 3 sample translations are printed alongside their references and a BLEU score is reported.

### What gets saved

| File | Contents |
|---|---|
| `checkpoints/en_tokenizer.json` | Trained English BPE tokeniser |
| `checkpoints/hi_tokenizer.json` | Trained Hindi BPE tokeniser |
| `checkpoints/config.json` | All training arguments |
| `checkpoints/epoch_XX.pt` | Model + optimiser state, loss, BLEU |
| `checkpoints/best_model.pt` | Checkpoint with highest validation BLEU |
| `checkpoints/history.json` | Per-epoch loss and BLEU history |

---

## Inference

*Interactive REPL* — type English sentences, get Hindi translations:

```bash
python translate.py --checkpoint checkpoints/best_model.pt
```

```text
=======================================================
  English → Hindi Translator  (type 'quit' to exit)
=======================================================

  English : The government announced a new policy today.
  Hindi   : सरकार ने आज एक नई नीति की घोषणा की।

  English : quit
```

*Single sentence (non-interactive):*

```bash
python translate.py \
  --checkpoint checkpoints/best_model.pt \
  --sentence "He went to the market."
```

*Custom checkpoint directory:*

```bash
python translate.py \
  --checkpoint my_run/best_model.pt \
  --checkpoint_dir my_run
```

---

## Model Architecture

The model strictly follows Vaswani et al. (2017). All components are implemented in `utils.py`, `encoder.py`, and `decoder.py`.

| Hyperparameter | Default | Paper (base) |
|---|---|---|
| d_model | 512 | 512 |
| FFN hidden dim | 2048 | 2048 |
| Attention heads | 8 | 8 |
| Encoder/Decoder layers | 6 | 6 |
| Head dimension (d_k) | 64 | 64 |
| Dropout | 0.1 | 0.1 |
| Max sequence length | 200 | — |
| Vocab size (EN / HI) | 10,000 each | 37K shared |
| Total parameters | ~59.4M | ~65M |

*Attention* uses an additive mask (0 = attend, −∞ = block) applied inside scaled dot-product attention before softmax. Three masks are built per batch:

- *Encoder padding mask* — blocks `<PAD>` positions in the source
- *Decoder causal mask* — upper-triangular block, prevents attending to future positions
- *Decoder combined mask* — causal + target padding, merged via `torch.clamp`

---

## Training Details

| Setting | Value |
|---|---|
| Loss | Cross-entropy, label smoothing ε=0.1, ignore `<PAD>` |
| Optimiser | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning rate | 1e-4 (fixed) |
| Gradient clipping | max norm = 1.0 |
| Decoder input | Hindi with `<START>` prepended (teacher forcing) |
| Decoder target | Hindi with `<END>` appended |
| BLEU eval | Greedy decoding on 200 validation samples per epoch |

---

## Known Limitations

- *No LR schedule* — the original paper uses a warm-up + inverse sqrt decay; this uses a fixed LR.
- *Greedy decoding only* — `beam_size` is plumbed in `translate.py` but beam search is not implemented. Beam search (k=4) typically adds 1–3 BLEU points.
- *No weight tying* — the output projection and Hindi embedding matrix are separate. Tying them reduces parameters and often improves performance.
- *ReLU vs GELU* — the FFN uses ReLU; most modern implementations use GELU.
- *BLEU on 200 samples* — full validation BLEU would be more reliable but is slower.

---
## Team Members

- K V Shashank Pai (BT2024250)
- K.Sai Tushaar (BT2024022)
- Muppana Jatin (BT2024127)
- Tejas Kollipara (BT2024147)
---

## References

- Vaswani et al., Attention Is All You Need, NeurIPS 2017
- Sennrich et al., Neural MT of Rare Words with Subword Units, ACL 2016
- Kunchukuttan et al., The IIT Bombay English-Hindi Parallel Corpus, LREC 2018
- Papineni et al., BLEU: a Method for Automatic Evaluation of MT, ACL 2002
