"""
metrics.py
----------
BLEU score implementation from scratch (no external library needed).

Implements corpus-level BLEU (Papineni et al., 2002) with:
  - Modified n-gram precision for n = 1..4
  - Brevity penalty
  - Geometric mean of precisions
"""

import math
from collections import Counter
from typing import List


def _ngrams(tokens: List[str], n: int) -> Counter:
    """Return a Counter of all n-grams in `tokens`."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(hypotheses: List[List[str]],
                         references: List[List[str]],
                         n: int) -> float:
    """
    Modified n-gram precision for a single n.

    For each hypothesis sentence:
      1. Count n-gram occurrences in hypothesis
      2. Clip each n-gram count to its max in ANY reference
      3. Sum clipped counts / total hypothesis n-grams
    """
    numerator = 0
    denominator = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_ngrams = _ngrams(hyp, n)
        ref_ngrams = _ngrams(ref, n)

        # Clip
        clipped = {gram: min(cnt, ref_ngrams.get(gram, 0))
                   for gram, cnt in hyp_ngrams.items()}

        numerator   += sum(clipped.values())
        denominator += max(len(hyp) - n + 1, 0)

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _brevity_penalty(hypotheses: List[List[str]],
                     references: List[List[str]]) -> float:
    """
    BP = exp(1 - r/c) if c < r else 1
    where c = total hypothesis length, r = closest reference length.
    """
    c = sum(len(h) for h in hypotheses)
    r = sum(len(ref) for ref in references)
    if c == 0:
        return 0.0
    if c >= r:
        return 1.0
    return math.exp(1 - r / c)


def corpus_bleu(hypotheses: List[str],
                references: List[str],
                max_n: int = 4,
                weights=None) -> float:
    """
    Compute corpus-level BLEU score.

    Args:
        hypotheses : list of predicted sentences (raw strings)
        references : list of reference sentences (raw strings)
        max_n      : highest order of n-gram (default 4)
        weights    : n-gram weights (default uniform 1/max_n each)

    Returns:
        BLEU score in [0, 1]
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Character-level tokenisation (matches our training setup)
    hyp_tokens = [list(h) for h in hypotheses]
    ref_tokens = [list(r) for r in references]

    bp = _brevity_penalty(hyp_tokens, ref_tokens)
    if bp == 0.0:
        return 0.0

    log_avg = 0.0
    for n, w in enumerate(weights, start=1):
        p = _modified_precision(hyp_tokens, ref_tokens, n)
        if p == 0.0:
            return 0.0
        log_avg += w * math.log(p)

    return bp * math.exp(log_avg)


def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Convenience wrapper for a single sentence pair."""
    return corpus_bleu([hypothesis], [reference], max_n=max_n)
