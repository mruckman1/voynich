"""
Phase 12, Module 12.6: Latin Character N-Gram Model
=====================================================
Provides character-level scoring for Latin word candidates using
trigram log-probabilities. Used as a final fallback scorer when
word-level bigram transition probabilities return zero signal.

The model learns character trigram distributions from the Latin corpus
and scores candidates by average log-probability of their character
trigrams. This captures morphological plausibility: real Latin words
like 'herbam' score higher than garbage like 'xzqvm'.

Phase 12  ·  Voynich Convergence Attack
"""

import math
from collections import Counter
from typing import Dict, List, Tuple


class LatinCharNgramModel:
    """
    Character-level n-gram language model trained on Latin corpus tokens.

    Scores words by average log-probability of overlapping character
    n-grams (default: trigrams). Uses Laplace smoothing to handle
    unseen n-grams without returning -inf.

    The model prepends '^' and appends '$' sentinel characters to
    capture word-boundary patterns (e.g., common Latin prefixes and
    suffixes).
    """

    START = '^'
    END = '$'

    def __init__(self, order: int = 3, smoothing: float = 0.01):
        self.order = order
        self.smoothing = smoothing
        self._ngram_counts: Dict[str, int] = {}
        self._context_counts: Dict[str, int] = {}
        self._vocab_size: int = 0
        self._trained: bool = False

    def train(self, corpus_tokens: List[str]) -> None:
        """
        Train on a list of Latin corpus tokens.

        Extracts character n-grams with boundary sentinels and builds
        conditional probability tables: P(c_i | c_{i-n+1}...c_{i-1}).
        """
        ngram_counts: Counter = Counter()
        context_counts: Counter = Counter()
        char_set: set = set()

        for word in corpus_tokens:
            w = word.lower()
            padded = self.START * (self.order - 1) + w + self.END
            char_set.update(padded)

            for i in range(len(padded) - self.order + 1):
                ngram = padded[i:i + self.order]
                context = padded[i:i + self.order - 1]
                ngram_counts[ngram] += 1
                context_counts[context] += 1

        self._ngram_counts = dict(ngram_counts)
        self._context_counts = dict(context_counts)
        self._vocab_size = len(char_set)
        self._trained = True

    def score_word(self, word: str) -> float:
        """
        Average log-probability of a word's character trigrams.

        Returns a float in roughly [-15, 0] where higher = more Latin-like.
        Uses Laplace-smoothed conditional probability.
        """
        if not self._trained:
            return -float('inf')

        w = word.lower()
        padded = self.START * (self.order - 1) + w + self.END

        n_ngrams = len(padded) - self.order + 1
        if n_ngrams <= 0:
            return -float('inf')

        total_log_prob = 0.0
        for i in range(n_ngrams):
            ngram = padded[i:i + self.order]
            context = padded[i:i + self.order - 1]

            ngram_count = self._ngram_counts.get(ngram, 0)
            context_count = self._context_counts.get(context, 0)

            prob = (ngram_count + self.smoothing) / (
                context_count + self.smoothing * self._vocab_size
            )
            total_log_prob += math.log(prob)

        return total_log_prob / n_ngrams

    def score_candidates(
        self, candidates: List[str]
    ) -> List[Tuple[float, str]]:
        """Score and rank candidates by character n-gram plausibility."""
        scored = [(self.score_word(c), c) for c in candidates]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored

    def get_stats(self) -> Dict:
        """Return training statistics for logging."""
        return {
            'order': self.order,
            'unique_ngrams': len(self._ngram_counts),
            'unique_contexts': len(self._context_counts),
            'vocab_size': self._vocab_size,
            'smoothing': self.smoothing,
        }
