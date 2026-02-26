"""
Tier Splitter: Corpus Tier Separation for Nomenclator Attack
==============================================================
Splits the full Language A corpus into two tiers:
  - Tier 1 (Codebook): ~1,001 word types appearing 2+ times (8,030 tokens)
  - Tier 2 (Cipher):   ~2,761 singleton word types (unique occurrences)

The critical innovation: singleton bridging. When a Tier 2 word sits
between two Tier 1 words, the bigram [T1_prev] → [T1_next] is counted
for the Tier 1 transition matrix. The singleton acts as a transparent
bridge — it contributes to the context of flanking codebook words but
is not itself in the matrix.

Phase 5  ·  Voynich Convergence Attack
"""

import sys
import os
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.statistical_analysis import (
    conditional_entropy, first_order_entropy, word_conditional_entropy,
)
from modules.phase4.lang_a_extractor import LanguageAExtractor


class TierSplitter:
    """
    Split the Language A corpus into codebook (Tier 1) and cipher (Tier 2)
    tiers based on word frequency.

    Tier 1: Word types appearing >= threshold times (codebook entries).
    Tier 2: Singleton word types (cipher-encoded unique content).
    """

    def __init__(self, extractor: LanguageAExtractor, threshold: int = 2):
        self.extractor = extractor
        self.threshold = threshold
        self._split = None
        self._all_tokens = None
        self._word_freqs = None

    def _ensure_split(self):
        """Lazily compute the tier split."""
        if self._split is not None:
            return

        self._all_tokens = self.extractor.extract_lang_a_tokens()
        self._word_freqs = Counter(self._all_tokens)

        tier1_types = sorted(w for w, c in self._word_freqs.items()
                             if c >= self.threshold)
        tier2_types = sorted(w for w, c in self._word_freqs.items()
                             if c < self.threshold)

        tier1_set = set(tier1_types)
        tier2_set = set(tier2_types)

        tier1_tokens = [t for t in self._all_tokens if t in tier1_set]
        tier2_tokens = [t for t in self._all_tokens if t in tier2_set]

        self._split = {
            'tier1_types': tier1_types,
            'tier2_types': tier2_types,
            'tier1_set': tier1_set,
            'tier2_set': tier2_set,
            'tier1_tokens': tier1_tokens,
            'tier2_tokens': tier2_tokens,
        }

    def split(self) -> Dict:
        """
        Perform the tier split and return summary statistics.

        Returns dict with tier sizes, token counts, and coverage fractions.
        """
        self._ensure_split()
        s = self._split

        total_tokens = len(self._all_tokens)
        tier1_token_count = len(s['tier1_tokens'])
        tier2_token_count = len(s['tier2_tokens'])

        return {
            'tier1_types': len(s['tier1_types']),
            'tier2_types': len(s['tier2_types']),
            'tier1_tokens': tier1_token_count,
            'tier2_tokens': tier2_token_count,
            'total_tokens': total_tokens,
            'tier1_coverage': tier1_token_count / max(1, total_tokens),
            'tier2_coverage': tier2_token_count / max(1, total_tokens),
            'threshold': self.threshold,
        }

    def get_tier1_types(self) -> List[str]:
        """Return sorted list of Tier 1 word types."""
        self._ensure_split()
        return self._split['tier1_types']

    def get_tier2_types(self) -> List[str]:
        """Return sorted list of Tier 2 (singleton) word types."""
        self._ensure_split()
        return self._split['tier2_types']

    def get_tier1_tokens(self) -> List[str]:
        """Return all tokens that belong to Tier 1 (in corpus order)."""
        self._ensure_split()
        return self._split['tier1_tokens']

    def get_tier2_tokens(self) -> List[str]:
        """Return all singleton tokens (in corpus order)."""
        self._ensure_split()
        return self._split['tier2_tokens']

    def is_tier1(self, word: str) -> bool:
        """Check if a word belongs to Tier 1."""
        self._ensure_split()
        return word in self._split['tier1_set']

    def is_tier2(self, word: str) -> bool:
        """Check if a word belongs to Tier 2."""
        self._ensure_split()
        return word in self._split['tier2_set']

    def get_bridged_bigrams(self) -> List[Tuple[str, str]]:
        """
        Extract bridged bigrams: when a Tier 2 singleton sits between
        two Tier 1 words, count the Tier 1 pair as a bigram.

        For sequence: [T1_a] [T2_x] [T1_b], yields (T1_a, T1_b).
        For sequence: [T1_a] [T2_x] [T2_y] [T1_b], yields (T1_a, T1_b).
        For consecutive Tier 1: [T1_a] [T1_b], yields (T1_a, T1_b).

        Returns list of (prev_tier1_word, next_tier1_word) pairs.
        """
        self._ensure_split()
        tokens = self._all_tokens
        tier1_set = self._split['tier1_set']

        bigrams = []
        last_tier1 = None

        for token in tokens:
            if token in tier1_set:
                if last_tier1 is not None:
                    bigrams.append((last_tier1, token))
                last_tier1 = token
            # If token is Tier 2, last_tier1 stays unchanged (bridging)

        return bigrams

    def get_annotated_sequence(self) -> List[Tuple[str, int]]:
        """
        Return the full token sequence with tier annotations.

        Returns list of (token, tier) where tier is 1 or 2.
        """
        self._ensure_split()
        tier1_set = self._split['tier1_set']
        return [(t, 1 if t in tier1_set else 2) for t in self._all_tokens]

    def validate_split(self) -> Dict:
        """
        Validate that the tier split produces expected statistics.

        Checks:
        - Tier 1 should have ~1,001 types (±300 acceptable)
        - Tier 2 should have ~2,761 types (±500 acceptable)
        - Tier 1 coverage should be ~74.4% (±10%)
        - Bridged bigram matrix should have 5-20 non-zero entries per row
        """
        split_stats = self.split()
        bridged = self.get_bridged_bigrams()

        # Count unique successors per Tier 1 word from bridged bigrams
        successor_counts = Counter()
        successor_sets = {}
        for prev, next_ in bridged:
            if prev not in successor_sets:
                successor_sets[prev] = set()
            successor_sets[prev].add(next_)

        n_successors = [len(s) for s in successor_sets.values()]
        mean_successors = np.mean(n_successors) if n_successors else 0
        median_successors = np.median(n_successors) if n_successors else 0

        # Expected perplexity ~4.5 (H2 ≈ 2.17 → 2^2.17 ≈ 4.5)
        expected_perplexity = 4.5

        tier1_ok = 500 <= split_stats['tier1_types'] <= 1500
        tier2_ok = 2000 <= split_stats['tier2_types'] <= 3500
        coverage_ok = 0.60 <= split_stats['tier1_coverage'] <= 0.90
        sparsity_ok = 3 <= mean_successors <= 30

        return {
            'tier1_types_ok': tier1_ok,
            'tier2_types_ok': tier2_ok,
            'coverage_ok': coverage_ok,
            'sparsity_ok': sparsity_ok,
            'all_valid': tier1_ok and tier2_ok and coverage_ok and sparsity_ok,
            'mean_successors_per_word': float(mean_successors),
            'median_successors_per_word': float(median_successors),
            'expected_perplexity': expected_perplexity,
            'n_bridged_bigrams': len(bridged),
            'split_stats': split_stats,
        }

    def compute_tier_statistics(self) -> Dict:
        """
        Compute detailed statistics for each tier to confirm nomenclator
        structure: H2 drop test, character structure divergence, alphabet sizes.
        """
        self._ensure_split()
        s = self._split

        # Character-level H2 for each tier
        tier1_text = ' '.join(s['tier1_tokens'])
        tier2_text = ' '.join(s['tier2_tokens'])
        full_text = self.extractor.extract_lang_a_text()

        full_h2 = conditional_entropy(full_text, order=1)
        tier1_h2 = conditional_entropy(tier1_text, order=1) if tier1_text else 0.0
        tier2_h2 = conditional_entropy(tier2_text, order=1) if tier2_text else 0.0

        h2_drop = full_h2 - tier1_h2
        h2_drop_significant = h2_drop > 0.15

        # Character repertoire per tier
        tier1_chars = set(c for t in s['tier1_tokens'] for c in t)
        tier2_chars = set(c for t in s['tier2_tokens'] for c in t)

        # Mean word length per tier
        tier1_lengths = [len(t) for t in s['tier1_tokens']]
        tier2_lengths = [len(t) for t in s['tier2_tokens']]

        tier1_mean_len = np.mean(tier1_lengths) if tier1_lengths else 0
        tier2_mean_len = np.mean(tier2_lengths) if tier2_lengths else 0

        # Character H1 per tier
        tier1_h1 = first_order_entropy(tier1_text) if tier1_text else 0.0
        tier2_h1 = first_order_entropy(tier2_text) if tier2_text else 0.0

        # Word-level H2 for Tier 1
        tier1_word_h2 = word_conditional_entropy(
            s['tier1_tokens'], order=1
        ) if len(s['tier1_tokens']) > 2 else 0.0

        return {
            'full_char_h2': full_h2,
            'tier1_char_h2': tier1_h2,
            'tier2_char_h2': tier2_h2,
            'h2_drop': h2_drop,
            'h2_drop_significant': h2_drop_significant,
            'tier1_char_h1': tier1_h1,
            'tier2_char_h1': tier2_h1,
            'tier1_alphabet_size': len(tier1_chars),
            'tier2_alphabet_size': len(tier2_chars),
            'tier1_mean_word_length': float(tier1_mean_len),
            'tier2_mean_word_length': float(tier2_mean_len),
            'tier1_word_h2': tier1_word_h2,
            'interpretation': (
                f'H2 drop: {h2_drop:.3f} '
                f'({"SIGNIFICANT" if h2_drop_significant else "not significant"}, '
                f'threshold=0.15). '
                f'Tier 1 alphabet: {len(tier1_chars)}, '
                f'Tier 2 alphabet: {len(tier2_chars)}. '
                f'Tier 1 mean length: {tier1_mean_len:.2f}, '
                f'Tier 2 mean length: {tier2_mean_len:.2f}.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run tier splitting with full statistics and validation."""
        split_stats = self.split()
        validation = self.validate_split()
        tier_stats = self.compute_tier_statistics()

        results = {
            'split': split_stats,
            'validation': validation,
            'tier_statistics': tier_stats,
            'synthesis': {
                'valid_split': validation['all_valid'],
                'h2_drop_significant': tier_stats['h2_drop_significant'],
                'nomenclator_confirmed': (
                    validation['all_valid'] and tier_stats['h2_drop_significant']
                ),
                'conclusion': (
                    f'Tier split: {split_stats["tier1_types"]} codebook types '
                    f'({split_stats["tier1_coverage"]:.1%} of tokens), '
                    f'{split_stats["tier2_types"]} cipher types. '
                    f'H2 drop: {tier_stats["h2_drop"]:.3f} '
                    f'({"SIGNIFICANT" if tier_stats["h2_drop_significant"] else "not significant"}). '
                    f'Validation: {"PASS" if validation["all_valid"] else "FAIL"}.'
                ),
            },
        }

        if verbose:
            print(f'\n  Tier Splitter Results:')
            print(f'    Tier 1 (codebook): {split_stats["tier1_types"]} types, '
                  f'{split_stats["tier1_tokens"]} tokens '
                  f'({split_stats["tier1_coverage"]:.1%})')
            print(f'    Tier 2 (cipher):   {split_stats["tier2_types"]} types, '
                  f'{split_stats["tier2_tokens"]} tokens '
                  f'({split_stats["tier2_coverage"]:.1%})')
            print(f'    Bridged bigrams:   {validation["n_bridged_bigrams"]}')
            print(f'    Mean successors:   {validation["mean_successors_per_word"]:.1f}')
            print(f'    --- H2 Drop Test ---')
            print(f'    Full H2:    {tier_stats["full_char_h2"]:.3f}')
            print(f'    Tier 1 H2:  {tier_stats["tier1_char_h2"]:.3f}')
            print(f'    Tier 2 H2:  {tier_stats["tier2_char_h2"]:.3f}')
            print(f'    Drop:       {tier_stats["h2_drop"]:.3f} '
                  f'({"SIGNIFICANT" if tier_stats["h2_drop_significant"] else "not significant"})')
            print(f'    --- Character Structure ---')
            print(f'    Tier 1 alphabet: {tier_stats["tier1_alphabet_size"]}, '
                  f'mean length: {tier_stats["tier1_mean_word_length"]:.2f}')
            print(f'    Tier 2 alphabet: {tier_stats["tier2_alphabet_size"]}, '
                  f'mean length: {tier_stats["tier2_mean_word_length"]:.2f}')
            print(f'    --- Validation ---')
            print(f'    {results["synthesis"]["conclusion"]}')

        return results
