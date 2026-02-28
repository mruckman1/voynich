"""
Homophone Merger — Path B Step 2
==================================
If HomophoneDetector finds significant variant groups, merge them and
re-measure corpus statistics. If the hypothesis holds:
  - Zipf exponent should rise toward 1.0 (from 0.743)
  - TTR should normalize
  - Vocabulary should shrink from 1001 to 200-400 groups

Phase 6  ·  Voynich Convergence Attack
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

from modules.statistical_analysis import (
    word_conditional_entropy, zipf_analysis, first_order_entropy,
    conditional_entropy,
)
from modules.phase5.tier_splitter import TierSplitter
from modules.phase4.lang_a_extractor import LanguageAExtractor

class HomophoneMerger:
    """
    Merge homophone variant groups and re-compute corpus statistics
    to validate or reject the homophonic substitution hypothesis.
    """

    def __init__(self, groups: Dict[str, List[str]],
                 splitter: TierSplitter,
                 extractor: LanguageAExtractor):
        """
        Parameters:
            groups: {canonical_word: [variant1, variant2, ...]} from HomophoneDetector
            splitter: TierSplitter for corpus access
            extractor: LanguageAExtractor for full token sequence
        """
        self.groups = groups
        self.splitter = splitter
        self.extractor = extractor
        self._merge_map = None
        self._merged_tokens = None

    def build_merge_map(self) -> Dict[str, str]:
        """Build {any_variant: canonical_form} lookup."""
        if self._merge_map is not None:
            return self._merge_map

        self._merge_map = {}
        for canonical, variants in self.groups.items():
            for variant in variants:
                self._merge_map[variant] = canonical
        return self._merge_map

    def merge_corpus(self) -> List[str]:
        """
        Replace all variant tokens in the Tier 1 corpus with their
        canonical form. Non-grouped words pass through unchanged.

        Returns the merged token sequence.
        """
        if self._merged_tokens is not None:
            return self._merged_tokens

        merge_map = self.build_merge_map()
        original_tokens = self.splitter.get_tier1_tokens()

        self._merged_tokens = [
            merge_map.get(t, t) for t in original_tokens
        ]
        return self._merged_tokens

    def compute_original_statistics(self) -> Dict:
        """Compute statistics on the original (unmerged) Tier 1 corpus."""
        tokens = self.splitter.get_tier1_tokens()
        text = ' '.join(tokens)
        zipf = zipf_analysis(tokens)

        return {
            'n_tokens': len(tokens),
            'n_types': len(set(tokens)),
            'ttr': len(set(tokens)) / max(1, len(tokens)),
            'zipf_exponent': zipf['zipf_exponent'],
            'zipf_r_squared': zipf['r_squared'],
            'word_h2': word_conditional_entropy(tokens, order=1),
            'char_h2': conditional_entropy(text, order=1),
        }

    def compute_merged_statistics(self) -> Dict:
        """Compute statistics on the merged corpus."""
        tokens = self.merge_corpus()
        text = ' '.join(tokens)
        zipf = zipf_analysis(tokens)

        return {
            'n_tokens': len(tokens),
            'n_types': len(set(tokens)),
            'ttr': len(set(tokens)) / max(1, len(tokens)),
            'zipf_exponent': zipf['zipf_exponent'],
            'zipf_r_squared': zipf['r_squared'],
            'word_h2': word_conditional_entropy(tokens, order=1),
            'char_h2': conditional_entropy(text, order=1),
        }

    def compare_before_after(self) -> Dict:
        """Side-by-side comparison of pre/post merge statistics."""
        before = self.compute_original_statistics()
        after = self.compute_merged_statistics()

        return {
            'before': before,
            'after': after,
            'deltas': {
                'types_reduction': before['n_types'] - after['n_types'],
                'types_reduction_pct': (before['n_types'] - after['n_types']) / max(1, before['n_types']),
                'ttr_change': after['ttr'] - before['ttr'],
                'zipf_change': after['zipf_exponent'] - before['zipf_exponent'],
                'word_h2_change': after['word_h2'] - before['word_h2'],
                'char_h2_change': after['char_h2'] - before['char_h2'],
            },
        }

    def build_merged_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build word-level transition matrix on the merged vocabulary.
        Returns (matrix, vocab) for use in ReducedSAA.
        """
        tokens = self.merge_corpus()
        freqs = Counter(tokens)
        vocab = [w for w, _ in freqs.most_common()]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        counts = np.zeros((n, n), dtype=float)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 in word_to_idx and w2 in word_to_idx:
                counts[word_to_idx[w1]][word_to_idx[w2]] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = counts / row_sums

        return matrix, vocab

    def validate_hypothesis(self) -> Dict:
        """
        Check if merging produces the expected statistical improvements
        that would confirm homophonic substitution.

        Expected changes if homophony is real:
        - Zipf exponent rises toward 1.0 (from 0.743)
        - Vocabulary reduces to 200-400 groups
        - TTR decreases (fewer types for same token count)
        """
        comparison = self.compare_before_after()
        before = comparison['before']
        after = comparison['after']
        deltas = comparison['deltas']

        zipf_improved = after['zipf_exponent'] > before['zipf_exponent']
        zipf_toward_1 = abs(after['zipf_exponent'] - 1.0) < abs(before['zipf_exponent'] - 1.0)
        vocab_reduced_significantly = deltas['types_reduction_pct'] > 0.3
        vocab_in_range = 100 <= after['n_types'] <= 600

        hypothesis_supported = (
            zipf_improved and
            zipf_toward_1 and
            vocab_reduced_significantly
        )

        return {
            'zipf_improved': zipf_improved,
            'zipf_toward_1': zipf_toward_1,
            'vocab_reduced_significantly': vocab_reduced_significantly,
            'vocab_in_range': vocab_in_range,
            'hypothesis_supported': hypothesis_supported,
            'before_zipf': before['zipf_exponent'],
            'after_zipf': after['zipf_exponent'],
            'before_types': before['n_types'],
            'after_types': after['n_types'],
            'reduction_pct': deltas['types_reduction_pct'],
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run full merge analysis."""
        comparison = self.compare_before_after()
        validation = self.validate_hypothesis()

        results = {
            'n_groups': len(self.groups),
            'n_merge_entries': len(self.build_merge_map()),
            'comparison': comparison,
            'validation': validation,
            'synthesis': {
                'hypothesis_supported': validation['hypothesis_supported'],
                'conclusion': (
                    f'Merged {comparison["before"]["n_types"]} types → '
                    f'{comparison["after"]["n_types"]} types '
                    f'({validation["reduction_pct"]:.1%} reduction). '
                    f'Zipf: {validation["before_zipf"]:.3f} → {validation["after_zipf"]:.3f}. '
                    f'Hypothesis {"SUPPORTED" if validation["hypothesis_supported"] else "NOT SUPPORTED"}.'
                ),
            },
        }

        if verbose:
            b = comparison['before']
            a = comparison['after']
            d = comparison['deltas']
            print(f'\n  Homophone Merger:')
            print(f'    Groups merged: {len(self.groups)}')
            print(f'    --- Before/After ---')
            print(f'    Types:      {b["n_types"]} → {a["n_types"]} '
                  f'(Δ={d["types_reduction"]}, {d["types_reduction_pct"]:.1%} reduction)')
            print(f'    TTR:        {b["ttr"]:.4f} → {a["ttr"]:.4f}')
            print(f'    Zipf:       {b["zipf_exponent"]:.3f} → {a["zipf_exponent"]:.3f} '
                  f'(Δ={d["zipf_change"]:+.3f})')
            print(f'    Word H2:    {b["word_h2"]:.3f} → {a["word_h2"]:.3f}')
            print(f'    Char H2:    {b["char_h2"]:.3f} → {a["char_h2"]:.3f}')
            print(f'    --- Validation ---')
            print(f'    Zipf improved:        {validation["zipf_improved"]}')
            print(f'    Zipf toward 1.0:      {validation["zipf_toward_1"]}')
            print(f'    Vocab reduced > 30%:  {validation["vocab_reduced_significantly"]}')
            print(f'    Vocab in range 100-600: {validation["vocab_in_range"]}')
            print(f'    HYPOTHESIS: '
                  f'{"SUPPORTED" if validation["hypothesis_supported"] else "NOT SUPPORTED"}')

        return results
