"""
Tier 1 Transition Matrix Builder
==================================
Builds the word-level bigram transition matrix for Tier 1 (codebook)
words only, with singleton bridging.

When a Tier 2 singleton appears between two Tier 1 words, the bigram
[T1_prev] → [T1_next] is counted. The singleton acts as a transparent
bridge — it contributes to the next-word probability of the flanking
Tier 1 words but is not itself in the matrix.

Key metric: T_v should have ~1,001 rows, each with 5-20 non-zero
entries (given H2 ≈ 2.17, expected perplexity ≈ 4.5 successors per word).

Phase 5  ·  Voynich Convergence Attack
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

from modules.phase5.tier_splitter import TierSplitter

class Tier1MatrixBuilder:
    """
    Build and validate the Tier 1 word-level transition matrix with
    singleton bridging.
    """

    def __init__(self, splitter: TierSplitter):
        self.splitter = splitter
        self._matrix = None
        self._vocab = None
        self._word_to_idx = None

    def build_voynich_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build T_v: the Tier 1 word-level bigram transition matrix.

        Uses bridged bigrams from TierSplitter — singletons between
        two Tier 1 words are transparent, so [T1_a] [T2_x] [T1_b]
        counts as a bigram (T1_a, T1_b).

        Returns:
            (matrix, vocab) where matrix[i][j] = P(word_j | word_i)
            and vocab is the list of Tier 1 word types.
        """
        if self._matrix is not None:
            return self._matrix, self._vocab

        self._vocab = self.splitter.get_tier1_types()
        self._word_to_idx = {w: i for i, w in enumerate(self._vocab)}
        n = len(self._vocab)

        counts = np.zeros((n, n), dtype=float)

        bridged = self.splitter.get_bridged_bigrams()
        for prev_word, next_word in bridged:
            if prev_word in self._word_to_idx and next_word in self._word_to_idx:
                i = self._word_to_idx[prev_word]
                j = self._word_to_idx[next_word]
                counts[i][j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._matrix = counts / row_sums

        return self._matrix, self._vocab

    def get_raw_counts(self) -> Tuple[np.ndarray, List[str]]:
        """Return the raw bigram count matrix (unnormalized)."""
        vocab = self.splitter.get_tier1_types()
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        counts = np.zeros((n, n), dtype=float)
        bridged = self.splitter.get_bridged_bigrams()
        for prev_word, next_word in bridged:
            if prev_word in word_to_idx and next_word in word_to_idx:
                counts[word_to_idx[prev_word]][word_to_idx[next_word]] += 1

        return counts, vocab

    def validate_matrix(self) -> Dict:
        """
        Validate the transition matrix against expected properties.

        Checks:
        - Row count matches Tier 1 vocabulary size
        - Each row has 5-20 non-zero entries (perplexity ~4.5)
        - Matrix is properly normalized (rows sum to 1)
        - No pathological sparsity or density
        """
        matrix, vocab = self.build_voynich_matrix()
        n = len(vocab)

        nonzero_per_row = np.array([(matrix[i] > 0).sum() for i in range(n)])
        mean_nonzero = float(np.mean(nonzero_per_row))
        median_nonzero = float(np.median(nonzero_per_row))

        row_entropies = []
        for i in range(n):
            row = matrix[i]
            h = -sum(p * math.log2(p) for p in row if p > 0)
            row_entropies.append(h)
        row_entropies = np.array(row_entropies)

        mean_entropy = float(np.mean(row_entropies))
        mean_perplexity = float(2 ** mean_entropy) if mean_entropy > 0 else 0

        row_sums = matrix.sum(axis=1)
        rows_normalized = bool(np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-6))

        active_rows = int((nonzero_per_row > 0).sum())

        total_entries = n * n
        nonzero_total = int((matrix > 0).sum())
        sparsity = 1.0 - (nonzero_total / max(1, total_entries))

        sparsity_ok = 5 <= mean_nonzero <= 30
        perplexity_ok = 2.0 <= mean_perplexity <= 10.0

        return {
            'n_rows': n,
            'active_rows': active_rows,
            'mean_nonzero_per_row': mean_nonzero,
            'median_nonzero_per_row': median_nonzero,
            'mean_row_entropy': mean_entropy,
            'mean_perplexity': mean_perplexity,
            'rows_normalized': rows_normalized,
            'sparsity': sparsity,
            'total_nonzero_entries': nonzero_total,
            'sparsity_ok': sparsity_ok,
            'perplexity_ok': perplexity_ok,
            'valid': sparsity_ok and perplexity_ok and rows_normalized,
        }

    def compute_matrix_statistics(self) -> Dict:
        """Compute detailed statistics about the transition matrix."""
        matrix, vocab = self.build_voynich_matrix()
        n = len(vocab)

        top_transitions = []
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > 0.1:
                    top_transitions.append({
                        'from': vocab[i],
                        'to': vocab[j],
                        'probability': float(matrix[i][j]),
                    })
        top_transitions.sort(key=lambda x: -x['probability'])

        out_degrees = [(vocab[i], int((matrix[i] > 0).sum()))
                       for i in range(n)]
        out_degrees.sort(key=lambda x: -x[1])

        frobenius = float(np.linalg.norm(matrix, 'fro'))

        return {
            'matrix_shape': (n, n),
            'frobenius_norm': frobenius,
            'top_transitions': top_transitions[:20],
            'most_connected': out_degrees[:20],
            'least_connected': out_degrees[-10:],
        }

    def run(self, verbose: bool = True) -> Dict:
        """Build and validate the Tier 1 transition matrix."""
        matrix, vocab = self.build_voynich_matrix()
        validation = self.validate_matrix()
        stats = self.compute_matrix_statistics()

        results = {
            'matrix_shape': validation['n_rows'],
            'validation': validation,
            'statistics': stats,
            'synthesis': {
                'valid': validation['valid'],
                'conclusion': (
                    f'Tier 1 transition matrix: {validation["n_rows"]}x'
                    f'{validation["n_rows"]}, '
                    f'{validation["active_rows"]} active rows, '
                    f'mean perplexity={validation["mean_perplexity"]:.1f}, '
                    f'sparsity={validation["sparsity"]:.3f}. '
                    f'Validation: {"PASS" if validation["valid"] else "FAIL"}.'
                ),
            },
        }

        if verbose:
            print(f'\n  Tier 1 Transition Matrix:')
            print(f'    Shape:           {validation["n_rows"]}x{validation["n_rows"]}')
            print(f'    Active rows:     {validation["active_rows"]}')
            print(f'    Mean nonzero/row:{validation["mean_nonzero_per_row"]:.1f}')
            print(f'    Mean perplexity: {validation["mean_perplexity"]:.1f}')
            print(f'    Mean row entropy:{validation["mean_row_entropy"]:.3f}')
            print(f'    Sparsity:        {validation["sparsity"]:.3f}')
            print(f'    Normalized:      {validation["rows_normalized"]}')
            print(f'    Valid:           {validation["valid"]}')
            if stats['top_transitions']:
                print(f'    --- Top Transitions ---')
                for t in stats['top_transitions'][:5]:
                    print(f'      {t["from"]} → {t["to"]}: {t["probability"]:.3f}')

        return results
