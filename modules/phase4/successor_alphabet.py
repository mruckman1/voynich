"""
Approach 2: Successor as Alphabet (SAA) — Transition Matrix Matching
======================================================================
If the codebook model is correct, the Voynich word transition matrix
is a scrambled version of the Latin word-bigram matrix. Find the
permutation that unscrambles it.

Uses Hungarian algorithm and simulated annealing on a REDUCED vocabulary
(top N most-frequent words) for tractability. The full corpus may have
3000+ word types; we restrict to the top ~100 for matrix matching.
"""

import sys
import os
import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scipy.optimize import linear_sum_assignment

from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase4.latin_herbal_corpus import LatinHerbalCorpus


# Maximum vocabulary size for matrix matching (keeps runtime tractable)
MAX_VOCAB_SIZE = 80


def _build_reduced_transition_matrix(
    tokens: List[str], top_n: int = MAX_VOCAB_SIZE
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a word-level transition matrix restricted to the top-N
    most frequent words. Tokens not in the top-N are skipped during
    bigram counting (transitions are only counted between top-N words).
    """
    freqs = Counter(tokens)
    vocab = [w for w, _ in freqs.most_common(top_n)]
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


class SuccessorAlphabetAttack:
    """
    Find the permutation mapping Voynich word indices to Latin word
    indices by matching transition matrices.

    Operates on a REDUCED vocabulary (top MAX_VOCAB_SIZE words) for
    tractability. The full Language A corpus has thousands of types;
    the codebook hypothesis predicts the top ~100 words carry most
    of the structural information.
    """

    def __init__(self, extractor: LanguageAExtractor,
                 latin_corpus: LatinHerbalCorpus,
                 botanical_cribs: Optional[List[Dict]] = None):
        self.extractor = extractor
        self.latin_corpus = latin_corpus
        self.botanical_cribs = botanical_cribs or []

    def _get_reduced_matrices(self) -> Tuple[np.ndarray, List[str],
                                              np.ndarray, List[str]]:
        """Build reduced transition matrices for both Voynich and Latin."""
        voynich_tokens = self.extractor.extract_lang_a_tokens()
        latin_tokens = self.latin_corpus.get_tokens()

        voynich_mat, voynich_vocab = _build_reduced_transition_matrix(
            voynich_tokens, MAX_VOCAB_SIZE
        )
        latin_mat, latin_vocab = _build_reduced_transition_matrix(
            latin_tokens, MAX_VOCAB_SIZE
        )
        return voynich_mat, voynich_vocab, latin_mat, latin_vocab

    def hungarian_matching(self) -> Dict:
        """
        Use the Hungarian algorithm to find the optimal permutation
        minimizing row-wise cost between transition matrices.

        Both matrices are restricted to top-N vocabulary.
        """
        voynich_mat, voynich_vocab, latin_mat, latin_vocab = \
            self._get_reduced_matrices()

        n_v = len(voynich_vocab)
        n_l = len(latin_vocab)
        n = min(n_v, n_l)

        # Truncate to common size
        v_mat = voynich_mat[:n, :n]
        l_mat = latin_mat[:n, :n]

        # Build cost matrix: cost[i][j] = squared difference of
        # row i of Voynich vs row j of Latin (row-wise Frobenius)
        cost = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                cost[i][j] = float(np.sum((v_mat[i] - l_mat[j]) ** 2))

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_v and c < n_l:
                mapping[voynich_vocab[r]] = latin_vocab[c]

        total_cost = sum(cost[r][c] for r, c in zip(row_ind, col_ind))
        normalized_cost = total_cost / max(n * n, 1)

        return {
            'mapping': mapping,
            'total_cost': float(total_cost),
            'normalized_cost': float(normalized_cost),
            'n_matched': n,
            'quality': 'good' if normalized_cost < 0.3 else 'poor',
        }

    def simulated_annealing_matching(self, n_iter: int = 20000,
                                      seed: int = 42) -> Dict:
        """
        Simulated annealing search for the optimal permutation.
        Uses an incremental cost update for speed.
        """
        rng = random.Random(seed)

        voynich_mat, voynich_vocab, latin_mat, latin_vocab = \
            self._get_reduced_matrices()

        n_v = len(voynich_vocab)
        n_l = len(latin_vocab)

        # Initialize: random assignment of each Voynich word to a Latin word
        current = [rng.randint(0, n_l - 1) for _ in range(n_v)]

        # Apply crib constraints
        voynich_to_idx = {w: i for i, w in enumerate(voynich_vocab)}
        latin_to_idx = {w: i for i, w in enumerate(latin_vocab)}
        fixed_indices = set()

        for crib in self.botanical_cribs:
            v_word = crib.get('voynich_word', '')
            candidates = crib.get('candidates', [])
            if v_word in voynich_to_idx and candidates:
                v_idx = voynich_to_idx[v_word]
                for cand in candidates:
                    if cand in latin_to_idx:
                        current[v_idx] = latin_to_idx[cand]
                        if crib.get('confidence') == 'MODERATE':
                            fixed_indices.add(v_idx)
                        break

        # Vectorized cost function
        def cost_vec(perm):
            perm_arr = np.array(perm)
            # Build permuted Latin matrix from current assignment
            l_rows = latin_mat[perm_arr][:, perm_arr]  # n_v x n_v
            v_sub = voynich_mat[:n_v, :n_v]
            return float(np.sum((v_sub - l_rows) ** 2))

        # Incremental cost: only recompute rows/columns involving changed index
        def delta_cost(perm, idx, old_val, new_val):
            """Compute cost change from switching perm[idx] old_val -> new_val."""
            delta = 0.0
            perm_new = list(perm)
            perm_new[idx] = new_val
            # Affected: row idx and column idx in the difference matrix
            for j in range(n_v):
                lj = perm[j]
                # Row idx: voynich_mat[idx, j] vs latin_mat[new_val, lj]
                v_val = voynich_mat[idx, j]
                old_l = latin_mat[old_val, lj] if old_val < n_l and lj < n_l else 0
                new_l = latin_mat[new_val, lj] if new_val < n_l and lj < n_l else 0
                delta += (v_val - new_l) ** 2 - (v_val - old_l) ** 2

                if j != idx:
                    # Column idx: voynich_mat[j, idx] vs latin_mat[lj, new_val]
                    v_val2 = voynich_mat[j, idx]
                    old_l2 = latin_mat[lj, old_val] if lj < n_l and old_val < n_l else 0
                    new_l2 = latin_mat[lj, new_val] if lj < n_l and new_val < n_l else 0
                    delta += (v_val2 - new_l2) ** 2 - (v_val2 - old_l2) ** 2

            return delta

        current_cost = cost_vec(current)
        best = current[:]
        best_cost = current_cost

        non_fixed = [i for i in range(n_v) if i not in fixed_indices]
        if not non_fixed:
            non_fixed = list(range(n_v))

        T0 = max(1.0, current_cost / max(n_v, 1))
        T_min = 0.001

        for iteration in range(n_iter):
            T = T0 * (T_min / T0) ** (iteration / n_iter)

            idx = rng.choice(non_fixed)
            old_val = current[idx]
            new_val = rng.randint(0, n_l - 1)
            if new_val == old_val:
                continue

            d = delta_cost(current, idx, old_val, new_val)

            if d < 0 or rng.random() < math.exp(-d / max(T, 1e-10)):
                current[idx] = new_val
                current_cost += d

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

        mapping = {}
        for i, l_idx in enumerate(best):
            if i < len(voynich_vocab) and l_idx < len(latin_vocab):
                mapping[voynich_vocab[i]] = latin_vocab[l_idx]

        return {
            'mapping': mapping,
            'best_cost': float(best_cost),
            'normalized_cost': float(best_cost) / max(n_v * n_v, 1),
            'n_iterations': n_iter,
            'n_fixed_cribs': len(fixed_indices),
            'n_vocab': n_v,
            'quality': 'good' if best_cost / max(n_v * n_v, 1) < 0.3 else 'poor',
        }

    def evaluate_permutation(self, mapping: Dict) -> Dict:
        """Score a candidate permutation."""
        voynich_freqs = self.extractor.compute_word_frequencies()

        # Crib satisfaction
        crib_satisfied = 0
        crib_total = len(self.botanical_cribs)
        for crib in self.botanical_cribs:
            v_word = crib['voynich_word']
            if v_word in mapping and mapping[v_word] in crib['candidates']:
                crib_satisfied += 1

        # Function word plausibility
        latin_function = {'et', 'in', 'est', 'ad', 'cum', 'de', 'contra',
                         'habet', 'per', 'pro', 'fac', 'da', 'super'}
        top_5 = [w for w, _ in voynich_freqs.most_common(5)]
        func_matches = sum(1 for w in top_5 if mapping.get(w, '') in latin_function)

        decoded = self.decode_with_permutation(mapping, max_tokens=50)

        return {
            'crib_satisfaction': crib_satisfied,
            'crib_total': crib_total,
            'crib_rate': crib_satisfied / max(1, crib_total),
            'function_word_matches': func_matches,
            'decoded_sample': decoded,
        }

    def decode_with_permutation(self, mapping: Dict,
                                 max_tokens: int = 100) -> str:
        """Apply the codebook mapping to produce decoded text."""
        tokens = self.extractor.extract_lang_a_tokens()[:max_tokens]
        return ' '.join(mapping.get(t, f'[{t}]') for t in tokens)

    def run(self, verbose: bool = True) -> Dict:
        """Run the SAA attack with both methods."""
        if verbose:
            print(f'    Vocabulary restricted to top {MAX_VOCAB_SIZE} words')

        hungarian = self.hungarian_matching()
        sa = self.simulated_annealing_matching()

        hungarian_eval = self.evaluate_permutation(hungarian['mapping'])
        sa_eval = self.evaluate_permutation(sa['mapping'])

        if sa['normalized_cost'] < hungarian['normalized_cost']:
            best_method = 'simulated_annealing'
            best_mapping = sa['mapping']
            best_eval = sa_eval
            best_cost = sa['normalized_cost']
        else:
            best_method = 'hungarian'
            best_mapping = hungarian['mapping']
            best_eval = hungarian_eval
            best_cost = hungarian['normalized_cost']

        synthesis = {
            'best_method': best_method,
            'best_normalized_cost': best_cost,
            'plausible_decryption': best_cost < 0.3,
            'crib_satisfaction_rate': best_eval['crib_rate'],
            'decoded_sample': best_eval['decoded_sample'][:200],
            'conclusion': (
                f'SAA attack: best method={best_method}, '
                f'cost={best_cost:.3f}. '
                f'{"Plausible candidate decryption." if best_cost < 0.3 else "No plausible decryption found."}'
            ),
        }

        results = {
            'vocab_size_used': MAX_VOCAB_SIZE,
            'hungarian': {
                'cost': hungarian['total_cost'],
                'normalized_cost': hungarian['normalized_cost'],
                'quality': hungarian['quality'],
                'mapping_sample': dict(list(hungarian['mapping'].items())[:10]),
                'evaluation': hungarian_eval,
            },
            'simulated_annealing': {
                'cost': sa['best_cost'],
                'normalized_cost': sa['normalized_cost'],
                'quality': sa['quality'],
                'n_fixed_cribs': sa['n_fixed_cribs'],
                'mapping_sample': dict(list(sa['mapping'].items())[:10]),
                'evaluation': sa_eval,
            },
            'best_mapping': dict(list(best_mapping.items())[:20]),
            'synthesis': synthesis,
        }

        if verbose:
            print(f'\n  Approach 2: Successor Alphabet Attack (SAA)')
            print(f'    --- Hungarian Algorithm ---')
            print(f'    Normalized cost: {hungarian["normalized_cost"]:.4f}')
            print(f'    Quality: {hungarian["quality"]}')
            print(f'    --- Simulated Annealing ---')
            print(f'    Normalized cost: {sa["normalized_cost"]:.4f}')
            print(f'    Quality: {sa["quality"]}')
            print(f'    Fixed cribs: {sa["n_fixed_cribs"]}')
            print(f'    --- Best Result ---')
            print(f'    Method: {best_method}')
            print(f'    Crib satisfaction: {best_eval["crib_rate"]:.1%}')
            print(f'    Decoded sample: {best_eval["decoded_sample"][:80]}...')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
