"""
Constrained Simulated Annealing Attack (SAA) on Tier 1
========================================================
Multi-component cost function SAA that finds the permutation π mapping
Voynich Tier 1 words to Latin words such that:

  Cost = α × D(T_v, π(T_L))     [Frobenius matrix distance]
       + β × Crib_violations     [broken botanical/rank constraints]
       + γ × Rank_deviation      [frequency rank distance]
       + δ × Topic_incoherence   [NMF topic penalty]

Replaces Phase 4's simple SAA which:
  - Used only 80 words (now ~1,001)
  - Had only 20,000 iterations (now 100,000)
  - Had no rank pairing (all words got same candidate list)
  - Had no NMF topic penalties

Phase 5  ·  Voynich Convergence Attack
"""

import sys
import os
import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase5.nmf_scaffold import NMFScaffold


# Known Latin herbal phrases for validation
RECOGNIZABLE_PHRASES = [
    'calida et sicca',
    'frigida et humida',
    'calida et humida',
    'frigida et sicca',
    'in primo gradu',
    'in secundo gradu',
    'in tertio gradu',
    'in quarto gradu',
    'habet virtutem',
    'valet contra',
    'contra dolorem',
    'recipe',
    'et est probatum',
    'et sanabitur',
    'cum aqua',
    'cum vino',
    'cum melle',
    'da bibere',
    'et coque',
    'contra venenum',
]


class ConstrainedSAA:
    """
    Multi-component constrained SAA for Tier 1 codebook decryption.

    Operates on the full Tier 1 vocabulary (~1,001 types) with a
    four-component cost function: matrix distance, crib violations,
    rank deviation, and NMF topic incoherence.
    """

    def __init__(
        self,
        voynich_matrix: np.ndarray,
        voynich_vocab: List[str],
        latin_matrix: np.ndarray,
        latin_vocab: List[str],
        rank_pairs: Dict[str, List[str]],
        nmf_scaffold: Optional[NMFScaffold] = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3,
        delta: float = 0.2,
    ):
        """
        Parameters:
            voynich_matrix: Tier 1 transition matrix (N_v × N_v)
            voynich_vocab:  Tier 1 word list (length N_v)
            latin_matrix:   Latin transition matrix (N_l × N_l)
            latin_vocab:    Latin word list (length N_l)
            rank_pairs:     {voynich_word: [latin_candidate1, ...]}
            nmf_scaffold:   Optional NMFScaffold for topic coherence
            alpha:          Weight for matrix distance
            beta:           Weight for crib violations
            gamma:          Weight for rank deviation
            delta:          Weight for topic incoherence
        """
        self.v_matrix = voynich_matrix
        self.v_vocab = voynich_vocab
        self.l_matrix = latin_matrix
        self.l_vocab = latin_vocab
        self.rank_pairs = rank_pairs
        self.nmf_scaffold = nmf_scaffold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.n_v = len(voynich_vocab)
        self.n_l = len(latin_vocab)
        self.v_word_to_idx = {w: i for i, w in enumerate(voynich_vocab)}
        self.l_word_to_idx = {w: i for i, w in enumerate(latin_vocab)}

        # Build candidate index sets for efficient crib checking
        self._candidate_indices = {}
        for v_word, candidates in rank_pairs.items():
            if v_word in self.v_word_to_idx:
                v_idx = self.v_word_to_idx[v_word]
                l_indices = set()
                for c in candidates:
                    if c in self.l_word_to_idx:
                        l_indices.add(self.l_word_to_idx[c])
                if l_indices:
                    self._candidate_indices[v_idx] = l_indices

        # Voynich word rank lookup
        self._v_rank = {w: i for i, w in enumerate(voynich_vocab)}
        self._l_rank = {w: i for i, w in enumerate(latin_vocab)}

        # NMF shared-topic matrix (precomputed for speed)
        self._topic_matrix = None
        if nmf_scaffold is not None:
            self._topic_matrix = nmf_scaffold.build_topic_coherence_matrix(voynich_vocab)

    def cost(self, perm: List[int]) -> float:
        """
        Compute the full multi-component cost of a permutation.

        Parameters:
            perm: List of length N_v where perm[i] is the Latin word index
                  that Voynich word i maps to.
        """
        c = 0.0
        if self.alpha > 0:
            c += self.alpha * self._matrix_distance(perm)
        if self.beta > 0:
            c += self.beta * self._crib_violations(perm)
        if self.gamma > 0:
            c += self.gamma * self._rank_deviation(perm)
        if self.delta > 0 and self._topic_matrix is not None:
            c += self.delta * self._topic_incoherence(perm)
        return c

    def _matrix_distance(self, perm: List[int]) -> float:
        """
        Frobenius norm of the difference between the Voynich transition
        matrix and the permuted Latin matrix.

        D = ||T_v - π(T_L)||_F
        """
        perm_arr = np.array(perm)

        # Clamp indices to valid range
        valid = perm_arr < self.n_l
        if not valid.all():
            perm_arr = np.clip(perm_arr, 0, self.n_l - 1)

        # Build permuted Latin matrix
        l_permuted = self.l_matrix[perm_arr][:, perm_arr]
        diff = self.v_matrix[:self.n_v, :self.n_v] - l_permuted[:self.n_v, :self.n_v]
        return float(np.sum(diff ** 2))

    def _crib_violations(self, perm: List[int]) -> float:
        """
        Count how many crib constraints are violated.

        A crib is violated if perm[v_idx] is not in the candidate set
        for that Voynich word.
        """
        violations = 0
        for v_idx, valid_l_indices in self._candidate_indices.items():
            if v_idx < len(perm) and perm[v_idx] not in valid_l_indices:
                violations += 1
        return float(violations)

    def _rank_deviation(self, perm: List[int]) -> float:
        """
        Sum of |rank_voynich(w) - rank_latin(π(w))| / N.

        Penalizes mappings that pair words of very different frequency ranks.
        """
        total = 0.0
        for v_idx in range(self.n_v):
            l_idx = perm[v_idx]
            total += abs(v_idx - l_idx)
        return total / max(self.n_v, 1)

    def _topic_incoherence(self, perm: List[int]) -> float:
        """
        Penalty for NMF topic violations.

        For each pair of Voynich words sharing an NMF topic, check if
        their Latin mappings co-occur in the Latin corpus. If not, add
        a penalty.

        Uses precomputed topic matrix for speed — only checks pairs
        where topic_matrix[i][j] = 1 (shared topic).
        """
        if self._topic_matrix is None or self.nmf_scaffold is None:
            return 0.0

        latin_cooc = self.nmf_scaffold.build_latin_cooccurrence_set()
        penalty = 0.0
        n_checked = 0

        # Sample pairs to avoid O(N^2) on every cost evaluation
        # Check a random subset of shared-topic pairs
        for i in range(self.n_v):
            for j in range(i + 1, min(i + 20, self.n_v)):
                if self._topic_matrix[i][j] > 0:
                    l_i = perm[i]
                    l_j = perm[j]
                    if l_i < self.n_l and l_j < self.n_l:
                        l_word_i = self.l_vocab[l_i]
                        l_word_j = self.l_vocab[l_j]
                        if l_word_i in latin_cooc:
                            if l_word_j not in latin_cooc[l_word_i]:
                                penalty += 1.0
                        else:
                            penalty += 1.0
                        n_checked += 1

        return penalty / max(n_checked, 1)

    def _delta_cost(self, perm: List[int], idx: int,
                     old_val: int, new_val: int) -> float:
        """
        Compute the incremental cost change from switching perm[idx]
        from old_val to new_val. Only recomputes affected components.
        """
        delta = 0.0

        # Matrix distance delta (rows and columns involving idx)
        if self.alpha > 0:
            for j in range(self.n_v):
                lj = perm[j]
                if lj >= self.n_l:
                    continue

                # Row idx
                v_val = self.v_matrix[idx, j]
                old_l = self.l_matrix[old_val, lj] if old_val < self.n_l else 0
                new_l = self.l_matrix[new_val, lj] if new_val < self.n_l else 0
                delta += self.alpha * ((v_val - new_l) ** 2 - (v_val - old_l) ** 2)

                if j != idx:
                    # Column idx
                    v_val2 = self.v_matrix[j, idx]
                    old_l2 = self.l_matrix[lj, old_val] if old_val < self.n_l else 0
                    new_l2 = self.l_matrix[lj, new_val] if new_val < self.n_l else 0
                    delta += self.alpha * ((v_val2 - new_l2) ** 2 - (v_val2 - old_l2) ** 2)

        # Crib violation delta
        if self.beta > 0 and idx in self._candidate_indices:
            valid = self._candidate_indices[idx]
            old_violated = 1.0 if old_val not in valid else 0.0
            new_violated = 1.0 if new_val not in valid else 0.0
            delta += self.beta * (new_violated - old_violated)

        # Rank deviation delta
        if self.gamma > 0:
            old_dev = abs(idx - old_val)
            new_dev = abs(idx - new_val)
            delta += self.gamma * (new_dev - old_dev) / max(self.n_v, 1)

        return delta

    def run(self, n_iter: int = 100000, T_0: float = 1.0,
            T_final: float = 0.001, seed: int = 42,
            log_interval: int = 10000) -> Dict:
        """
        Run the constrained SAA with exponential cooling schedule.

        Parameters:
            n_iter:       Number of iterations (default 100,000)
            T_0:          Initial temperature
            T_final:      Final temperature
            seed:         Random seed
            log_interval: Log cost every N iterations

        Returns:
            Dict with mapping, cost trajectory, decoded samples.
        """
        rng = random.Random(seed)

        # Initialize: assign each Voynich word to its best rank-paired candidate
        current = []
        fixed_indices = set()

        for v_idx in range(self.n_v):
            v_word = self.v_vocab[v_idx]
            if v_word in self.rank_pairs and self.rank_pairs[v_word]:
                # Use first candidate
                cand = self.rank_pairs[v_word][0]
                if cand in self.l_word_to_idx:
                    current.append(self.l_word_to_idx[cand])
                else:
                    current.append(min(v_idx, self.n_l - 1))
            else:
                current.append(min(v_idx, self.n_l - 1))

        current_cost = self.cost(current)
        best = current[:]
        best_cost = current_cost

        # Non-fixed indices (all indices are mutable, but cribs guide initialization)
        non_fixed = list(range(self.n_v))

        cost_trajectory = [(0, current_cost)]

        for iteration in range(1, n_iter + 1):
            T = T_0 * (T_final / T_0) ** (iteration / n_iter)

            idx = rng.choice(non_fixed)
            old_val = current[idx]
            new_val = rng.randint(0, self.n_l - 1)
            if new_val == old_val:
                continue

            d = self._delta_cost(current, idx, old_val, new_val)

            if d < 0 or rng.random() < math.exp(-d / max(T, 1e-10)):
                current[idx] = new_val
                current_cost += d

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            if iteration % log_interval == 0:
                cost_trajectory.append((iteration, current_cost))

        cost_trajectory.append((n_iter, best_cost))

        # Build mapping
        mapping = {}
        for i, l_idx in enumerate(best):
            if i < len(self.v_vocab) and l_idx < len(self.l_vocab):
                mapping[self.v_vocab[i]] = self.l_vocab[l_idx]

        return {
            'mapping': mapping,
            'best_cost': float(best_cost),
            'normalized_cost': float(best_cost) / max(self.n_v * self.n_v, 1),
            'n_iterations': n_iter,
            'n_voynich_words': self.n_v,
            'n_latin_words': self.n_l,
            'cost_trajectory': cost_trajectory,
            'parameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'delta': self.delta,
                'T_0': T_0,
                'T_final': T_final,
            },
        }

    def validate_mapping(self, mapping: Dict,
                          page_tokens: Dict[str, List[str]]) -> Dict:
        """
        Validate the mapping by checking for recognizable Latin herbal
        phrases in decoded page text.

        Parameters:
            mapping:     {voynich_word: latin_word} from the SAA
            page_tokens: {folio_id: [token, ...]} from LanguageAExtractor

        Returns dict with phrase detection results per page.
        """
        page_results = {}
        best_page = None
        best_count = 0

        for folio, tokens in page_tokens.items():
            decoded = self.decode_text(mapping, tokens)
            decoded_lower = decoded.lower()

            found_phrases = []
            for phrase in RECOGNIZABLE_PHRASES:
                if phrase in decoded_lower:
                    found_phrases.append(phrase)

            page_results[folio] = {
                'n_phrases_found': len(found_phrases),
                'phrases': found_phrases,
                'decoded_preview': decoded[:200],
            }

            if len(found_phrases) > best_count:
                best_count = len(found_phrases)
                best_page = folio

        passes_threshold = best_count >= 3

        return {
            'page_results': page_results,
            'best_page': best_page,
            'best_phrase_count': best_count,
            'passes_threshold': passes_threshold,
            'n_pages_tested': len(page_tokens),
            'conclusion': (
                f'Validation: {best_count} recognizable phrases on best page '
                f'({best_page}). '
                f'{"PASSES" if passes_threshold else "FAILS"} threshold (≥3).'
            ),
        }

    def decode_text(self, mapping: Dict, tokens: List[str]) -> str:
        """
        Apply the Tier 1 mapping to a token sequence.
        Unmapped words (Tier 2 singletons) are marked as [SINGLETON].
        """
        return ' '.join(mapping.get(t, f'[{t}]') for t in tokens)

    def evaluate_crib_satisfaction(self, mapping: Dict) -> Dict:
        """Check how many rank-paired cribs are satisfied by the mapping."""
        satisfied = 0
        total = 0
        details = []

        for v_word, candidates in self.rank_pairs.items():
            if not candidates:
                continue
            total += 1
            mapped_to = mapping.get(v_word, '')
            is_satisfied = mapped_to in candidates
            if is_satisfied:
                satisfied += 1
            details.append({
                'voynich_word': v_word,
                'mapped_to': mapped_to,
                'candidates': candidates[:3],
                'satisfied': is_satisfied,
            })

        return {
            'satisfied': satisfied,
            'total': total,
            'rate': satisfied / max(1, total),
            'details': details[:20],
        }
