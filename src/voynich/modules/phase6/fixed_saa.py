"""
Fixed Simulated Annealing Attack — Path A
===========================================
Fixes Phase 5's SAA with three structural changes:

1. Bijection enforcement: swap-based mutations maintain a 1:1 mapping
   at all times (no surjective collapse)
2. Top-N lock: the N most frequent Voynich words are hard-locked to
   their best rank-paired Latin candidate and cannot be modified
3. Inverted weights: β=2.0 (cribs dominant), γ=1.0 (rank strong),
   α=0.3 (matrix as regularizer), δ=0.2 (topics)

Phase 6  ·  Voynich Convergence Attack
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

from voynich.modules.phase5.nmf_scaffold import NMFScaffold
from voynich.modules.phase5.constrained_saa import ConstrainedSAA

RECOGNIZABLE_PHRASES = [
    'calida et sicca', 'frigida et humida', 'calida et humida',
    'frigida et sicca', 'in primo gradu', 'in secundo gradu',
    'in tertio gradu', 'in quarto gradu', 'habet virtutem',
    'valet contra', 'contra dolorem', 'recipe', 'et est probatum',
    'et sanabitur', 'cum aqua', 'cum vino', 'cum melle',
    'da bibere', 'et coque', 'contra venenum',
]

class FixedSAA(ConstrainedSAA):
    """
    Bijection-enforced SAA with inverted cost weights and top-N locking.

    Key differences from Phase 5 ConstrainedSAA:
    - Swap-based mutations (exchange perm[i] and perm[j]) instead of
      single-point mutations, maintaining bijectivity
    - Top-N words locked to their best rank-paired candidate
    - Inverted weights: cribs > rank > matrix (was matrix > cribs > rank)
    """

    def __init__(
        self,
        voynich_matrix: np.ndarray,
        voynich_vocab: List[str],
        latin_matrix: np.ndarray,
        latin_vocab: List[str],
        rank_pairs: Dict[str, List[str]],
        nmf_scaffold: Optional[NMFScaffold] = None,
        n_locked: int = 50,
        alpha: float = 0.3,
        beta: float = 2.0,
        gamma: float = 1.0,
        delta: float = 0.2,
    ):
        super().__init__(
            voynich_matrix, voynich_vocab, latin_matrix, latin_vocab,
            rank_pairs, nmf_scaffold, alpha, beta, gamma, delta,
        )
        self.n_locked = n_locked

    def _initialize_bijective(self, rng: random.Random) -> Tuple[List[int], Set[int]]:
        """
        Build initial bijective permutation.

        Top-N words get their best rank-paired candidate (with priority).
        Remaining words get the next-best unused candidate.
        Unmatched words get the next available Latin index.

        Returns:
            (permutation, locked_indices) — locked indices cannot be modified.
        """
        perm = [-1] * self.n_v
        used_l_indices: Set[int] = set()
        locked_indices: Set[int] = set()

        for v_idx in range(min(self.n_locked, self.n_v)):
            v_word = self.v_vocab[v_idx]
            assigned = False

            if v_word in self.rank_pairs:
                for cand in self.rank_pairs[v_word]:
                    if cand in self.l_word_to_idx:
                        l_idx = self.l_word_to_idx[cand]
                        if l_idx not in used_l_indices:
                            perm[v_idx] = l_idx
                            used_l_indices.add(l_idx)
                            locked_indices.add(v_idx)
                            assigned = True
                            break

            if not assigned:
                for l_idx in range(self.n_l):
                    if l_idx not in used_l_indices:
                        perm[v_idx] = l_idx
                        used_l_indices.add(l_idx)
                        locked_indices.add(v_idx)
                        break

        for v_idx in range(self.n_v):
            if perm[v_idx] != -1:
                continue

            v_word = self.v_vocab[v_idx]
            assigned = False

            if v_word in self.rank_pairs:
                for cand in self.rank_pairs[v_word]:
                    if cand in self.l_word_to_idx:
                        l_idx = self.l_word_to_idx[cand]
                        if l_idx not in used_l_indices:
                            perm[v_idx] = l_idx
                            used_l_indices.add(l_idx)
                            assigned = True
                            break

            if not assigned:
                for l_idx in range(self.n_l):
                    if l_idx not in used_l_indices:
                        perm[v_idx] = l_idx
                        used_l_indices.add(l_idx)
                        break

        for v_idx in range(self.n_v):
            if perm[v_idx] == -1:
                perm[v_idx] = v_idx % self.n_l

        return perm, locked_indices

    def _delta_cost_swap(self, perm: List[int], i: int, j: int) -> float:
        """
        Compute incremental cost change from swapping perm[i] and perm[j].
        Only recomputes affected row/column contributions.
        """
        old_li, old_lj = perm[i], perm[j]
        delta = 0.0

        if self.alpha > 0:
            for k in range(self.n_v):
                lk = perm[k]
                if lk >= self.n_l:
                    continue
                if k == i or k == j:
                    continue

                vi_k = self.v_matrix[i, k]
                old_l = self.l_matrix[old_li, lk] if old_li < self.n_l else 0
                new_l = self.l_matrix[old_lj, lk] if old_lj < self.n_l else 0
                delta += self.alpha * ((vi_k - new_l) ** 2 - (vi_k - old_l) ** 2)

                vk_i = self.v_matrix[k, i]
                old_l2 = self.l_matrix[lk, old_li] if old_li < self.n_l else 0
                new_l2 = self.l_matrix[lk, old_lj] if old_lj < self.n_l else 0
                delta += self.alpha * ((vk_i - new_l2) ** 2 - (vk_i - old_l2) ** 2)

                vj_k = self.v_matrix[j, k]
                old_l3 = self.l_matrix[old_lj, lk] if old_lj < self.n_l else 0
                new_l3 = self.l_matrix[old_li, lk] if old_li < self.n_l else 0
                delta += self.alpha * ((vj_k - new_l3) ** 2 - (vj_k - old_l3) ** 2)

                vk_j = self.v_matrix[k, j]
                old_l4 = self.l_matrix[lk, old_lj] if old_lj < self.n_l else 0
                new_l4 = self.l_matrix[lk, old_li] if old_li < self.n_l else 0
                delta += self.alpha * ((vk_j - new_l4) ** 2 - (vk_j - old_l4) ** 2)

            if old_li < self.n_l and old_lj < self.n_l:
                vi_j = self.v_matrix[i, j]
                old_cross = self.l_matrix[old_li, old_lj]
                new_cross = self.l_matrix[old_lj, old_li]
                delta += self.alpha * ((vi_j - new_cross) ** 2 - (vi_j - old_cross) ** 2)

                vj_i = self.v_matrix[j, i]
                old_cross2 = self.l_matrix[old_lj, old_li]
                new_cross2 = self.l_matrix[old_li, old_lj]
                delta += self.alpha * ((vj_i - new_cross2) ** 2 - (vj_i - old_cross2) ** 2)

        if self.beta > 0:
            for idx, old_val, new_val in [(i, old_li, old_lj), (j, old_lj, old_li)]:
                if idx in self._candidate_indices:
                    valid = self._candidate_indices[idx]
                    old_v = 1.0 if old_val not in valid else 0.0
                    new_v = 1.0 if new_val not in valid else 0.0
                    delta += self.beta * (new_v - old_v)

        if self.gamma > 0:
            old_dev = abs(i - old_li) + abs(j - old_lj)
            new_dev = abs(i - old_lj) + abs(j - old_li)
            delta += self.gamma * (new_dev - old_dev) / max(self.n_v, 1)

        return delta

    def run(self, n_iter: int = 100000, T_0: float = 1.0,
            T_final: float = 0.001, seed: int = 42,
            log_interval: int = 10000) -> Dict:
        """
        Run swap-based SAA with bijection enforcement.

        Returns dict with mapping, cost trajectory, validation info.
        """
        rng = random.Random(seed)

        current, locked_indices = self._initialize_bijective(rng)
        non_locked = [i for i in range(self.n_v) if i not in locked_indices]

        if not non_locked:
            mapping = {}
            for i, l_idx in enumerate(current):
                if i < len(self.v_vocab) and l_idx < len(self.l_vocab):
                    mapping[self.v_vocab[i]] = self.l_vocab[l_idx]
            return {
                'mapping': mapping,
                'best_cost': self.cost(current),
                'normalized_cost': self.cost(current) / max(self.n_v * self.n_v, 1),
                'n_iterations': 0,
                'cost_trajectory': [],
                'parameters': self._get_params(T_0, T_final),
            }

        current_cost = self.cost(current)
        best = current[:]
        best_cost = current_cost

        cost_trajectory = [(0, current_cost)]
        n_accepted = 0

        for iteration in range(1, n_iter + 1):
            T = T_0 * (T_final / T_0) ** (iteration / n_iter)

            i = rng.choice(non_locked)
            j = rng.choice(non_locked)
            if i == j:
                continue

            d = self._delta_cost_swap(current, i, j)

            if d < 0 or rng.random() < math.exp(-d / max(T, 1e-10)):
                current[i], current[j] = current[j], current[i]
                current_cost += d
                n_accepted += 1

                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost

            if iteration % log_interval == 0:
                cost_trajectory.append((iteration, current_cost))

        cost_trajectory.append((n_iter, best_cost))

        mapping = {}
        for i, l_idx in enumerate(best):
            if i < len(self.v_vocab) and l_idx < len(self.l_vocab):
                mapping[self.v_vocab[i]] = self.l_vocab[l_idx]

        reverse = {}
        collisions = 0
        for v, l in mapping.items():
            if l in reverse:
                collisions += 1
            reverse[l] = v

        return {
            'mapping': mapping,
            'best_cost': float(best_cost),
            'normalized_cost': float(best_cost) / max(self.n_v * self.n_v, 1),
            'n_iterations': n_iter,
            'n_voynich_words': self.n_v,
            'n_latin_words': self.n_l,
            'n_locked': len(locked_indices),
            'n_accepted': n_accepted,
            'acceptance_rate': n_accepted / max(n_iter, 1),
            'is_bijective': collisions == 0,
            'n_collisions': collisions,
            'cost_trajectory': cost_trajectory,
            'parameters': self._get_params(T_0, T_final),
        }

    def _get_params(self, T_0, T_final):
        return {
            'alpha': self.alpha, 'beta': self.beta,
            'gamma': self.gamma, 'delta': self.delta,
            'n_locked': self.n_locked,
            'T_0': T_0, 'T_final': T_final,
        }

    def validate_mapping(self, mapping: Dict,
                         page_tokens: Dict[str, List[str]]) -> Dict:
        """Validate mapping by detecting recognizable Latin phrases."""
        page_results = {}
        best_page = None
        best_count = 0

        for folio, tokens in page_tokens.items():
            decoded = self.decode_text(mapping, tokens)
            decoded_lower = decoded.lower()

            found_phrases = [p for p in RECOGNIZABLE_PHRASES
                             if p in decoded_lower]

            page_results[folio] = {
                'n_phrases_found': len(found_phrases),
                'phrases': found_phrases,
                'decoded_preview': decoded[:200],
            }

            if len(found_phrases) > best_count:
                best_count = len(found_phrases)
                best_page = folio

        return {
            'page_results': page_results,
            'best_page': best_page,
            'best_phrase_count': best_count,
            'passes_threshold': best_count >= 3,
            'n_pages_tested': len(page_tokens),
        }

    def evaluate_crib_satisfaction(self, mapping: Dict) -> Dict:
        """Check crib constraint satisfaction."""
        satisfied = 0
        total = 0

        for v_word, candidates in self.rank_pairs.items():
            if not candidates:
                continue
            total += 1
            if mapping.get(v_word, '') in candidates:
                satisfied += 1

        return {
            'satisfied': satisfied,
            'total': total,
            'rate': satisfied / max(1, total),
        }
