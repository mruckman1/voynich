"""
Reduced SAA — Path B Step 3
==============================
Run SAA on the reduced (merged) vocabulary from HomophoneMerger.
With ~200-400 groups instead of 1001 types, the problem is smaller
and bijection to a similarly-sized Latin vocabulary is more feasible.

Phase 6  ·  Voynich Convergence Attack
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

from modules.phase6.homophone_merger import HomophoneMerger
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.fixed_saa import FixedSAA
from modules.phase5.rank_paired_cribs import RankPairedCribs
from modules.phase5.nmf_scaffold import NMFScaffold

class ReducedSAA:
    """
    Run FixedSAA on the reduced (homophone-merged) vocabulary.

    The merged vocabulary is typically 200-400 canonical words instead
    of 1001 types. The Latin corpus is trimmed to match this size,
    enabling a true bijective mapping.
    """

    def __init__(self,
                 merger: HomophoneMerger,
                 latin_corpus: ImprovedLatinCorpus,
                 rank_pairs: Optional[Dict[str, List[str]]] = None,
                 nmf_scaffold: Optional[NMFScaffold] = None,
                 alpha: float = 0.3,
                 beta: float = 2.0,
                 gamma: float = 1.0,
                 delta: float = 0.2):
        self.merger = merger
        self.latin_corpus = latin_corpus
        self.rank_pairs = rank_pairs or {}
        self.nmf_scaffold = nmf_scaffold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self._reduced_mapping = None

    def prepare_matrices(self) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
        """
        Build the Voynich merged matrix and Latin matrix of matching size.

        Returns:
            (v_matrix, v_vocab, l_matrix, l_vocab)
        """
        v_matrix, v_vocab = self.merger.build_merged_transition_matrix()

        n_merged = len(v_vocab)
        l_matrix, l_vocab = self.latin_corpus.build_transition_matrix(top_n=n_merged)

        return v_matrix, v_vocab, l_matrix, l_vocab

    def build_reduced_rank_pairs(self, v_vocab: List[str],
                                 l_vocab: List[str]) -> Dict[str, List[str]]:
        """
        Build rank-paired cribs for the reduced vocabulary.
        Uses simple rank pairing: Voynich word at rank k gets
        Latin candidates at ranks k-2 to k+2.
        """
        pairs = {}
        for v_idx, v_word in enumerate(v_vocab):
            if v_word in self.rank_pairs:
                candidates = [c for c in self.rank_pairs[v_word]
                              if c in set(l_vocab)]
                if candidates:
                    pairs[v_word] = candidates
                    continue

            candidates = []
            for offset in range(-2, 3):
                l_idx = v_idx + offset
                if 0 <= l_idx < len(l_vocab):
                    candidates.append(l_vocab[l_idx])
            pairs[v_word] = candidates

        return pairs

    def run_saa(self, n_iter: int = 100000, seed: int = 42) -> Dict:
        """Run FixedSAA on the reduced vocabulary."""
        v_matrix, v_vocab, l_matrix, l_vocab = self.prepare_matrices()

        if len(v_vocab) == 0 or len(l_vocab) == 0:
            return {
                'error': 'Empty vocabulary after merging',
                'n_voynich': len(v_vocab),
                'n_latin': len(l_vocab),
            }

        reduced_pairs = self.build_reduced_rank_pairs(v_vocab, l_vocab)

        n_locked = min(20, len(v_vocab) // 5)

        saa = FixedSAA(
            voynich_matrix=v_matrix,
            voynich_vocab=v_vocab,
            latin_matrix=l_matrix,
            latin_vocab=l_vocab,
            rank_pairs=reduced_pairs,
            nmf_scaffold=None,
            n_locked=n_locked,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=0.0,
        )

        results = saa.run(n_iter=n_iter, seed=seed)
        self._reduced_mapping = results.get('mapping', {})
        results['n_reduced_vocab'] = len(v_vocab)
        results['n_latin_vocab'] = len(l_vocab)

        return results

    def expand_mapping(self) -> Dict[str, str]:
        """
        Expand the reduced mapping back to the full 1001-word vocabulary.
        All variants in a homophone group get the same Latin word as
        their canonical.

        Returns {voynich_word: latin_word} for all 1001 Tier 1 types.
        """
        if self._reduced_mapping is None:
            return {}

        full_mapping = {}
        merge_map = self.merger.build_merge_map()

        for v_word, l_word in self._reduced_mapping.items():
            full_mapping[v_word] = l_word

        for variant, canonical in merge_map.items():
            if canonical in self._reduced_mapping:
                full_mapping[variant] = self._reduced_mapping[canonical]

        return full_mapping

    def run(self, verbose: bool = True, n_iter: int = 100000) -> Dict:
        """Run reduced SAA and return full results."""
        v_matrix, v_vocab, l_matrix, l_vocab = self.prepare_matrices()

        if verbose:
            print(f'\n  Reduced SAA:')
            print(f'    Merged Voynich vocab: {len(v_vocab)} types')
            print(f'    Latin vocab:          {len(l_vocab)} types')
            print(f'    Running SAA ({n_iter:,} iterations)...')

        saa_results = self.run_saa(n_iter=n_iter)

        expanded = self.expand_mapping()
        saa_results['expanded_mapping_size'] = len(expanded)

        if self._reduced_mapping:
            extractor = self.merger.extractor
            page_tokens = extractor.extract_lang_a_by_folio()

            saa_temp = FixedSAA(
                voynich_matrix=v_matrix,
                voynich_vocab=v_vocab,
                latin_matrix=l_matrix,
                latin_vocab=l_vocab,
                rank_pairs={},
            )
            validation = saa_temp.validate_mapping(expanded, page_tokens)
            saa_results['validation'] = validation

            all_tokens = extractor.extract_lang_a_tokens()[:100]
            decoded = ' '.join(expanded.get(t, f'[{t}]') for t in all_tokens)
            saa_results['decoded_sample'] = decoded[:300]

            if verbose:
                print(f'    Best cost:       {saa_results.get("best_cost", "N/A")}')
                print(f'    Bijective:       {saa_results.get("is_bijective", "N/A")}')
                print(f'    Expanded mapping: {len(expanded)} words')
                print(f'    Best page phrases: {validation.get("best_phrase_count", 0)}')
                print(f'    Decoded sample:  {decoded[:100]}...')

        results = {
            'saa_results': saa_results,
            'expanded_mapping': expanded,
            'synthesis': {
                'n_reduced_vocab': len(v_vocab),
                'best_cost': saa_results.get('best_cost', float('inf')),
                'is_bijective': saa_results.get('is_bijective', False),
                'best_phrase_count': saa_results.get('validation', {}).get(
                    'best_phrase_count', 0),
                'conclusion': (
                    f'Reduced SAA on {len(v_vocab)} merged types: '
                    f'cost={saa_results.get("best_cost", "N/A")}, '
                    f'bijective={saa_results.get("is_bijective", "N/A")}, '
                    f'phrases={saa_results.get("validation", {}).get("best_phrase_count", 0)}'
                ),
            },
        }

        return results
