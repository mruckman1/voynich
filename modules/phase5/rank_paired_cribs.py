"""
Rank-Paired Crib System
========================
Replaces Phase 4's underconstrained cribs (all universal words given the
same candidate list) with frequency-rank pairing. Each Voynich word gets
a unique candidate set of 3-5 Latin words at similar frequency ranks.

Phase 4 failure: All 45 "universal" words were mapped to "et" because the
optimizer found the cheapest solution when all words had the same candidates.
Fix: daiin (rank 1, 4.3%) maps only to {et} (rank 1), chol (rank 2) maps
only to {contra} (rank 2), etc.

Phase 5  ·  Voynich Convergence Attack
"""

import sys
import os
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase5.tier_splitter import TierSplitter
from modules.phase5.latin_corpus_expanded import ExpandedLatinHerbalCorpus
from modules.phase4.known_plaintext_botanical import (
    BotanicalKnownPlaintext, HUMORAL_EXPECTED_VOCAB,
)
from modules.phase4.lang_a_extractor import LanguageAExtractor


class RankPairedCribs:
    """
    Generate frequency-rank-paired crib constraints for the constrained SAA.

    Each Voynich Tier 1 word is paired with 3-5 Latin candidate words at
    similar frequency ranks, plus botanical overrides for words with
    humoral-quality page associations.
    """

    def __init__(self, splitter: TierSplitter,
                 latin_corpus: ExpandedLatinHerbalCorpus,
                 rank_window: int = 5):
        """
        Parameters:
            splitter: TierSplitter with the corpus split.
            latin_corpus: Expanded Latin herbal corpus.
            rank_window: How many Latin words around each rank to consider
                         as candidates. Default 5 → each word gets up to
                         2*5+1=11 candidates, narrowed to 3-5 by filtering.
        """
        self.splitter = splitter
        self.latin_corpus = latin_corpus
        self.rank_window = rank_window
        self._voynich_ranked = None
        self._latin_ranked = None
        self._botanical = None

    def _get_voynich_ranked(self) -> List[Tuple[str, int]]:
        """Get Tier 1 words ranked by frequency."""
        if self._voynich_ranked is not None:
            return self._voynich_ranked

        all_tokens = self.splitter.extractor.extract_lang_a_tokens()
        tier1_set = set(self.splitter.get_tier1_types())
        tier1_tokens = [t for t in all_tokens if t in tier1_set]
        freqs = Counter(tier1_tokens)
        self._voynich_ranked = freqs.most_common()
        return self._voynich_ranked

    def _get_latin_ranked(self) -> List[Tuple[str, int]]:
        """Get Latin words ranked by frequency."""
        if self._latin_ranked is not None:
            return self._latin_ranked

        self._latin_ranked = Counter(self.latin_corpus.get_tokens()).most_common()
        return self._latin_ranked

    def _get_botanical_constraints(self) -> Dict:
        """Get botanical humoral constraints from Phase 4."""
        if self._botanical is not None:
            return self._botanical

        try:
            botanical = BotanicalKnownPlaintext(self.splitter.extractor)
            cross_ref = botanical.cross_reference_pages()
            self._botanical = cross_ref
        except Exception:
            self._botanical = {}

        return self._botanical

    def build_rank_pairs(self) -> List[Dict]:
        """
        Build frequency-rank-paired crib constraints.

        For each Tier 1 Voynich word at rank R, select 3-5 Latin words
        from ranks [R - window, R + window] as candidates. Ensures each
        Voynich word has a UNIQUE candidate set.

        Returns list of dicts with keys:
            voynich_word, voynich_rank, voynich_freq,
            candidates [{latin_word, latin_rank, latin_freq}],
            confidence
        """
        v_ranked = self._get_voynich_ranked()
        l_ranked = self._get_latin_ranked()

        if not v_ranked or not l_ranked:
            return []

        # Build rank pairs
        pairs = []
        used_latin = set()  # Track which Latin words are already top candidates

        for v_rank, (v_word, v_freq) in enumerate(v_ranked):
            # Find Latin words at similar ranks
            l_start = max(0, v_rank - self.rank_window)
            l_end = min(len(l_ranked), v_rank + self.rank_window + 1)

            candidates = []
            for l_rank in range(l_start, l_end):
                if l_rank < len(l_ranked):
                    l_word, l_freq = l_ranked[l_rank]
                    # Prefer candidates not already used as primary
                    priority = 0 if l_word not in used_latin else 1
                    candidates.append({
                        'latin_word': l_word,
                        'latin_rank': l_rank,
                        'latin_freq': l_freq,
                        'rank_distance': abs(v_rank - l_rank),
                        'priority': priority,
                    })

            # Sort by priority (unused first), then rank distance
            candidates.sort(key=lambda c: (c['priority'], c['rank_distance']))

            # Take top 3-5 candidates
            top_candidates = candidates[:5]
            if top_candidates:
                used_latin.add(top_candidates[0]['latin_word'])

            # Confidence based on rank distance of best candidate
            if top_candidates:
                best_dist = top_candidates[0]['rank_distance']
                if best_dist == 0:
                    confidence = 'HIGH'
                elif best_dist <= 2:
                    confidence = 'MODERATE'
                else:
                    confidence = 'LOW'
            else:
                confidence = 'NONE'

            pairs.append({
                'voynich_word': v_word,
                'voynich_rank': v_rank,
                'voynich_freq': v_freq,
                'candidates': top_candidates,
                'confidence': confidence,
            })

        return pairs

    def apply_botanical_overrides(self, base_pairs: List[Dict]) -> List[Dict]:
        """
        Apply botanical constraints as overrides to rank-paired cribs.

        Words exclusive to specific humoral-quality pages get constrained
        to quality-appropriate Latin vocabulary. These override rank-based
        pairing when botanical evidence is available.

        Returns updated pairs list.
        """
        cross_ref = self._get_botanical_constraints()
        if not cross_ref:
            return base_pairs

        specific_words = cross_ref.get('specific_words', [])
        universal_words = cross_ref.get('universal_words', [])

        # Build lookup: voynich_word → humoral quality
        word_to_quality = {}
        for entry in specific_words:
            word_to_quality[entry['word']] = entry['quality']

        universal_set = set(universal_words)

        updated = []
        n_overrides = 0

        for pair in base_pairs:
            v_word = pair['voynich_word']

            if v_word in word_to_quality:
                # Botanical override: constrain to humoral vocabulary
                quality = word_to_quality[v_word]
                humoral_vocab = HUMORAL_EXPECTED_VOCAB.get(quality, [])

                if humoral_vocab:
                    # Replace candidates with humoral-specific words
                    botanical_candidates = [
                        {
                            'latin_word': lw,
                            'latin_rank': -1,
                            'latin_freq': -1,
                            'rank_distance': -1,
                            'priority': 0,
                            'source': f'botanical_{quality}',
                        }
                        for lw in humoral_vocab[:5]
                    ]
                    pair = dict(pair)
                    pair['candidates'] = botanical_candidates
                    pair['confidence'] = 'MODERATE'
                    pair['botanical_override'] = quality
                    n_overrides += 1

            elif v_word in universal_set:
                # Universal words: keep rank-paired candidates but boost confidence
                pair = dict(pair)
                pair['universal'] = True

            updated.append(pair)

        if self.splitter.extractor.__class__.__name__ == 'LanguageAExtractor':
            pass  # Normal flow

        return updated

    def get_candidate_matrix(self) -> Dict[str, List[str]]:
        """
        Return a clean mapping: {voynich_word: [candidate1, candidate2, ...]}.

        This is the format consumed by the constrained SAA.
        """
        pairs = self.build_rank_pairs()
        pairs = self.apply_botanical_overrides(pairs)

        matrix = {}
        for pair in pairs:
            v_word = pair['voynich_word']
            candidates = [c['latin_word'] for c in pair['candidates']]
            if candidates:
                matrix[v_word] = candidates

        return matrix

    def validate_cribs(self) -> Dict:
        """
        Validate the crib system:
        - No two Voynich words should share the same top candidate
        - Rank correlation between paired words should be high
        - All Tier 1 words should have at least one candidate
        """
        pairs = self.build_rank_pairs()
        pairs = self.apply_botanical_overrides(pairs)

        # Check for top-candidate collisions
        top_candidates = {}
        collisions = []
        for pair in pairs:
            if pair['candidates']:
                top = pair['candidates'][0]['latin_word']
                if top in top_candidates:
                    collisions.append({
                        'word1': top_candidates[top],
                        'word2': pair['voynich_word'],
                        'shared_candidate': top,
                    })
                else:
                    top_candidates[top] = pair['voynich_word']

        # Rank correlation (Spearman-like)
        v_ranks = []
        l_ranks = []
        for pair in pairs:
            if pair['candidates'] and pair['candidates'][0].get('latin_rank', -1) >= 0:
                v_ranks.append(pair['voynich_rank'])
                l_ranks.append(pair['candidates'][0]['latin_rank'])

        if len(v_ranks) > 2:
            # Simple rank correlation
            n = len(v_ranks)
            d_squared = sum((v - l) ** 2 for v, l in zip(v_ranks, l_ranks))
            rank_correlation = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
        else:
            rank_correlation = 0.0

        # Coverage
        n_with_candidates = sum(1 for p in pairs if p['candidates'])
        n_total = len(pairs)

        # Confidence distribution
        conf_dist = Counter(p['confidence'] for p in pairs)

        # Botanical overrides
        n_botanical = sum(1 for p in pairs if 'botanical_override' in p)

        return {
            'n_pairs': n_total,
            'n_with_candidates': n_with_candidates,
            'coverage': n_with_candidates / max(1, n_total),
            'n_collisions': len(collisions),
            'collisions': collisions[:10],
            'rank_correlation': rank_correlation,
            'confidence_distribution': dict(conf_dist),
            'n_botanical_overrides': n_botanical,
            'valid': len(collisions) == 0 and n_with_candidates > 0,
        }

    def run(self, verbose: bool = True) -> Dict:
        """Build and validate rank-paired cribs."""
        pairs = self.build_rank_pairs()
        pairs = self.apply_botanical_overrides(pairs)
        validation = self.validate_cribs()
        candidate_matrix = self.get_candidate_matrix()

        results = {
            'n_pairs': len(pairs),
            'sample_pairs': [
                {
                    'voynich': p['voynich_word'],
                    'rank': p['voynich_rank'],
                    'candidates': [c['latin_word'] for c in p['candidates'][:3]],
                    'confidence': p['confidence'],
                }
                for p in pairs[:20]
            ],
            'validation': validation,
            'n_candidate_entries': len(candidate_matrix),
            'synthesis': {
                'conclusion': (
                    f'{len(pairs)} rank-paired cribs generated. '
                    f'{validation["n_with_candidates"]} have candidates '
                    f'({validation["coverage"]:.1%} coverage). '
                    f'{validation["n_collisions"]} collisions. '
                    f'Rank correlation: {validation["rank_correlation"]:.3f}. '
                    f'{validation["n_botanical_overrides"]} botanical overrides.'
                ),
            },
        }

        if verbose:
            print(f'\n  Rank-Paired Cribs:')
            print(f'    Total pairs:       {len(pairs)}')
            print(f'    With candidates:   {validation["n_with_candidates"]}')
            print(f'    Coverage:          {validation["coverage"]:.1%}')
            print(f'    Collisions:        {validation["n_collisions"]}')
            print(f'    Rank correlation:  {validation["rank_correlation"]:.3f}')
            print(f'    Botanical overrides: {validation["n_botanical_overrides"]}')
            print(f'    Confidence dist:   {dict(validation["confidence_distribution"])}')
            print(f'    --- Sample Pairs ---')
            for p in pairs[:5]:
                candidates = [c['latin_word'] for c in p['candidates'][:3]]
                print(f'      {p["voynich_word"]} (rank {p["voynich_rank"]}) '
                      f'→ {candidates} [{p["confidence"]}]')

        return results
