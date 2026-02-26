"""
Phase 12, Module 12.2: Budgeted CSP Decoder
=============================================
Replaces Phase 11's CSPPhoneticDecoder with a folio-aware decoder that:
  1. Budgets token frequency per folio (exponential penalty for overuse)
  2. Injects humoral cribs from Phase 4's botanical identifications
  3. Scores across multiple skeleton candidates from Module 12.1

This eliminates the "hora/quae/oleo" repetition problem and produces
contextually appropriate vocabulary for identified plant folios.

Phase 12  ·  Voynich Convergence Attack
"""

import Levenshtein
from collections import Counter
from typing import Dict, List, Optional, Tuple

from modules.phase11.csp_decoder import SIGLA_MAP, SUFFIX_MAP, FUNCTION_WORDS
from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer, dynamic_levenshtein_threshold

from data.botanical_identifications import PLANT_IDS, HUMORAL_QUALITIES
from modules.phase4.latin_herbal_corpus import (
    LATIN_QUALITY_WORDS, LATIN_PROPERTY_WORDS, LATIN_BODY_WORDS,
    LATIN_SUBSTANCE_WORDS, LATIN_PLANT_NAMES,
)


# Build humoral vocabulary mapping
# Maps each humoral category to Latin words contextually appropriate for it
HUMORAL_VOCAB: Dict[str, List[str]] = {
    'hot_dry': (
        [w for w in LATIN_QUALITY_WORDS if w.startswith('calid')]
        + [w for w in LATIN_QUALITY_WORDS if w.startswith('sicc')]
        + ['cholera', 'ignis', 'virtutem', 'emmenagogam', 'purgativam',
           'stypticam', 'digestivam', 'carminativam']
    ),
    'hot_wet': (
        [w for w in LATIN_QUALITY_WORDS if w.startswith('calid')]
        + [w for w in LATIN_QUALITY_WORDS if w.startswith('humid')]
        + ['sanguis', 'aer', 'cordialem', 'emollientem', 'laxativam',
           'nutritivu', 'confortativam']
    ),
    'cold_dry': (
        [w for w in LATIN_QUALITY_WORDS if w.startswith('frigid')]
        + [w for w in LATIN_QUALITY_WORDS if w.startswith('sicc')]
        + ['melancholia', 'terra', 'stypticam', 'vulnerariam',
           'constringit', 'desiccat']
    ),
    'cold_wet': (
        [w for w in LATIN_QUALITY_WORDS if w.startswith('frigid')]
        + [w for w in LATIN_QUALITY_WORDS if w.startswith('humid')]
        + ['phlegma', 'aqua', 'soporificam', 'anodynam', 'refrigerat',
           'sedandi', 'emollientem']
    ),
}


# Minimum score threshold: if best candidate scores below this after
# budget penalties, bracket the token rather than emitting a spammed word
MIN_SCORE_THRESHOLD = 0.01


class BudgetedCSPDecoder:
    """
    Folio-aware CSP decoder with frequency budgeting and humoral injection.

    Improvements over Phase 11's CSPPhoneticDecoder:
      - Multi-skeleton scoring via FuzzySkeletonizer
      - Exponential decay penalty for overused words
      - Humoral crib boosting for identified folios
      - Minimum score gate to prevent emitting heavily penalized words
    """

    def __init__(
        self,
        latin_skeletonizer: LatinPhoneticSkeletonizer,
        fuzzy_skeletonizer: FuzzySkeletonizer,
        corpus_tokens: List[str],
        folio_metadata: Optional[Dict] = None,
    ):
        """
        Args:
            latin_skeletonizer: Phase 11 Latin skeleton index
            fuzzy_skeletonizer: Phase 12 branching Voynich skeletonizer
            corpus_tokens: Full Latin corpus token list (for frequency stats)
            folio_metadata: Optional dict (folio_id -> plant data), defaults to PLANT_IDS
        """
        self.l_skel = latin_skeletonizer
        self.f_skel = fuzzy_skeletonizer
        self.l_unigrams = latin_skeletonizer.unigram_counts
        self.total_corpus_tokens = len(corpus_tokens)
        self.folio_metadata = folio_metadata or PLANT_IDS

        # Word frequency ratios (precomputed, folio-length-independent)
        self.word_freq_ratio: Dict[str, float] = {}
        for word, count in self.l_unigrams.items():
            self.word_freq_ratio[word] = count / max(1, self.total_corpus_tokens)

        # Folio token count (set dynamically in decode_folio)
        self._folio_token_count: int = 40

        # Per-folio emission tracking (reset each folio)
        self.emission_counts: Dict[str, int] = {}

    def _get_humoral_boost_words(self, folio_id: str) -> set:
        """Get the set of words that should receive humoral boosting for a folio."""
        if not folio_id or folio_id not in self.folio_metadata:
            return set()

        humoral_cat = self.folio_metadata[folio_id].get('humoral', '')
        if humoral_cat and humoral_cat in HUMORAL_VOCAB:
            return set(HUMORAL_VOCAB[humoral_cat])
        return set()

    def _score_candidates(
        self,
        valid_latin_words: List[str],
        prefix: str,
        suffix: str,
        skeleton_weight: float,
        humoral_words: set,
    ) -> List[Tuple[float, str]]:
        """
        Score Latin word candidates with prefix/suffix sigla, frequency
        budgeting, and humoral boosting.

        Returns list of (score, word) tuples.
        """
        l_prefixes = SIGLA_MAP.get(prefix, [''])
        l_suffixes = SUFFIX_MAP.get(suffix, [''])
        scored = []

        for l_word in valid_latin_words:
            score = 0.0

            # Prefix sigla match (+10)
            if prefix and any(l_word.startswith(lp) for lp in l_prefixes):
                score += 10.0

            # Suffix sigla match (+10)
            if suffix and any(l_word.endswith(ls) for ls in l_suffixes):
                score += 10.0

            # Frequency tiebreaker (normalized)
            score += self.l_unigrams.get(l_word, 0) / max(1, self.total_corpus_tokens)

            # Humoral crib boost (+50)
            if l_word in humoral_words:
                score += 50.0

            # Frequency budget penalty (exponential decay for overused words)
            # Expected count scales with actual folio length
            freq_ratio = self.word_freq_ratio.get(l_word, 0.0)
            expected = freq_ratio * self._folio_token_count
            expected = max(expected, 0.5)  # Floor at 0.5 to allow rare words once
            current_count = self.emission_counts.get(l_word, 0)
            if current_count > expected:
                overage = current_count - expected
                score *= 0.1 ** overage

            # Weight by skeleton probability
            score *= skeleton_weight

            scored.append((score, l_word))

        return scored

    def find_best_match(self, v_token: str, folio_id: str = None) -> str:
        """
        Find the best Latin match for a Voynich token using multi-skeleton
        scoring with frequency budgeting and humoral injection.

        Args:
            v_token: Voynich token string
            folio_id: Optional folio identifier for humoral boosting

        Returns:
            Best Latin word, or [v_token] if unresolvable
        """
        # 1. Function Word Bypass
        if v_token in FUNCTION_WORDS:
            word = FUNCTION_WORDS[v_token]
            self.emission_counts[word] = self.emission_counts.get(word, 0) + 1
            return word

        # 2. Parse Morphology
        pref, stem, suf = self.f_skel.v_morphemer._strip_affixes(v_token)

        # 3. Get multiple skeleton candidates (Phase 12 branching)
        skeleton_candidates = self.f_skel.get_skeleton_candidates(stem)
        if not skeleton_candidates:
            return f"[{v_token}]"

        # 4. Get humoral boost words for this folio
        humoral_words = self._get_humoral_boost_words(folio_id)

        # 5. Score across all skeleton branches
        all_scored: List[Tuple[float, str]] = []

        for target_skel, skel_weight in skeleton_candidates:
            # Exact skeleton match
            valid_words = self.l_skel.skeleton_index.get(target_skel, [])

            if not valid_words:
                # Fuzzy match with dynamic threshold
                max_dist = dynamic_levenshtein_threshold(target_skel)
                if max_dist > 0:
                    closest_skel = None
                    min_dist = float('inf')
                    for l_skel in self.l_skel.skeleton_index.keys():
                        dist = Levenshtein.distance(target_skel, l_skel)
                        if dist < min_dist and dist <= max_dist:
                            min_dist = dist
                            closest_skel = l_skel
                    if closest_skel:
                        valid_words = self.l_skel.skeleton_index.get(closest_skel, [])

            if valid_words:
                scored = self._score_candidates(
                    valid_words, pref, suf, skel_weight, humoral_words
                )
                all_scored.extend(scored)

        if not all_scored:
            return f"[{v_token}]"

        # 6. Sort by score descending, pick best
        all_scored.sort(key=lambda x: -x[0])
        best_score, best_word = all_scored[0]

        # Minimum score gate: if budget penalties have crushed the score,
        # bracket the token rather than emitting a spammed word
        if best_score < MIN_SCORE_THRESHOLD:
            return f"[{v_token}]"

        # Track emission
        self.emission_counts[best_word] = self.emission_counts.get(best_word, 0) + 1

        return best_word

    def decode_folio(self, tokens: List[str], folio_id: str = None) -> str:
        """
        Decode a full folio's worth of Voynich tokens.

        Resets emission counts at the start of each folio to ensure
        frequency budgets are per-folio.

        Args:
            tokens: List of Voynich tokens for this folio
            folio_id: Folio identifier (e.g., 'f1v', 'f2r')

        Returns:
            Decoded Latin text string
        """
        self.emission_counts = {}  # Reset per folio
        self._folio_token_count = len(tokens)  # Scale budgets to actual folio size
        decoded_words = []

        for token in tokens:
            word = self.find_best_match(token, folio_id=folio_id)
            decoded_words.append(word)

        return ' '.join(decoded_words)
