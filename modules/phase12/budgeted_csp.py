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

MIN_SCORE_THRESHOLD = 20.0

MAX_CANDIDATE_AMBIGUITY = 15

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
        enable_graduated_csp: bool = False,
        high_threshold: float = 20.0,
        medium_threshold: float = 10.0,
        enable_selective_function_words: bool = False,
        function_word_max_density: float = 1.5,
        function_word_window_size: int = 20,
    ):
        """
        Args:
            latin_skeletonizer: Phase 11 Latin skeleton index
            fuzzy_skeletonizer: Phase 12 branching Voynich skeletonizer
            corpus_tokens: Full Latin corpus token list (for frequency stats)
            folio_metadata: Optional dict (folio_id -> plant data), defaults to PLANT_IDS
            enable_graduated_csp: Enable three-tier scoring (HIGH/MEDIUM/LOW)
            high_threshold: Score threshold for direct resolution (default 20.0)
            medium_threshold: Score threshold for passing candidates to n-gram solver
            enable_selective_function_words: Enable density-gated function words
            function_word_max_density: Max density ratio relative to corpus frequency
            function_word_window_size: Window size for density calculation
        """
        self.l_skel = latin_skeletonizer
        self.f_skel = fuzzy_skeletonizer
        self.l_unigrams = latin_skeletonizer.unigram_counts
        self.total_corpus_tokens = len(corpus_tokens)
        self.folio_metadata = folio_metadata or PLANT_IDS

        self.word_freq_ratio: Dict[str, float] = {}
        for word, count in self.l_unigrams.items():
            self.word_freq_ratio[word] = count / max(1, self.total_corpus_tokens)

        self._folio_token_count: int = 40

        self.emission_counts: Dict[str, int] = {}

        self.enable_graduated_csp = enable_graduated_csp
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.medium_candidates: Dict[int, List[Tuple[float, str]]] = {}
        self._current_medium_candidates: Optional[List[Tuple[float, str]]] = None

        self.enable_selective_function_words = enable_selective_function_words
        self.function_word_max_density = function_word_max_density
        self.function_word_window_size = function_word_window_size
        self._decoded_so_far: List[str] = []
        self._function_word_expected_freq: Dict[str, float] = {}
        for v_char, l_word in FUNCTION_WORDS.items():
            self._function_word_expected_freq[l_word] = self.word_freq_ratio.get(l_word, 0.001)

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

            if prefix and any(l_word.startswith(lp) for lp in l_prefixes):
                score += 10.0

            if suffix and any(l_word.endswith(ls) for ls in l_suffixes):
                score += 10.0

            score += self.l_unigrams.get(l_word, 0) / max(1, self.total_corpus_tokens)

            if l_word in humoral_words:
                score += 50.0

            freq_ratio = self.word_freq_ratio.get(l_word, 0.0)
            expected = freq_ratio * self._folio_token_count
            expected = max(expected, 0.5)
            current_count = self.emission_counts.get(l_word, 0)
            if current_count > expected:
                overage = current_count - expected
                score *= 0.1 ** overage

            score *= skeleton_weight

            scored.append((score, l_word))

        return scored

    def _check_function_word_density(self, candidate_word: str) -> bool:
        """
        Check if adding this function word would exceed the density threshold
        in the local window.

        Returns True if the word is safe to emit (density within bounds).
        """
        window_start = max(0, len(self._decoded_so_far) - self.function_word_window_size)
        window = self._decoded_so_far[window_start:]

        if not window:
            return True

        current_count = sum(1 for w in window if w == candidate_word)
        window_size = len(window)

        expected_freq = self._function_word_expected_freq.get(candidate_word, 0.001)
        expected_in_window = expected_freq * window_size
        expected_in_window = max(expected_in_window, 0.5)

        max_allowed = self.function_word_max_density * expected_in_window

        return current_count < max_allowed

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
        self._current_medium_candidates = None

        if self.enable_selective_function_words and v_token in FUNCTION_WORDS:
            candidate = FUNCTION_WORDS[v_token]
            if self._check_function_word_density(candidate):
                self.emission_counts[candidate] = self.emission_counts.get(candidate, 0) + 1
                return candidate
            return f"[{v_token}]"

        pref, stem, suf = self.f_skel.v_morphemer._strip_affixes(v_token)

        skeleton_candidates = self.f_skel.get_skeleton_candidates(stem)
        if not skeleton_candidates:
            return f"[{v_token}]"

        humoral_words = self._get_humoral_boost_words(folio_id)

        all_scored: List[Tuple[float, str]] = []

        for target_skel, skel_weight in skeleton_candidates:
            valid_words = self.l_skel.skeleton_index.get(target_skel, [])

            if not valid_words:
                max_dist = dynamic_levenshtein_threshold(target_skel)
                if max_dist > 0:
                    closest_skel = None
                    min_dist = float('inf')
                    for l_skel in self.l_skel.skeleton_index.keys():
                        dist = Levenshtein.distance(target_skel, l_skel)
                        if dist <= max_dist and (dist < min_dist or (dist == min_dist and (closest_skel is None or l_skel < closest_skel))):
                            min_dist = dist
                            closest_skel = l_skel
                    if closest_skel:
                        valid_words = self.l_skel.skeleton_index.get(closest_skel, [])

            if len(valid_words) > MAX_CANDIDATE_AMBIGUITY:
                continue

            if valid_words:
                scored = self._score_candidates(
                    valid_words, pref, suf, skel_weight, humoral_words
                )
                all_scored.extend(scored)

        if not all_scored:
            return f"[{v_token}]"

        all_scored.sort(key=lambda x: (-x[0], x[1]))
        best_score, best_word = all_scored[0]

        effective_threshold = self.high_threshold if self.enable_graduated_csp else MIN_SCORE_THRESHOLD

        if best_score >= effective_threshold:
            self.emission_counts[best_word] = self.emission_counts.get(best_word, 0) + 1
            return best_word

        if self.enable_graduated_csp and best_score >= self.medium_threshold:
            self._current_medium_candidates = all_scored[:5]
            return f"[{v_token}]"

        return f"[{v_token}]"

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
        self.emission_counts = {}
        self.medium_candidates = {}
        self._decoded_so_far = []
        self._folio_token_count = len(tokens)
        decoded_words = []

        for idx, token in enumerate(tokens):
            self._current_medium_candidates = None
            word = self.find_best_match(token, folio_id=folio_id)
            decoded_words.append(word)
            self._decoded_so_far.append(word)

            if self._current_medium_candidates is not None:
                self.medium_candidates[idx] = self._current_medium_candidates

        return ' '.join(decoded_words)
