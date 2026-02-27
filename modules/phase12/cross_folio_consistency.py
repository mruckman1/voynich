"""
Phase 12, Module 12.5: Cross-Folio Consistency Engine
=====================================================
Post-pipeline pass that identifies skeleton->word mappings resolved
consistently across multiple folios and applies them to fill remaining
brackets in folios where the same skeleton was left unresolved.

In a consistent cipher, the same ciphertext should map to the same
plaintext. This module exploits that property: if skeleton 'K-L'
resolves to 'aqua' on 8 folios, it should resolve to 'aqua' on the
remaining folios too.

Phase 12  ·  Voynich Convergence Attack
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer

# Regex for UNRESOLVED brackets
UNRESOLVED_RE = re.compile(r'\[([^_\]]+)_UNRESOLVED\]|<([^_>]+)_UNRESOLVED>')


class CrossFolioConsistencyEngine:
    """
    Two-pass cross-folio consistency resolver.

    Pass 1 (collect): Scans all resolved folios, builds a mapping from
    Voynich token skeleton -> {latin_word: count_across_folios}.

    Pass 2 (apply): For each remaining [token_UNRESOLVED] bracket,
    checks if the token's skeleton has a high-agreement mapping.
    If agreement >= threshold and occurrences >= min, resolves it.
    """

    def __init__(
        self,
        fuzzy_skeletonizer: FuzzySkeletonizer,
        min_agreement: float = 0.6,
        min_occurrences: int = 3,
    ):
        """
        Args:
            fuzzy_skeletonizer: Phase 12 Voynich skeletonizer for skeleton lookup
            min_agreement: Fraction of resolved occurrences that must agree
                on the same Latin word (e.g., 0.6 = 60%)
            min_occurrences: Minimum number of resolved occurrences of a
                skeleton across all folios to consider it for consistency
        """
        self.f_skel = fuzzy_skeletonizer
        self.min_agreement = min_agreement
        self.min_occurrences = min_occurrences

        # skeleton -> {latin_word: count}
        self._skeleton_mappings: Dict[str, Counter] = defaultdict(Counter)
        # voynich_token -> skeleton (cache)
        self._token_skeleton_cache: Dict[str, Optional[str]] = {}
        # Computed consistent mappings (skeleton -> latin_word)
        self._consistent_mappings: Dict[str, str] = {}

    def _get_primary_skeleton(self, voynich_token: str) -> Optional[str]:
        """Get the primary (highest-weight) skeleton for a Voynich token."""
        if voynich_token in self._token_skeleton_cache:
            return self._token_skeleton_cache[voynich_token]

        # Strip affixes to get stem before skeleton lookup
        pref, stem, suf = self.f_skel.v_morphemer._strip_affixes(voynich_token)
        candidates = self.f_skel.get_skeleton_candidates(stem)
        if not candidates:
            self._token_skeleton_cache[voynich_token] = None
            return None

        primary = candidates[0][0]  # highest-weight skeleton
        self._token_skeleton_cache[voynich_token] = primary
        return primary

    def collect_folio(
        self,
        resolved_text: str,
        original_tokens: List[str],
        folio_id: str,
    ) -> None:
        """
        Pass 1: Collect skeleton->word mappings from a resolved folio.

        For each non-bracketed word in the resolved text, look up the
        corresponding Voynich token's skeleton and record the mapping.

        Args:
            resolved_text: Final resolved text (with [token_UNRESOLVED] brackets)
            original_tokens: Original Voynich tokens for this folio
            folio_id: Folio identifier (for logging)
        """
        resolved_words = resolved_text.split()
        n = min(len(resolved_words), len(original_tokens))

        for i in range(n):
            word = resolved_words[i]
            # Skip brackets
            if word.startswith('[') or word.startswith('<'):
                continue

            v_token = original_tokens[i]
            skeleton = self._get_primary_skeleton(v_token)
            if skeleton:
                self._skeleton_mappings[skeleton][word] += 1

    def compute_consistent_mappings(self) -> Dict[str, str]:
        """
        After all folios are collected, compute the globally consistent
        skeleton -> latin_word mappings.

        A mapping is consistent if:
        - The skeleton appears resolved in >= min_occurrences total
        - The winning Latin word has >= min_agreement fraction of total

        Returns:
            Dict of skeleton -> latin_word for consistent mappings
        """
        self._consistent_mappings = {}

        for skeleton, word_counts in self._skeleton_mappings.items():
            total = sum(word_counts.values())
            if total < self.min_occurrences:
                continue

            best_word, best_count = word_counts.most_common(1)[0]
            agreement = best_count / total

            if agreement >= self.min_agreement:
                self._consistent_mappings[skeleton] = best_word

        return dict(self._consistent_mappings)

    def apply_consistency(
        self,
        resolved_text: str,
        original_tokens: List[str],
    ) -> Tuple[str, Dict]:
        """
        Pass 2: Apply cross-folio consistency to fill remaining brackets.

        For each [token_UNRESOLVED] bracket, check if the token's skeleton
        has a consistent global mapping. If so, resolve it.

        Args:
            resolved_text: Final resolved text with UNRESOLVED brackets
            original_tokens: Original Voynich tokens for this folio

        Returns:
            (updated_text, stats_dict) where stats_dict contains:
                'applied': number of brackets resolved by consistency
                'skipped': number of brackets without consistent mapping
        """
        resolved_words = resolved_text.split()
        n = min(len(resolved_words), len(original_tokens))

        applied_count = 0
        skipped_count = 0

        for i in range(n):
            word = resolved_words[i]
            match = UNRESOLVED_RE.match(word)
            if not match:
                continue

            voynich_token = match.group(1) or match.group(2)
            skeleton = self._get_primary_skeleton(voynich_token)

            if skeleton and skeleton in self._consistent_mappings:
                resolved_words[i] = self._consistent_mappings[skeleton]
                applied_count += 1
            else:
                skipped_count += 1

        stats = {
            'applied': applied_count,
            'skipped': skipped_count,
            'unique_consistent_mappings': len(self._consistent_mappings),
        }

        return ' '.join(resolved_words), stats
