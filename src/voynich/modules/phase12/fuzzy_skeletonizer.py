"""
Phase 12, Module 12.1: Fuzzy Skeletonizer
==========================================
Replaces the rigid VoynichPhoneticSkeletonizer with a branching version
that handles y/o semi-consonants and returns multiple weighted skeleton
candidates instead of a single rigid string.

Key improvements over Phase 11:
  1. Semi-consonant branching: y → J class, o → P class when phonetically
     plausible (stem-initial or intervocalic position)
  2. Dynamic Levenshtein thresholds based on skeleton length
  3. Returns List[Tuple[str, float]] instead of a single skeleton

Phase 12  ·  Voynich Convergence Attack
"""

from typing import List, Tuple
from itertools import product

from voynich.modules.phase11.phonetic_skeletonizer import (
    VOYNICH_CONSONANT_CLASSES,
    LATIN_CONSONANT_CLASSES,
)

VOYNICH_VOWELS = {'a', 'e', 'o', 'y'}

SEMI_CONSONANT_MAP = {
    'y': 'J',
    'o': 'P',
}

def dynamic_levenshtein_threshold(skeleton: str) -> int:
    """
    Compute the maximum allowed Levenshtein distance for a skeleton
    based on its segment count.

    Short skeletons (1-2 segments) require exact match to prevent
    spurious matches like 'hora' for every 1-consonant stem.
    Longer skeletons can tolerate more edit distance.

    Args:
        skeleton: A '-'-joined consonant skeleton (e.g., 'K-L' or 'K-N-S-T')

    Returns:
        Maximum allowed Levenshtein distance (0, 1, or 2)
    """
    if not skeleton:
        return 0
    n_segments = len(skeleton.split('-'))
    if n_segments <= 4:
        return 0
    else:
        return 1

def _is_vowel_position(stem: str, idx: int) -> bool:
    """Check if the character at idx is surrounded by vowels or at stem-start."""
    if idx == 0:
        return True

    prev_is_vowel = (idx > 0 and stem[idx - 1] in VOYNICH_VOWELS)
    next_is_vowel = (idx < len(stem) - 1 and stem[idx + 1] in VOYNICH_VOWELS)

    return prev_is_vowel and next_is_vowel

class FuzzySkeletonizer:
    """
    Produces multiple weighted consonant skeleton candidates from a
    Voynich stem, branching on semi-consonant y/o positions.

    Unlike Phase 11's VoynichPhoneticSkeletonizer which returns a single
    skeleton, this returns a ranked list of (skeleton, weight) tuples.
    """

    def __init__(self, v_morphemer):
        """
        Args:
            v_morphemer: VoynichMorphemer instance (Phase 7) for affix stripping
        """
        self.v_morphemer = v_morphemer

    def get_skeleton_candidates(self, v_stem: str) -> List[Tuple[str, float]]:
        """
        Generate all plausible consonant skeletons for a Voynich stem,
        branching on y/o semi-consonant positions.

        Args:
            v_stem: A stripped Voynich stem (no prefix/suffix)

        Returns:
            List of (skeleton_string, probability_weight) tuples.
            Primary (vowel-skip) branch gets 0.7, secondary (consonant) gets 0.3.
            If multiple branch points exist, weights are multiplied.
        """
        if not v_stem:
            return []

        segments = []
        i = 0
        while i < len(v_stem):
            if i < len(v_stem) - 1:
                digraph = v_stem[i:i + 2]
                if digraph in VOYNICH_CONSONANT_CLASSES:
                    segments.append(VOYNICH_CONSONANT_CLASSES[digraph])
                    i += 2
                    continue

            char = v_stem[i]

            if char in VOYNICH_CONSONANT_CLASSES:
                segments.append(VOYNICH_CONSONANT_CLASSES[char])
                i += 1
                continue

            if char in SEMI_CONSONANT_MAP and _is_vowel_position(v_stem, i):
                segments.append((None, SEMI_CONSONANT_MAP[char]))
                i += 1
                continue

            i += 1

        if not segments:
            return []

        branch_points = []
        for seg in segments:
            if isinstance(seg, tuple):
                branch_points.append(seg)
            else:
                branch_points.append((seg,))

        candidates = {}
        for combo in product(*branch_points):
            consonants = [c for c in combo if c is not None]

            deduped = []
            for c in consonants:
                if not deduped or c != deduped[-1]:
                    deduped.append(c)

            if not deduped:
                continue

            skeleton = '-'.join(deduped)

            weight = 1.0
            branch_idx = 0
            for seg in segments:
                if isinstance(seg, tuple):
                    chosen = combo[branch_idx]
                    if chosen is None:
                        weight *= 0.7
                    else:
                        weight *= 0.3
                    branch_idx += 1
                else:
                    branch_idx += 1

            if skeleton in candidates:
                candidates[skeleton] = max(candidates[skeleton], weight)
            else:
                candidates[skeleton] = weight

        result = sorted(candidates.items(), key=lambda x: -x[1])
        return result
