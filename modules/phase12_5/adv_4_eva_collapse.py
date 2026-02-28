"""
Phase 12.5, Test 4: EVA Collapse — The "EVA Illusion"
======================================================
Proves that morphological paradigms and CSP resolutions are not just
artifacts of how the modern EVA alphabet treats benched gallows
(cth, ckh, cph) as multiple letters.

By collapsing all complex EVA ligatures into single unique characters
and re-running the pipeline, we show that the decryption is immune
to transliteration disputes — the underlying phonetic skeletons
remain identical regardless of character representation.

Phase 12.5  ·  Voynich Convergence Attack
"""

import copy
import re
from typing import Dict, List, Tuple

from modules.phase11.phonetic_skeletonizer import (
    LatinPhoneticSkeletonizer,
    VOYNICH_CONSONANT_CLASSES,
)
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver
from modules.phase7.voynich_morphemer import VoynichMorphemer
from orchestrators._utils import _resolution_rate

COLLAPSE_MAP = {
    'cth': 'X',
    'ckh': 'Y',
    'cph': 'Z',
    'cfh': 'W',
    'sh':  'S',
    'ch':  'C',
}

COLLAPSED_CONSONANT_CLASSES = {
    'X': 'T',
    'Y': 'K',
    'Z': 'P',
    'W': 'P',
    'S': 'S',
    'C': 'K',
}

_COLLAPSE_ORDER = sorted(COLLAPSE_MAP.items(), key=lambda x: -len(x[0]))

def _collapse_token(token: str) -> str:
    """Apply ligature collapse to a single token (longest-match-first)."""
    result = token
    for old, new in _COLLAPSE_ORDER:
        result = result.replace(old, new)
    return result

def _collapse_tokens(tokens: List[str]) -> List[str]:
    """Apply ligature collapse to all tokens."""
    return [_collapse_token(t) for t in tokens]

class CollapsedFuzzySkeletonizer(FuzzySkeletonizer):
    """
    FuzzySkeletonizer variant that recognizes collapsed ligature characters.

    Overrides the parent's skeleton extraction to handle single-char
    ligatures (C, S, X, Y, Z, W) in addition to standard EVA consonants.
    """

    def get_skeleton_candidates(self, v_stem: str):
        """
        Generate skeleton candidates, treating collapsed characters
        as their consonant-class equivalents.
        """
        from modules.phase12.fuzzy_skeletonizer import (
            SEMI_CONSONANT_MAP, _is_vowel_position,
        )
        from itertools import product

        if not v_stem:
            return []

        extended_classes = dict(VOYNICH_CONSONANT_CLASSES)
        extended_classes.update(COLLAPSED_CONSONANT_CLASSES)

        segments = []
        i = 0
        while i < len(v_stem):
            if i < len(v_stem) - 1:
                digraph = v_stem[i:i + 2]
                if digraph in extended_classes:
                    segments.append(extended_classes[digraph])
                    i += 2
                    continue

            char = v_stem[i]

            if char in extended_classes:
                segments.append(extended_classes[char])
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
                    weight *= 0.7 if chosen is None else 0.3
                    branch_idx += 1
                else:
                    branch_idx += 1

            if skeleton in candidates:
                candidates[skeleton] = max(candidates[skeleton], weight)
            else:
                candidates[skeleton] = weight

        return sorted(candidates.items(), key=lambda x: -x[1])

class CollapsedMorphemer:
    """
    Wrapper around VoynichMorphemer that updates prefix/suffix sets
    to account for collapsed ligature characters.
    """

    def __init__(self, v_morphemer: VoynichMorphemer):
        self._inner = v_morphemer

        self.valid_prefixes = set(v_morphemer.valid_prefixes)
        self.valid_suffixes = set(v_morphemer.valid_suffixes)

        for pref in list(self.valid_prefixes):
            collapsed = _collapse_token(pref)
            if collapsed != pref:
                self.valid_prefixes.add(collapsed)

        for suf in list(self.valid_suffixes):
            collapsed = _collapse_token(suf)
            if collapsed != suf:
                self.valid_suffixes.add(collapsed)

    def _strip_affixes(self, word: str):
        """Strip affixes using both original and collapsed affix sets."""
        best_pref, best_suff = '', ''

        for p_len in range(min(5, len(word) - 1), 0, -1):
            if word[:p_len] in self.valid_prefixes:
                best_pref = word[:p_len]
                break

        remainder = word[len(best_pref):]

        for s_len in range(min(5, len(remainder) - 1), 0, -1):
            if remainder[-s_len:] in self.valid_suffixes:
                best_suff = remainder[-s_len:]
                break

        stem = remainder[:-len(best_suff)] if best_suff else remainder
        return best_pref, stem, best_suff

class EvaCollapseTest:
    """
    Adversarial test proving the decryption is immune to EVA
    transliteration disputes by collapsing ligatures into single chars.
    """

    def __init__(
        self,
        budgeted_decoder: BudgetedCSPDecoder,
        scaffolder: SyntacticScaffolder,
        ngram_solver: NgramMaskSolver,
        latin_skeletonizer: LatinPhoneticSkeletonizer,
        v_morphemer: VoynichMorphemer,
        corpus_tokens: List[str],
        folio_metadata: Dict = None,
        herbal_matrix=None,
        herbal_vocab=None,
    ):
        self.decoder = budgeted_decoder
        self.scaffolder = scaffolder
        self.solver = ngram_solver
        self.l_skel = latin_skeletonizer
        self.v_morphemer = v_morphemer
        self.corpus_tokens = corpus_tokens
        self.folio_metadata = folio_metadata
        self.herbal_matrix = herbal_matrix
        self.herbal_vocab = herbal_vocab

    def _decode_standard(
        self, tokens: List[str], folio_id: str,
    ) -> str:
        """Run standard (uncollapsed) Phase 12 pipeline."""
        self.decoder.emission_counts = {}
        self.decoder._folio_token_count = len(tokens)

        decoded = self.decoder.decode_folio(tokens, folio_id=folio_id)
        scaffolded = self.scaffolder.scaffold(decoded)
        resolved, _ = self.solver.solve_folio(scaffolded, folio_id=folio_id)
        return resolved

    def _decode_collapsed(
        self, tokens: List[str], folio_id: str,
    ) -> str:
        """Run collapsed-ligature Phase 12 pipeline."""
        collapsed_tokens = _collapse_tokens(tokens)

        collapsed_morph = CollapsedMorphemer(self.v_morphemer)
        collapsed_fuzzy = CollapsedFuzzySkeletonizer(collapsed_morph)

        collapsed_decoder = BudgetedCSPDecoder(
            self.l_skel, collapsed_fuzzy,
            self.corpus_tokens, self.folio_metadata,
        )

        collapsed_scaffolder = SyntacticScaffolder(collapsed_morph)

        collapsed_solver = NgramMaskSolver(
            self.herbal_matrix, self.herbal_vocab,
            self.l_skel, collapsed_fuzzy,
            min_confidence_ratio=3.0,
        )

        decoded = collapsed_decoder.decode_folio(collapsed_tokens, folio_id=folio_id)
        scaffolded = collapsed_scaffolder.scaffold(decoded)
        resolved, _ = collapsed_solver.solve_folio(scaffolded, folio_id=folio_id)
        return resolved

    def run(
        self,
        by_folio: Dict[str, List[str]],
        folio_limit: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the EVA collapse test.

        Args:
            by_folio: Dict of folio_id -> token list
            folio_limit: Number of folios to process
            verbose: Print progress

        Returns:
            Results dict with agreement rate and pass/fail
        """
        total_words = 0
        matching_words = 0
        per_folio = {}

        for folio, tokens in list(by_folio.items())[:folio_limit]:
            if len(tokens) < 5:
                continue

            standard = self._decode_standard(tokens, folio)
            collapsed = self._decode_collapsed(tokens, folio)

            std_words = standard.split()
            col_words = collapsed.split()

            n_compare = min(len(std_words), len(col_words))
            n_match = sum(
                1 for s, c in zip(std_words[:n_compare], col_words[:n_compare])
                if s == c
            )

            total_words += n_compare
            matching_words += n_match

            folio_rate = n_match / max(1, n_compare)
            per_folio[folio] = {
                'total_words': n_compare,
                'matching_words': n_match,
                'agreement_rate': round(folio_rate, 4),
            }

            if verbose:
                print(f'    {folio}: {folio_rate:.1%} agreement '
                      f'({n_match}/{n_compare} words)')

        overall_rate = matching_words / max(1, total_words)
        test_pass = overall_rate > 0.90

        if verbose:
            verdict = 'PASS' if test_pass else 'FAIL'
            print(f'    Overall: {overall_rate:.1%} word-level agreement')
            print(f'    Verdict: {verdict} (threshold: >90%)')

        return {
            'test': 'eva_collapse',
            'folio_limit': folio_limit,
            'total_words': total_words,
            'matching_words': matching_words,
            'agreement_rate': round(overall_rate, 4),
            'per_folio': per_folio,
            'pass': test_pass,
        }
