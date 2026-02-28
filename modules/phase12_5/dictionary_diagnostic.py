"""
Phase 12.5: Dictionary Coverage Diagnostic
===========================================
Audits the Latin dictionary to quantify exactly how many unresolved tokens
fail because their consonant skeleton has zero dictionary entries vs. tokens
that had candidates but were rejected by the solver.

This separates "dictionary ceiling" failures (fixable by vocabulary expansion)
from "solver threshold" failures (fixable by tuning).

Phase 12.5  ·  Voynich Convergence Attack
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer, dynamic_levenshtein_threshold
from modules.phase12.budgeted_csp import BudgetedCSPDecoder
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver

import Levenshtein

UNRESOLVED_RE = re.compile(r'\[([^_\]]+)_UNRESOLVED\]|<([^_>]+)_UNRESOLVED>')

class DictionaryDiagnostic:
    """
    Categorizes every unresolved Voynich token into:
      - ZERO_MATCH:  skeleton has no dictionary entries (dictionary expansion helps)
      - MATCH_BUT_UNRESOLVED: skeleton matched entries but solver rejected
      - NO_SKELETON: token couldn't produce a valid consonant skeleton
    """

    def __init__(
        self,
        budgeted_decoder: BudgetedCSPDecoder,
        scaffolder: SyntacticScaffolder,
        ngram_solver: NgramMaskSolver,
        latin_skel: LatinPhoneticSkeletonizer,
        fuzzy_skel: FuzzySkeletonizer,
        l_tokens: list,
    ):
        self.decoder = budgeted_decoder
        self.scaffolder = scaffolder
        self.solver = ngram_solver
        self.latin_skel = latin_skel
        self.fuzzy_skel = fuzzy_skel
        self.l_tokens = l_tokens

    def _decode_tokens(self, tokens: List[str], folio_id: str = None) -> str:
        """Run tokens through the full Phase 12 pipeline."""
        self.decoder.emission_counts = {}
        self.decoder._folio_token_count = len(tokens)

        decoded = self.decoder.decode_folio(tokens, folio_id=folio_id)
        scaffolded = self.scaffolder.scaffold(decoded)
        resolved, _ = self.solver.solve_folio(scaffolded, folio_id=folio_id)
        return resolved

    def _has_skeleton_match(self, skeleton: str) -> bool:
        """Check if a skeleton has any entries in the Latin skeleton index."""
        if not skeleton:
            return False
        if skeleton in self.latin_skel.skeleton_index:
            return True
        threshold = dynamic_levenshtein_threshold(skeleton)
        if threshold > 0:
            for latin_skel_key in self.latin_skel.skeleton_index:
                dist = Levenshtein.distance(skeleton, latin_skel_key)
                if dist <= threshold:
                    return True
        return False

    def _categorize_token(self, voynich_token: str) -> Tuple[str, List[str]]:
        """
        Categorize an unresolved token.

        Returns:
            (category, skeletons) where category is one of:
            'ZERO_MATCH', 'MATCH_IN_MATRIX', 'MATCH_NOT_IN_MATRIX', 'NO_SKELETON'

            MATCH_IN_MATRIX = has candidates AND at least one is in transition matrix
            MATCH_NOT_IN_MATRIX = has candidates but NONE are in transition matrix
        """
        prefix, stem, suffix = self.fuzzy_skel.v_morphemer._strip_affixes(voynich_token)

        if not stem:
            return 'NO_SKELETON', []

        candidates = self.fuzzy_skel.get_skeleton_candidates(stem)

        if not candidates:
            return 'NO_SKELETON', []

        skeletons = [skel for skel, _ in candidates]

        has_match = any(self._has_skeleton_match(skel) for skel in skeletons)

        if not has_match:
            return 'ZERO_MATCH', skeletons

        matrix_vocab = set(self.solver.word_to_idx.keys())
        for skel in skeletons:
            words = self.latin_skel.skeleton_index.get(skel, [])
            if any(w in matrix_vocab for w in words):
                return 'MATCH_IN_MATRIX', skeletons
            threshold = dynamic_levenshtein_threshold(skel)
            if threshold > 0:
                for l_skel in self.latin_skel.skeleton_index:
                    dist = Levenshtein.distance(skel, l_skel)
                    if dist <= threshold:
                        fuzz_words = self.latin_skel.skeleton_index[l_skel]
                        if any(w in matrix_vocab for w in fuzz_words):
                            return 'MATCH_IN_MATRIX', skeletons

        return 'MATCH_NOT_IN_MATRIX', skeletons

    def run(
        self,
        by_folio: Dict[str, List[str]],
        folio_limit: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the dictionary coverage diagnostic.

        Args:
            by_folio: Dict of folio_id -> token list
            folio_limit: Number of folios to process
            verbose: Print progress

        Returns:
            Results dict with coverage analysis
        """
        if verbose:
            print('\n  Running dictionary coverage diagnostic...')

        folio_items = list(by_folio.items())[:folio_limit]

        all_unresolved: List[str] = []
        total_words = 0
        total_resolved = 0

        for i, (folio_id, tokens) in enumerate(folio_items):
            if len(tokens) < 5:
                continue

            resolved_text = self._decode_tokens(tokens, folio_id=folio_id)
            words = resolved_text.split()
            total_words += len(words)

            for w in words:
                if not (w.startswith('[') or w.startswith('<')):
                    total_resolved += 1

            for match in UNRESOLVED_RE.finditer(resolved_text):
                token = match.group(1) or match.group(2)
                all_unresolved.append(token)

            if verbose and (i + 1) % 5 == 0:
                print(f'    Processed {i + 1}/{len(folio_items)} folios...')

        category_counts = Counter()
        zero_match_skeletons: Counter = Counter()
        match_in_matrix_skeletons: Counter = Counter()
        match_not_in_matrix_skeletons: Counter = Counter()
        no_skeleton_tokens: Counter = Counter()
        zero_match_tokens: Counter = Counter()
        match_not_in_matrix_tokens: Counter = Counter()

        for token in all_unresolved:
            category, skeletons = self._categorize_token(token)
            category_counts[category] += 1

            if category == 'ZERO_MATCH':
                zero_match_tokens[token] += 1
                for skel in skeletons:
                    zero_match_skeletons[skel] += 1
            elif category == 'MATCH_IN_MATRIX':
                for skel in skeletons:
                    match_in_matrix_skeletons[skel] += 1
            elif category == 'MATCH_NOT_IN_MATRIX':
                match_not_in_matrix_tokens[token] += 1
                for skel in skeletons:
                    match_not_in_matrix_skeletons[skel] += 1
            elif category == 'NO_SKELETON':
                no_skeleton_tokens[token] += 1

        total_unresolved = len(all_unresolved)
        matrix_vocab_size = len(self.solver.word_to_idx)

        results = {
            'test': 'dictionary_diagnostic',
            'pass': True,
            'corpus_stats': {
                'latin_types': len(set(self.l_tokens)),
                'latin_tokens': len(self.l_tokens),
                'latin_skeletons': len(self.latin_skel.skeleton_index),
                'matrix_vocab_size': matrix_vocab_size,
            },
            'resolution_stats': {
                'folios_processed': len(folio_items),
                'total_words': total_words,
                'total_resolved': total_resolved,
                'total_unresolved': total_unresolved,
                'resolution_rate': total_resolved / max(1, total_words),
            },
            'category_breakdown': {
                'ZERO_MATCH': {
                    'count': category_counts.get('ZERO_MATCH', 0),
                    'pct': category_counts.get('ZERO_MATCH', 0) / max(1, total_unresolved),
                    'description': 'Skeleton has no dictionary entries — dictionary expansion needed',
                },
                'MATCH_IN_MATRIX': {
                    'count': category_counts.get('MATCH_IN_MATRIX', 0),
                    'pct': category_counts.get('MATCH_IN_MATRIX', 0) / max(1, total_unresolved),
                    'description': 'Candidates exist in transition matrix but solver rejected (ambiguous/ratio)',
                },
                'MATCH_NOT_IN_MATRIX': {
                    'count': category_counts.get('MATCH_NOT_IN_MATRIX', 0),
                    'pct': category_counts.get('MATCH_NOT_IN_MATRIX', 0) / max(1, total_unresolved),
                    'description': 'Candidates exist but NONE in transition matrix (score 0.0)',
                },
                'NO_SKELETON': {
                    'count': category_counts.get('NO_SKELETON', 0),
                    'pct': category_counts.get('NO_SKELETON', 0) / max(1, total_unresolved),
                    'description': 'Token could not produce a valid consonant skeleton',
                },
            },
            'top_50_zero_match_skeletons': [
                {'skeleton': skel, 'frequency': freq}
                for skel, freq in zero_match_skeletons.most_common(50)
            ],
            'top_50_match_in_matrix_skeletons': [
                {'skeleton': skel, 'frequency': freq}
                for skel, freq in match_in_matrix_skeletons.most_common(50)
            ],
            'top_50_match_not_in_matrix_tokens': [
                {'token': tok, 'frequency': freq}
                for tok, freq in match_not_in_matrix_tokens.most_common(50)
            ],
            'top_50_zero_match_tokens': [
                {'token': tok, 'frequency': freq}
                for tok, freq in zero_match_tokens.most_common(50)
            ],
            'all_zero_match_skeletons': [
                {'skeleton': skel, 'frequency': freq}
                for skel, freq in zero_match_skeletons.most_common()
            ],
        }

        if verbose:
            print(f'\n  Dictionary Coverage Diagnostic Results:')
            print(f'    Latin corpus: {len(set(self.l_tokens))} types, '
                  f'{len(self.latin_skel.skeleton_index)} unique skeletons')
            print(f'    Transition matrix: {matrix_vocab_size} words')
            print(f'    Folios processed: {len(folio_items)}')
            print(f'    Total words: {total_words}')
            print(f'    Resolved: {total_resolved} '
                  f'({100 * total_resolved / max(1, total_words):.1f}%)')
            print(f'    Unresolved: {total_unresolved}')
            print(f'\n    Category Breakdown:')
            for cat in ['ZERO_MATCH', 'MATCH_IN_MATRIX', 'MATCH_NOT_IN_MATRIX', 'NO_SKELETON']:
                c = category_counts.get(cat, 0)
                pct = 100 * c / max(1, total_unresolved)
                print(f'      {cat:25s}: {c:5d} ({pct:5.1f}%)')

            if zero_match_skeletons:
                print(f'\n    Top 20 ZERO_MATCH skeletons (priority for expansion):')
                for skel, freq in zero_match_skeletons.most_common(20):
                    print(f'      {skel:20s} × {freq}')

            if zero_match_tokens:
                print(f'\n    Top 20 ZERO_MATCH tokens:')
                for tok, freq in zero_match_tokens.most_common(20):
                    print(f'      {tok:20s} × {freq}')

            if match_not_in_matrix_tokens:
                print(f'\n    Top 20 MATCH_NOT_IN_MATRIX tokens (candidates exist but OOV):')
                for tok, freq in match_not_in_matrix_tokens.most_common(20):
                    print(f'      {tok:20s} × {freq}')

        return results
