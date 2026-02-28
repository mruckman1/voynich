"""
Phase 12.5, Test 3: Polyglot Dictionary — The "A Priori Latin Bias"
====================================================================
Quantifies how much "Latin" the text actually is by running the Phase 12
CSP pipeline with Italian and Occitan dictionaries instead of Latin.

If Latin resolves at the highest rate, Italian close behind, and Occitan
lower, this provides empirical proof that the output is best understood
as an isomorphic projection of a pan-Romance medical vernacular into
standard Latin.

Phase 12.5  ·  Voynich Convergence Attack
"""

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder, HUMORAL_VOCAB
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver

_DICT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dictionaries')
LATIN_DICT_PATH = os.path.join(_DICT_DIR, 'phase5_latin_dict.json')
ITALIAN_DICT_PATH = os.path.join(_DICT_DIR, 'romance_italian_dict.json')
OCCITAN_DICT_PATH = os.path.join(_DICT_DIR, 'romance_occitan_dict.json')

def _load_dict(filepath: str) -> List[str]:
    """Load a dictionary JSON array file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        words = json.load(f)
    seen = set()
    unique = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique

def _resolution_rate(decoded_text: str) -> float:
    """Compute the fraction of words that are NOT bracketed."""
    words = decoded_text.split()
    if not words:
        return 0.0
    brackets = sum(1 for w in words if w.startswith('[') or w.startswith('<'))
    return 1.0 - (brackets / len(words))

class PolyglotDictTest:
    """
    Adversarial test quantifying the Latin specificity of the Voynich
    decoding by comparing resolution rates across Romance dictionaries.
    """

    def __init__(
        self,
        fuzzy_skeletonizer: FuzzySkeletonizer,
        scaffolder: SyntacticScaffolder,
        herbal_matrix: np.ndarray,
        herbal_vocab: List[str],
        folio_metadata: Dict = None,
    ):
        self.f_skel = fuzzy_skeletonizer
        self.scaffolder = scaffolder
        self.herbal_matrix = herbal_matrix
        self.herbal_vocab = herbal_vocab
        self.folio_metadata = folio_metadata

    def _build_pipeline_for_dict(
        self,
        word_list: List[str],
    ) -> Tuple[BudgetedCSPDecoder, NgramMaskSolver]:
        """
        Build a Phase 12 pipeline using an alternative dictionary.

        Creates a new LatinPhoneticSkeletonizer from the given word list,
        then wires it into a BudgetedCSPDecoder and NgramMaskSolver.
        """
        skel = LatinPhoneticSkeletonizer(word_list)

        decoder = BudgetedCSPDecoder(
            skel, self.f_skel, word_list, self.folio_metadata
        )

        solver = NgramMaskSolver(
            self.herbal_matrix, self.herbal_vocab,
            skel, self.f_skel,
            humoral_vocab=HUMORAL_VOCAB,
            min_confidence_ratio=3.0,
        )

        return decoder, solver

    def _decode_with_dict(
        self,
        by_folio: Dict[str, List[str]],
        word_list: List[str],
        folio_limit: int,
        label: str,
        verbose: bool,
    ) -> Dict:
        """Run the Phase 12 pipeline with an alternative dictionary."""
        decoder, solver = self._build_pipeline_for_dict(word_list)

        total_words = 0
        total_brackets = 0
        per_folio = {}

        for folio, tokens in list(by_folio.items())[:folio_limit]:
            if len(tokens) < 5:
                continue

            decoded = decoder.decode_folio(tokens, folio_id=folio)
            scaffolded = self.scaffolder.scaffold(decoded)
            resolved, stats = solver.solve_folio(scaffolded, folio_id=folio)

            words = resolved.split()
            n_words = len(words)
            n_brackets = sum(1 for w in words
                             if w.startswith('[') or w.startswith('<'))
            rate = _resolution_rate(resolved)

            total_words += n_words
            total_brackets += n_brackets
            per_folio[folio] = {
                'words': n_words,
                'brackets': n_brackets,
                'rate': round(rate, 4),
            }

        overall_rate = 1.0 - (total_brackets / max(1, total_words))

        if verbose:
            print(f'    {label}: {overall_rate:.1%} resolution '
                  f'({len(word_list)} dict words, '
                  f'{total_words} output words, {total_brackets} brackets)')

        return {
            'label': label,
            'dict_size': len(word_list),
            'overall_rate': round(overall_rate, 4),
            'total_words': total_words,
            'total_brackets': total_brackets,
            'per_folio': per_folio,
        }

    def run(
        self,
        by_folio: Dict[str, List[str]],
        folio_limit: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the polyglot dictionary test.

        Args:
            by_folio: Dict of folio_id -> token list
            folio_limit: Number of folios to process
            verbose: Print progress

        Returns:
            Results dict with per-language rates and pass/fail
        """
        latin_words = _load_dict(LATIN_DICT_PATH)
        italian_words = _load_dict(ITALIAN_DICT_PATH)
        occitan_words = _load_dict(OCCITAN_DICT_PATH)

        if verbose:
            print(f'    Latin dict: {len(latin_words)} words')
            print(f'    Italian dict: {len(italian_words)} words')
            print(f'    Occitan dict: {len(occitan_words)} words')

        latin_results = self._decode_with_dict(
            by_folio, latin_words, folio_limit, 'Latin', verbose,
        )
        italian_results = self._decode_with_dict(
            by_folio, italian_words, folio_limit, 'Italian', verbose,
        )
        occitan_results = self._decode_with_dict(
            by_folio, occitan_words, folio_limit, 'Occitan', verbose,
        )

        latin_rate = latin_results['overall_rate']
        italian_rate = italian_results['overall_rate']
        occitan_rate = occitan_results['overall_rate']

        test_pass = latin_rate >= italian_rate >= occitan_rate

        if verbose:
            verdict = 'PASS' if test_pass else 'FAIL'
            print(f'    Verdict: {verdict} '
                  f'(Latin {latin_rate:.1%} >= Italian {italian_rate:.1%} '
                  f'>= Occitan {occitan_rate:.1%})')

        return {
            'test': 'polyglot_dict',
            'folio_limit': folio_limit,
            'latin': latin_results,
            'italian': italian_results,
            'occitan': occitan_results,
            'pass': test_pass,
        }
