"""
Phase 12.5, Test 2: Domain Swap — The "Texas Sharpshooter"
===========================================================
Proves that the pipeline didn't "force" the text to be a herbal just
because it was given a herbal transition matrix. By swapping in Bible
and Legal transition matrices (while keeping the same dictionary), we
show that the Voynich transitions inherently match a botanical text.

If the Herbal matrix yields ~74.5% resolution but Bible and Legal
matrices cause massive resolution drops, the Voynich text is genuinely
a herbal — not just any Latin text projected through the skeleton.

Phase 12.5  ·  Voynich Convergence Attack
"""

import os
import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder, HUMORAL_VOCAB
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver

# Data file paths (relative to project root)
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'corpora')
VULGATE_PATH = os.path.join(_DATA_DIR, 'latin_vulgate_sample.txt')
LEGAL_PATH = os.path.join(_DATA_DIR, 'corpus_juris_civilis.txt')


def _load_and_tile_corpus(filepath: str, target_tokens: int = 30012) -> List[str]:
    """
    Load a corpus text file and tile (loop) it to match the target token count.

    Args:
        filepath: Path to the corpus text file
        target_tokens: Number of tokens to tile to (default matches herbal corpus)

    Returns:
        List of tokens of length target_tokens
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Clean: lowercase, strip punctuation, split to tokens
    raw = raw.lower()
    raw = re.sub(r'[^a-z\s]', '', raw)
    base_tokens = raw.split()

    if not base_tokens:
        return []

    # Tile by repeating
    tiled = []
    while len(tiled) < target_tokens:
        tiled.extend(base_tokens)
    return tiled[:target_tokens]


def build_transition_matrix_from_tokens(
    tokens: List[str], top_n: int = 1001,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a word-level transition probability matrix P(w_j | w_i).

    Args:
        tokens: Corpus token list
        top_n: Number of most-frequent words to include

    Returns:
        (matrix[n x n], vocab[n])
    """
    freqs = Counter(tokens)
    vocab = [w for w, _ in freqs.most_common(top_n)]
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)

    counts = np.zeros((n, n), dtype=float)
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        if w1 in word_to_idx and w2 in word_to_idx:
            counts[word_to_idx[w1]][word_to_idx[w2]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, vocab


def _resolution_rate(decoded_text: str) -> float:
    """Compute the fraction of words that are NOT bracketed."""
    words = decoded_text.split()
    if not words:
        return 0.0
    brackets = sum(1 for w in words if w.startswith('[') or w.startswith('<'))
    return 1.0 - (brackets / len(words))


class DomainSwapTest:
    """
    Adversarial test proving that the Voynich text's transition
    statistics match a herbal domain, not just any Latin domain.
    """

    def __init__(
        self,
        budgeted_decoder: BudgetedCSPDecoder,
        scaffolder: SyntacticScaffolder,
        latin_skeletonizer: LatinPhoneticSkeletonizer,
        fuzzy_skeletonizer: FuzzySkeletonizer,
        herbal_matrix: np.ndarray,
        herbal_vocab: List[str],
    ):
        self.decoder = budgeted_decoder
        self.scaffolder = scaffolder
        self.l_skel = latin_skeletonizer
        self.f_skel = fuzzy_skeletonizer
        self.herbal_matrix = herbal_matrix
        self.herbal_vocab = herbal_vocab

    def _decode_with_matrix(
        self,
        by_folio: Dict[str, List[str]],
        matrix: np.ndarray,
        vocab: List[str],
        folio_limit: int,
        label: str,
        verbose: bool,
    ) -> Dict:
        """Run the Phase 12 pipeline with a specific transition matrix."""
        solver = NgramMaskSolver(
            matrix, vocab, self.l_skel, self.f_skel,
            humoral_vocab=HUMORAL_VOCAB,
            min_confidence_ratio=3.0,
        )

        total_words = 0
        total_brackets = 0
        per_folio = {}

        for folio, tokens in list(by_folio.items())[:folio_limit]:
            if len(tokens) < 5:
                continue

            # Reset decoder per folio
            self.decoder.emission_counts = {}
            self.decoder._folio_token_count = len(tokens)

            decoded = self.decoder.decode_folio(tokens, folio_id=folio)
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
                  f'({total_words} words, {total_brackets} brackets)')

        return {
            'label': label,
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
        Run the domain swap test with Herbal, Bible, and Legal matrices.

        Args:
            by_folio: Dict of folio_id -> token list
            folio_limit: Number of folios to process
            verbose: Print progress

        Returns:
            Results dict with per-domain rates and pass/fail
        """
        # Load and tile alternative corpora
        if verbose:
            print('    Loading alternative corpora...')

        bible_tokens = _load_and_tile_corpus(VULGATE_PATH)
        legal_tokens = _load_and_tile_corpus(LEGAL_PATH)

        # Build transition matrices
        bible_matrix, bible_vocab = build_transition_matrix_from_tokens(bible_tokens)
        legal_matrix, legal_vocab = build_transition_matrix_from_tokens(legal_tokens)

        if verbose:
            print(f'    Bible corpus: {len(bible_tokens)} tokens, '
                  f'{bible_matrix.shape[0]}x{bible_matrix.shape[1]} matrix')
            print(f'    Legal corpus: {len(legal_tokens)} tokens, '
                  f'{legal_matrix.shape[0]}x{legal_matrix.shape[1]} matrix')

        # Run pipeline with each matrix
        herbal_results = self._decode_with_matrix(
            by_folio, self.herbal_matrix, self.herbal_vocab,
            folio_limit, 'Herbal', verbose,
        )
        bible_results = self._decode_with_matrix(
            by_folio, bible_matrix, bible_vocab,
            folio_limit, 'Bible', verbose,
        )
        legal_results = self._decode_with_matrix(
            by_folio, legal_matrix, legal_vocab,
            folio_limit, 'Legal', verbose,
        )

        herbal_rate = herbal_results['overall_rate']
        bible_rate = bible_results['overall_rate']
        legal_rate = legal_results['overall_rate']

        test_pass = herbal_rate > bible_rate and herbal_rate > legal_rate

        if verbose:
            verdict = 'PASS' if test_pass else 'FAIL'
            print(f'    Verdict: {verdict} '
                  f'(Herbal {herbal_rate:.1%} > Bible {bible_rate:.1%}, '
                  f'Legal {legal_rate:.1%})')

        return {
            'test': 'domain_swap',
            'folio_limit': folio_limit,
            'herbal': herbal_results,
            'bible': bible_results,
            'legal': legal_results,
            'pass': test_pass,
        }
