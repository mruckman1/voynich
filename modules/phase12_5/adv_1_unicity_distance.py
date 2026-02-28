"""
Phase 12.5, Test 1: Unicity Distance — The "Procrustean Bed"
=============================================================
Proves that the Fuzzy Skeletonizer is not so loose that it turns any
random text into Latin. Establishes the unicity distance baseline by
running scrambled and frequency-matched random EVA text through the
Phase 12 pipeline.

If the real Voynich resolves at ~74.5% but scrambled/random controls
resolve at <15%, the skeletonizer is mathematically rigorous — it
requires the specific underlying phonetic structure of the Voynich.

Phase 12.5  ·  Voynich Convergence Attack
"""

import random
import re
from collections import Counter
from typing import Dict, List, Optional

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver
from orchestrators._utils import _resolution_rate

EVA_CHARS = list('oainedylrsktchpfqmgx')

def _scramble_tokens(tokens: List[str], seed: int) -> List[str]:
    """Shuffle characters within each token, preserving word lengths."""
    rng = random.Random(seed)
    scrambled = []
    for token in tokens:
        chars = list(token)
        rng.shuffle(chars)
        scrambled.append(''.join(chars))
    return scrambled

def _build_eva_frequency_dist(tokens: List[str]) -> List[float]:
    """Build character frequency distribution from real Voynich tokens."""
    char_counts = Counter()
    for token in tokens:
        for ch in token:
            if ch in set(EVA_CHARS):
                char_counts[ch] += 1
    total = sum(char_counts.values()) or 1
    return [char_counts.get(ch, 0) / total for ch in EVA_CHARS]

def _generate_random_tokens(tokens: List[str], char_weights: List[float],
                            seed: int) -> List[str]:
    """Generate random EVA tokens matching the word-length distribution."""
    rng = random.Random(seed)
    random_tokens = []
    for token in tokens:
        length = len(token)
        chars = rng.choices(EVA_CHARS, weights=char_weights, k=length)
        random_tokens.append(''.join(chars))
    return random_tokens

class UnicityDistanceTest:
    """
    Adversarial test proving the Phase 12 pipeline requires genuine
    Voynich phonetic structure, not just any EVA-like character sequence.
    """

    def __init__(
        self,
        budgeted_decoder: BudgetedCSPDecoder,
        scaffolder: SyntacticScaffolder,
        ngram_solver: NgramMaskSolver,
    ):
        self.decoder = budgeted_decoder
        self.scaffolder = scaffolder
        self.solver = ngram_solver

    def _decode_tokens(self, tokens: List[str], folio_id: str = None) -> str:
        """Run tokens through the full Phase 12 pipeline."""
        self.decoder.emission_counts = {}
        self.decoder._folio_token_count = len(tokens)

        decoded = self.decoder.decode_folio(tokens, folio_id=folio_id)
        scaffolded = self.scaffolder.scaffold(decoded)
        resolved, _ = self.solver.solve_folio(scaffolded, folio_id=folio_id)
        return resolved

    def run(
        self,
        by_folio: Dict[str, List[str]],
        folio_id: str = 'f1r',
        n_trials: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the unicity distance test.

        Args:
            by_folio: Dict of folio_id -> token list from extractor
            folio_id: Target folio to test (default f1r)
            n_trials: Number of randomization trials
            verbose: Print progress

        Returns:
            Results dict with real/scrambled/random rates and pass/fail
        """
        if folio_id not in by_folio:
            folio_id = next(iter(by_folio))
            if verbose:
                print(f'    (folio not found, falling back to {folio_id})')

        real_tokens = by_folio[folio_id]

        if verbose:
            print(f'    Target folio: {folio_id} ({len(real_tokens)} tokens)')

        real_decoded = self._decode_tokens(real_tokens, folio_id=folio_id)
        real_rate = _resolution_rate(real_decoded)
        if verbose:
            print(f'    Real resolution rate: {real_rate:.1%}')

        scrambled_rates = []
        for trial in range(n_trials):
            scrambled = _scramble_tokens(real_tokens, seed=42 + trial)
            decoded = self._decode_tokens(scrambled, folio_id=folio_id)
            rate = _resolution_rate(decoded)
            scrambled_rates.append(rate)

        mean_scrambled = sum(scrambled_rates) / len(scrambled_rates)
        if verbose:
            print(f'    Scrambled mean rate: {mean_scrambled:.1%} '
                  f'(range {min(scrambled_rates):.1%}–{max(scrambled_rates):.1%})')

        char_weights = _build_eva_frequency_dist(real_tokens)
        random_rates = []
        for trial in range(n_trials):
            random_tokens = _generate_random_tokens(
                real_tokens, char_weights, seed=1000 + trial
            )
            decoded = self._decode_tokens(random_tokens, folio_id=folio_id)
            rate = _resolution_rate(decoded)
            random_rates.append(rate)

        mean_random = sum(random_rates) / len(random_rates)
        if verbose:
            print(f'    Random mean rate: {mean_random:.1%} '
                  f'(range {min(random_rates):.1%}–{max(random_rates):.1%})')

        threshold = 0.15
        test_pass = mean_scrambled < threshold and mean_random < threshold
        if verbose:
            verdict = 'PASS' if test_pass else 'FAIL'
            print(f'    Verdict: {verdict} '
                  f'(threshold: <{threshold:.0%} for controls)')

        return {
            'test': 'unicity_distance',
            'folio': folio_id,
            'n_trials': n_trials,
            'real_rate': round(real_rate, 4),
            'scrambled_mean_rate': round(mean_scrambled, 4),
            'scrambled_rates': [round(r, 4) for r in scrambled_rates],
            'random_mean_rate': round(mean_random, 4),
            'random_rates': [round(r, 4) for r in random_rates],
            'threshold': threshold,
            'pass': test_pass,
        }
