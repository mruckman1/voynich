"""
Selective Matching Test: Vowel preservation + length constraints for selectivity.
=================================================================================
Tests whether enriching skeleton representations with vowel position information
and/or length constraints creates selectivity — i.e., whether real Voynich tokens
match Latin dictionary entries at a higher rate than random character tokens.

Four conditions tested:
  1. baseline          — current consonant-only skeleton matching
  2. vowel_aware       — skeleton preserves vowel positions as 'V' markers
  3. length_constrained — consonant skeleton + token/candidate length ratio filter
  4. combined          — vowel-aware + length constraint

Usage:
  uv run cli.py --robustness selective_matching
"""

import random
import time
from collections import defaultdict
from itertools import product
from typing import Dict, List, Callable

from voynich.modules.phase11.phonetic_skeletonizer import (
    LATIN_CONSONANT_CLASSES,
    VOYNICH_CONSONANT_CLASSES,
)
from voynich.modules.phase12.fuzzy_skeletonizer import (
    VOYNICH_VOWELS,
    SEMI_CONSONANT_MAP,
    _is_vowel_position,
)
from voynich.modules.robustness.multiple_baselines import (
    EVA_CHARS,
    _build_eva_char_weights,
)


# Length ratio bounds for length-constrained matching
LENGTH_RATIO_MIN = 0.5
LENGTH_RATIO_MAX = 2.0


class SelectiveMatchingTest:
    """Test whether vowel preservation and length constraints create skeleton matching selectivity."""

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, n_null_trials: int = 5, seed: int = 42) -> Dict:
        """Run all 4 conditions and compute selectivity ratios."""
        t0 = time.time()

        # Gather all unique Voynich tokens
        by_folio = self.components['by_folio']
        all_tokens = []
        for tokens in by_folio.values():
            all_tokens.extend(tokens)
        real_tokens = sorted(set(all_tokens))

        # Build character weights for null generation
        char_weights = _build_eva_char_weights(all_tokens)

        if self.verbose:
            print(f'\n{"=" * 70}')
            print('SELECTIVE MATCHING TEST')
            print('=' * 70)
            print(f'  Unique Voynich tokens: {len(real_tokens):,d}')
            print(f'  Null trials: {n_null_trials}')

        # Build vowel-aware Latin index (one-time cost)
        if self.verbose:
            print('  Building vowel-aware Latin index...')
        va_index = self._build_vowel_aware_latin_index()
        consonant_index_size = len(self.components['latin_skel'].skeleton_index)
        if self.verbose:
            print(f'    Vowel-aware skeletons: {len(va_index):,d}')
            print(f'    Consonant-only skeletons: {consonant_index_size:,d}')

        # Build candidate word lists for length-constrained conditions
        # (need actual words, not just skeleton matches)
        latin_skel = self.components['latin_skel']

        # Measure all 4 conditions
        conditions = {}
        condition_specs = [
            ('baseline', lambda t: self._match_baseline(t)),
            ('vowel_aware', lambda t: self._match_vowel_aware(t, va_index)),
            ('length_constrained', lambda t: self._match_length_constrained(t)),
            ('combined', lambda t: self._match_combined(t, va_index)),
        ]

        for cond_name, match_fn in condition_specs:
            if self.verbose:
                print(f'  Measuring {cond_name}...')
            conditions[cond_name] = self._measure_condition(
                match_fn, real_tokens, char_weights, n_null_trials, seed,
            )

        decision = self._compute_decision(conditions)
        elapsed = time.time() - t0

        result = {
            'conditions': {
                name: {
                    'real_match_rate': round(d['real_match_rate'], 4),
                    'null_match_rate': round(d['null_match_rate'], 4),
                    'selectivity': round(d['selectivity'], 2),
                }
                for name, d in conditions.items()
            },
            'decision': decision,
            'diagnostics': {
                'unique_tokens': len(real_tokens),
                'va_index_size': len(va_index),
                'consonant_index_size': consonant_index_size,
                'n_null_trials': n_null_trials,
            },
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_results(result)

        return result

    # ── Vowel-aware skeleton builders ──────────────────────────────────

    def _voynich_vowel_aware_skeleton(self, v_stem: str) -> List[str]:
        """Build vowel-aware skeleton(s) for a Voynich token.

        Returns list of skeleton strings (multiple due to y/o branching).
        Each skeleton uses '-'-joined elements: consonant classes + 'V' for vowels.
        """
        if not v_stem:
            return []

        elements = []  # str (fixed) or tuple (branch options)
        i = 0
        while i < len(v_stem):
            # 1. Check digraph
            if i < len(v_stem) - 1:
                digraph = v_stem[i:i + 2]
                if digraph in VOYNICH_CONSONANT_CLASSES:
                    elements.append(VOYNICH_CONSONANT_CLASSES[digraph])
                    i += 2
                    continue

            char = v_stem[i]

            # 2. Single-char consonant
            if char in VOYNICH_CONSONANT_CLASSES:
                elements.append(VOYNICH_CONSONANT_CLASSES[char])
                i += 1
                continue

            # 3. Semi-consonant (y, o) at vowel position: branch
            if char in SEMI_CONSONANT_MAP and _is_vowel_position(v_stem, i):
                elements.append(('V', SEMI_CONSONANT_MAP[char]))
                i += 1
                continue

            # 4. Plain vowel (a, e, and non-branching o, y)
            if char in VOYNICH_VOWELS:
                elements.append('V')
                i += 1
                continue

            # 5. Unknown character — skip
            i += 1

        if not elements:
            return []

        # Expand branches
        branch_points = []
        for elem in elements:
            if isinstance(elem, tuple):
                branch_points.append(elem)
            else:
                branch_points.append((elem,))

        results = set()
        for combo in product(*branch_points):
            skeleton = '-'.join(combo)
            results.add(skeleton)

        return list(results)

    def _latin_vowel_aware_skeleton(self, word: str) -> str:
        """Build vowel-aware skeleton for a Latin word.

        Consonants emit their class (with adjacent same-class dedup).
        Non-consonants emit 'V'. Dedup resets after each vowel.
        """
        word = word.lower()
        elements = []
        last_consonant = ''

        for char in word:
            if char in LATIN_CONSONANT_CLASSES:
                mapped = LATIN_CONSONANT_CLASSES[char]
                if mapped != last_consonant:
                    elements.append(mapped)
                    last_consonant = mapped
            else:
                elements.append('V')
                last_consonant = ''  # reset dedup after vowel

        return '-'.join(elements) if elements else ''

    def _build_vowel_aware_latin_index(self) -> Dict[str, List[str]]:
        """Build vowel-aware skeleton index from the Latin vocabulary."""
        latin_skel = self.components['latin_skel']
        index = defaultdict(list)

        sorted_vocab = sorted(
            latin_skel.vocab,
            key=lambda w: (-latin_skel.unigram_counts[w], w),
        )
        for word in sorted_vocab:
            skel = self._latin_vowel_aware_skeleton(word)
            if skel:
                index[skel].append(word)

        return dict(index)

    # ── Matching functions ─────────────────────────────────────────────

    def _match_baseline(self, token: str) -> bool:
        """Condition 1: Current consonant-only skeleton matching."""
        fuzzy_skel = self.components['fuzzy_skel']
        latin_skel = self.components['latin_skel']

        skeleton_candidates = fuzzy_skel.get_skeleton_candidates(token)
        for skel, _ in skeleton_candidates:
            if skel in latin_skel.skeleton_index:
                return True
        return False

    def _match_vowel_aware(self, token: str, va_index: Dict) -> bool:
        """Condition 2: Vowel-aware skeleton matching."""
        skeletons = self._voynich_vowel_aware_skeleton(token)
        for skel in skeletons:
            if skel in va_index:
                return True
        return False

    def _match_length_constrained(self, token: str) -> bool:
        """Condition 3: Consonant-only + length ratio filter."""
        fuzzy_skel = self.components['fuzzy_skel']
        latin_skel = self.components['latin_skel']
        token_len = len(token)

        skeleton_candidates = fuzzy_skel.get_skeleton_candidates(token)
        for skel, _ in skeleton_candidates:
            words = latin_skel.skeleton_index.get(skel, [])
            for word in words:
                ratio = token_len / len(word) if len(word) > 0 else 0
                if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
                    return True
        return False

    def _match_combined(self, token: str, va_index: Dict) -> bool:
        """Condition 4: Vowel-aware + length constraint."""
        token_len = len(token)
        skeletons = self._voynich_vowel_aware_skeleton(token)
        for skel in skeletons:
            words = va_index.get(skel, [])
            for word in words:
                ratio = token_len / len(word) if len(word) > 0 else 0
                if LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX:
                    return True
        return False

    # ── Null token generation ──────────────────────────────────────────

    def _generate_null_tokens(
        self,
        real_tokens: List[str],
        char_weights: List[float],
        rng: random.Random,
    ) -> List[str]:
        """Generate random EVA character strings matching real token lengths."""
        null_tokens = []
        for token in real_tokens:
            length = len(token)
            chars = rng.choices(EVA_CHARS, weights=char_weights, k=length)
            null_tokens.append(''.join(chars))
        return null_tokens

    # ── Core measurement ───────────────────────────────────────────────

    def _measure_condition(
        self,
        match_fn: Callable,
        real_tokens: List[str],
        char_weights: List[float],
        n_null_trials: int,
        seed: int,
    ) -> Dict:
        """Measure real_match_rate, null_match_rate, and selectivity for one condition."""
        # Real match rate
        real_matches = sum(1 for t in real_tokens if match_fn(t))
        real_match_rate = real_matches / len(real_tokens) if real_tokens else 0

        # Null match rates
        null_rates = []
        for trial in range(n_null_trials):
            rng = random.Random(seed + trial)
            null_tokens = self._generate_null_tokens(real_tokens, char_weights, rng)
            null_matches = sum(1 for t in null_tokens if match_fn(t))
            null_rates.append(null_matches / len(null_tokens) if null_tokens else 0)

        null_match_rate = sum(null_rates) / len(null_rates) if null_rates else 0

        selectivity = (
            real_match_rate / null_match_rate
            if null_match_rate > 0
            else float('inf') if real_match_rate > 0
            else 1.0
        )

        return {
            'real_match_rate': real_match_rate,
            'null_match_rate': null_match_rate,
            'selectivity': selectivity,
            'null_trials': [round(r, 4) for r in null_rates],
        }

    # ── Decision matrix ────────────────────────────────────────────────

    def _compute_decision(self, conditions: Dict) -> str:
        """Compute overall decision based on combined selectivity."""
        combined = conditions.get('combined', {})
        sel = combined.get('selectivity', 1.0)

        if sel >= 2.0:
            return 'HIGH'
        elif sel >= 1.3:
            return 'MODERATE'
        elif sel >= 1.1:
            return 'LOW'
        else:
            return 'NONE'

    # ── Pretty printing ────────────────────────────────────────────────

    def _print_results(self, results: Dict) -> None:
        conds = results['conditions']

        print(f'\n  {"Condition":<24s} {"Real Match":>12s} {"Null Match":>12s} '
              f'{"Selectivity":>12s}')
        print('  ' + '-' * 62)

        for name in ['baseline', 'vowel_aware', 'length_constrained', 'combined']:
            c = conds[name]
            print(f'  {name:<24s} {100*c["real_match_rate"]:>11.1f}% '
                  f'{100*c["null_match_rate"]:>11.1f}% '
                  f'{c["selectivity"]:>11.2f}x')

        diag = results['diagnostics']
        print(f'\n  Vowel-aware index: {diag["va_index_size"]:,d} skeletons '
              f'(vs {diag["consonant_index_size"]:,d} consonant-only)')

        decision = results['decision']
        decision_labels = {
            'HIGH': 'HIGH — vowel-aware matching is significantly discriminative',
            'MODERATE': 'MODERATE — some discrimination, worth pursuing',
            'LOW': 'LOW — marginal selectivity, probably not worth the complexity',
            'NONE': 'NONE — no selectivity gain from vowel preservation',
        }
        print(f'\n  Decision: {decision_labels.get(decision, decision)}')
        print(f'  Elapsed: {results["elapsed_seconds"]:.1f}s')
