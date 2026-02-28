"""
Test 8a: Cardan Grille Test
=============================
Generates fake Voynich text using Rugg's Cardan grille method, then runs
it through the decoding pipeline.

If grille text resolves at ~13% (like random): pipeline distinguishes
genuine cipher from mechanical generation.

If grille text resolves at ~53% (like real): pipeline can't distinguish,
suggesting resolution may be an artifact of statistical regularity.
"""
import random
import time
from typing import Dict, List, Tuple

from orchestrators.robustness import run_full_pipeline


# EVA alphabet patterns and frequencies (approximate from the Voynich manuscript)
EVA_CONSONANTS = ['k', 'd', 'ch', 'sh', 'l', 'r', 's', 'p', 't', 'f', 'n']
EVA_VOWELS = ['o', 'a', 'e', 'y', 'i']
EVA_CONSONANT_WEIGHTS = [15, 12, 10, 8, 8, 6, 5, 4, 4, 3, 2]
EVA_VOWEL_WEIGHTS = [30, 20, 15, 10, 5]

# Syllable patterns observed in Voynich text with approximate frequencies
SYLLABLE_PATTERNS = ['CV', 'CVC', 'CVCV', 'V', 'VC']
SYLLABLE_PATTERN_WEIGHTS = [30, 25, 15, 15, 15]


class CardanGrilleTest:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, n_trials: int = 10) -> Dict:
        by_folio = self.components['by_folio']
        t0 = time.time()

        # Get real resolution for comparison
        if self.verbose:
            print('  Running real Voynich baseline...')
        real_result = run_full_pipeline(self.components, verbose=False)
        real_resolution = real_result['overall_resolution']

        # Run grille trials
        trial_results = []
        for trial in range(n_trials):
            if self.verbose:
                print(f'  Grille trial {trial + 1}/{n_trials}...')

            rng = random.Random(42 + trial)
            grille_folios = self._generate_grille_folios(by_folio, rng)
            result = run_full_pipeline(
                self.components, by_folio_override=grille_folios, verbose=False,
            )
            rate = result['overall_resolution']
            trial_results.append(rate)

        mean_res = sum(trial_results) / len(trial_results)
        std_res = (sum((r - mean_res) ** 2 for r in trial_results) / len(trial_results)) ** 0.5
        elapsed = time.time() - t0

        result = {
            'test': 'cardan_grille',
            'real_resolution': round(real_resolution, 4),
            'grille_results': {
                'mean': round(mean_res, 4),
                'std': round(std_res, 4),
                'min': round(min(trial_results), 4),
                'max': round(max(trial_results), 4),
                'trials': [round(r, 4) for r in trial_results],
            },
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _generate_grille_folios(self, by_folio: Dict, rng: random.Random) -> Dict:
        """Generate grille text matching real Voynich folio structure."""
        table = self._build_syllable_table(rng)
        grille = self._build_grille(rng)

        grille_folios = {}
        offset = 0
        for folio, tokens in by_folio.items():
            n_tokens = len(tokens)
            # Compute target word lengths from real tokens
            target_lengths = [len(t.split('_')[0]) for t in tokens]
            grille_tokens = self._generate_words(
                table, grille, n_tokens, target_lengths, rng, offset,
            )
            grille_folios[folio] = grille_tokens
            offset += n_tokens
        return grille_folios

    def _build_syllable_table(self, rng: random.Random, rows: int = 12, cols: int = 12) -> List[List[str]]:
        """Build a table of Voynich-like syllables using EVA character frequencies."""
        table = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                pattern = _weighted_choice(rng, SYLLABLE_PATTERNS, SYLLABLE_PATTERN_WEIGHTS)
                syllable = self._generate_syllable(pattern, rng)
                row.append(syllable)
            table.append(row)
        return table

    def _generate_syllable(self, pattern: str, rng: random.Random) -> str:
        """Generate a single syllable from a CV pattern."""
        result = []
        for char in pattern:
            if char == 'C':
                result.append(_weighted_choice(rng, EVA_CONSONANTS, EVA_CONSONANT_WEIGHTS))
            elif char == 'V':
                result.append(_weighted_choice(rng, EVA_VOWELS, EVA_VOWEL_WEIGHTS))
        return ''.join(result)

    def _build_grille(self, rng: random.Random, rows: int = 12, cols: int = 12) -> List[List[Tuple[int, int]]]:
        """Build multiple grille configurations with 2-4 holes each.

        Returns a list of grille configs (each a list of (row, col) positions).
        Multiple grilles allow generating diverse word lengths.
        """
        all_positions = [(r, c) for r in range(rows) for c in range(cols)]
        grilles = []
        # Create grilles with 1-4 holes for different word lengths
        for n_holes in [1, 2, 2, 3, 3, 3, 4, 4]:
            positions = rng.sample(all_positions, n_holes)
            grilles.append(positions)
        return grilles

    def _generate_words(
        self, table, grilles, n_words, target_lengths, rng, base_offset,
    ) -> List[str]:
        """Generate n_words of grille text by sliding grilles across the table."""
        rows = len(table)
        cols = len(table[0])
        words = []

        for i in range(n_words):
            target_len = target_lengths[i] if i < len(target_lengths) else 5
            # Pick grille that best matches target word length
            best_grille = self._select_grille(grilles, target_len)
            offset = base_offset + i

            word = ''
            for row, col in best_grille:
                actual_col = (col + offset) % cols
                actual_row = (row + (offset // cols)) % rows
                word += table[actual_row][actual_col]

            # Trim or pad to approximately match target length
            if len(word) > target_len + 2:
                word = word[:target_len + 1]

            words.append(word)

        return words

    def _select_grille(self, grilles, target_len: int):
        """Select the grille whose hole count best matches the target word length."""
        # Average syllable length ~2 chars, so holes * 2 ≈ word length
        best = grilles[0]
        best_diff = abs(len(best) * 2 - target_len)
        for g in grilles[1:]:
            diff = abs(len(g) * 2 - target_len)
            if diff < best_diff:
                best = g
                best_diff = diff
        return best

    def _print_report(self, result):
        real = result['real_resolution']
        grille = result['grille_results']

        print(f'\nCardan Grille Test ({len(grille["trials"])} trials)')
        print('=' * 70)
        print()

        for i, rate in enumerate(grille['trials']):
            print(f'  Trial {i+1:>2}: {100*rate:.1f}%')

        print()
        print(f'  Mean:    {100*grille["mean"]:.1f}% +/- {100*grille["std"]:.1f}%')
        print(f'  Range:   [{100*grille["min"]:.1f}%, {100*grille["max"]:.1f}%]')
        print()
        print(f'  Real Voynich:    {100*real:.1f}%')
        print(f'  Random baseline: ~13.8%')
        print(f'  Grille text:     {100*grille["mean"]:.1f}%')

        if grille['mean'] < 0.20:
            verdict = 'CLOSE TO RANDOM — pipeline distinguishes cipher from hoax'
        elif grille['mean'] < 0.35:
            verdict = 'PARTIAL DISCRIMINATION — some cipher-like structure detected'
        else:
            verdict = 'CLOSE TO REAL — pipeline cannot distinguish (concern)'
        print(f'\n  Verdict: {verdict}')
        print(f'  Elapsed: {result["elapsed_seconds"]:.1f}s')


def _weighted_choice(rng, items, weights):
    """Weighted random choice compatible with Python's random module."""
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1]
