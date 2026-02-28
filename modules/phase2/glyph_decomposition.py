"""
Model 6: Glyph Decomposition (Wrong Alphabet Hypothesis)
==========================================================
Tests whether the EVA transcription alphabet artificially deflates entropy
by treating multi-glyph units as separate characters.

This model does NOT generate from plaintext. Instead, it retranscribes the
actual Voynich corpus into alternative alphabets and re-analyzes.

Critical test: Does any re-alphabetization bring H2 > 2.5 (into the range
where character-level ciphers could explain the manuscript)?

Historical plausibility: HIGH — there is active scholarly debate about
glyph boundaries (Stolfi, Zandbergen, Bennett).
Priority: MEDIUM
"""

from typing import Dict, List

from modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS
from modules.statistical_analysis import full_statistical_profile
from data.glyph_alphabets import (
    retranscribe, ALTERNATIVE_ALPHABETS, list_alphabets, get_alphabet_info
)
from data.voynich_corpus import get_all_tokens

H2_CIPHER_THRESHOLD = 2.5

class GlyphDecomposition(Phase2GenerativeModel):
    """
    Model 6: Re-alphabetization analysis.

    Unlike other models, this does NOT generate synthetic text.
    It retranscribes the real Voynich corpus with alternative glyph
    boundaries and checks whether the statistical picture changes.
    """

    MODEL_NAME = 'glyph_decomposition'
    MODEL_PRIORITY = 'MEDIUM'

    def __init__(self, alphabet_name: str = 'ligature_merged',
                 merge_ligatures: bool = True,
                 split_gallows: bool = False,
                 seed: int = 42, **kwargs):
        params = {
            'alphabet_name': alphabet_name,
            'merge_ligatures': merge_ligatures,
            'split_gallows': split_gallows,
            'seed': seed,
        }
        super().__init__(**params)
        self.alphabet_name = alphabet_name

    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Retranscribe the actual Voynich corpus into the target alphabet.
        The plaintext parameter is ignored — this model works on the real corpus.
        """
        tokens = get_all_tokens()
        if not tokens:
            return ''

        eva_text = ' '.join(tokens)

        text = eva_text

        if self.params.get('split_gallows', False) and self.alphabet_name != 'gallows_split':
            text = retranscribe(text, 'gallows_split')

        if self.alphabet_name != 'eva_standard':
            text = retranscribe(text, self.alphabet_name)

        if not self.params.get('merge_ligatures', True) and self.alphabet_name == 'ligature_merged':
            text = eva_text

        return text

    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """
        Return grid of alphabet + option combinations.
        This grid is small since there are only ~6 alphabets × 2 × 2 options.
        """
        alphabets = list_alphabets()

        if resolution == 'coarse':
            return [{'alphabet_name': a, 'seed': 42}
                    for a in alphabets]

        grid = []
        for a in alphabets:
            for merge in [True, False]:
                for split in [True, False]:
                    if a == 'gallows_split' and split:
                        continue
                    if a == 'ligature_merged' and not merge:
                        continue
                    grid.append({
                        'alphabet_name': a,
                        'merge_ligatures': merge,
                        'split_gallows': split,
                        'seed': 42,
                    })
        return grid

    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Critical test: Does this re-alphabetization bring H2 into cipher range?

        If H2 > 2.5 under any alternative alphabet, the EVA alphabet is
        the problem and all Phase 1 ciphers need re-testing.
        """
        h2 = generated_profile.get('entropy', {}).get('H2', 0.0)
        h1 = generated_profile.get('entropy', {}).get('H1', 0.0)

        passes = h2 > H2_CIPHER_THRESHOLD
        h2_delta = h2 - VOYNICH_TARGETS['H2']

        return {
            'passes': passes,
            'description': (
                f'H2 = {h2:.4f} (target > {H2_CIPHER_THRESHOLD} for cipher range). '
                f'Delta from EVA baseline: {h2_delta:+.4f}'
            ),
            'details': {
                'alphabet': self.alphabet_name,
                'H2': h2,
                'H1': h1,
                'H2_delta_from_baseline': h2_delta,
                'in_cipher_range': passes,
                'threshold': H2_CIPHER_THRESHOLD,
            },
        }

    def run_all_alphabets(self, verbose: bool = True) -> Dict:
        """
        Quick analysis: retranscribe with every available alphabet and
        compare H2 values. This is fast (seconds per alphabet).

        Returns:
            {alphabets: [{name, H2, H1, H2_delta, in_cipher_range, profile}...],
             any_in_cipher_range: bool,
             best_alphabet: str,
             best_H2: float}
        """
        results = []

        for alpha_name in list_alphabets():
            model = GlyphDecomposition(alphabet_name=alpha_name)
            text = model.generate()

            if not text:
                continue

            profile = model.get_profile(text)
            crit = model.critical_test(profile)

            entry = {
                'name': alpha_name,
                'alphabet_info': get_alphabet_info(alpha_name),
                'H2': crit['details']['H2'],
                'H1': crit['details']['H1'],
                'H3': profile.get('entropy', {}).get('H3', 0),
                'H2_delta': crit['details']['H2_delta_from_baseline'],
                'in_cipher_range': crit['passes'],
                'TTR': profile.get('zipf', {}).get('type_token_ratio', 0),
                'zipf_exponent': profile.get('zipf', {}).get('zipf_exponent', 0),
                'vocab_size': profile.get('zipf', {}).get('vocabulary_size', 0),
            }
            results.append(entry)

            if verbose:
                marker = ' *** CIPHER RANGE ***' if crit['passes'] else ''
                print(f'  [{alpha_name:20s}] H2 = {entry["H2"]:.4f} '
                      f'(delta {entry["H2_delta"]:+.4f}){marker}')

        results.sort(key=lambda r: -r['H2'])

        any_match = any(r['in_cipher_range'] for r in results)
        best = results[0] if results else {}

        return {
            'alphabets': results,
            'any_in_cipher_range': any_match,
            'best_alphabet': best.get('name', ''),
            'best_H2': best.get('H2', 0),
            'conclusion': (
                'EVA alphabet IS the problem — re-test all Phase 1 ciphers'
                if any_match else
                'H2 remains anomalously low under all re-alphabetizations. '
                'The encoding unit is genuinely larger than a character.'
            ),
        }
