"""
Test 1a: Reversed Voynich Text
================================
Runs the full decoding pipeline on Voynich folios with tokens in reversed
order to test whether word order carries information.

If resolution drops significantly: word order is a meaningful signal and
the bigram model contributes genuine sequential information.

If resolution stays near baseline: the pipeline resolves based on
word-level properties (skeleton matching, frequency, cross-folio consistency),
not sequence.
"""
import json
import os
from typing import Dict

from orchestrators.robustness import run_full_pipeline


class ReversedTextTest:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        by_folio = self.components['by_folio']
        folio_metadata = self.components['folio_metadata']

        # Load forward baseline from Phase 12 output
        phase12_path = os.path.join('./output/phase12', 'phase12_reconstruction.json')
        if os.path.exists(phase12_path):
            with open(phase12_path) as f:
                phase12_data = json.load(f)
            forward = self._compute_rates_from_translations(
                phase12_data['final_translations'], folio_metadata,
            )
        else:
            if self.verbose:
                print('  Phase 12 output not found, running forward pipeline...')
            fwd_result = run_full_pipeline(self.components, verbose=False)
            forward = {
                'overall': fwd_result['overall_resolution'],
                'lang_a': fwd_result['lang_a_resolution'],
                'lang_b': fwd_result['lang_b_resolution'],
            }

        # Build reversed folios
        if self.verbose:
            print('  Reversing token order for all folios...')
        reversed_folios = {}
        for folio, tokens in by_folio.items():
            reversed_folios[folio] = list(reversed(tokens))

        # Run full pipeline on reversed tokens
        if self.verbose:
            print('  Running pipeline on reversed text...')
        rev_result = run_full_pipeline(
            self.components, by_folio_override=reversed_folios, verbose=False,
        )
        reversed_rates = {
            'overall': rev_result['overall_resolution'],
            'lang_a': rev_result['lang_a_resolution'],
            'lang_b': rev_result['lang_b_resolution'],
        }

        delta_overall = reversed_rates['overall'] - forward['overall']
        delta_a = reversed_rates['lang_a'] - forward['lang_a']
        delta_b = reversed_rates['lang_b'] - forward['lang_b']

        result = {
            'test': 'reversed_text',
            'forward': {k: round(v, 4) for k, v in forward.items()},
            'reversed': {k: round(v, 4) for k, v in reversed_rates.items()},
            'delta': {
                'overall': round(delta_overall, 4),
                'lang_a': round(delta_a, 4),
                'lang_b': round(delta_b, 4),
            },
            'interpretation': self._interpret(delta_overall),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _compute_rates_from_translations(self, translations, folio_metadata):
        import re
        total_words = total_brackets = 0
        a_words = a_brackets = b_words = b_brackets = 0
        for folio, text in translations.items():
            wc = len(text.split())
            bc = len(re.findall(r'\[[^\]]+\]|<[^>]+>', text))
            total_words += wc
            total_brackets += bc
            meta = folio_metadata.get(folio, {})
            if meta.get('language') == 'B':
                b_words += wc
                b_brackets += bc
            else:
                a_words += wc
                a_brackets += bc
        return {
            'overall': (total_words - total_brackets) / max(1, total_words),
            'lang_a': (a_words - a_brackets) / max(1, a_words),
            'lang_b': (b_words - b_brackets) / max(1, b_words),
        }

    def _interpret(self, delta):
        pp = abs(delta) * 100
        if pp > 5:
            return (f'Word order accounts for {pp:.1f}pp of resolution. '
                    'The bigram model contributes genuine sequential signal.')
        elif pp > 2:
            return (f'Moderate word-order sensitivity ({pp:.1f}pp). '
                    'Some sequential signal exists but most resolution is order-independent.')
        else:
            return (f'Minimal word-order sensitivity ({pp:.1f}pp). '
                    'Pipeline resolves based on word-level properties '
                    '(skeleton matching, frequency, cross-folio consistency), not sequence.')

    def _print_report(self, result):
        print('\nReversed Text Test')
        print('=' * 55)
        print()
        print(f'{"":20} {"Forward":>10} {"Reversed":>10} {"Delta":>10}')
        print('-' * 55)
        for key in ['overall', 'lang_a', 'lang_b']:
            label = {'overall': 'Overall resolution',
                     'lang_a': 'Lang A', 'lang_b': 'Lang B'}[key]
            fwd = result['forward'][key]
            rev = result['reversed'][key]
            d = result['delta'][key]
            sign = '+' if d >= 0 else ''
            print(f'{label:<20} {100 * fwd:>9.1f}% {100 * rev:>9.1f}% '
                  f'{sign}{100 * d:>8.1f}pp')
        print()
        print(f'Interpretation: {result["interpretation"]}')
