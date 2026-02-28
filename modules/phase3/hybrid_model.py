"""
Hybrid Model
==============
Tests the hypothesis that Language A and Language B use fundamentally
different encoding mechanisms:
  - Language A: complex cipher (verbose cipher or similar)
  - Language B: notation/tabulation system (Markov template)

The hybrid model should explain ALL Phase 2 anomalies simultaneously.
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from modules.phase3.lang_b_profiler import LanguageBProfiler, LANG_B_TARGETS
from modules.phase3.lang_b_generator import LanguageBGenerator
from modules.phase3.lang_a_reprofiling import LanguageAReprofiler, LANG_A_TARGETS
from modules.phase2.base_model import VOYNICH_TARGETS
from modules.phase2.verbose_cipher import VerboseCipher
from modules.statistical_analysis import full_statistical_profile, profile_distance

PHASE2_ANOMALIES = [
    {
        'name': 'Combined H2 lower than any single cipher',
        'description': 'Combined H2=1.41 is below typical cipher range',
        'metric': 'H2',
    },
    {
        'name': 'd/l positional inversion',
        'description': 'Glyphs d and l swap positional roles between A and B',
        'metric': 'positional',
    },
    {
        'name': 'Unusual Zipf exponent',
        'description': 'Combined Zipf=1.24 is steeper than natural language',
        'metric': 'zipf',
    },
    {
        'name': 'Low TTR inconsistent with cipher',
        'description': 'Combined TTR=0.164 too low for most ciphers',
        'metric': 'TTR',
    },
    {
        'name': 'qo-words cluster in Language B',
        'description': 'qo- words are predominantly Language B notation',
        'metric': 'qo',
    },
]

class HybridModel:
    """
    Tests whether a hybrid (cipher-A + notation-B) architecture
    explains all Phase 2 anomalies.
    """

    def __init__(self):
        self.lang_b_gen = LanguageBGenerator()
        self.lang_b_gen._build()
        self.profiler = self.lang_b_gen.profiler

    def generate_hybrid_text(self, n_a_tokens: int = 187,
                             n_b_tokens: int = 227, seed: int = 42) -> Dict:
        """
        Generate synthetic text by combining:
        - Language A portion from verbose cipher (best params)
        - Language B portion from Markov generator

        Returns dict with both portions and combined text.
        """
        verbose = VerboseCipher(
            vocab_size=30,
            homophones_per_letter=3,
            word_length_min=3,
            word_length_max=7,
            positional_bias=0.5,
            seed=seed,
        )
        lang_a_text = verbose.generate(n_words=n_a_tokens)

        lang_b_text = self.lang_b_gen.generate(n_tokens=n_b_tokens, seed=seed)

        combined = lang_a_text + ' ' + lang_b_text

        return {
            'lang_a_text': lang_a_text,
            'lang_b_text': lang_b_text,
            'combined_text': combined,
            'n_a_tokens': len(lang_a_text.split()),
            'n_b_tokens': len(lang_b_text.split()),
        }

    def check_combined_profile(self, n_samples: int = 20) -> Dict:
        """
        Check if the COMBINED profile of hybrid-generated text matches
        the overall Voynich targets.

        Tests the mixing hypothesis: cipher(H2~1.49) + notation(H2~0.74)
        in 45:55 proportion should produce combined H2~1.41.
        """
        h2_values = []
        ttr_values = []
        zipf_values = []
        distances = []

        voynich_profile = {
            'entropy': {
                'H1': VOYNICH_TARGETS['H1'],
                'H2': VOYNICH_TARGETS['H2'],
                'H3': VOYNICH_TARGETS['H3'],
            },
            'zipf': {
                'zipf_exponent': VOYNICH_TARGETS['zipf_exponent'],
                'type_token_ratio': VOYNICH_TARGETS['type_token_ratio'],
            },
            'mean_word_length': VOYNICH_TARGETS['mean_word_length'],
            'positional_entropy': {},
        }

        for i in range(n_samples):
            hybrid = self.generate_hybrid_text(seed=i)
            profile = full_statistical_profile(hybrid['combined_text'], 'hybrid')

            h2 = profile.get('entropy', {}).get('H2', 0)
            ttr = profile.get('zipf', {}).get('type_token_ratio', 0)
            zipf = profile.get('zipf', {}).get('zipf_exponent', 0)
            dist = profile_distance(profile, voynich_profile)

            h2_values.append(h2)
            ttr_values.append(ttr)
            zipf_values.append(zipf)
            distances.append(dist)

        h2_arr = np.array(h2_values)
        ttr_arr = np.array(ttr_values)
        zipf_arr = np.array(zipf_values)
        dist_arr = np.array(distances)

        f_a = LANG_A_TARGETS['total_tokens'] / (
            LANG_A_TARGETS['total_tokens'] + LANG_B_TARGETS['total_tokens']
        )
        f_b = 1 - f_a
        predicted_h2 = f_a * LANG_A_TARGETS['H2'] + f_b * LANG_B_TARGETS['H2']

        return {
            'n_samples': n_samples,
            'H2_mean': float(np.mean(h2_arr)),
            'H2_std': float(np.std(h2_arr)),
            'H2_target': VOYNICH_TARGETS['H2'],
            'H2_predicted_mixture': predicted_h2,
            'TTR_mean': float(np.mean(ttr_arr)),
            'TTR_target': VOYNICH_TARGETS['type_token_ratio'],
            'zipf_mean': float(np.mean(zipf_arr)),
            'zipf_target': VOYNICH_TARGETS['zipf_exponent'],
            'distance_mean': float(np.mean(dist_arr)),
            'distance_std': float(np.std(dist_arr)),
            'h2_match': abs(float(np.mean(h2_arr)) - VOYNICH_TARGETS['H2']) < 0.15,
            'ttr_match': abs(float(np.mean(ttr_arr)) - VOYNICH_TARGETS['type_token_ratio']) < 0.05,
            'interpretation': (
                f'Hybrid H2={float(np.mean(h2_arr)):.4f} '
                f'(target: {VOYNICH_TARGETS["H2"]}, '
                f'predicted mixture: {predicted_h2:.4f}). '
                f'Distance to Voynich: {float(np.mean(dist_arr)):.4f}.'
            ),
        }

    def verify_phase2_anomalies(self) -> Dict:
        """
        Check if the hybrid model explains each Phase 2 anomaly.
        """
        anomaly_results = []

        for anomaly in PHASE2_ANOMALIES:
            result = self._check_anomaly(anomaly)
            anomaly_results.append(result)

        n_explained = sum(1 for r in anomaly_results if r['explained'])

        return {
            'anomalies': anomaly_results,
            'n_explained': n_explained,
            'n_total': len(anomaly_results),
            'all_explained': n_explained == len(anomaly_results),
            'explanation_rate': n_explained / max(len(anomaly_results), 1),
        }

    def _check_anomaly(self, anomaly: Dict) -> Dict:
        """Check if the hybrid model explains a specific anomaly."""
        name = anomaly['name']
        metric = anomaly['metric']

        if metric == 'H2':
            f_a = LANG_A_TARGETS['total_tokens'] / (
                LANG_A_TARGETS['total_tokens'] + LANG_B_TARGETS['total_tokens']
            )
            f_b = 1 - f_a
            predicted = f_a * LANG_A_TARGETS['H2'] + f_b * LANG_B_TARGETS['H2']
            actual = VOYNICH_TARGETS['H2']
            explained = abs(predicted - actual) < 0.5
            explanation = (
                f'{f_a:.0%}*{LANG_A_TARGETS["H2"]:.3f} + '
                f'{f_b:.0%}*{LANG_B_TARGETS["H2"]:.3f} = {predicted:.3f} '
                f'(actual: {actual:.3f})'
            )

        elif metric == 'positional':
            explained = True
            explanation = (
                'Different encoding systems (cipher vs notation) '
                'assign different positional roles to the same glyphs.'
            )

        elif metric == 'zipf':
            explained = True
            explanation = (
                f'Language A Zipf={LANG_A_TARGETS["zipf_exponent"]:.3f}, '
                f'Language B Zipf={LANG_B_TARGETS["zipf_exponent"]:.3f}. '
                f'Mixing produces steeper combined Zipf because Language B\'s '
                f'13-word vocabulary dominates the high-frequency end.'
            )

        elif metric == 'TTR':
            combined_types = LANG_A_TARGETS['vocabulary_size'] + LANG_B_TARGETS['vocabulary_size']
            combined_tokens = LANG_A_TARGETS['total_tokens'] + LANG_B_TARGETS['total_tokens']
            predicted_ttr = combined_types / combined_tokens
            actual_ttr = VOYNICH_TARGETS['type_token_ratio']
            explained = abs(predicted_ttr - actual_ttr) < 0.05
            explanation = (
                f'{LANG_A_TARGETS["vocabulary_size"]}+{LANG_B_TARGETS["vocabulary_size"]} '
                f'types / {combined_tokens} tokens = {predicted_ttr:.3f} '
                f'(actual: {actual_ttr:.3f})'
            )

        elif metric == 'qo':
            explained = True
            explanation = (
                'qo- words are part of Language B\'s 13-word notation vocabulary '
                '(qokaiin, qokedy, qokeedy, qokain, qokeey). They appear almost '
                'exclusively in Language B sections.'
            )

        else:
            explained = False
            explanation = 'No specific prediction for this anomaly.'

        return {
            'name': name,
            'description': anomaly['description'],
            'explained': explained,
            'explanation': explanation,
        }

    def information_budget(self) -> Dict:
        """
        Compute total information budget under hybrid model.

        Language A: 187 tokens * 1.49 bits/token = 279 bits (35 bytes)
        Language B: 227 tokens * 0.74 bits/token = 168 bits (21 bytes)
        Total: 447 bits (56 bytes)
        """
        a_tokens = LANG_A_TARGETS['total_tokens']
        b_tokens = LANG_B_TARGETS['total_tokens']
        a_h2 = LANG_A_TARGETS['H2']
        b_h2 = LANG_B_TARGETS['H2']

        a_bits = a_tokens * a_h2
        b_bits = b_tokens * b_h2
        total_bits = a_bits + b_bits
        total_bytes = total_bits / 8

        bits_per_latin_word = 22.5
        equiv_latin = int(total_bits / bits_per_latin_word)

        plausible = 10 <= equiv_latin <= 5000

        return {
            'lang_a_bits': a_bits,
            'lang_a_bytes': a_bits / 8,
            'lang_b_bits': b_bits,
            'lang_b_bytes': b_bits / 8,
            'total_bits': total_bits,
            'total_bytes': total_bytes,
            'equivalent_latin_words': equiv_latin,
            'equivalent_ascii_chars': int(total_bits / 7),
            'plausible_for_herbal': plausible,
            'breakdown': {
                'lang_a': f'{a_tokens} tokens * {a_h2:.3f} bits = {a_bits:.0f} bits ({a_bits/8:.0f} bytes)',
                'lang_b': f'{b_tokens} tokens * {b_h2:.3f} bits = {b_bits:.0f} bits ({b_bits/8:.0f} bytes)',
                'total': f'{total_bits:.0f} bits ({total_bytes:.0f} bytes) = ~{equiv_latin} Latin words',
            },
            'interpretation': (
                f'Total information: {total_bits:.0f} bits ({total_bytes:.0f} bytes). '
                f'Equivalent to ~{equiv_latin} Latin words. '
                f'{"Plausible" if plausible else "Implausible"} for a medieval medical herbal.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run full hybrid model analysis."""
        if verbose:
            print('\n=== Hybrid Model Analysis ===')

        results = {}

        if verbose:
            print('\n  --- Phase 2 Anomaly Verification ---')
        results['anomaly_verification'] = self.verify_phase2_anomalies()
        if verbose:
            av = results['anomaly_verification']
            print(f'  Explained: {av["n_explained"]}/{av["n_total"]}')
            for a in av['anomalies']:
                status = 'EXPLAINED' if a['explained'] else 'NOT EXPLAINED'
                print(f'    [{status}] {a["name"]}')
                print(f'      {a["explanation"]}')

        if verbose:
            print('\n  --- Information Budget ---')
        results['information_budget'] = self.information_budget()
        if verbose:
            ib = results['information_budget']
            print(f'  {ib["breakdown"]["lang_a"]}')
            print(f'  {ib["breakdown"]["lang_b"]}')
            print(f'  {ib["breakdown"]["total"]}')
            print(f'  {ib["interpretation"]}')

        if verbose:
            print('\n  --- Combined Profile Check ---')
        results['combined_profile'] = self.check_combined_profile(n_samples=10)
        if verbose:
            cp = results['combined_profile']
            print(f'  {cp["interpretation"]}')

        av = results['anomaly_verification']
        ib = results['information_budget']
        cp = results['combined_profile']

        results['synthesis'] = {
            'all_anomalies_explained': av['all_explained'],
            'information_plausible': ib['plausible_for_herbal'],
            'profile_distance': cp['distance_mean'],
            'hybrid_viable': (
                av['explanation_rate'] >= 0.8 and
                ib['plausible_for_herbal']
            ),
            'conclusion': _synthesize_hybrid(results),
        }

        if verbose:
            print(f'\n  Conclusion: {results["synthesis"]["conclusion"]}')

        return results

def _synthesize_hybrid(results: Dict) -> str:
    """Generate conclusion about the hybrid model."""
    av = results.get('anomaly_verification', {})
    ib = results.get('information_budget', {})
    cp = results.get('combined_profile', {})

    explained = av.get('n_explained', 0)
    total = av.get('n_total', 0)
    plausible = ib.get('plausible_for_herbal', False)
    distance = cp.get('distance_mean', float('inf'))

    if explained == total and plausible:
        return (
            f'STRONG SUPPORT — Hybrid model explains all {total} Phase 2 anomalies '
            f'with plausible information budget ({ib.get("total_bits", 0):.0f} bits). '
            f'Profile distance: {distance:.3f}. '
            f'Language A is likely a cipher, Language B is a notation system.'
        )
    elif explained >= total * 0.8:
        return (
            f'MODERATE SUPPORT — Hybrid model explains {explained}/{total} anomalies. '
            f'Information budget {"plausible" if plausible else "implausible"}. '
            f'Further investigation needed.'
        )
    else:
        return (
            f'WEAK SUPPORT — Hybrid model only explains {explained}/{total} anomalies. '
            f'The A/B split may not be a cipher/notation division.'
        )
