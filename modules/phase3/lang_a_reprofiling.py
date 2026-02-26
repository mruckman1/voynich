"""
Language A Re-profiling
========================
Re-runs the Phase 1 null framework against Language A's own metrics
(H2=1.49) instead of the combined corpus (H2=1.41), to check if
Language A alone falls within any cipher family's null distribution.

Also tests the verbose cipher model specifically for Language A.
"""

import sys
import os
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase2.cross_cutting import LanguageABSplitter
from modules.null_framework import (
    NullDistributionEngine, CIPHER_FAMILIES, SOURCE_LANGUAGES
)
from modules.statistical_analysis import (
    full_statistical_profile, compute_all_entropy, zipf_analysis
)
from modules.phase2.verbose_cipher import VerboseCipher
from modules.phase2.base_model import VOYNICH_TARGETS


# Language A targets from Phase 2 cross-cutting results
LANG_A_TARGETS = {
    'H1': 3.634,
    'H2': 1.487,
    'H3': 1.021,
    'zipf_exponent': 0.931,
    'type_token_ratio': 0.305,
    'vocabulary_size': 57,
    'total_tokens': 187,
    'mean_word_length': 4.43,
}


class LanguageAReprofiler:
    """
    Re-runs null hypothesis framework specifically for Language A metrics.
    """

    def __init__(self, n_samples: int = 100, verbose: bool = True):
        self.n_samples = n_samples
        self.verbose = verbose
        self.splitter = LanguageABSplitter()
        self.lang_a_text = ''
        self.lang_a_profile = {}

    def compute_lang_a_profile(self) -> Dict:
        """Extract Language A text and compute its full statistical profile."""
        lang_a, _ = self.splitter.split_corpus()
        self.lang_a_text = lang_a

        if not lang_a:
            return {}

        self.lang_a_profile = full_statistical_profile(lang_a, 'language_a')
        return self.lang_a_profile

    def run_null_against_lang_a(self) -> Dict:
        """
        Run the null distribution engine and compare against Language A
        targets instead of the combined Voynich targets.

        For each cipher_family x source_language:
        1. Generate encrypted samples
        2. Compute percentile rank of Language A's H2=1.49
        3. Compute p-value
        """
        if not self.lang_a_profile:
            self.compute_lang_a_profile()

        engine = NullDistributionEngine(
            n_samples=self.n_samples,
            n_words=LANG_A_TARGETS['total_tokens'],
            verbose=False,
        )

        # Override the Voynich metrics with Language A metrics
        engine.voynich_metrics = {
            'H1': LANG_A_TARGETS['H1'],
            'H2': LANG_A_TARGETS['H2'],
            'H3': LANG_A_TARGETS['H3'],
            'zipf_exponent': LANG_A_TARGETS['zipf_exponent'],
            'type_token_ratio': LANG_A_TARGETS['type_token_ratio'],
            'mean_word_length': LANG_A_TARGETS['mean_word_length'],
        }

        # Run distributions for all cipher x language combinations
        percentile_ranks = {}
        p_values = {}
        compatible_families = []

        for cipher_name in CIPHER_FAMILIES:
            percentile_ranks[cipher_name] = {}
            p_values[cipher_name] = {}

            for lang in SOURCE_LANGUAGES:
                if self.verbose:
                    print(f'    Testing {cipher_name} x {lang}...', end='')

                samples = engine.generate_reference_corpus(cipher_name, lang)
                if len(samples) < 10:
                    if self.verbose:
                        print(' SKIP')
                    continue

                dist = engine.compute_metric_distributions(samples)

                # Compute p-values for each metric
                lang_p_values = {}
                for metric in ['H2', 'H3', 'zipf_exponent', 'type_token_ratio']:
                    target_val = engine.voynich_metrics.get(metric, 0)
                    metric_dist = dist.get(metric, np.array([]))
                    if len(metric_dist) > 0:
                        p = engine.compute_p_value(target_val, metric_dist)
                        lang_p_values[metric] = p
                    else:
                        lang_p_values[metric] = 1.0

                p_values[cipher_name][lang] = lang_p_values

                # Percentile ranks
                lang_percentiles = {}
                for metric in ['H2', 'H3', 'zipf_exponent', 'type_token_ratio']:
                    target_val = engine.voynich_metrics.get(metric, 0)
                    metric_dist = dist.get(metric, np.array([]))
                    if len(metric_dist) > 0:
                        percentile = float(np.mean(metric_dist <= target_val) * 100)
                        lang_percentiles[metric] = percentile

                percentile_ranks[cipher_name][lang] = lang_percentiles

                # A cipher family is "compatible" if Language A's H2
                # is within the 5th-95th percentile range
                h2_dist = dist.get('H2', np.array([]))
                if len(h2_dist) > 0:
                    p5 = np.percentile(h2_dist, 5)
                    p95 = np.percentile(h2_dist, 95)
                    h2_compatible = p5 <= LANG_A_TARGETS['H2'] <= p95
                else:
                    h2_compatible = False

                if h2_compatible:
                    compatible_families.append({
                        'cipher': cipher_name,
                        'language': lang,
                        'h2_range': (float(p5), float(p95)),
                        'h2_p_value': lang_p_values.get('H2', 1.0),
                    })

                if self.verbose:
                    h2_p = lang_p_values.get('H2', 1.0)
                    status = 'COMPATIBLE' if h2_compatible else 'incompatible'
                    print(f' H2 p={h2_p:.3f} [{status}]')

        return {
            'percentile_ranks': percentile_ranks,
            'p_values': p_values,
            'compatible_families': compatible_families,
            'any_family_compatible': len(compatible_families) > 0,
            'n_compatible': len(compatible_families),
        }

    def test_verbose_cipher_for_lang_a(self) -> Dict:
        """
        Test if the verbose cipher model can match Language A's H2=1.49.

        Phase 2 verbose cipher failed on combined corpus (H2=2.08) but
        Language A's H2=1.49 is closer to the verbose cipher range.
        """
        # Run a coarse sweep first
        model = VerboseCipher(seed=42)
        sweep_results = model.run_sweep(
            resolution='coarse',
            n_best=10,
            verbose=self.verbose,
        )

        # Score against Language A targets instead of combined
        lang_a_scored = []
        for entry in sweep_results.get('best_results', []):
            profile = entry.get('score', {})
            h2 = profile.get('entropy', {}).get('H2', 0)
            ttr_val = profile.get('zipf_summary', {}).get('type_token_ratio', 0)
            zipf_val = profile.get('zipf_summary', {}).get('zipf_exponent', 0)

            h2_delta = abs(h2 - LANG_A_TARGETS['H2'])
            ttr_delta = abs(ttr_val - LANG_A_TARGETS['type_token_ratio'])
            zipf_delta = abs(zipf_val - LANG_A_TARGETS['zipf_exponent'])

            distance = math.sqrt(
                9.0 * h2_delta ** 2 +
                4.0 * ttr_delta ** 2 +
                4.0 * zipf_delta ** 2
            )

            lang_a_scored.append({
                'params': entry.get('params', {}),
                'H2': h2,
                'TTR': ttr_val,
                'zipf': zipf_val,
                'H2_delta': h2_delta,
                'TTR_delta': ttr_delta,
                'zipf_delta': zipf_delta,
                'distance_to_lang_a': distance,
            })

        lang_a_scored.sort(key=lambda x: x['distance_to_lang_a'])
        best = lang_a_scored[0] if lang_a_scored else {}

        return {
            'best_distance': best.get('distance_to_lang_a', float('inf')),
            'best_params': best.get('params', {}),
            'best_H2': best.get('H2', 0),
            'best_TTR': best.get('TTR', 0),
            'best_zipf': best.get('zipf', 0),
            'n_tested': sweep_results.get('n_tested', 0),
            'top_results': lang_a_scored[:5],
            'consistent': best.get('distance_to_lang_a', float('inf')) < 1.0,
            'interpretation': (
                f'Best verbose cipher distance to Language A: '
                f'{best.get("distance_to_lang_a", float("inf")):.4f}. '
                f'H2={best.get("H2", 0):.4f} (target: {LANG_A_TARGETS["H2"]}). '
                f'{"CONSISTENT" if best.get("distance_to_lang_a", float("inf")) < 1.0 else "INCONSISTENT"} '
                f'with Language A being a verbose cipher.'
            ),
        }

    def compare_a_vs_combined(self) -> Dict:
        """
        Compare Language A's anomaly status vs the combined corpus.

        If Language A alone is LESS anomalous, blending with Language B
        was masking a viable cipher family.
        """
        # Combined corpus targets
        combined_targets = {
            'H2': VOYNICH_TARGETS['H2'],           # 1.4058
            'type_token_ratio': VOYNICH_TARGETS['type_token_ratio'],  # 0.1643
            'zipf_exponent': VOYNICH_TARGETS['zipf_exponent'],        # 1.2437
        }

        # Language A targets
        lang_a_targets = {
            'H2': LANG_A_TARGETS['H2'],             # 1.487
            'type_token_ratio': LANG_A_TARGETS['type_token_ratio'],  # 0.305
            'zipf_exponent': LANG_A_TARGETS['zipf_exponent'],        # 0.931
        }

        # Natural language reference ranges (Latin medical text)
        natural_ranges = {
            'H2': (1.2, 2.0),
            'type_token_ratio': (0.15, 0.40),
            'zipf_exponent': (0.8, 1.2),
        }

        comparisons = {}
        for metric, (low, high) in natural_ranges.items():
            combined_val = combined_targets[metric]
            lang_a_val = lang_a_targets[metric]

            combined_in_range = low <= combined_val <= high
            lang_a_in_range = low <= lang_a_val <= high

            comparisons[metric] = {
                'combined': combined_val,
                'lang_a': lang_a_val,
                'natural_range': (low, high),
                'combined_in_range': combined_in_range,
                'lang_a_in_range': lang_a_in_range,
                'lang_a_more_natural': lang_a_in_range and not combined_in_range,
            }

        n_combined_in = sum(1 for c in comparisons.values() if c['combined_in_range'])
        n_lang_a_in = sum(1 for c in comparisons.values() if c['lang_a_in_range'])
        unmasked = sum(1 for c in comparisons.values() if c['lang_a_more_natural'])

        return {
            'comparisons': comparisons,
            'combined_metrics_in_range': n_combined_in,
            'lang_a_metrics_in_range': n_lang_a_in,
            'metrics_unmasked_by_split': unmasked,
            'lang_a_less_anomalous': n_lang_a_in > n_combined_in,
            'interpretation': (
                f'Language A: {n_lang_a_in}/3 metrics in natural range. '
                f'Combined: {n_combined_in}/3 metrics in natural range. '
                f'{unmasked} metrics were masked by Language B blending. '
                f'Language A is {"LESS" if n_lang_a_in > n_combined_in else "NOT less"} '
                f'anomalous than the combined corpus.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run full Language A re-profiling."""
        if verbose:
            print('\n=== Language A Re-profiling ===')

        results = {}

        # Compute Language A profile
        if verbose:
            print('\n  --- Language A Profile ---')
        profile = self.compute_lang_a_profile()
        if verbose and profile:
            ent = profile.get('entropy', {})
            zipf = profile.get('zipf', {})
            print(f'  H1={ent.get("H1", 0):.4f}  '
                  f'H2={ent.get("H2", 0):.4f}  '
                  f'H3={ent.get("H3", 0):.4f}')
            print(f'  Zipf={zipf.get("zipf_exponent", 0):.4f}  '
                  f'TTR={zipf.get("type_token_ratio", 0):.4f}')
            print(f'  Vocabulary: {zipf.get("vocabulary_size", 0)} types')

        results['lang_a_profile'] = {
            'entropy': profile.get('entropy', {}),
            'zipf_exponent': profile.get('zipf', {}).get('zipf_exponent', 0),
            'type_token_ratio': profile.get('zipf', {}).get('type_token_ratio', 0),
            'vocabulary_size': profile.get('zipf', {}).get('vocabulary_size', 0),
            'token_count': profile.get('token_count', 0),
        }

        # A vs Combined comparison
        if verbose:
            print('\n  --- Language A vs Combined Corpus ---')
        results['a_vs_combined'] = self.compare_a_vs_combined()
        if verbose:
            print(f'  {results["a_vs_combined"]["interpretation"]}')

        # Null framework against Language A
        if verbose:
            print('\n  --- Null Framework (Language A targets) ---')
        results['null_comparison'] = self.run_null_against_lang_a()
        if verbose:
            nc = results['null_comparison']
            print(f'  Compatible families: {nc["n_compatible"]}')
            for cf in nc['compatible_families']:
                print(f'    {cf["cipher"]} x {cf["language"]}: '
                      f'H2 range [{cf["h2_range"][0]:.3f}, {cf["h2_range"][1]:.3f}], '
                      f'p={cf["h2_p_value"]:.3f}')

        # Verbose cipher test
        if verbose:
            print('\n  --- Verbose Cipher for Language A ---')
        results['verbose_cipher'] = self.test_verbose_cipher_for_lang_a()
        if verbose:
            print(f'  {results["verbose_cipher"]["interpretation"]}')

        # Synthesis
        results['synthesis'] = {
            'lang_a_less_anomalous': results['a_vs_combined']['lang_a_less_anomalous'],
            'any_cipher_compatible': results['null_comparison']['any_family_compatible'],
            'verbose_cipher_viable': results['verbose_cipher']['consistent'],
            'conclusion': _synthesize_lang_a(results),
        }

        if verbose:
            print(f'\n  Conclusion: {results["synthesis"]["conclusion"]}')

        return results


def _synthesize_lang_a(results: Dict) -> str:
    """Generate conclusion about Language A."""
    less_anomalous = results.get('a_vs_combined', {}).get('lang_a_less_anomalous', False)
    compatible = results.get('null_comparison', {}).get('any_family_compatible', False)
    verbose_ok = results.get('verbose_cipher', {}).get('consistent', False)

    parts = []
    if less_anomalous:
        parts.append('Language A is less anomalous than the combined corpus')
    if compatible:
        n = results['null_comparison']['n_compatible']
        parts.append(f'{n} cipher family/language combinations are compatible')
    if verbose_ok:
        parts.append('verbose cipher model is consistent with Language A')

    if parts:
        return (
            'PROMISING — ' + '; '.join(parts) + '. '
            'Language B was masking Language A\'s cipher signature.'
        )
    else:
        return (
            'INCONCLUSIVE — Language A remains anomalous even in isolation. '
            'The encoding may not be a conventional cipher.'
        )
