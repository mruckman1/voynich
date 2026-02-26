"""
Phase 2 Null Distribution Engine
===================================
Generates null distributions for all six Phase 2 generative models,
following the pattern established by NullDistributionEngine in
modules/null_framework.py.

For each model × language combination, generates N synthetic texts
and computes their statistical profiles to build reference distributions.
"""

import sys
import os
import time
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.statistical_analysis import full_statistical_profile
from modules.phase2.base_model import VOYNICH_TARGETS
from modules.phase2.verbose_cipher import VerboseCipher
from modules.phase2.syllabary_code import SyllabaryCode
from modules.phase2.slot_machine import SlotMachine
from modules.phase2.steganographic_carrier import SteganographicCarrier
from modules.phase2.grammar_induction import GrammarInduction
from modules.phase2.glyph_decomposition import GlyphDecomposition
from modules.strategy1_parameter_search import generate_medical_plaintext
from data.medieval_text_templates import generate_italian_text, generate_german_text


MODEL_REGISTRY = {
    'verbose_cipher': VerboseCipher,
    'syllabary_code': SyllabaryCode,
    'slot_machine': SlotMachine,
    'steganographic_carrier': SteganographicCarrier,
    'grammar_induction': GrammarInduction,
    'glyph_decomposition': GlyphDecomposition,
}

METRICS = ['H1', 'H2', 'H3', 'zipf_exponent', 'type_token_ratio', 'mean_word_length']


def _get_plaintext(language: str, n_words: int = 600) -> str:
    """Generate plaintext in the specified language."""
    if language == 'latin':
        return generate_medical_plaintext(n_words)
    elif language == 'italian':
        return generate_italian_text(n_words)
    elif language == 'german':
        return generate_german_text(n_words)
    return generate_medical_plaintext(n_words)


class Phase2NullEngine:
    """
    Generate null distributions for Phase 2 models.

    For each model × language pair, generates n_samples synthetic texts
    with randomized seeds and computes their statistical profiles.
    """

    def __init__(self, n_samples: int = 50, verbose: bool = True):
        self.n_samples = n_samples
        self.verbose = verbose
        self.distributions = {}  # model -> language -> metric -> [values]

    def generate_model_distributions(self, model_name: str,
                                     source_language: str) -> Dict:
        """
        Generate reference distributions for a single model × language pair.

        Returns:
            {metric: {mean, std, min, max, percentile_5, percentile_95, values}}
        """
        if model_name not in MODEL_REGISTRY:
            return {'error': f'Unknown model: {model_name}'}

        model_class = MODEL_REGISTRY[model_name]
        metric_values = {m: [] for m in METRICS}

        for i in range(self.n_samples):
            seed = 42 + i * 137  # Reproducible but varied seeds

            try:
                if model_name in ('glyph_decomposition', 'steganographic_carrier'):
                    # These models don't use plaintext
                    model = model_class(seed=seed)
                    text = model.generate(n_words=500)
                elif model_name == 'grammar_induction':
                    # Grammar induction needs special handling
                    model = model_class(
                        max_rules=20, max_symbols=50,
                        evolution_generations=50, seed=seed
                    )
                    text = model.generate(n_words=500)
                else:
                    plaintext = _get_plaintext(source_language)
                    model = model_class(seed=seed)
                    text = model.generate(plaintext=plaintext, n_words=500)

                if not text or len(text.split()) < 20:
                    continue

                profile = full_statistical_profile(text, f'{model_name}_{i}')

                metric_values['H1'].append(profile['entropy']['H1'])
                metric_values['H2'].append(profile['entropy']['H2'])
                metric_values['H3'].append(profile['entropy']['H3'])
                metric_values['zipf_exponent'].append(
                    profile['zipf']['zipf_exponent'])
                metric_values['type_token_ratio'].append(
                    profile['zipf']['type_token_ratio'])
                metric_values['mean_word_length'].append(
                    profile.get('mean_word_length', 0))

            except Exception as e:
                if self.verbose:
                    print(f'  Sample {i} failed: {e}')
                continue

        # Compute distribution statistics
        dist = {}
        for metric, values in metric_values.items():
            if not values:
                dist[metric] = {'error': 'No valid samples'}
                continue

            arr = np.array(values)
            dist[metric] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'percentile_5': float(np.percentile(arr, 5)),
                'percentile_95': float(np.percentile(arr, 95)),
                'n_samples': len(values),
            }

        return dist

    def compute_p_values(self, model_name: str,
                         source_language: str) -> Dict:
        """
        Compute p-values for Voynich metrics against model distributions.

        Returns:
            {metric: {voynich_value, percentile_rank, p_value, anomalous}}
        """
        dist = self.distributions.get(model_name, {}).get(source_language, {})
        if not dist:
            dist = self.generate_model_distributions(model_name, source_language)
            # Cache
            if model_name not in self.distributions:
                self.distributions[model_name] = {}
            self.distributions[model_name][source_language] = dist

        results = {}
        for metric in METRICS:
            voynich_val = VOYNICH_TARGETS.get(metric, 0)
            metric_dist = dist.get(metric, {})

            if 'error' in metric_dist:
                results[metric] = {'error': metric_dist['error']}
                continue

            mean = metric_dist.get('mean', 0)
            std = metric_dist.get('std', 1)

            # Two-sided p-value (how unusual is the Voynich value?)
            if std > 0:
                z_score = abs(voynich_val - mean) / std
                # Approximate p-value from z-score
                p_value = 2.0 * (1.0 - _normal_cdf(z_score))
            else:
                p_value = 0.0 if voynich_val != mean else 1.0

            # Percentile rank
            percentile = _estimate_percentile(voynich_val, mean, std)

            results[metric] = {
                'voynich_value': voynich_val,
                'model_mean': mean,
                'model_std': std,
                'z_score': z_score if std > 0 else float('inf'),
                'percentile_rank': percentile,
                'p_value': p_value,
                'anomalous': p_value < 0.05,
            }

        return results

    def run_all(self, models: Optional[List[str]] = None,
                languages: Optional[List[str]] = None) -> Dict:
        """
        Run all models × languages.

        Returns:
            {model: {language: {distributions, p_values, anomaly_summary}}}
        """
        if models is None:
            models = list(MODEL_REGISTRY.keys())
        if languages is None:
            languages = ['latin', 'italian', 'german']

        t0 = time.time()
        results = {}

        for model_name in models:
            if self.verbose:
                print(f'\n--- Null distributions: {model_name} ---')

            results[model_name] = {}

            # Some models don't vary by language
            if model_name in ('glyph_decomposition', 'steganographic_carrier'):
                lang = 'n/a'
                if self.verbose:
                    print(f'  (language-independent model)')
                dist = self.generate_model_distributions(model_name, lang)
                if model_name not in self.distributions:
                    self.distributions[model_name] = {}
                self.distributions[model_name][lang] = dist

                p_values = self.compute_p_values(model_name, lang)
                n_anomalous = sum(1 for v in p_values.values()
                                  if isinstance(v, dict) and v.get('anomalous', False))

                results[model_name][lang] = {
                    'distributions': dist,
                    'p_values': p_values,
                    'n_anomalous': n_anomalous,
                    'n_metrics': len(METRICS),
                }
                continue

            for language in languages:
                if self.verbose:
                    print(f'  Language: {language} ({self.n_samples} samples)')

                dist = self.generate_model_distributions(model_name, language)
                if model_name not in self.distributions:
                    self.distributions[model_name] = {}
                self.distributions[model_name][language] = dist

                p_values = self.compute_p_values(model_name, language)
                n_anomalous = sum(1 for v in p_values.values()
                                  if isinstance(v, dict) and v.get('anomalous', False))

                results[model_name][language] = {
                    'distributions': dist,
                    'p_values': p_values,
                    'n_anomalous': n_anomalous,
                    'n_metrics': len(METRICS),
                }

                if self.verbose:
                    print(f'    {n_anomalous}/{len(METRICS)} metrics anomalous')

        elapsed = time.time() - t0
        if self.verbose:
            print(f'\nTotal null distribution time: {elapsed:.1f}s')

        return {
            'results': results,
            'n_samples': self.n_samples,
            'elapsed_seconds': elapsed,
        }


# ============================================================================
# HELPERS
# ============================================================================

def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using the error function."""
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _estimate_percentile(value: float, mean: float, std: float) -> float:
    """Estimate percentile rank assuming normal distribution."""
    if std <= 0:
        return 50.0
    z = (value - mean) / std
    return _normal_cdf(z) * 100
