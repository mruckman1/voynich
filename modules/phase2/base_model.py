"""
Phase 2 Base Model
===================
Abstract base class for all six super-character generative models.
Provides a uniform interface for the discrimination sweep and orchestrator,
with shared scoring implementations that reuse Phase 1 statistical tools.
"""

import sys
import os
import math
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.statistical_analysis import (
    full_statistical_profile, profile_distance, compute_all_entropy,
    zipf_analysis
)


# ============================================================================
# VOYNICH TARGET CONSTANTS (from output/null_distributions.json)
# ============================================================================

VOYNICH_TARGETS = {
    'H1': 3.7065,
    'H2': 1.4058,
    'H3': 0.8983,
    'zipf_exponent': 1.2437,
    'type_token_ratio': 0.1643,
    'mean_word_length': 4.908,
    'fsa_states': 56,
    'nmf_effective_rank': 12,
}

# Quick discrimination thresholds (from research document)
TRIPLE_THRESHOLDS = {
    'H2': 0.1,          # |H2 - 1.4058| < 0.1
    'TTR': 0.02,        # |TTR - 0.1643| < 0.02
    'zipf_exponent': 0.15,  # |Zipf - 1.2437| < 0.15
}


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class Phase2GenerativeModel(ABC):
    """
    Abstract base class for all Phase 2 super-character generative models.

    Each model must implement:
      - generate(): produce synthetic Voynich-like text
      - parameter_grid(): return the parameter sweep grid
      - critical_test(): run model-specific pass/fail test

    Shared implementations:
      - quick_score(): H2/TTR/Zipf triple check
      - full_score(): all 17 constraints via ConstraintModel
      - run_sweep(): full parameter sweep with ranking
    """

    MODEL_NAME: str = 'base'
    MODEL_PRIORITY: str = 'MEDIUM'

    @abstractmethod
    def __init__(self, **params):
        """Initialize with model-specific parameters."""
        self.params = params
        self.seed = params.get('seed', 42)
        self.rng = random.Random(self.seed)

    @abstractmethod
    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Generate synthetic Voynich-like text.

        Parameters:
            plaintext: source plaintext (some models ignore this)
            n_words: target number of output tokens

        Returns:
            Space-separated token string.
        """

    @abstractmethod
    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """
        Return the parameter sweep grid for this model.

        Parameters:
            resolution: 'coarse' (~10-25), 'medium' (~50-200), 'fine' (~500+)

        Returns:
            List of parameter dicts, each suitable for __init__(**params).
        """

    @abstractmethod
    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Run model-specific critical test.

        Parameters:
            generated_profile: output of full_statistical_profile() on generated text

        Returns:
            {passes: bool, details: {...}, description: str}
        """

    def get_profile(self, text: str) -> Dict:
        """Compute full statistical profile of generated text."""
        return full_statistical_profile(text, label=f'{self.MODEL_NAME}_output')

    def quick_score(self, generated_profile: Dict) -> Dict:
        """
        Score against the H2/TTR/Zipf triple for discrimination sweep.

        Returns:
            {H2: float, TTR: float, zipf_exponent: float,
             H2_delta: float, TTR_delta: float, zipf_delta: float,
             triple_match: bool, distance: float}
        """
        entropy = generated_profile.get('entropy', {})
        zipf = generated_profile.get('zipf', {})

        h2 = entropy.get('H2', 0.0)
        ttr = zipf.get('type_token_ratio', 0.0)
        zipf_exp = zipf.get('zipf_exponent', 0.0)

        h2_delta = abs(h2 - VOYNICH_TARGETS['H2'])
        ttr_delta = abs(ttr - VOYNICH_TARGETS['type_token_ratio'])
        zipf_delta = abs(zipf_exp - VOYNICH_TARGETS['zipf_exponent'])

        triple_match = (
            h2_delta < TRIPLE_THRESHOLDS['H2'] and
            ttr_delta < TRIPLE_THRESHOLDS['TTR'] and
            zipf_delta < TRIPLE_THRESHOLDS['zipf_exponent']
        )

        # Composite distance (weighted)
        distance = math.sqrt(
            9.0 * h2_delta ** 2 +      # H2 weighted heavily
            4.0 * ttr_delta ** 2 +      # TTR weighted moderately
            4.0 * zipf_delta ** 2       # Zipf weighted moderately
        )

        return {
            'H2': h2,
            'TTR': ttr,
            'zipf_exponent': zipf_exp,
            'H2_delta': h2_delta,
            'TTR_delta': ttr_delta,
            'zipf_delta': zipf_delta,
            'triple_match': triple_match,
            'distance': distance,
        }

    def full_score(self, generated_profile: Dict) -> Dict:
        """
        Score against the full Voynich target profile using profile_distance.

        Returns:
            {distance: float, entropy: {...}, zipf: {...}, quick: {...}}
        """
        # Build a Voynich-like profile dict for comparison
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

        dist = profile_distance(generated_profile, voynich_profile)
        quick = self.quick_score(generated_profile)

        return {
            'profile_distance': dist,
            'entropy': generated_profile.get('entropy', {}),
            'zipf_summary': {
                'zipf_exponent': generated_profile.get('zipf', {}).get('zipf_exponent', 0),
                'type_token_ratio': generated_profile.get('zipf', {}).get('type_token_ratio', 0),
                'vocabulary_size': generated_profile.get('zipf', {}).get('vocabulary_size', 0),
                'total_tokens': generated_profile.get('zipf', {}).get('total_tokens', 0),
            },
            'mean_word_length': generated_profile.get('mean_word_length', 0),
            'quick': quick,
        }

    def run_sweep(self, plaintext: str = '', resolution: str = 'medium',
                  n_best: int = 20, verbose: bool = True) -> Dict:
        """
        Run parameter sweep, return ranked results.

        Parameters:
            plaintext: source text for generation
            resolution: parameter grid resolution
            n_best: number of top results to keep
            verbose: print progress

        Returns:
            {model: str, resolution: str, n_tested: int,
             best_results: [...], triple_matches: [...],
             best_distance: float, elapsed_seconds: float}
        """
        grid = self.parameter_grid(resolution)
        results = []
        triple_matches = []
        t0 = time.time()

        for i, params in enumerate(grid):
            try:
                # Create fresh model instance with these params
                model = self.__class__(**params)
                text = model.generate(plaintext=plaintext)

                if not text or len(text.split()) < 10:
                    continue

                profile = model.get_profile(text)
                score = model.full_score(profile)
                crit = model.critical_test(profile)

                entry = {
                    'params': params,
                    'score': score,
                    'critical_test': crit,
                    'sample_output': ' '.join(text.split()[:20]),
                }
                results.append(entry)

                if score['quick']['triple_match']:
                    triple_matches.append(entry)

                if verbose and (i + 1) % max(1, len(grid) // 10) == 0:
                    print(f'  [{self.MODEL_NAME}] {i+1}/{len(grid)} tested, '
                          f'{len(triple_matches)} triple matches')

            except Exception as e:
                if verbose:
                    print(f'  [{self.MODEL_NAME}] params {i+1} failed: {e}')
                continue

        # Sort by profile distance (lower = better)
        results.sort(key=lambda r: r['score']['profile_distance'])
        elapsed = time.time() - t0

        if verbose:
            print(f'  [{self.MODEL_NAME}] Sweep complete: {len(results)} valid, '
                  f'{len(triple_matches)} triple matches, {elapsed:.1f}s')

        return {
            'model': self.MODEL_NAME,
            'resolution': resolution,
            'n_tested': len(grid),
            'n_valid': len(results),
            'best_results': results[:n_best],
            'triple_matches': triple_matches[:n_best],
            'best_distance': results[0]['score']['profile_distance'] if results else float('inf'),
            'elapsed_seconds': elapsed,
        }

    def get_params(self) -> Dict:
        """Return the current model parameters."""
        return dict(self.params)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.params})'
