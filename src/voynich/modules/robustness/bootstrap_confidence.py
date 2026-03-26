"""
Test 5b: Bootstrap Confidence Intervals
=========================================
Runs the pipeline N times with all key parameters simultaneously jittered
by ±10%. Reports resolution as mean ± std with 95% CI.
"""
import random
import time
from typing import Dict

from voynich.phases.robustness import (
    build_default_solver_kwargs, run_full_pipeline,
)
from voynich.modules.phase12.ngram_mask_solver import NgramMaskSolver

# Parameters to jitter and their defaults (must match solver kwarg names)
BOOTSTRAP_PARAMS = {
    'min_confidence_ratio': 5.0,
    'dual_context_ratio_factor': 0.6,
    'unigram_backoff_ratio_factor': 1.5,
    'char_ngram_min_score_gap': 0.5,
    'pos_backoff_weight': 0.1,
    'pos_backoff_min_confidence': 5.0,
    'adaptive_confidence_2_cand_factor': 0.75,
    'adaptive_confidence_few_cand_factor': 0.9,
    'single_cand_min_char_score': -6.0,
}

N_BOOTSTRAP_RUNS = 50
JITTER_FRACTION = 0.10


class BootstrapConfidence:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        resolutions = []
        lang_a_rates = []
        lang_b_rates = []
        run_details = []

        if self.verbose:
            print(f'\n  Running {N_BOOTSTRAP_RUNS} bootstrap trials '
                  f'(±{100 * JITTER_FRACTION:.0f}% jitter)...')

        for trial in range(N_BOOTSTRAP_RUNS):
            t0 = time.time()

            # Jitter all parameters
            rng = random.Random(trial)
            overrides = {}
            for param, default_val in BOOTSTRAP_PARAMS.items():
                factor = rng.uniform(
                    1.0 - JITTER_FRACTION, 1.0 + JITTER_FRACTION,
                )
                overrides[param] = default_val * factor

            # Build solver with jittered params
            solver_kwargs = build_default_solver_kwargs(
                self.components['pos_tagger'],
                self.components['pos_matrix'],
                self.components['pos_vocab'],
                self.components['char_ngram_model'],
                self.components['illustration_prior'],
                **overrides,
            )
            solver = NgramMaskSolver(
                self.components['trans_matrix'],
                self.components['trans_vocab'],
                self.components['latin_skel'],
                self.components['fuzzy_skel'],
                **solver_kwargs,
            )
            solver.set_corpus_frequencies(self.components['l_tokens'])

            # Run pipeline
            pipeline_result = run_full_pipeline(
                self.components, solver_override=solver, verbose=False,
            )

            elapsed = time.time() - t0
            resolutions.append(pipeline_result['overall_resolution'])
            lang_a_rates.append(pipeline_result['lang_a_resolution'])
            lang_b_rates.append(pipeline_result['lang_b_resolution'])

            run_details.append({
                'trial': trial,
                'overrides': {k: round(v, 4) for k, v in overrides.items()},
                'resolution': round(pipeline_result['overall_resolution'], 4),
                'lang_a': round(pipeline_result['lang_a_resolution'], 4),
                'lang_b': round(pipeline_result['lang_b_resolution'], 4),
                'elapsed': round(elapsed, 2),
            })

            if self.verbose and (trial + 1) % 10 == 0:
                mean_so_far = sum(resolutions) / len(resolutions)
                print(f'    Trial {trial + 1}/{N_BOOTSTRAP_RUNS}: '
                      f'mean resolution so far = {100 * mean_so_far:.1f}%  '
                      f'({elapsed:.1f}s/trial)')

        # Compute statistics
        n = len(resolutions)
        mean_res = sum(resolutions) / n
        std_res = (sum((r - mean_res) ** 2 for r in resolutions) / n) ** 0.5
        sorted_res = sorted(resolutions)
        ci_low = sorted_res[max(0, int(n * 0.025))]
        ci_high = sorted_res[min(n - 1, int(n * 0.975))]
        min_res = min(resolutions)
        max_res = max(resolutions)

        mean_a = sum(lang_a_rates) / n
        mean_b = sum(lang_b_rates) / n

        # Safety assessment
        safe_runs = sum(1 for r in resolutions if r > 0.40)

        result = {
            'test': 'bootstrap_confidence',
            'n_runs': n,
            'jitter_fraction': JITTER_FRACTION,
            'parameters_jittered': list(BOOTSTRAP_PARAMS.keys()),
            'statistics': {
                'mean': round(mean_res, 4),
                'std': round(std_res, 4),
                'ci_95_low': round(ci_low, 4),
                'ci_95_high': round(ci_high, 4),
                'min': round(min_res, 4),
                'max': round(max_res, 4),
                'mean_lang_a': round(mean_a, 4),
                'mean_lang_b': round(mean_b, 4),
            },
            'safety': {
                'runs_above_40pct': safe_runs,
                'fraction_safe': round(safe_runs / n, 4),
            },
            'interpretation': self._interpret(mean_res, std_res, safe_runs, n),
            'runs': run_details,
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _interpret(self, mean, std, safe_runs, n):
        pct_std = std * 100
        if pct_std < 2:
            robustness = 'VERY ROBUST'
            desc = 'Result barely moves with ±10% jitter.'
        elif pct_std < 5:
            robustness = 'MODERATELY ROBUST'
            desc = 'Normal sensitivity to parameter perturbation.'
        else:
            robustness = 'SENSITIVE'
            desc = 'Result depends heavily on parameter choices.'

        safety_desc = (
            f'{safe_runs}/{n} runs ({100 * safe_runs / n:.0f}%) remain '
            f'in the safe operating region (>40% resolution).'
        )

        return f'{robustness}: {desc} {safety_desc}'

    def _print_report(self, result):
        stats = result['statistics']

        print(f'\nBootstrap Confidence Intervals '
              f'({result["n_runs"]} runs, ±{100 * result["jitter_fraction"]:.0f}% jitter)')
        print('=' * 55)
        print(f'\nResolution:')
        print(f'  Mean:   {100 * stats["mean"]:.1f}%')
        print(f'  Std:    {100 * stats["std"]:.1f}%')
        print(f'  95% CI: [{100 * stats["ci_95_low"]:.1f}%, '
              f'{100 * stats["ci_95_high"]:.1f}%]')
        print(f'  Min:    {100 * stats["min"]:.1f}%')
        print(f'  Max:    {100 * stats["max"]:.1f}%')
        print(f'  Lang A: {100 * stats["mean_lang_a"]:.1f}% (mean)')
        print(f'  Lang B: {100 * stats["mean_lang_b"]:.1f}% (mean)')

        safety = result['safety']
        print(f'\nSafety:')
        print(f'  Runs in safe region (>40%): '
              f'{safety["runs_above_40pct"]}/{result["n_runs"]}')

        print(f'\n{result["interpretation"]}')

        # ASCII histogram
        resolutions = [r['resolution'] for r in result['runs']]
        self._print_histogram(resolutions)

    def _print_histogram(self, values):
        """Print a simple ASCII histogram."""
        if not values:
            return
        min_v = min(values)
        max_v = max(values)
        n_bins = 10
        if max_v == min_v:
            print(f'\n  All values: {100 * min_v:.1f}%')
            return

        bin_width = (max_v - min_v) / n_bins
        bins = [0] * n_bins
        for v in values:
            idx = min(int((v - min_v) / bin_width), n_bins - 1)
            bins[idx] += 1

        max_count = max(bins)
        print(f'\n  Distribution:')
        for i, count in enumerate(bins):
            lo = min_v + i * bin_width
            hi = lo + bin_width
            bar = '#' * int(30 * count / max(1, max_count))
            print(f'  {100 * lo:5.1f}-{100 * hi:5.1f}% | {bar} ({count})')
