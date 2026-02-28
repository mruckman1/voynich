"""
Test 5c: Ablation Cascade
===========================
Measures the contribution of each pipeline improvement by disabling them
one at a time (individual ablation) and adding them one at a time
(cumulative build).

Two modes:
  1. Individual ablation: disable ONE improvement, run pipeline, measure drop
  2. Cumulative build: add improvements one at a time in chronological order
"""
import time
from typing import Dict, List, Optional, Set

from orchestrators.robustness import (
    build_default_solver_kwargs, run_full_pipeline, NgramMaskSolver,
)


# Each improvement with how to disable it.
# 'solver_kwarg' items are toggled in solver construction.
# 'pipeline_feature' items are toggled via disabled_features in run_full_pipeline.
IMPROVEMENTS = [
    {
        'name': 'Cross-folio consistency',
        'type': 'pipeline_feature',
        'feature': 'cross_folio_consistency',
    },
    {
        'name': 'Relaxed consistency',
        'type': 'pipeline_feature',
        'feature': 'relaxed_consistency',
    },
    {
        'name': 'Graduated CSP',
        'type': 'pipeline_feature',
        'feature': 'graduated_csp',
    },
    {
        'name': 'POS backoff',
        'type': 'pipeline_feature',
        'feature': 'pos_backoff',
    },
    {
        'name': 'Character n-gram',
        'type': 'pipeline_feature',
        'feature': 'char_ngram',
    },
    {
        'name': 'Iterative refinement',
        'type': 'pipeline_feature',
        'feature': 'iterative_refinement',
    },
    {
        'name': 'Unigram backoff',
        'type': 'solver_kwarg',
        'kwarg': 'enable_unigram_backoff',
    },
    {
        'name': 'Adaptive confidence',
        'type': 'solver_kwarg',
        'kwarg': 'enable_adaptive_confidence',
    },
    {
        'name': 'Single-cand char rescue',
        'type': 'solver_kwarg',
        'kwarg': 'enable_single_cand_char_rescue',
    },
    {
        'name': 'Illustration prior',
        'type': 'solver_kwarg',
        'kwarg': 'enable_illustration_prior',
    },
    {
        'name': 'Function word recovery',
        'type': 'solver_kwarg',
        'kwarg': 'enable_function_word_recovery',
    },
]

# All pipeline-level features (for disabling all at once)
ALL_PIPELINE_FEATURES = {
    imp['feature'] for imp in IMPROVEMENTS if imp['type'] == 'pipeline_feature'
}

# All solver-level kwargs that can be toggled (for disabling all at once)
ALL_SOLVER_TOGGLES = {
    imp['kwarg'] for imp in IMPROVEMENTS if imp['type'] == 'solver_kwarg'
}


class AblationCascade:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        t0 = time.time()

        # Full pipeline baseline
        if self.verbose:
            print('  Running full pipeline baseline...')
        baseline = run_full_pipeline(self.components, verbose=False)
        baseline_rate = baseline['overall_resolution']

        # Individual ablation
        if self.verbose:
            print('  Running individual ablation...')
        individual_results = self._run_individual_ablation(baseline_rate)

        # Cumulative build
        if self.verbose:
            print('  Running cumulative build...')
        cumulative_results = self._run_cumulative_build()

        elapsed = time.time() - t0

        result = {
            'test': 'ablation_cascade',
            'baseline_resolution': round(baseline_rate, 4),
            'individual_ablation': individual_results,
            'cumulative_build': cumulative_results,
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _build_solver_with_overrides(self, **overrides) -> NgramMaskSolver:
        """Build a new NgramMaskSolver with specific kwarg overrides."""
        c = self.components
        kwargs = build_default_solver_kwargs(
            c['pos_tagger'], c['pos_matrix'], c['pos_vocab'],
            c['char_ngram_model'], c['illustration_prior'],
            **overrides,
        )
        solver = NgramMaskSolver(
            c['trans_matrix'], c['trans_vocab'],
            c['latin_skel'], c['fuzzy_skel'],
            **kwargs,
        )
        solver.set_corpus_frequencies(c['l_tokens'])
        return solver

    def _run_individual_ablation(self, baseline_rate: float) -> List[Dict]:
        """Disable one improvement at a time, measure resolution drop."""
        results = []
        for imp in IMPROVEMENTS:
            name = imp['name']
            if self.verbose:
                print(f'    - {name}...')

            if imp['type'] == 'pipeline_feature':
                rate = run_full_pipeline(
                    self.components,
                    disabled_features={imp['feature']},
                    verbose=False,
                )['overall_resolution']
            else:
                solver = self._build_solver_with_overrides(**{imp['kwarg']: False})
                rate = run_full_pipeline(
                    self.components,
                    solver_override=solver,
                    verbose=False,
                )['overall_resolution']

            delta = rate - baseline_rate
            results.append({
                'name': name,
                'resolution': round(rate, 4),
                'delta': round(delta, 4),
            })

        results.sort(key=lambda x: x['delta'])
        return results

    def _run_cumulative_build(self) -> List[Dict]:
        """Start with all features off, add one at a time chronologically."""
        results = []

        # Base: all features disabled
        if self.verbose:
            print('    - Base (all off)...')
        all_off_overrides = {kwarg: False for kwarg in ALL_SOLVER_TOGGLES}
        base_solver = self._build_solver_with_overrides(**all_off_overrides)
        base_rate = run_full_pipeline(
            self.components,
            solver_override=base_solver,
            disabled_features=ALL_PIPELINE_FEATURES.copy(),
            verbose=False,
        )['overall_resolution']
        results.append({
            'step': 0,
            'name': 'Base pipeline (all off)',
            'resolution': round(base_rate, 4),
            'gain': 0.0,
        })

        # Track which features are currently enabled
        enabled_pipeline: Set[str] = set()
        enabled_solver_overrides = dict(all_off_overrides)
        prev_rate = base_rate

        for i, imp in enumerate(IMPROVEMENTS, 1):
            name = imp['name']
            if self.verbose:
                print(f'    + {name}...')

            if imp['type'] == 'pipeline_feature':
                enabled_pipeline.add(imp['feature'])
            else:
                enabled_solver_overrides[imp['kwarg']] = True

            still_disabled = ALL_PIPELINE_FEATURES - enabled_pipeline
            solver = self._build_solver_with_overrides(**enabled_solver_overrides)
            rate = run_full_pipeline(
                self.components,
                solver_override=solver,
                disabled_features=still_disabled if still_disabled else None,
                verbose=False,
            )['overall_resolution']

            gain = rate - prev_rate
            results.append({
                'step': i,
                'name': f'+ {name}',
                'resolution': round(rate, 4),
                'gain': round(gain, 4),
            })
            prev_rate = rate

        return results

    def _print_report(self, result):
        baseline = result['baseline_resolution']
        individual = result['individual_ablation']
        cumulative = result['cumulative_build']

        print(f'\nAblation Cascade Analysis')
        print('=' * 70)

        print(f'\nIndividual Ablation (disable one, keep rest):')
        print(f'  {"Improvement":<30} {"Resolution":>12} {"Delta":>10}')
        print('  ' + '-' * 54)
        print(f'  {"Full pipeline":<30} {100*baseline:>11.1f}% {"baseline":>10}')
        for r in individual:
            print(f'  {"- " + r["name"]:<30} {100*r["resolution"]:>11.1f}% '
                  f'{100*r["delta"]:>+9.1f}pp')

        print(f'\nCumulative Build (add one at a time):')
        print(f'  {"Step":<4} {"Improvement":<32} {"Resolution":>12} {"Gain":>10}')
        print('  ' + '-' * 60)
        for r in cumulative:
            if r['step'] == 0:
                print(f'  {r["step"]:<4} {r["name"]:<32} '
                      f'{100*r["resolution"]:>11.1f}% {"":>10}')
            else:
                print(f'  {r["step"]:<4} {r["name"]:<32} '
                      f'{100*r["resolution"]:>11.1f}% {100*r["gain"]:>+9.1f}pp')

        # Summary
        if individual:
            largest = individual[0]  # sorted by delta ascending (most negative first)
            smallest = individual[-1]
            print(f'\n  Largest contributor:  {largest["name"]} at {100*largest["delta"]:+.1f}pp')
            print(f'  Smallest contributor: {smallest["name"]} at {100*smallest["delta"]:+.1f}pp')
            total_gain = baseline - cumulative[0]['resolution'] if cumulative else 0
            if total_gain > 0 and largest['delta'] < 0:
                pct = abs(largest['delta']) / total_gain * 100
                print(f'  Largest accounts for {pct:.0f}% of total gain')
        print(f'  Elapsed: {result["elapsed_seconds"]:.1f}s')
