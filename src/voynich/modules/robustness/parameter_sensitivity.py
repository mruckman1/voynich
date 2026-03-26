"""
Test 5a: Parameter Sensitivity Sweep
======================================
For each key parameter, sweeps through a range of values and records
resolution rate at each point. Identifies the "safe operating region"
where real resolution is high AND the parameter choice is defensible.
"""
import random
import re
import time
from typing import Dict, List

from voynich.phases.robustness import (
    build_default_solver_kwargs, run_full_pipeline, _count_brackets,
)
from voynich.core.config import (
    MIN_CONFIDENCE_RATIO, CSP_HIGH_CONFIDENCE_THRESHOLD,
    CROSS_FOLIO_MIN_AGREEMENT, CHAR_NGRAM_MIN_SCORE_GAP,
    FUNCTION_WORD_MAX_DENSITY, DUAL_CONTEXT_RATIO_FACTOR,
    ENABLE_GRADUATED_CSP, CSP_MEDIUM_CONFIDENCE_THRESHOLD,
    ENABLE_SELECTIVE_FUNCTION_WORDS, FUNCTION_WORD_WINDOW_SIZE,
)
from voynich.modules.phase12.ngram_mask_solver import NgramMaskSolver
from voynich.modules.phase12.budgeted_csp import BudgetedCSPDecoder

# Parameter sweep definitions: (config_name, default, sweep_values, affects)
# 'affects' = 'solver', 'decoder', or 'consistency'
SWEEP_PARAMS = [
    ('MIN_CONFIDENCE_RATIO', MIN_CONFIDENCE_RATIO,
     [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 'solver'),
    ('CSP_HIGH_CONFIDENCE_THRESHOLD', CSP_HIGH_CONFIDENCE_THRESHOLD,
     [10, 14, 18, 20, 22, 26, 30], 'decoder'),
    ('CROSS_FOLIO_MIN_AGREEMENT', CROSS_FOLIO_MIN_AGREEMENT,
     [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'consistency'),
    ('CHAR_NGRAM_MIN_SCORE_GAP', CHAR_NGRAM_MIN_SCORE_GAP,
     [0.1, 0.2, 0.3, 0.5, 0.7, 1.0], 'solver'),
    ('FUNCTION_WORD_MAX_DENSITY', FUNCTION_WORD_MAX_DENSITY,
     [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 'decoder'),
    ('DUAL_CONTEXT_RATIO_FACTOR', DUAL_CONTEXT_RATIO_FACTOR,
     [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], 'solver'),
]


class ParameterSensitivity:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        all_sweep_results = {}

        for param_name, default_val, values, affects in SWEEP_PARAMS:
            if self.verbose:
                print(f'\n  Sweeping {param_name} (current: {default_val})...')

            sweep_results = []
            for val in values:
                t0 = time.time()
                is_default = (val == default_val)

                result = self._run_with_override(param_name, val, affects)
                elapsed = time.time() - t0

                entry = {
                    'value': val,
                    'resolution': round(result['overall_resolution'], 4),
                    'lang_a': round(result['lang_a_resolution'], 4),
                    'lang_b': round(result['lang_b_resolution'], 4),
                    'is_default': is_default,
                    'elapsed': round(elapsed, 2),
                }
                sweep_results.append(entry)

                if self.verbose:
                    marker = ' <-- current' if is_default else ''
                    print(f'    {param_name}={val:<6}  '
                          f'resolution={100 * entry["resolution"]:.1f}%  '
                          f'({elapsed:.1f}s){marker}')

            # Find safe region (resolution > 40%)
            safe_values = [r for r in sweep_results if r['resolution'] > 0.40]
            resolution_range = (
                (min(r['resolution'] for r in safe_values),
                 max(r['resolution'] for r in safe_values))
                if safe_values else (0, 0)
            )

            all_sweep_results[param_name] = {
                'default': default_val,
                'affects': affects,
                'sweep': sweep_results,
                'safe_region_size': len(safe_values),
                'total_values': len(values),
                'resolution_range_in_safe': [
                    round(resolution_range[0], 4),
                    round(resolution_range[1], 4),
                ],
            }

        # Summary
        summary = []
        for param_name, data in all_sweep_results.items():
            safe_frac = data['safe_region_size'] / max(1, data['total_values'])
            robustness = 'ROBUST' if safe_frac >= 0.5 else 'SENSITIVE'
            summary.append({
                'parameter': param_name,
                'safe_region': f'{data["safe_region_size"]}/{data["total_values"]}',
                'resolution_range': (
                    f'{100 * data["resolution_range_in_safe"][0]:.1f}%-'
                    f'{100 * data["resolution_range_in_safe"][1]:.1f}%'
                ),
                'verdict': robustness,
            })

        result = {
            'test': 'parameter_sensitivity',
            'sweeps': all_sweep_results,
            'summary': summary,
        }

        if self.verbose:
            self._print_summary(result)

        return result

    def _run_with_override(self, param_name, value, affects):
        """Run the pipeline with a single parameter overridden."""
        components = self.components

        if affects == 'solver':
            # Build new solver with overridden param
            override_map = {
                'MIN_CONFIDENCE_RATIO': 'min_confidence_ratio',
                'CHAR_NGRAM_MIN_SCORE_GAP': 'char_ngram_min_score_gap',
                'DUAL_CONTEXT_RATIO_FACTOR': 'dual_context_ratio_factor',
            }
            kwarg_name = override_map[param_name]
            solver_kwargs = build_default_solver_kwargs(
                components['pos_tagger'],
                components['pos_matrix'],
                components['pos_vocab'],
                components['char_ngram_model'],
                components['illustration_prior'],
                **{kwarg_name: value},
            )
            solver = NgramMaskSolver(
                components['trans_matrix'],
                components['trans_vocab'],
                components['latin_skel'],
                components['fuzzy_skel'],
                **solver_kwargs,
            )
            solver.set_corpus_frequencies(components['l_tokens'])
            return run_full_pipeline(
                components, solver_override=solver, verbose=False,
            )

        elif affects == 'decoder':
            # Build new decoder with overridden param
            decoder_kwargs = {
                'enable_graduated_csp': ENABLE_GRADUATED_CSP,
                'high_threshold': CSP_HIGH_CONFIDENCE_THRESHOLD,
                'medium_threshold': CSP_MEDIUM_CONFIDENCE_THRESHOLD,
                'enable_selective_function_words': ENABLE_SELECTIVE_FUNCTION_WORDS,
                'function_word_max_density': FUNCTION_WORD_MAX_DENSITY,
                'function_word_window_size': FUNCTION_WORD_WINDOW_SIZE,
            }
            decoder_override_map = {
                'CSP_HIGH_CONFIDENCE_THRESHOLD': 'high_threshold',
                'FUNCTION_WORD_MAX_DENSITY': 'function_word_max_density',
            }
            kwarg_name = decoder_override_map[param_name]
            decoder_kwargs[kwarg_name] = value

            decoder = BudgetedCSPDecoder(
                components['latin_skel'],
                components['fuzzy_skel'],
                components['l_tokens'],
                None,  # PLANT_IDS — pass None for speed, not critical for sweep
                **decoder_kwargs,
            )
            return run_full_pipeline(
                components, decoder_override=decoder, verbose=False,
            )

        elif affects == 'consistency':
            # Re-run pipeline with different consistency threshold
            return run_full_pipeline(
                components, consistency_min_agreement=value, verbose=False,
            )

    def _print_summary(self, result):
        print(f'\n{"=" * 70}')
        print('Parameter Sensitivity Summary')
        print('=' * 70)
        print()
        print(f'{"Parameter":<35} {"Safe Region":>13} '
              f'{"Resolution Range":>18} {"Verdict":>10}')
        print('-' * 80)
        for s in result['summary']:
            print(f'{s["parameter"]:<35} {s["safe_region"]:>13} '
                  f'{s["resolution_range"]:>18} {s["verdict"]:>10}')
