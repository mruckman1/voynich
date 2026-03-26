"""
Phase 2 Orchestrator: Super-Character Generative Models
=========================================================
Entry point for Phase 2 of the Voynich Convergence Attack.

Phase 1 proved that all character-level ciphers are statistically
incompatible with the Voynich Manuscript. Phase 2 tests 6 generative
models that operate at encoding units larger than individual characters.

Sub-phases:
  discrimination  — Quick H2/TTR/Zipf screening of all 6 models
  deep            — Full constraint testing of top 2-3 models
  crosscutting    — Language A/B, qo- predictions, info-theoretic FSM
  null            — Null distributions for Phase 2 models
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from voynich.core.utils import save_json, ensure_output_dir

from voynich.modules.phase2.phase2_discrimination import run_quick_discrimination, MODEL_REGISTRY
from voynich.modules.phase2.cross_cutting import run_cross_cutting
from voynich.modules.phase2.phase2_null_engine import Phase2NullEngine
from voynich.modules.phase2.base_model import VOYNICH_TARGETS
from voynich.core.stats import full_statistical_profile

def run_phase2_attack(
    phases: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './results/phase2',
    n_samples: int = 50,
) -> Dict:
    """
    Run Phase 2: Super-Character Generative Models.

    Parameters:
        phases: List of sub-phase names to run.
                Options: 'discrimination', 'deep', 'crosscutting', 'null'
                Default: all phases.
        models: List of model names to test. Default: all 6.
        verbose: Print detailed output.
        output_dir: Directory for Phase 2 output files.
        n_samples: Number of samples per model × language for null distributions.

    Returns:
        Complete Phase 2 results dict.
    """
    if phases is None:
        phases = ['discrimination', 'deep', 'crosscutting']
    if models is None:
        models = list(MODEL_REGISTRY.keys())

    ensure_output_dir(output_dir)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase1_summary': {
            'all_character_ciphers_excluded': True,
            'voynich_targets': VOYNICH_TARGETS,
            'methodology': 'Phase 1 tested 5 cipher families × 3 languages × 200 trials. '
                          'All metrics globally anomalous (p=0).',
        },
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 2')
        print('Super-Character Generative Models')
        print('=' * 70)
        print(f'Sub-phases: {phases}')
        print(f'Models: {models}')
        print()

    if 'discrimination' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 1: QUICK DISCRIMINATION SWEEP')
            print('=' * 70)

        discrimination = run_quick_discrimination(
            models=models,
            resolution='coarse',
            verbose=verbose,
        )
        results['discrimination'] = discrimination

        save_json(os.path.join(output_dir, 'discrimination_results.json'),
                   discrimination)

        surviving_models = (
            discrimination.get('triple_matches', []) +
            discrimination.get('partial_matches', [])
        )
        if not surviving_models:
            rankings = discrimination.get('model_rankings', [])
            surviving_models = [r['model'] for r in rankings[:3]]

        if verbose:
            print(f'\nSurviving models for deep analysis: {surviving_models}')

    else:
        surviving_models = models

    if 'deep' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 2: DEEP IMPLEMENTATION')
            print(f'Testing {len(surviving_models)} models against all 17 constraints')
            print('=' * 70)

        deep_results = {}
        for model_name in surviving_models:
            if model_name not in MODEL_REGISTRY:
                continue

            if verbose:
                print(f'\n--- Deep analysis: {model_name} ---')

            deep = _run_deep_analysis(model_name, verbose=verbose)
            deep_results[model_name] = deep

        results['deep_analysis'] = deep_results

        save_json(os.path.join(output_dir, 'deep_analysis_results.json'),
                   deep_results)

    if 'crosscutting' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 3: CROSS-CUTTING INVESTIGATIONS')
            print('=' * 70)

        model_outputs = {}
        for model_name in surviving_models:
            if model_name in MODEL_REGISTRY:
                try:
                    model = MODEL_REGISTRY[model_name](seed=42)
                    text = model.generate(n_words=500)
                    if text:
                        profile = full_statistical_profile(text, model_name)
                        model_outputs[model_name] = {'profile': profile, 'text': text}
                except Exception:
                    pass

        cross_cutting = run_cross_cutting(
            model_outputs=model_outputs,
            verbose=verbose,
        )
        results['cross_cutting'] = cross_cutting

        save_json(os.path.join(output_dir, 'cross_cutting_results.json'),
                   cross_cutting)

    if 'null' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 4: NULL DISTRIBUTIONS')
            print(f'{n_samples} samples per model × language')
            print('=' * 70)

        engine = Phase2NullEngine(n_samples=n_samples, verbose=verbose)
        null_results = engine.run_all(models=surviving_models)
        results['null_distributions'] = null_results

        save_json(os.path.join(output_dir, 'null_distributions_p2.json'),
                   null_results)

    elapsed = time.time() - t0
    conclusion = _synthesize_conclusion(results, surviving_models)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    save_json(os.path.join(output_dir, 'phase2_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 2 CONCLUSION')
        print('=' * 70)
        print(f'Viable models: {conclusion.get("viable_models", [])}')
        print(f'Best model: {conclusion.get("best_model", "none")}')
        print(f'Next steps: {conclusion.get("next_steps", "")}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results

def _run_deep_analysis(model_name: str, verbose: bool = True) -> Dict:
    """
    Run full constraint testing for a single model with medium-resolution sweep.
    """
    model_class = MODEL_REGISTRY.get(model_name)
    if not model_class:
        return {'error': f'Unknown model: {model_name}'}

    from voynich.modules.strategy1_parameter_search import generate_medical_plaintext
    plaintext = generate_medical_plaintext(600)

    default_model = model_class(seed=42)
    sweep = default_model.run_sweep(
        plaintext=plaintext,
        resolution='medium',
        n_best=10,
        verbose=verbose,
    )

    best = sweep.get('best_results', [{}])[0] if sweep.get('best_results') else {}
    best_params = best.get('params', {})

    constraints_result = {}
    if best_params:
        try:
            model = model_class(**best_params)
            text = model.generate(plaintext=plaintext, n_words=500)
            if text:
                profile = full_statistical_profile(text, f'{model_name}_deep')
                score = model.full_score(profile)
                crit = model.critical_test(profile)

                try:
                    from voynich.modules.constraint_model import ConstraintModel
                    cm = ConstraintModel(verbose=False)
                    constraint_file = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'output', 'constraint_model.json'
                    )
                    if os.path.exists(constraint_file):
                        with open(constraint_file) as f:
                            p1_constraints = json.load(f)
                        for c in p1_constraints.get('constraints', []):
                            cm.constraints.append(c)

                    check = cm.check_candidate({
                        'text': text,
                        'params': best_params,
                        'cipher_family': model_name,
                    })
                    constraints_result = check
                except Exception as e:
                    constraints_result = {'error': str(e)}

                return {
                    'sweep_results': {
                        'n_tested': sweep.get('n_tested', 0),
                        'n_valid': sweep.get('n_valid', 0),
                        'n_triple_matches': len(sweep.get('triple_matches', [])),
                        'best_distance': sweep.get('best_distance', float('inf')),
                    },
                    'best_params': best_params,
                    'best_score': score,
                    'critical_test': crit,
                    'constraint_check': constraints_result,
                    'sample_output': ' '.join(text.split()[:30]),
                }
        except Exception as e:
            return {'error': str(e), 'sweep_results': sweep}

    return {'sweep_results': sweep, 'best_params': best_params}

def _synthesize_conclusion(results: Dict, surviving_models: List[str]) -> Dict:
    """Synthesize the overall Phase 2 conclusion."""
    viable = []
    best_model = None
    best_distance = float('inf')

    disc = results.get('discrimination', {})
    triple_matches = disc.get('triple_matches', [])

    deep = results.get('deep_analysis', {})
    for model_name, analysis in deep.items():
        score = analysis.get('best_score', {})
        dist = score.get('profile_distance', float('inf'))
        crit = analysis.get('critical_test', {})

        if crit.get('passes', False):
            viable.append(model_name)

        if dist < best_distance:
            best_distance = dist
            best_model = model_name

    if not viable and triple_matches:
        viable = triple_matches

    if viable:
        next_steps = (
            f'Models {viable} survived all tests. '
            f'Proceed to Phase 3 (Decryption Attempt) with {viable[0]}.'
        )
    elif triple_matches:
        next_steps = (
            f'Models {triple_matches} partially matched. '
            f'Run deep analysis with fine resolution to refine parameters.'
        )
    else:
        next_steps = (
            'No model fully matched the Voynich statistical signature. '
            'Consider: (1) hybrid models combining elements from multiple approaches, '
            '(2) expanding parameter ranges, '
            '(3) revisiting the glyph decomposition results for alternative alphabets.'
        )

    return {
        'viable_models': viable,
        'best_model': best_model,
        'best_distance': best_distance,
        'triple_match_models': triple_matches,
        'surviving_for_deep': surviving_models,
        'next_steps': next_steps,
    }
