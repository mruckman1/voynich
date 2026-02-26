"""
Phase 3 Orchestrator: Language B First
========================================
Entry point for Phase 3 of the Voynich Convergence Attack.

Phase 2 proved that all 6 super-character models are statistically
incompatible with the Voynich Manuscript. However, the Language A/B
split revealed that Language B has extreme regularity (13 words,
H2=0.74) making it the ideal attack entry point.

Sub-phases:
  profiling    — Language B statistical profiling and word family classification
  two_pattern  — Test edy/aiin semantic correlation hypotheses
  onset        — Onset decomposition and 4x5 grid analysis
  generator    — Markov chain synthetic generator + matrix analysis
  reprofiling  — Language A null framework re-test
  hybrid       — Hybrid model (cipher-A + notation-B) verification

Usage:
  python convergence_attack_p3.py                  # Run all sub-phases
  python convergence_attack_p3.py --profiling      # Language B profiling only
  python convergence_attack_p3.py --two-pattern    # Two-pattern attack only
  python convergence_attack_p3.py --onset          # Onset decomposition only
  python convergence_attack_p3.py --generator      # Markov generator only
  python convergence_attack_p3.py --reprofiling    # Language A re-profiling only
  python convergence_attack_p3.py --hybrid         # Hybrid model only
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.phase3.lang_b_profiler import LanguageBProfiler, LANG_B_TARGETS
from modules.phase3.two_pattern_attack import TwoPatternAttack
from modules.phase3.onset_decomposition import OnsetDecomposition
from modules.phase3.lang_b_generator import LanguageBGenerator
from modules.phase3.lang_a_reprofiling import LanguageAReprofiler, LANG_A_TARGETS
from modules.phase3.hybrid_model import HybridModel
from modules.phase2.base_model import VOYNICH_TARGETS


# ============================================================================
# PHASE 3 ORCHESTRATOR
# ============================================================================

def run_phase3_attack(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/phase3',
) -> Dict:
    """
    Run Phase 3: Language B First.

    Parameters:
        phases: Sub-phases to run. Default: all.
        verbose: Print detailed output.
        output_dir: Directory for Phase 3 output files.

    Returns:
        Complete Phase 3 results dict.
    """
    if phases is None:
        phases = ['profiling', 'two_pattern', 'onset', 'generator',
                  'reprofiling', 'hybrid']

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase2_summary': {
            'all_super_character_models_excluded': True,
            'language_ab_split_discovered': True,
            'lang_b_vocabulary_size': LANG_B_TARGETS['vocabulary_size'],
            'lang_b_H2': LANG_B_TARGETS['H2'],
            'lang_a_H2': LANG_A_TARGETS['H2'],
            'combined_voynich_targets': VOYNICH_TARGETS,
        },
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 3')
        print('Language B First')
        print('=' * 70)
        print(f'Sub-phases: {phases}')
        print(f'Language B: {LANG_B_TARGETS["vocabulary_size"]} words, '
              f'H2={LANG_B_TARGETS["H2"]}')
        print(f'Language A: {LANG_A_TARGETS["vocabulary_size"]} words, '
              f'H2={LANG_A_TARGETS["H2"]}')
        print()

    # Shared profiler instance (avoids redundant corpus extraction)
    profiler = LanguageBProfiler()

    # ---- Sub-phase 1: Language B Profiling ----
    if 'profiling' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 1: LANGUAGE B PROFILING')
            print('=' * 70)

        profiling_results = profiler.run(verbose=verbose)
        results['profiling'] = profiling_results

        _save_json(os.path.join(output_dir, 'lang_b_profile.json'),
                   profiling_results)

    # ---- Sub-phase 2: Two-Pattern Attack (HIGHEST priority) ----
    if 'two_pattern' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 2: TWO-PATTERN HYPOTHESIS (HIGHEST PRIORITY)')
            print('=' * 70)

        attack1 = TwoPatternAttack(profiler=profiler)
        two_pattern_results = attack1.run(verbose=verbose)
        results['two_pattern'] = two_pattern_results

        _save_json(os.path.join(output_dir, 'two_pattern_results.json'),
                   two_pattern_results)

    # ---- Sub-phase 3: Onset Decomposition (HIGH priority) ----
    if 'onset' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 3: ONSET DECOMPOSITION (HIGH PRIORITY)')
            print('=' * 70)

        attack2 = OnsetDecomposition(profiler=profiler)
        onset_results = attack2.run(verbose=verbose)
        results['onset'] = onset_results

        _save_json(os.path.join(output_dir, 'onset_decomposition_results.json'),
                   onset_results)

    # ---- Sub-phase 4: Markov Generator (HIGH priority) ----
    if 'generator' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 4: LANGUAGE B MARKOV GENERATOR (HIGH PRIORITY)')
            print('=' * 70)

        attack3 = LanguageBGenerator(profiler=profiler)
        generator_results = attack3.run(verbose=verbose)
        results['generator'] = generator_results

        _save_json(os.path.join(output_dir, 'lang_b_generator_results.json'),
                   generator_results)

    # ---- Sub-phase 5: Language A Re-profiling ----
    if 'reprofiling' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 5: LANGUAGE A RE-PROFILING')
            print('=' * 70)

        reprofiler = LanguageAReprofiler(n_samples=50, verbose=verbose)
        reprofiling_results = reprofiler.run(verbose=verbose)
        results['reprofiling'] = reprofiling_results

        _save_json(os.path.join(output_dir, 'lang_a_reprofiling_results.json'),
                   reprofiling_results)

    # ---- Sub-phase 6: Hybrid Model ----
    if 'hybrid' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 6: HYBRID MODEL VERIFICATION')
            print('=' * 70)

        hybrid = HybridModel()
        hybrid_results = hybrid.run(verbose=verbose)
        results['hybrid'] = hybrid_results

        _save_json(os.path.join(output_dir, 'hybrid_model_results.json'),
                   hybrid_results)

    # ---- Synthesis & Conclusion ----
    elapsed = time.time() - t0
    conclusion = _synthesize_phase3(results)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    _save_json(os.path.join(output_dir, 'phase3_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 3 CONCLUSION')
        print('=' * 70)
        for key, value in conclusion.items():
            print(f'  {key}: {value}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results


# ============================================================================
# SYNTHESIS
# ============================================================================

def _synthesize_phase3(results: Dict) -> Dict:
    """
    Synthesize Phase 3 findings into a conclusion.

    Key questions:
    1. What does Language B encode? (notation vs cipher)
    2. What do the two word families mean?
    3. What do the onsets encode?
    4. Is Language A a different mechanism?
    5. Does the hybrid model explain all anomalies?
    """
    conclusion = {}

    # 1. Language B mechanism
    gen = results.get('generator', {})
    matrix = gen.get('matrix_analysis', {})
    total_info = gen.get('total_information', {})

    if total_info.get('markov_bits', 0) < 300:
        conclusion['lang_b_mechanism'] = 'notation_system'
        conclusion['lang_b_info_content'] = (
            f'{total_info.get("markov_bits", 0):.0f} bits — too little for cipher, '
            f'consistent with notation/tabulation system'
        )
    else:
        conclusion['lang_b_mechanism'] = 'uncertain'
        conclusion['lang_b_info_content'] = (
            f'{total_info.get("markov_bits", 0):.0f} bits'
        )

    # 2. Two-pattern verdict
    tp = results.get('two_pattern', {})
    synthesis = tp.get('synthesis', {})
    conclusion['two_pattern_verdict'] = synthesis.get('strongest_hypothesis', 'unknown')

    # 3. Onset encoding
    onset = results.get('onset', {})
    onset_synth = onset.get('synthesis', {})
    conclusion['onset_encoding'] = (
        f'{onset_synth.get("n_unique_onsets", "?")} onsets, '
        f'{onset_synth.get("onset_entropy_bits", 0):.2f} bits/word, '
        f'planet mapping: {"yes" if onset_synth.get("planet_consistent") else "no"}'
    )

    # 4. Language A
    reprof = results.get('reprofiling', {})
    reprof_synth = reprof.get('synthesis', {})
    conclusion['lang_a_mechanism'] = reprof_synth.get('conclusion', 'unknown')

    # 5. Hybrid model
    hybrid = results.get('hybrid', {})
    hybrid_synth = hybrid.get('synthesis', {})
    conclusion['hybrid_viable'] = hybrid_synth.get('hybrid_viable', False)
    conclusion['hybrid_conclusion'] = hybrid_synth.get('conclusion', 'unknown')

    # Overall assessment
    if hybrid_synth.get('hybrid_viable') and total_info.get('markov_bits', 0) < 300:
        conclusion['overall'] = (
            'CONVERGENT — Language B is a notation system encoding '
            f'~{total_info.get("markov_bits", 0):.0f} bits. '
            f'Language A is likely a separate cipher. '
            f'The hybrid model explains all Phase 2 anomalies.'
        )
    else:
        conclusion['overall'] = (
            'PARTIAL — Some Phase 3 tests converge but the full picture '
            'remains incomplete. See individual sub-phase results.'
        )

    return conclusion


# ============================================================================
# UTILITIES
# ============================================================================

def _save_json(filepath: str, data: Dict):
    """Save results to JSON, handling non-serializable types."""
    def default_handler(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, float) and (obj != obj):  # NaN check
            return None
        if obj == float('inf') or obj == float('-inf'):
            return str(obj)
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=default_handler)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    phases_to_run = []

    if '--profiling' in sys.argv or '-p' in sys.argv:
        phases_to_run.append('profiling')
    if '--two-pattern' in sys.argv or '-t' in sys.argv:
        phases_to_run.append('two_pattern')
    if '--onset' in sys.argv or '-o' in sys.argv:
        phases_to_run.append('onset')
    if '--generator' in sys.argv or '-g' in sys.argv:
        phases_to_run.append('generator')
    if '--reprofiling' in sys.argv or '-r' in sys.argv:
        phases_to_run.append('reprofiling')
    if '--hybrid' in sys.argv or '-h' in sys.argv:
        phases_to_run.append('hybrid')

    if not phases_to_run:
        phases_to_run = None  # Run all

    verbose = '--quiet' not in sys.argv and '-q' not in sys.argv

    results = run_phase3_attack(
        phases=phases_to_run,
        verbose=verbose,
    )

    print(f'\nPhase 3 complete. Results in ./output/phase3/')
