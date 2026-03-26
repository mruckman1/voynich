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
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from voynich.core.utils import save_json, ensure_output_dir

from voynich.modules.phase3.lang_b_profiler import LanguageBProfiler, LANG_B_TARGETS
from voynich.modules.phase3.two_pattern_attack import TwoPatternAttack
from voynich.modules.phase3.onset_decomposition import OnsetDecomposition
from voynich.modules.phase3.lang_b_generator import LanguageBGenerator
from voynich.modules.phase3.lang_a_reprofiling import LanguageAReprofiler, LANG_A_TARGETS
from voynich.modules.phase3.hybrid_model import HybridModel
from voynich.modules.phase2.base_model import VOYNICH_TARGETS

def run_phase3_attack(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './results/phase3',
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

    ensure_output_dir(output_dir)
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

    profiler = LanguageBProfiler()

    if 'profiling' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 1: LANGUAGE B PROFILING')
            print('=' * 70)

        profiling_results = profiler.run(verbose=verbose)
        results['profiling'] = profiling_results

        save_json(os.path.join(output_dir, 'lang_b_profile.json'),
                   profiling_results)

    if 'two_pattern' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 2: TWO-PATTERN HYPOTHESIS (HIGHEST PRIORITY)')
            print('=' * 70)

        attack1 = TwoPatternAttack(profiler=profiler)
        two_pattern_results = attack1.run(verbose=verbose)
        results['two_pattern'] = two_pattern_results

        save_json(os.path.join(output_dir, 'two_pattern_results.json'),
                   two_pattern_results)

    if 'onset' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 3: ONSET DECOMPOSITION (HIGH PRIORITY)')
            print('=' * 70)

        attack2 = OnsetDecomposition(profiler=profiler)
        onset_results = attack2.run(verbose=verbose)
        results['onset'] = onset_results

        save_json(os.path.join(output_dir, 'onset_decomposition_results.json'),
                   onset_results)

    if 'generator' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 4: LANGUAGE B MARKOV GENERATOR (HIGH PRIORITY)')
            print('=' * 70)

        attack3 = LanguageBGenerator(profiler=profiler)
        generator_results = attack3.run(verbose=verbose)
        results['generator'] = generator_results

        save_json(os.path.join(output_dir, 'lang_b_generator_results.json'),
                   generator_results)

    if 'reprofiling' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 5: LANGUAGE A RE-PROFILING')
            print('=' * 70)

        reprofiler = LanguageAReprofiler(n_samples=50, verbose=verbose)
        reprofiling_results = reprofiler.run(verbose=verbose)
        results['reprofiling'] = reprofiling_results

        save_json(os.path.join(output_dir, 'lang_a_reprofiling_results.json'),
                   reprofiling_results)

    if 'hybrid' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 6: HYBRID MODEL VERIFICATION')
            print('=' * 70)

        hybrid = HybridModel()
        hybrid_results = hybrid.run(verbose=verbose)
        results['hybrid'] = hybrid_results

        save_json(os.path.join(output_dir, 'hybrid_model_results.json'),
                   hybrid_results)

    elapsed = time.time() - t0
    conclusion = _synthesize_phase3(results)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    save_json(os.path.join(output_dir, 'phase3_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 3 CONCLUSION')
        print('=' * 70)
        for key, value in conclusion.items():
            print(f'  {key}: {value}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results

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

    tp = results.get('two_pattern', {})
    synthesis = tp.get('synthesis', {})
    conclusion['two_pattern_verdict'] = synthesis.get('strongest_hypothesis', 'unknown')

    onset = results.get('onset', {})
    onset_synth = onset.get('synthesis', {})
    conclusion['onset_encoding'] = (
        f'{onset_synth.get("n_unique_onsets", "?")} onsets, '
        f'{onset_synth.get("onset_entropy_bits", 0):.2f} bits/word, '
        f'planet mapping: {"yes" if onset_synth.get("planet_consistent") else "no"}'
    )

    reprof = results.get('reprofiling', {})
    reprof_synth = reprof.get('synthesis', {})
    conclusion['lang_a_mechanism'] = reprof_synth.get('conclusion', 'unknown')

    hybrid = results.get('hybrid', {})
    hybrid_synth = hybrid.get('synthesis', {})
    conclusion['hybrid_viable'] = hybrid_synth.get('hybrid_viable', False)
    conclusion['hybrid_conclusion'] = hybrid_synth.get('conclusion', 'unknown')

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
