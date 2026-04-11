"""Voynich CLI — flat command dispatch with lazy imports.

Usage:
    voynich phase1                     # Run Phase 1 convergence attack
    voynich phase12                    # Run Phase 12 reconstruction
    voynich adversarial                # Run Phase 12.5 adversarial suite
    voynich robustness                 # Run all robustness tests
    voynich all                        # Run all phases sequentially
    voynich list                       # List available commands

Sub-phase flags (passed via sys.argv[2:]):
    voynich phase2 --discrimination
    voynich phase6 --path-a --path-b
    voynich adversarial --unicity
    voynich phase13 --correlation --folios 20
    voynich robustness skeleton
"""
import os
import sys
import time


# ── Sub-phase flag parsing ──────────────────────────────────────────

def _parse_subphase_flags(phase_key, remaining_args):
    """Parse sub-phase flags from remaining CLI args for a given phase."""
    kwargs = {}

    if phase_key == '12.5':
        phases = []
        flag_map = {
            '--unicity': 'unicity', '--domain-swap': 'domain_swap',
            '--polyglot': 'polyglot', '--eva-collapse': 'eva_collapse',
            '--ablation': 'ablation', '--compositionality': 'compositionality',
            '--dictionary-diagnostic': 'dictionary_diagnostic',
        }
        for flag, name in flag_map.items():
            if flag in remaining_args:
                phases.append(name)
        if phases:
            kwargs['phases'] = phases
        if '--folios' in remaining_args:
            idx = remaining_args.index('--folios')
            if idx + 1 < len(remaining_args):
                try: kwargs['folio_limit'] = int(remaining_args[idx + 1])
                except ValueError: pass
        return kwargs

    phase_num = int(phase_key) if isinstance(phase_key, (int, float)) else None
    try: phase_num = int(phase_key)
    except (ValueError, TypeError): pass

    if phase_num == 2:
        phases = []
        for flag, name in [('--discrimination', 'discrimination'), ('-d', 'discrimination'),
                           ('--deep', 'deep'), ('--crosscutting', 'crosscutting'),
                           ('-x', 'crosscutting'), ('--null', 'null'), ('-n', 'null')]:
            if flag in remaining_args and name not in phases:
                phases.append(name)
        if phases: kwargs['phases'] = phases

    elif phase_num == 3:
        phases = []
        for flag, name in [('--profiling', 'profiling'), ('-p', 'profiling'),
                           ('--two-pattern', 'two_pattern'), ('-t', 'two_pattern'),
                           ('--onset', 'onset'), ('-o', 'onset'),
                           ('--generator', 'generator'), ('-g', 'generator'),
                           ('--reprofiling', 'reprofiling'), ('-r', 'reprofiling'),
                           ('--hybrid', 'hybrid'), ('-h', 'hybrid')]:
            if flag in remaining_args and name not in phases:
                phases.append(name)
        if phases: kwargs['phases'] = phases

    elif phase_num == 4:
        phases = []
        for flag, name in [('--extraction', 'extraction'), ('-e', 'extraction'),
                           ('--latin-corpus', 'latin_corpus'), ('-l', 'latin_corpus'),
                           ('--model-a1', 'model_a1'), ('-1', 'model_a1'),
                           ('--model-a2', 'model_a2'), ('-2', 'model_a2'),
                           ('--model-a3', 'model_a3'), ('-3', 'model_a3'),
                           ('--botanical', 'botanical'), ('-b', 'botanical'),
                           ('--saa', 'saa'), ('-s', 'saa'),
                           ('--gradient', 'gradient'), ('-g', 'gradient'),
                           ('--multi-lang', 'multi_lang'), ('-m', 'multi_lang')]:
            if flag in remaining_args and name not in phases:
                phases.append(name)
        if phases: kwargs['phases'] = phases

    elif phase_num == 5:
        phases = []
        for flag, name in [('--tier-split', 'tier_split'), ('-t', 'tier_split'),
                           ('--latin-corpus', 'latin_corpus'), ('-l', 'latin_corpus'),
                           ('--rank-cribs', 'rank_cribs'), ('-r', 'rank_cribs'),
                           ('--nmf', 'nmf_scaffold'), ('-n', 'nmf_scaffold'),
                           ('--attack-a', 'attack_a'), ('-a', 'attack_a'),
                           ('--attack-b', 'attack_b'), ('-b', 'attack_b'),
                           ('--cross-validate', 'cross_validate'), ('-c', 'cross_validate')]:
            if flag in remaining_args and name not in phases:
                phases.append(name)
        if phases: kwargs['phases'] = phases
        if '--quick' in remaining_args:
            kwargs['saa_iterations'] = 1000
            kwargs['latin_corpus_size'] = 10000

    elif phase_num == 6:
        paths = []
        for flag, name in [('--path-a', 'path_a'), ('-a', 'path_a'),
                           ('--path-b', 'path_b'), ('-b', 'path_b'),
                           ('--path-c', 'path_c'), ('-c', 'path_c')]:
            if flag in remaining_args and name not in paths:
                paths.append(name)
        if paths: kwargs['paths'] = paths
        if '--quick' in remaining_args:
            kwargs['saa_iterations'] = 1000
            kwargs['latin_corpus_size'] = 10000

    elif phase_num == 13:
        phases = []
        for flag, name in [('--decode', 'decode'), ('--correlation', 'correlation')]:
            if flag in remaining_args:
                phases.append(name)
        if phases: kwargs['phases'] = phases
        if '--folios' in remaining_args:
            idx = remaining_args.index('--folios')
            if idx + 1 < len(remaining_args):
                try: kwargs['folio_limit'] = int(remaining_args[idx + 1])
                except ValueError: pass

    elif phase_num == 14:
        phases = []
        for flag, name in [('--vocabulary', 'vocabulary'), ('--concentration', 'concentration'),
                           ('--templates', 'templates'), ('--collocations', 'collocations'),
                           ('--significance', 'significance'), ('--langb', 'langb')]:
            if flag in remaining_args:
                phases.append(name)
        if phases: kwargs['phases'] = phases
        if '--folios' in remaining_args:
            idx = remaining_args.index('--folios')
            if idx + 1 < len(remaining_args):
                try: kwargs['folio_limit'] = int(remaining_args[idx + 1])
                except ValueError: pass
        if '--trials' in remaining_args:
            idx = remaining_args.index('--trials')
            if idx + 1 < len(remaining_args):
                try: kwargs['n_trials'] = int(remaining_args[idx + 1])
                except ValueError: pass

    return kwargs


# ── Output directory helpers ────────────────────────────────────────

def _get_output_dir():
    """Get output dir from --output-dir flag or default."""
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return './results'


def _phase_dir(phase_key):
    """Derive per-phase output dir."""
    root = _get_output_dir()
    if phase_key == 1:
        return {'output_dir': root}
    dir_suffix = str(phase_key).replace('.', '_')
    return {'output_dir': os.path.join(root, f'phase{dir_suffix}')}


def _verbose():
    return '--quiet' not in sys.argv and '-q' not in sys.argv


# ── Command functions ───────────────────────────────────────────────

def cmd_phase1():
    from voynich.phases.phase1 import run_convergence_attack
    t0 = time.time()
    result = run_convergence_attack(verbose=_verbose(), **_phase_dir(1))
    print(f"\nPhase 1 completed in {time.time() - t0:.1f}s")
    return {1: result}


def cmd_phase2():
    from voynich.phases.phase2 import run_phase2_attack
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(2))
    kwargs.update(_parse_subphase_flags(2, sys.argv[2:]))
    result = run_phase2_attack(**kwargs)
    print(f"\nPhase 2 completed in {time.time() - t0:.1f}s")
    return {2: result}


def cmd_phase3():
    from voynich.phases.phase3 import run_phase3_attack
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(3))
    kwargs.update(_parse_subphase_flags(3, sys.argv[2:]))
    result = run_phase3_attack(**kwargs)
    print(f"\nPhase 3 completed in {time.time() - t0:.1f}s")
    return {3: result}


def cmd_phase4():
    from voynich.phases.phase4 import run_phase4_attack
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(4))
    kwargs.update(_parse_subphase_flags(4, sys.argv[2:]))
    result = run_phase4_attack(**kwargs)
    print(f"\nPhase 4 completed in {time.time() - t0:.1f}s")
    return {4: result}


def cmd_phase5():
    from voynich.phases.phase5 import run_phase5_attack
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(5))
    kwargs.update(_parse_subphase_flags(5, sys.argv[2:]))
    result = run_phase5_attack(**kwargs)
    print(f"\nPhase 5 completed in {time.time() - t0:.1f}s")
    return {5: result}


def cmd_phase6():
    from voynich.phases.phase6 import run_phase6_attack
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(6))
    kwargs.update(_parse_subphase_flags(6, sys.argv[2:]))
    result = run_phase6_attack(**kwargs)
    print(f"\nPhase 6 completed in {time.time() - t0:.1f}s")
    return {6: result}


def cmd_phase7():
    from voynich.phases.phase7 import run_phase7_attack
    t0 = time.time()
    result = run_phase7_attack(verbose=_verbose(), **_phase_dir(7))
    print(f"\nPhase 7 completed in {time.time() - t0:.1f}s")
    return {7: result}


def cmd_phase8():
    from voynich.phases.phase8 import run_phase8_translation
    t0 = time.time()
    result = run_phase8_translation(verbose=_verbose(), **_phase_dir(8))
    print(f"\nPhase 8 completed in {time.time() - t0:.1f}s")
    return {8: result}


def cmd_phase9():
    from voynich.phases.phase9 import run_phase9_attack
    t0 = time.time()
    result = run_phase9_attack(verbose=_verbose(), **_phase_dir(9))
    print(f"\nPhase 9 completed in {time.time() - t0:.1f}s")
    return {9: result}


def cmd_phase10():
    from voynich.phases.phase10 import run_phase10_final_translation
    t0 = time.time()
    result = run_phase10_final_translation(verbose=_verbose(), **_phase_dir(10))
    print(f"\nPhase 10 completed in {time.time() - t0:.1f}s")
    return {10: result}


def cmd_phase11():
    from voynich.phases.phase11 import run_phase11_csp_translation
    t0 = time.time()
    result = run_phase11_csp_translation(verbose=_verbose(), **_phase_dir(11))
    print(f"\nPhase 11 completed in {time.time() - t0:.1f}s")
    return {11: result}


def cmd_phase12():
    from voynich.phases.phase12 import run_phase12_reconstruction
    t0 = time.time()
    result = run_phase12_reconstruction(verbose=_verbose(), **_phase_dir(12))
    print(f"\nPhase 12 completed in {time.time() - t0:.1f}s")
    return {12: result}


def cmd_adversarial():
    from voynich.phases.phase12_5 import run_phase12_5_adversarial
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir('12.5'))
    kwargs.update(_parse_subphase_flags('12.5', sys.argv[2:]))
    result = run_phase12_5_adversarial(**kwargs)
    print(f"\nPhase 12.5 completed in {time.time() - t0:.1f}s")
    return {'12.5': result}


def cmd_phase13():
    from voynich.phases.phase13 import run_phase13_synthesis
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(13))
    kwargs.update(_parse_subphase_flags(13, sys.argv[2:]))
    result = run_phase13_synthesis(**kwargs)
    print(f"\nPhase 13 completed in {time.time() - t0:.1f}s")
    return {13: result}


def cmd_phase14():
    from voynich.phases.phase14 import run_phase14
    t0 = time.time()
    kwargs = {'verbose': _verbose()}
    kwargs.update(_phase_dir(14))
    kwargs.update(_parse_subphase_flags(14, sys.argv[2:]))
    result = run_phase14(**kwargs)
    print(f"\nPhase 14 completed in {time.time() - t0:.1f}s")
    return {14: result}


def cmd_robustness():
    from voynich.phases.robustness import run_robustness_tests
    t0 = time.time()
    report_dir = _get_output_dir()
    tests = None
    remaining = sys.argv[2:]
    valid_tests = ['skeleton', 'reversed', 'consistency', 'sensitivity',
                   'bootstrap', 'bidirectional', 'baselines', 'ablation',
                   'grille', 'loo', 'discriminant', 'selectivity',
                   'selective_matching', 'tier1', 'tier2']
    for arg in remaining:
        if arg in valid_tests:
            tests = [arg]
            break
    result = run_robustness_tests(
        tests=tests, verbose=_verbose(),
        output_dir=os.path.join(report_dir, 'robustness'),
    )
    print(f"\nRobustness tests completed in {time.time() - t0:.1f}s")
    return {'robustness': result}


def cmd_all():
    from voynich.core.utils import build_combined_report
    phase_results = {}
    ordered = [
        ('phase1', cmd_phase1), ('phase2', cmd_phase2), ('phase3', cmd_phase3),
        ('phase4', cmd_phase4), ('phase5', cmd_phase5), ('phase6', cmd_phase6),
        ('phase7', cmd_phase7), ('phase8', cmd_phase8), ('phase9', cmd_phase9),
        ('phase10', cmd_phase10), ('phase11', cmd_phase11), ('phase12', cmd_phase12),
        ('adversarial', cmd_adversarial), ('phase13', cmd_phase13),
        ('phase14', cmd_phase14), ('robustness', cmd_robustness),
    ]
    for name, func in ordered:
        print(f'\n=== {name} ===')
        phase_results.update(func())
    report_dir = _get_output_dir()
    path = build_combined_report(phase_results, report_dir)
    print(f'\nCombined report saved to {path}')
    return phase_results


def cmd_list():
    from voynich.phases import list_phases
    print('Available commands:')
    for name in sorted(commands):
        print(f'  {name}')
    print('\nPhase registry:')
    for num, path in list_phases().items():
        label = f'{num:>5}' if isinstance(num, int) else f'{num:>5s}'
        print(f'  Phase {label}: {path}')
    return {}


# ── Command dispatch table ──────────────────────────────────────────

commands = {
    'phase1': cmd_phase1,
    'phase2': cmd_phase2,
    'phase3': cmd_phase3,
    'phase4': cmd_phase4,
    'phase5': cmd_phase5,
    'phase6': cmd_phase6,
    'phase7': cmd_phase7,
    'phase8': cmd_phase8,
    'phase9': cmd_phase9,
    'phase10': cmd_phase10,
    'phase11': cmd_phase11,
    'phase12': cmd_phase12,
    'adversarial': cmd_adversarial,
    'phase13': cmd_phase13,
    'phase14': cmd_phase14,
    'robustness': cmd_robustness,
    'all': cmd_all,
    'list': cmd_list,
}


# ── Entry point ─────────────────────────────────────────────────────

def main():
    # Guarantee deterministic hashing
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: voynich <command> [flags]")
        print(f"\nCommands: {', '.join(sorted(commands))}")
        print("\nExamples:")
        print("  voynich phase12              # Run Phase 12 reconstruction")
        print("  voynich adversarial          # Run adversarial defense suite")
        print("  voynich robustness           # Run all robustness tests")
        print("  voynich all                  # Run everything")
        print("  voynich phase2 --deep        # Run Phase 2 deep analysis only")
        print("  voynich phase13 --correlation --folios 20")
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    commands[sys.argv[1]]()


if __name__ == '__main__':
    main()
