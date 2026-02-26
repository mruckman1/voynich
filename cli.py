#!/usr/bin/env python
"""
Unified CLI entry point for the Voynich Convergence Attack.

Usage:
  uv run cli.py                          # Run master convergence attack (strategies 1-5)
  uv run cli.py --phased                 # Run phased attack (phases 1-4)
  uv run cli.py --phase 2                # Run phase 2 only
  uv run cli.py --phase 7                # Run phase 7 only
  uv run cli.py --phase 5 --quick        # Run phase 5 in quick mode
  uv run cli.py --all                    # Run all 12 phases sequentially
  uv run cli.py --output-dir ./results   # Custom output directory
  uv run cli.py --list                   # List all available phases

Sub-phase flags are forwarded to the phase orchestrator:
  uv run cli.py --phase 2 --discrimination
  uv run cli.py --phase 6 --path-a --path-b
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_subphase_flags(phase_num, remaining_args):
    """Parse sub-phase flags from remaining CLI args for a given phase.

    Returns kwargs dict to pass to the phase runner.
    """
    kwargs = {}

    if phase_num == 2:
        phases = []
        if '--discrimination' in remaining_args or '-d' in remaining_args:
            phases.append('discrimination')
        if '--deep' in remaining_args:
            phases.append('deep')
        if '--crosscutting' in remaining_args or '-x' in remaining_args:
            phases.append('crosscutting')
        if '--null' in remaining_args or '-n' in remaining_args:
            phases.append('null')
        if phases:
            kwargs['phases'] = phases

    elif phase_num == 3:
        phases = []
        if '--profiling' in remaining_args or '-p' in remaining_args:
            phases.append('profiling')
        if '--two-pattern' in remaining_args or '-t' in remaining_args:
            phases.append('two_pattern')
        if '--onset' in remaining_args or '-o' in remaining_args:
            phases.append('onset')
        if '--generator' in remaining_args or '-g' in remaining_args:
            phases.append('generator')
        if '--reprofiling' in remaining_args or '-r' in remaining_args:
            phases.append('reprofiling')
        if '--hybrid' in remaining_args or '-h' in remaining_args:
            phases.append('hybrid')
        if phases:
            kwargs['phases'] = phases

    elif phase_num == 4:
        phases = []
        if '--extraction' in remaining_args or '-e' in remaining_args:
            phases.append('extraction')
        if '--latin-corpus' in remaining_args or '-l' in remaining_args:
            phases.append('latin_corpus')
        if '--model-a1' in remaining_args or '-1' in remaining_args:
            phases.append('model_a1')
        if '--model-a2' in remaining_args or '-2' in remaining_args:
            phases.append('model_a2')
        if '--model-a3' in remaining_args or '-3' in remaining_args:
            phases.append('model_a3')
        if '--botanical' in remaining_args or '-b' in remaining_args:
            phases.append('botanical')
        if '--saa' in remaining_args or '-s' in remaining_args:
            phases.append('saa')
        if '--gradient' in remaining_args or '-g' in remaining_args:
            phases.append('gradient')
        if '--multi-lang' in remaining_args or '-m' in remaining_args:
            phases.append('multi_lang')
        if phases:
            kwargs['phases'] = phases

    elif phase_num == 5:
        phases = []
        if '--tier-split' in remaining_args or '-t' in remaining_args:
            phases.append('tier_split')
        if '--latin-corpus' in remaining_args or '-l' in remaining_args:
            phases.append('latin_corpus')
        if '--rank-cribs' in remaining_args or '-r' in remaining_args:
            phases.append('rank_cribs')
        if '--nmf' in remaining_args or '-n' in remaining_args:
            phases.append('nmf_scaffold')
        if '--attack-a' in remaining_args or '-a' in remaining_args:
            phases.append('attack_a')
        if '--attack-b' in remaining_args or '-b' in remaining_args:
            phases.append('attack_b')
        if '--cross-validate' in remaining_args or '-c' in remaining_args:
            phases.append('cross_validate')
        if phases:
            kwargs['phases'] = phases
        if '--quick' in remaining_args:
            kwargs['saa_iterations'] = 1000
            kwargs['latin_corpus_size'] = 10000

    elif phase_num == 6:
        paths = []
        if '--path-a' in remaining_args or '-a' in remaining_args:
            paths.append('path_a')
        if '--path-b' in remaining_args or '-b' in remaining_args:
            paths.append('path_b')
        if '--path-c' in remaining_args or '-c' in remaining_args:
            paths.append('path_c')
        if paths:
            kwargs['paths'] = paths
        if '--quick' in remaining_args:
            kwargs['saa_iterations'] = 1000
            kwargs['latin_corpus_size'] = 10000

    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description='Voynich Manuscript Convergence Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--phase', type=int, metavar='N',
                        help='Run a specific phase (2-12)')
    parser.add_argument('--phased', action='store_true',
                        help='Run the full phased attack (phases 1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 12 phases sequentially (1-12)')
    parser.add_argument('--output-dir', '-o', metavar='DIR',
                        help='Root output directory (default: ./output)')
    parser.add_argument('--list', action='store_true',
                        help='List all available phases')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    # Parse known args; pass remainder to phase orchestrators
    args, remaining = parser.parse_known_args()

    if args.list:
        from orchestrators import list_phases
        print('Available phases:')
        for num, path in list_phases().items():
            print(f'  Phase {num:2d}: {path}')
        return

    verbose = not args.quiet
    output_root = args.output_dir  # None means use each phase's default
    report_dir = output_root or './output'

    def _phase_dir(phase_num):
        """Derive per-phase output dir from --output-dir, or None for default."""
        if output_root is None:
            return {}
        if phase_num == 1:
            return {'output_dir': output_root}
        return {'output_dir': os.path.join(output_root, f'phase{phase_num}')}

    phase_results = {}

    if args.all:
        from convergence_attack import run_convergence_attack
        from orchestrators import get_phase_runner
        print('=== Phase 1: Convergence Attack (5 strategies) ===')
        phase_results[1] = run_convergence_attack(verbose=verbose, **_phase_dir(1))
        for phase_num in range(2, 13):
            print(f'\n=== Phase {phase_num} ===')
            phase_results[phase_num] = get_phase_runner(phase_num)(
                verbose=verbose, **_phase_dir(phase_num)
            )

    elif args.phase:
        from orchestrators import get_phase_runner
        runner = get_phase_runner(args.phase)

        # Build kwargs from sub-phase flags
        kwargs = {'verbose': verbose}
        kwargs.update(_phase_dir(args.phase))
        kwargs.update(_parse_subphase_flags(args.phase, remaining))

        phase_results[args.phase] = runner(**kwargs)

    elif args.phased:
        from convergence_attack import run_phased_attack
        phase_results[1] = run_phased_attack(verbose=verbose, **_phase_dir(1))

    else:
        from convergence_attack import run_convergence_attack
        phase_results[1] = run_convergence_attack(verbose=verbose, **_phase_dir(1))

    # Write combined report if any phases produced results
    if phase_results:
        from orchestrators._utils import build_combined_report
        path = build_combined_report(phase_results, report_dir)
        if verbose:
            print(f'\nCombined report saved to {path}')


if __name__ == '__main__':
    main()
