#!/usr/bin/env python
"""
Unified CLI entry point for the Voynich Convergence Attack.

Usage:
  uv run cli.py                          # Run master convergence attack (strategies 1-5)
  uv run cli.py --phased                 # Run phased attack (phases 1-4)
  uv run cli.py --phase 2                # Run phase 2 only
  uv run cli.py --phase 7                # Run phase 7 only
  uv run cli.py --phase 5 --quick        # Run phase 5 in quick mode
  uv run cli.py --phase 12.5             # Run adversarial defense suite
  uv run cli.py --all                    # Run all 13 phases sequentially
  uv run cli.py --output-dir ./results   # Custom output directory
  uv run cli.py --list                   # List all available phases

Phase 12.5 sub-phase flags:
  uv run cli.py --phase 12.5 --unicity       # Unicity distance test only
  uv run cli.py --phase 12.5 --domain-swap   # Domain swap test only
  uv run cli.py --phase 12.5 --polyglot      # Polyglot dictionary test only
  uv run cli.py --phase 12.5 --eva-collapse  # EVA collapse test only
  uv run cli.py --phase 12.5 --ablation            # Ablation study only
  uv run cli.py --phase 12.5 --compositionality    # Compositionality proof only
  uv run cli.py --phase 12.5 --dictionary-diagnostic  # Dictionary coverage audit

Phase 13 sub-phase flags:
  uv run cli.py --phase 13 --html           # HTML viewer only
  uv run cli.py --phase 13 --gloss          # English glosser only
  uv run cli.py --phase 13 --hitl           # Interactive HITL console
  uv run cli.py --phase 13 --whitepaper     # Whitepaper only
  uv run cli.py --phase 13 --correlation    # Illustration-text correlation
  uv run cli.py --phase 13 --folios 20      # Limit decode to 20 folios

Sub-phase flags are forwarded to the phase orchestrator:
  uv run cli.py --phase 2 --discrimination
  uv run cli.py --phase 6 --path-a --path-b
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _parse_subphase_flags(phase_key, remaining_args):
    """Parse sub-phase flags from remaining CLI args for a given phase.

    Args:
        phase_key: Phase number (int) or key (str, e.g. '12.5')
        remaining_args: Unparsed CLI arguments

    Returns kwargs dict to pass to the phase runner.
    """
    kwargs = {}

    if phase_key == '12.5':
        phases = []
        if '--unicity' in remaining_args:
            phases.append('unicity')
        if '--domain-swap' in remaining_args:
            phases.append('domain_swap')
        if '--polyglot' in remaining_args:
            phases.append('polyglot')
        if '--eva-collapse' in remaining_args:
            phases.append('eva_collapse')
        if '--ablation' in remaining_args:
            phases.append('ablation')
        if '--compositionality' in remaining_args:
            phases.append('compositionality')
        if '--dictionary-diagnostic' in remaining_args:
            phases.append('dictionary_diagnostic')
        if phases:
            kwargs['phases'] = phases
        if '--folios' in remaining_args:
            idx = remaining_args.index('--folios')
            if idx + 1 < len(remaining_args):
                try:
                    kwargs['folio_limit'] = int(remaining_args[idx + 1])
                except ValueError:
                    pass
        return kwargs

    phase_num = int(phase_key) if isinstance(phase_key, str) else phase_key

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

    elif phase_num == 13:
        phases = []
        if '--decode' in remaining_args:
            phases.append('decode')
        if '--html' in remaining_args:
            phases.append('html')
        if '--gloss' in remaining_args:
            phases.append('gloss')
        if '--hitl' in remaining_args:
            phases.append('hitl')
        if '--whitepaper' in remaining_args:
            phases.append('whitepaper')
        if '--correlation' in remaining_args:
            phases.append('correlation')
        if phases:
            kwargs['phases'] = phases
        if '--folios' in remaining_args:
            idx = remaining_args.index('--folios')
            if idx + 1 < len(remaining_args):
                try:
                    kwargs['folio_limit'] = int(remaining_args[idx + 1])
                except ValueError:
                    pass

    return kwargs

def main():
    if os.environ.get('PYTHONHASHSEED') != '0':
        os.environ['PYTHONHASHSEED'] = '0'
        os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(
        description='Voynich Manuscript Convergence Attack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--phase', type=str, metavar='N',
                        help='Run a specific phase (2-13, or 12.5)')
    parser.add_argument('--phased', action='store_true',
                        help='Run the full phased attack (phases 1-4)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 13 phases sequentially (1-13)')
    parser.add_argument('--output-dir', '-o', metavar='DIR',
                        help='Root output directory (default: ./output)')
    parser.add_argument('--list', action='store_true',
                        help='List all available phases')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args, remaining = parser.parse_known_args()

    if args.list:
        from orchestrators import list_phases
        print('Available phases:')
        for num, path in list_phases().items():
            label = f'{num:>5}' if isinstance(num, int) else f'{num:>5s}'
            print(f'  Phase {label}: {path}')
        return

    verbose = not args.quiet
    output_root = args.output_dir
    report_dir = output_root or './output'

    def _phase_dir(phase_key):
        """Derive per-phase output dir from --output-dir, or None for default."""
        if output_root is None:
            return {}
        if phase_key == 1:
            return {'output_dir': output_root}
        dir_suffix = str(phase_key).replace('.', '_')
        return {'output_dir': os.path.join(output_root, f'phase{dir_suffix}')}

    phase_results = {}

    if args.all:
        from convergence_attack import run_convergence_attack
        from orchestrators import get_phase_runner, list_phases
        print('=== Phase 1: Convergence Attack (5 strategies) ===')
        phase_results[1] = run_convergence_attack(verbose=verbose, **_phase_dir(1))
        for phase_key in list_phases():
            print(f'\n=== Phase {phase_key} ===')
            phase_results[phase_key] = get_phase_runner(phase_key)(
                verbose=verbose, **_phase_dir(phase_key)
            )

    elif args.phase:
        from orchestrators import get_phase_runner

        phase_key = args.phase
        try:
            phase_key_parsed = int(phase_key)
            phase_key = phase_key_parsed
        except ValueError:
            pass

        runner = get_phase_runner(phase_key)

        kwargs = {'verbose': verbose}
        kwargs.update(_phase_dir(phase_key))
        sub_key = args.phase if isinstance(phase_key, str) else phase_key
        kwargs.update(_parse_subphase_flags(sub_key, remaining))

        phase_results[phase_key] = runner(**kwargs)

    elif args.phased:
        from convergence_attack import run_phased_attack
        phase_results[1] = run_phased_attack(verbose=verbose, **_phase_dir(1))

    else:
        from convergence_attack import run_convergence_attack
        phase_results[1] = run_convergence_attack(verbose=verbose, **_phase_dir(1))

    if phase_results:
        from orchestrators._utils import build_combined_report
        path = build_combined_report(phase_results, report_dir)
        if verbose:
            print(f'\nCombined report saved to {path}')

if __name__ == '__main__':
    main()
