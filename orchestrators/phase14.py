"""
Phase 14: Semantic Coherence Analysis
======================================
Validates whether Phase 12 decoded text forms coherent medieval
medical recipe structures or is semantically random.

Four metrics compared against null distributions:
  1. Medical vocabulary rate (vs random vocab substitution)
  2. Semantic field concentration / entropy (vs shuffled word order)
  3. Recipe template matching (vs shuffled word order)
  4. Collocational coherence (vs random vocab substitution)

This is a pure validation phase — no pipeline changes.

Phase 14  ·  Voynich Convergence Attack
"""

import json
import os
import time
from typing import Dict, List, Optional

from orchestrators._utils import ensure_output_dir, save_json, vprint


def run_phase14(phases: Optional[List[str]] = None,
                verbose: bool = True,
                output_dir: str = './output/phase14',
                folio_limit: Optional[int] = None,
                n_trials: int = 1000,
                **_kwargs) -> Dict:
    """Run Phase 14: Semantic Coherence Analysis.

    Sub-phases:
        vocabulary    — medical vocabulary rate analysis
        concentration — semantic field entropy
        templates     — recipe template matching
        collocations  — collocational coherence
        significance  — null distribution + p-values
        langb         — Language B diagnostic comparison

    Requires Phase 12 output (output/phase12/phase12_reconstruction.json).
    """
    ensure_output_dir(output_dir)

    all_phases = ['vocabulary', 'concentration', 'templates',
                  'collocations', 'significance', 'langb']
    run_phases = set(phases or all_phases)

    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 14')
        print('Semantic Coherence Analysis')
        print('=' * 70)

    # ── Load Phase 12 output ────────────────────────────────────────
    p12_path = './output/phase12/phase12_reconstruction.json'
    if not os.path.exists(p12_path):
        msg = (f'Phase 12 output not found: {p12_path}\n'
               f'Run Phase 12 first: uv run cli.py --phase 12')
        if verbose:
            print(f'\n  ERROR: {msg}')
        return {'error': msg}

    vprint(verbose, f'\n  Loading Phase 12 output: {p12_path}')
    with open(p12_path) as f:
        p12 = json.load(f)

    translations = p12.get('final_translations', {})
    per_folio_stats = p12.get('per_folio_stats', {})

    if folio_limit:
        folio_ids = sorted(translations.keys())[:folio_limit]
        translations = {k: translations[k] for k in folio_ids}
        per_folio_stats = {k: per_folio_stats[k] for k in folio_ids
                          if k in per_folio_stats}

    vprint(verbose, f'  Loaded {len(translations)} folios')

    # ── Build analyzer ──────────────────────────────────────────────
    from modules.phase14.semantic_coherence import SemanticCoherenceAnalyzer
    analyzer = SemanticCoherenceAnalyzer(translations, per_folio_stats)

    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'purpose': 'Measures semantic coherence of Phase 12 decoded text',
        'folios_loaded': len(translations),
        'folio_limit': folio_limit,
        'n_trials': n_trials,
    }

    # ── Run core metrics (vocabulary, concentration, templates, collocations)
    if run_phases & {'vocabulary', 'concentration', 'templates', 'collocations'}:
        vprint(verbose, '\n' + '─' * 70)
        vprint(verbose, '  Running core metrics on all folios...')

        analysis = analyzer.analyze_all(folio_limit=folio_limit)

        if 'error' in analysis:
            vprint(verbose, f'  ERROR: {analysis["error"]}')
            results['analysis'] = analysis
        else:
            results['summary'] = analysis['summary']
            results['by_section'] = analysis['by_section']
            results['by_language'] = analysis['by_language']
            results['per_folio'] = analysis['per_folio']

            summ = analysis['summary']
            vprint(verbose,
                   f'  Analyzed {summ["folios_analyzed"]} folios '
                   f'({summ["folios_skipped"]} skipped, <5 resolved words)')

            if 'vocabulary' in run_phases:
                vprint(verbose, '')
                vprint(verbose,
                       f'  Medical Vocabulary Rate: '
                       f'{summ["overall_medical_rate"]:.1%}')
                if summ.get('field_distribution'):
                    top3 = list(summ['field_distribution'].items())[:5]
                    for field, pct in top3:
                        vprint(verbose, f'    → {field}: {pct:.1%}')

            if 'concentration' in run_phases:
                vprint(verbose, '')
                vprint(verbose,
                       f'  Field Entropy (normalized): '
                       f'{summ["overall_entropy"]:.4f} '
                       f'(0=focused, 1=scattered)')

            if 'templates' in run_phases:
                vprint(verbose, '')
                vprint(verbose,
                       f'  Template Coverage: '
                       f'{summ["overall_template_coverage"]:.1%} '
                       f'of classified words match recipe patterns')
                if summ.get('top_templates'):
                    vprint(verbose, '  Top templates:')
                    for t_info in summ['top_templates'][:5]:
                        tpl = ' → '.join(t_info['template'])
                        vprint(verbose,
                               f'    {tpl}: {t_info["count"]} matches')

            if 'collocations' in run_phases:
                vprint(verbose, '')
                vprint(verbose,
                       f'  Collocation Plausibility: '
                       f'{summ["overall_collocation_plausible"]:.1%}')

            # Section breakdown
            if verbose and analysis.get('by_section'):
                vprint(verbose, '')
                vprint(verbose, '  By manuscript section:')
                vprint(verbose,
                       f'    {"Section":<20s} {"Folios":>6s} '
                       f'{"Med%":>6s} {"Entropy":>8s} '
                       f'{"Templ%":>7s} {"Colloc%":>8s}')
                vprint(verbose, '    ' + '─' * 55)
                for sec, data in sorted(analysis['by_section'].items()):
                    vprint(verbose,
                           f'    {sec:<20s} {data["folios"]:>6d} '
                           f'{data["medical_rate"]:>5.1%} '
                           f'{data["entropy"]:>8.4f} '
                           f'{data["template_coverage"]:>6.1%} '
                           f'{data["collocation_plausible"]:>7.1%}')

            # Language comparison
            if verbose and analysis.get('by_language'):
                vprint(verbose, '')
                vprint(verbose, '  Language A vs B:')
                for lang, data in sorted(analysis['by_language'].items()):
                    vprint(verbose,
                           f'    Language {lang}: '
                           f'medical {data["medical_rate"]:.1%}, '
                           f'entropy {data["entropy"]:.4f}, '
                           f'templates {data["template_coverage"]:.1%}, '
                           f'collocations {data["collocation_plausible"]:.1%}')

    # ── Significance testing ────────────────────────────────────────
    if 'significance' in run_phases:
        vprint(verbose, '\n' + '─' * 70)
        vprint(verbose,
               f'  Running significance tests ({n_trials} trials per folio)...')

        significance = analyzer.compute_significance(
            folio_limit=folio_limit,
            n_trials=n_trials,
            verbose=verbose,
        )
        results['significance'] = significance

        if verbose and 'error' not in significance:
            vprint(verbose, '')
            for metric, data in significance.items():
                if 'error' in data:
                    vprint(verbose, f'  {metric}: {data["error"]}')
                    continue
                sig_str = ('SIGNIFICANT' if data['significant_005']
                           else 'not significant')
                vprint(verbose,
                       f'  {metric}:')
                vprint(verbose,
                       f'    Real: {data["real_mean"]:.4f}  '
                       f'Null ({data["null_method"]}): {data["null_mean"]:.4f}  '
                       f'Effect: {data["effect_size"]:+.4f}')
                vprint(verbose,
                       f'    p = {data["mean_p_value"]:.4f} ({sig_str})')

    # ── Language B diagnostic ───────────────────────────────────────
    if 'langb' in run_phases:
        vprint(verbose, '\n' + '─' * 70)
        vprint(verbose, '  Language B diagnostic...')

        langb = analyzer.language_b_diagnostic()
        results['language_b_diagnostic'] = langb

        if verbose:
            vprint(verbose,
                   f'    Lang A: {langb["lang_a_types"]} types, '
                   f'{langb["lang_a_tokens"]} tokens')
            vprint(verbose,
                   f'    Lang B: {langb["lang_b_types"]} types, '
                   f'{langb["lang_b_tokens"]} tokens')
            vprint(verbose,
                   f'    Overlap: {langb["overlap_types"]} types '
                   f'(Jaccard: {langb["jaccard_similarity"]:.3f})')
            vprint(verbose,
                   f'    A-only types: {langb["a_only_types"]}, '
                   f'B-only types: {langb["b_only_types"]}')

            if langb.get('field_comparison'):
                vprint(verbose, '    Field distributions:')
                vprint(verbose,
                       f'      {"Field":<16s} {"Lang A":>8s} {"Lang B":>8s}')
                vprint(verbose, '      ' + '─' * 32)
                for field, pcts in sorted(langb['field_comparison'].items()):
                    vprint(verbose,
                           f'      {field:<16s} '
                           f'{pcts["lang_a_pct"]:>7.1%} '
                           f'{pcts["lang_b_pct"]:>7.1%}')

    # ── Save and summarize ──────────────────────────────────────────
    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 2)

    report_path = os.path.join(output_dir, 'semantic_coherence.json')
    save_json(report_path, results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 14: SUMMARY')
        print('=' * 70)

        if 'summary' in results:
            s = results['summary']
            print(f'  Folios analyzed:        {s["folios_analyzed"]}')
            print(f'  Medical vocabulary:     {s["overall_medical_rate"]:.1%}')
            print(f'  Field entropy:          {s["overall_entropy"]:.4f}')
            print(f'  Template coverage:      {s["overall_template_coverage"]:.1%}')
            print(f'  Collocation plausible:  {s["overall_collocation_plausible"]:.1%}')

        if 'significance' in results and 'error' not in results.get('significance', {}):
            print('\n  Significance (p < 0.05):')
            for metric, data in results['significance'].items():
                if isinstance(data, dict) and 'significant_005' in data:
                    status = 'PASS' if data['significant_005'] else 'FAIL'
                    print(f'    {metric:<28s} p={data["mean_p_value"]:.4f} '
                          f'[{status}]')

        print(f'\n  Total time: {elapsed:.1f}s')
        print(f'  Report saved: {report_path}')

    return results
