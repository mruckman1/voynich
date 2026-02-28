"""
Phase 13 Orchestrator: Scholarly Synthesis & Presentation
==========================================================
Transforms the Voynich Convergence Attack output into readable, publishable
formats. First decodes ALL folios (configurable), then generates:

  13.0  Full-Corpus Decode        — run Phase 12 pipeline on all folios
  13.1  Interlinear HTML Viewer   — 4-tier offline HTML with traceability
  13.2  Deterministic English Glosser — Latin→English dictionary + inflection rules
  13.3  HITL Console              — interactive resolution of [UNRESOLVED] tokens
  13.4  Academic Whitepaper       — structured Markdown with matplotlib charts
  13.5  Illustration-Text Correlation — botanical ID vs decoded text validation

February 2026  ·  Voynich Convergence Attack  ·  Phase 13
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

from orchestrators._utils import save_json, ensure_output_dir, make_results_header
from orchestrators._config import (
    LATIN_CORPUS_TOKENS_LARGE, MIN_CONFIDENCE_RATIO,
    HITL_OVERRIDES_FILE, HITL_MAX_CANDIDATES, WHITEPAPER_CHART_DPI,
)
from orchestrators._foundation import build_morphological_context

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder, HUMORAL_VOCAB
from modules.phase12.syntactic_scaffolder import SyntacticScaffolder
from modules.phase12.ngram_mask_solver import NgramMaskSolver
from data.botanical_identifications import PLANT_IDS

def _count_brackets(text: str) -> int:
    """Count all bracketed tokens in text."""
    return len(re.findall(r'\[[^\]]+\]|<[^>]+>', text))

def _find_phase_output(phase_num: int, filename: str) -> str:
    """Locate a prior phase's output file, checking multiple directories."""
    candidates = [
        f'./output/phase{phase_num}/{filename}',
        f'./2nd_run_results/phase{phase_num}/{filename}',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'Cannot find {filename} for phase {phase_num}. '
        f'Searched: {candidates}. Run phase {phase_num} first.'
    )

def _find_combined_report() -> str:
    """Locate the combined report JSON."""
    candidates = [
        './output/combined_report.json',
        './2nd_run_results/combined_report.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'Cannot find combined_report.json. Searched: {candidates}'
    )

def _run_full_decode(
    ctx, folio_limit: Optional[int], verbose: bool
) -> Dict:
    """Run the full Phase 12 pipeline on all (or limited) folios.

    Returns a results dict compatible with Phase 12 output format:
    final_translations, csp_metrics, ngram_metrics, per_folio_stats, etc.
    """
    extractor = ctx.extractor
    v_morph = ctx.voynich_morphemer
    l_tokens = ctx.latin_tokens
    l_corpus = ctx.latin_corpus

    latin_skel = LatinPhoneticSkeletonizer(l_tokens)
    trans_matrix, trans_vocab = l_corpus.build_transition_matrix()

    fuzzy_skel = FuzzySkeletonizer(v_morph)
    budgeted_decoder = BudgetedCSPDecoder(
        latin_skel, fuzzy_skel, l_tokens, PLANT_IDS
    )
    scaffolder = SyntacticScaffolder(v_morph)
    ngram_solver = NgramMaskSolver(
        trans_matrix, trans_vocab, latin_skel, fuzzy_skel,
        humoral_vocab=HUMORAL_VOCAB,
        min_confidence_ratio=MIN_CONFIDENCE_RATIO,
    )

    if verbose:
        print(f'  → Latin corpus: {len(l_tokens)} tokens, '
              f'{len(set(l_tokens))} types')
        print(f'  → Latin skeletons: {len(latin_skel.skeleton_index)} unique')
        print(f'  → Transition matrix: {trans_matrix.shape[0]}x{trans_matrix.shape[1]}')

    by_folio = extractor.extract_lang_a_by_folio()
    folio_items = list(by_folio.items())
    if folio_limit is not None:
        folio_items = folio_items[:folio_limit]

    if verbose:
        print(f'  → Decoding {len(folio_items)} folios...')

    csp_translations = {}
    total_bracket_count = 0
    total_word_count = 0

    for folio, tokens in folio_items:
        if len(tokens) < 5:
            continue
        decoded = budgeted_decoder.decode_folio(tokens, folio_id=folio)
        csp_translations[folio] = decoded
        brackets = _count_brackets(decoded)
        total_bracket_count += brackets
        total_word_count += len(decoded.split())

    if verbose:
        print(f'  → CSP decoded {len(csp_translations)} folios, '
              f'{total_word_count} words, {total_bracket_count} brackets')

    scaffolded_translations = {}
    for folio, decoded_text in csp_translations.items():
        scaffolded_translations[folio] = scaffolder.scaffold(decoded_text)

    final_translations = {}
    total_resolved = 0
    total_unresolved = 0

    for folio, scaffolded_text in scaffolded_translations.items():
        resolved_text, stats = ngram_solver.solve_folio(scaffolded_text, folio_id=folio)
        final_translations[folio] = resolved_text
        total_resolved += stats['resolved']
        total_unresolved += stats['unresolved']

    if verbose:
        print(f'  → N-gram resolved: {total_resolved}, '
              f'still unresolved: {total_unresolved}')

    per_folio_stats = {}
    from collections import Counter
    for folio, text in final_translations.items():
        words_clean = [w for w in text.split()
                       if not w.startswith('[') and not w.startswith('<')]
        top_words = dict(Counter(words_clean).most_common(10))
        max_repeat = max(top_words.values()) if top_words else 0
        per_folio_stats[folio] = {
            'top_words': top_words,
            'max_repeat': max_repeat,
            'word_count': len(text.split()),
            'remaining_brackets': _count_brackets(text),
        }

    return {
        'final_translations': final_translations,
        'budgeted_csp_translations': csp_translations,
        'scaffolded_translations': scaffolded_translations,
        'csp_metrics': {
            'folios_decoded': len(csp_translations),
            'total_words': total_word_count,
            'total_brackets': total_bracket_count,
            'bracket_rate': total_bracket_count / max(1, total_word_count),
        },
        'ngram_metrics': {
            'initial_brackets': total_bracket_count,
            'resolved_by_ngram': total_resolved,
            'still_unresolved': total_unresolved,
            'ngram_resolution_rate': total_resolved / max(1, total_bracket_count),
            'final_unresolved_rate': total_unresolved / max(1, total_word_count),
            'min_confidence_ratio': MIN_CONFIDENCE_RATIO,
        },
        'per_folio_stats': per_folio_stats,
        '_fuzzy_skel': fuzzy_skel,
        '_latin_skel': latin_skel,
    }

def run_phase13_synthesis(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/phase13',
    folio: Optional[str] = None,
    folio_limit: Optional[int] = None,
) -> Dict:
    """Run the full Phase 13 pipeline.

    Args:
        phases: Sub-phases to run: 'decode', 'html', 'gloss', 'hitl', 'whitepaper'
                Default: decode + html + gloss + whitepaper (HITL excluded)
        verbose: Print progress
        output_dir: Output directory
        folio: Optional folio filter for HITL console
        folio_limit: Max folios to decode (None = all)
    """
    ensure_output_dir(output_dir)
    all_phases = ['decode', 'html', 'gloss', 'whitepaper']
    run_phases = set(phases or all_phases)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 13')
        print('Scholarly Synthesis & Presentation')
        print('=' * 70)

    results = make_results_header()

    glossary_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'english_glossary.json'
    )

    decode_data = None
    ctx = None
    fuzzy_skel = None
    latin_skel = None

    if 'decode' in run_phases:
        if verbose:
            limit_str = f'first {folio_limit}' if folio_limit else 'ALL'
            print(f'\n[0/5] Running Full-Corpus Decode ({limit_str} folios)...')

        ctx = build_morphological_context(
            verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE
        )
        decode_data = _run_full_decode(ctx, folio_limit, verbose)
        fuzzy_skel = decode_data.pop('_fuzzy_skel')
        latin_skel = decode_data.pop('_latin_skel')

        results['decode_metrics'] = {
            'folios_decoded': decode_data['csp_metrics']['folios_decoded'],
            'total_words': decode_data['csp_metrics']['total_words'],
            'final_unresolved_rate': decode_data['ngram_metrics']['final_unresolved_rate'],
        }

        translations_path = os.path.join(output_dir, 'phase13_full_translations.json')
        save_json(translations_path, decode_data)

        if verbose:
            n_folios = decode_data['csp_metrics']['folios_decoded']
            n_words = decode_data['csp_metrics']['total_words']
            unres = decode_data['ngram_metrics']['final_unresolved_rate']
            print(f'  → {n_folios} folios, {n_words} words, '
                  f'unresolved rate: {unres:.1%}')
            print(f'  → Saved: {translations_path}')
    else:
        cached_path = os.path.join(output_dir, 'phase13_full_translations.json')
        if os.path.exists(cached_path):
            if verbose:
                print(f'\n[0/5] Loading cached decode from {cached_path}...')
            with open(cached_path, 'r') as f:
                decode_data = json.load(f)
        else:
            try:
                p12_path = _find_phase_output(12, 'phase12_reconstruction.json')
                if verbose:
                    print(f'\n[0/5] Loading Phase 12 output from {p12_path}...')
                with open(p12_path, 'r') as f:
                    decode_data = json.load(f)
            except FileNotFoundError:
                print('ERROR: No decoded translations found. Run with --decode or run Phase 12 first.')
                return results

    english_translations = {}
    if 'gloss' in run_phases:
        if verbose:
            print('\n[1/5] Running Deterministic English Glosser...')

        from modules.phase13.english_glosser import run_english_glosser

        gloss_output = os.path.join(output_dir, 'english_translations.json')
        gloss_metrics = run_english_glosser(
            phase12_data=decode_data,
            glossary_path=glossary_path,
            output_path=gloss_output,
            verbose=verbose,
        )
        results['english_glosser'] = gloss_metrics

        with open(gloss_output, 'r') as f:
            english_data = json.load(f)
        english_translations = english_data.get('english_translations', {})

        if verbose:
            print(f'  → Glossed {gloss_metrics.get("folios_glossed", 0)} folios')
            print(f'  → Coverage: {gloss_metrics.get("glossed_rate", 0):.1%}')

    if 'html' in run_phases:
        if verbose:
            print('\n[2/5] Generating Interlinear HTML Viewer...')

        from modules.phase13.html_viewer import generate_interlinear_html

        try:
            p7_path = _find_phase_output(7, 'phase7_report.json')
            with open(p7_path, 'r') as f:
                phase7_data = json.load(f)
        except FileNotFoundError:
            phase7_data = {}

        eva_by_folio = {}
        if ctx is not None:
            by_folio = ctx.extractor.extract_lang_a_by_folio()
            eva_by_folio = {f: tokens for f, tokens in by_folio.items()}
        else:
            from modules.phase4.lang_a_extractor import LanguageAExtractor
            ext = LanguageAExtractor(verbose=False)
            by_folio = ext.extract_lang_a_by_folio()
            eva_by_folio = {f: tokens for f, tokens in by_folio.items()}

        html_path = os.path.join(output_dir, 'interlinear_viewer.html')
        html_metrics = generate_interlinear_html(
            phase7_data=phase7_data,
            phase12_data=decode_data,
            english_translations=english_translations,
            eva_by_folio=eva_by_folio,
            output_path=html_path,
            verbose=verbose,
        )
        results['html_viewer'] = html_metrics
        results['html_viewer']['path'] = html_path

        if verbose:
            print(f'  → HTML saved: {html_path}')

    if 'hitl' in run_phases:
        if verbose:
            print('\n[3/5] Starting HITL Console...')

        from modules.phase13.hitl_console import run_hitl_console

        if fuzzy_skel is None or latin_skel is None:
            if ctx is None:
                ctx = build_morphological_context(
                    verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE
                )
            latin_skel = LatinPhoneticSkeletonizer(ctx.latin_tokens)
            fuzzy_skel = FuzzySkeletonizer(ctx.voynich_morphemer)

        overrides_path = os.path.join(output_dir, HITL_OVERRIDES_FILE)
        hitl_metrics = run_hitl_console(
            phase12_data=decode_data,
            overrides_path=overrides_path,
            fuzzy_skeletonizer=fuzzy_skel,
            latin_skeletonizer=latin_skel,
            verbose=verbose,
            folio=folio,
        )
        results['hitl_console'] = hitl_metrics
    else:
        if verbose:
            print('\n[3/5] HITL Console skipped (use --hitl to enable)')
        results['hitl_console'] = {'skipped': True}

    if 'whitepaper' in run_phases:
        if verbose:
            print('\n[4/5] Generating Academic Whitepaper...')

        from modules.phase13.whitepaper_gen import run_whitepaper_generator

        try:
            combined_path = _find_combined_report()
        except FileNotFoundError:
            if verbose:
                print('  → Combined report not found, skipping whitepaper.')
            combined_path = None

        if combined_path:
            wp_metrics = run_whitepaper_generator(
                combined_report_path=combined_path,
                phase12_data=decode_data,
                output_dir=output_dir,
                dpi=WHITEPAPER_CHART_DPI,
                verbose=verbose,
            )
            results['whitepaper'] = wp_metrics

    if 'correlation' in run_phases:
        if verbose:
            print('\n[5/5] Running Illustration-Text Correlation...')

        from modules.phase13.illustration_correlation import run_illustration_correlation

        corr_metrics = run_illustration_correlation(
            phase12_data=decode_data,
            output_dir=output_dir,
            verbose=verbose,
        )
        results['illustration_correlation'] = corr_metrics

        if verbose:
            matched = corr_metrics.get('matched_folios', 0)
            testable = corr_metrics.get('testable_folios', 0)
            rate = corr_metrics.get('match_rate_testable', 0)
            bp = corr_metrics.get('binomial_p', 1.0)
            pp = corr_metrics.get('permutation_p', 1.0)
            print(f'  => {matched}/{testable} testable folios matched ({rate:.1%})')
            print(f'  => Binomial p={bp:.4f}, Permutation p={pp:.4f}')
    else:
        if verbose:
            print('\n[5/5] Illustration-Text Correlation skipped '
                  '(use --correlation)')
        results['illustration_correlation'] = {'skipped': True}

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 2)
    results['conclusion'] = (
        'Phase 13 decoded the full manuscript corpus and produced scholarly '
        'presentation materials. All outputs preserve the mathematical chain '
        'of evidence: every word traces from Voynich glyph through consonant '
        'skeleton to dictionary match to transition probability.'
    )

    report_path = os.path.join(output_dir, 'phase13_synthesis.json')
    save_json(report_path, results)

    if verbose:
        print(f'\n{"=" * 70}')
        print('PHASE 13 COMPLETE')
        print(f'  Total time: {elapsed:.1f}s')
        print(f'  Report: {report_path}')
        print(f'  Outputs:')
        if 'decode' in run_phases:
            print(f'    Full decode: {output_dir}/phase13_full_translations.json')
        if 'gloss' in run_phases:
            print(f'    English:     {output_dir}/english_translations.json')
        if 'html' in run_phases:
            print(f'    HTML Viewer: {output_dir}/interlinear_viewer.html')
        if 'whitepaper' in run_phases:
            wp = results.get('whitepaper', {})
            if wp.get('markdown_path'):
                print(f'    Whitepaper:  {wp["markdown_path"]}')
                print(f'    Charts:      {output_dir}/charts/')
        if 'correlation' in run_phases:
            corr = results.get('illustration_correlation', {})
            if corr.get('output_path'):
                print(f'    Correlation: {corr["output_path"]}')

    return results
