"""
Robustness Test Orchestrator
============================
Runs Tier 1 validation tests to preemptively close peer review attack vectors.

Tests:
  3a  Skeleton Length Analysis    — resolution rate by skeleton segment count
  1a  Reversed Text              — word-order sensitivity test
  4c  Consistency Significance   — per-mapping p-values + bidirectional check
  5a  Parameter Sensitivity      — sweep key params across ranges
  5b  Bootstrap Confidence       — jitter all params ±10%, 50 runs

Usage:
  uv run cli.py --robustness              # all tests
  uv run cli.py --robustness skeleton     # Test 3a only
  uv run cli.py --robustness reversed     # Test 1a only
  uv run cli.py --robustness consistency  # Test 4c only
  uv run cli.py --robustness sensitivity  # Test 5a only
  uv run cli.py --robustness bootstrap    # Test 5b only
"""
import json
import os
import re
import time
from typing import Dict, List, Optional

from orchestrators._utils import ensure_output_dir
from orchestrators._config import (
    LATIN_CORPUS_TOKENS_LARGE, VOYNICH_SECTIONS,
    MIN_CONFIDENCE_RATIO,
    MIN_CONFIDENCE_RATIO_LONG, LONG_SKELETON_SEGMENTS, ENABLE_LENGTH_SCALED_RATIO,
    ENABLE_BIDIRECTIONAL_SOLVING, MAX_SOLVING_PASSES,
    ENABLE_FUNCTION_WORD_RECOVERY, FUNCTION_WORD_TRIGRAM_THRESHOLD,
    DUAL_CONTEXT_RATIO_FACTOR, DUAL_CONTEXT_MAX_DISTANCE,
    ENABLE_UNIGRAM_BACKOFF, UNIGRAM_BACKOFF_RATIO_FACTOR, UNIGRAM_BACKOFF_MIN_SEGMENTS,
    ENABLE_POS_BACKOFF, POS_BACKOFF_WEIGHT, POS_BACKOFF_MIN_CONFIDENCE,
    ENABLE_CHAR_NGRAM_FALLBACK, CHAR_NGRAM_ORDER, CHAR_NGRAM_SMOOTHING,
    CHAR_NGRAM_MIN_SCORE_GAP, CHAR_NGRAM_MIN_SEGMENTS,
    CHAR_NGRAM_MAX_CONTEXT_DISTANCE, CHAR_NGRAM_REQUIRE_CONTEXT,
    ENABLE_CROSS_FOLIO_CONSISTENCY, CROSS_FOLIO_MIN_AGREEMENT, CROSS_FOLIO_MIN_OCCURRENCES,
    ENABLE_GRADUATED_CSP, CSP_HIGH_CONFIDENCE_THRESHOLD, CSP_MEDIUM_CONFIDENCE_THRESHOLD,
    ENABLE_SELECTIVE_FUNCTION_WORDS, FUNCTION_WORD_MAX_DENSITY, FUNCTION_WORD_WINDOW_SIZE,
    ENABLE_ILLUSTRATION_PRIOR, ILLUSTRATION_BOOSTED_RATIO_FACTOR, ILLUSTRATION_PRIOR_MIN_SEGMENTS,
    ENABLE_ADAPTIVE_CONFIDENCE, ADAPTIVE_CONFIDENCE_2_CAND_FACTOR,
    ADAPTIVE_CONFIDENCE_FEW_CAND_FACTOR,
    ENABLE_SINGLE_CAND_CHAR_RESCUE, SINGLE_CAND_MIN_SEGMENTS, SINGLE_CAND_MIN_CHAR_SCORE,
    ENABLE_RELAXED_CONSISTENCY, CROSS_FOLIO_MIN_OCCURRENCES_RELAXED,
    ENABLE_ITERATIVE_REFINEMENT, ITERATIVE_MAX_PASSES, ITERATIVE_MIN_IMPROVEMENT,
    ILLUSTRATION_TIER1_BOOST,
)
from orchestrators._foundation import build_morphological_context

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer
from modules.phase12.budgeted_csp import BudgetedCSPDecoder, HUMORAL_VOCAB
from modules.phase12.syntactic_scaffolder import (
    SyntacticScaffolder, build_pos_transition_matrix,
)
from modules.phase12.ngram_mask_solver import NgramMaskSolver
from modules.phase12.char_ngram_model import LatinCharNgramModel
from modules.phase12.cross_folio_consistency import CrossFolioConsistencyEngine

from data.botanical_identifications import PLANT_IDS


def _build_illustration_prior_safe():
    if not ENABLE_ILLUSTRATION_PRIOR:
        return None
    try:
        from data.folio_illustration_priors import build_illustration_prior
        return build_illustration_prior()
    except (ImportError, FileNotFoundError):
        return None


def _count_brackets(text: str) -> int:
    return len(re.findall(r'\[[^\]]+\]|<[^>]+>', text))


def _normalize_section(section: str, folio: str) -> str:
    from data.voynich_corpus import _infer_section
    if section == 'herbal':
        return _infer_section(folio)
    if section == 'zodiac':
        return 'astronomical'
    if section == 'text_only':
        return 'recipes'
    return section


def _get_folio_metadata(folio: str, page=None) -> Dict:
    from data.voynich_corpus import _infer_section, _infer_language, _infer_scribe
    if page is not None:
        section = getattr(page, 'section', None) or _infer_section(folio)
        language = getattr(page, 'language', None) or _infer_language(folio)
        scribe = getattr(page, 'hand', None) or _infer_scribe(folio)
    else:
        section = _infer_section(folio)
        language = _infer_language(folio)
        scribe = _infer_scribe(folio)
    section = _normalize_section(section, folio)
    return {'language': language, 'section': section, 'scribe': scribe}


def build_default_solver_kwargs(
    pos_tagger, pos_matrix, pos_vocab, char_ngram_model,
    illustration_prior, **overrides,
):
    """Build the standard solver kwargs dict, with optional overrides."""
    kwargs = dict(
        humoral_vocab=HUMORAL_VOCAB,
        min_confidence_ratio=MIN_CONFIDENCE_RATIO,
        pos_tagger=pos_tagger,
        pos_transition_matrix=pos_matrix,
        pos_vocab=pos_vocab,
        min_confidence_ratio_long=MIN_CONFIDENCE_RATIO_LONG,
        long_skeleton_segments=LONG_SKELETON_SEGMENTS,
        enable_length_scaled_ratio=ENABLE_LENGTH_SCALED_RATIO,
        enable_bidirectional=ENABLE_BIDIRECTIONAL_SOLVING,
        max_solving_passes=MAX_SOLVING_PASSES,
        enable_function_word_recovery=ENABLE_FUNCTION_WORD_RECOVERY,
        function_word_trigram_threshold=FUNCTION_WORD_TRIGRAM_THRESHOLD,
        dual_context_ratio_factor=DUAL_CONTEXT_RATIO_FACTOR,
        dual_context_max_distance=DUAL_CONTEXT_MAX_DISTANCE,
        enable_unigram_backoff=ENABLE_UNIGRAM_BACKOFF,
        unigram_backoff_ratio_factor=UNIGRAM_BACKOFF_RATIO_FACTOR,
        unigram_backoff_min_segments=UNIGRAM_BACKOFF_MIN_SEGMENTS,
        enable_pos_backoff=ENABLE_POS_BACKOFF,
        pos_backoff_weight=POS_BACKOFF_WEIGHT,
        pos_backoff_min_confidence=POS_BACKOFF_MIN_CONFIDENCE,
        enable_char_ngram_fallback=ENABLE_CHAR_NGRAM_FALLBACK,
        char_ngram_model=char_ngram_model,
        char_ngram_min_score_gap=CHAR_NGRAM_MIN_SCORE_GAP,
        char_ngram_min_segments=CHAR_NGRAM_MIN_SEGMENTS,
        char_ngram_max_context_distance=CHAR_NGRAM_MAX_CONTEXT_DISTANCE,
        char_ngram_require_context=CHAR_NGRAM_REQUIRE_CONTEXT,
        enable_illustration_prior=ENABLE_ILLUSTRATION_PRIOR,
        illustration_prior=illustration_prior,
        illustration_boosted_ratio_factor=ILLUSTRATION_BOOSTED_RATIO_FACTOR,
        illustration_prior_min_segments=ILLUSTRATION_PRIOR_MIN_SEGMENTS,
        enable_adaptive_confidence=ENABLE_ADAPTIVE_CONFIDENCE,
        adaptive_confidence_2_cand_factor=ADAPTIVE_CONFIDENCE_2_CAND_FACTOR,
        adaptive_confidence_few_cand_factor=ADAPTIVE_CONFIDENCE_FEW_CAND_FACTOR,
        enable_single_cand_char_rescue=ENABLE_SINGLE_CAND_CHAR_RESCUE,
        single_cand_min_segments=SINGLE_CAND_MIN_SEGMENTS,
        single_cand_min_char_score=SINGLE_CAND_MIN_CHAR_SCORE,
    )
    kwargs.update(overrides)
    return kwargs


def build_pipeline_components(verbose=True):
    """Build all Phase 12 pipeline components once for shared reuse."""
    if verbose:
        print('  Building Phase 12 pipeline components...')

    ctx = build_morphological_context(
        verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE
    )
    extractor = ctx.extractor
    v_morph = ctx.voynich_morphemer
    l_tokens = ctx.latin_tokens
    l_corpus = ctx.latin_corpus

    latin_skel = LatinPhoneticSkeletonizer(l_tokens)
    trans_matrix, trans_vocab = l_corpus.build_transition_matrix()

    fuzzy_skel = FuzzySkeletonizer(v_morph)
    decoder = BudgetedCSPDecoder(
        latin_skel, fuzzy_skel, l_tokens, PLANT_IDS,
        enable_graduated_csp=ENABLE_GRADUATED_CSP,
        high_threshold=CSP_HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold=CSP_MEDIUM_CONFIDENCE_THRESHOLD,
        enable_selective_function_words=ENABLE_SELECTIVE_FUNCTION_WORDS,
        function_word_max_density=FUNCTION_WORD_MAX_DENSITY,
        function_word_window_size=FUNCTION_WORD_WINDOW_SIZE,
    )
    scaffolder = SyntacticScaffolder(v_morph)

    pos_matrix, pos_vocab, pos_tagger = build_pos_transition_matrix(l_tokens)

    char_ngram_model = None
    if ENABLE_CHAR_NGRAM_FALLBACK:
        char_ngram_model = LatinCharNgramModel(
            order=CHAR_NGRAM_ORDER, smoothing=CHAR_NGRAM_SMOOTHING,
        )
        char_ngram_model.train(l_tokens)

    illustration_prior = _build_illustration_prior_safe()

    solver_kwargs = build_default_solver_kwargs(
        pos_tagger, pos_matrix, pos_vocab, char_ngram_model, illustration_prior,
    )
    solver = NgramMaskSolver(
        trans_matrix, trans_vocab, latin_skel, fuzzy_skel, **solver_kwargs,
    )
    solver.set_corpus_frequencies(l_tokens)

    # Extract folios from IVTFF
    by_folio = {}
    folio_metadata = {}
    if extractor._source == 'ivtff' and extractor._corpus is not None:
        for fid, page in extractor._corpus.pages.items():
            tokens = page.paragraph_text.split()
            if tokens:
                by_folio[fid] = tokens
                folio_metadata[fid] = _get_folio_metadata(fid, page=page)
    else:
        from data.voynich_corpus import SAMPLE_CORPUS
        for fid, data in SAMPLE_CORPUS.items():
            text = ' '.join(data.get('text', []))
            tokens = text.split()
            if tokens:
                by_folio[fid] = tokens
                folio_metadata[fid] = {
                    'language': data.get('lang', 'A'),
                    'section': data.get('section', 'unknown'),
                    'scribe': data.get('scribe', 0),
                }

    if verbose:
        print(f'    Latin corpus: {len(l_tokens)} tokens, '
              f'{len(set(l_tokens))} types')
        print(f'    Transition matrix: {trans_matrix.shape[0]}x{trans_matrix.shape[1]}')
        print(f'    Folios loaded: {len(by_folio)}')

    return {
        'extractor': extractor,
        'l_tokens': l_tokens,
        'l_corpus': l_corpus,
        'latin_skel': latin_skel,
        'trans_matrix': trans_matrix,
        'trans_vocab': trans_vocab,
        'fuzzy_skel': fuzzy_skel,
        'decoder': decoder,
        'scaffolder': scaffolder,
        'pos_tagger': pos_tagger,
        'pos_matrix': pos_matrix,
        'pos_vocab': pos_vocab,
        'char_ngram_model': char_ngram_model,
        'illustration_prior': illustration_prior,
        'solver': solver,
        'solver_kwargs': solver_kwargs,
        'by_folio': by_folio,
        'folio_metadata': folio_metadata,
    }


def run_full_pipeline(
    components, by_folio_override=None, solver_override=None,
    decoder_override=None, consistency_min_agreement=None,
    verbose=False,
):
    """Run the full Phase 12 pipeline and return final_translations + resolution rate.

    Reuses shared components. Allows overriding by_folio (for reversed text),
    solver (for parameter sweeps), decoder (for CSP threshold sweeps),
    and consistency params.
    """
    by_folio = by_folio_override or components['by_folio']
    folio_metadata = components['folio_metadata']
    decoder = decoder_override or components['decoder']
    scaffolder = components['scaffolder']
    solver = solver_override or components['solver']
    fuzzy_skel = components['fuzzy_skel']

    # CSP decode
    csp_translations = {}
    folio_medium_candidates = {}
    for folio, tokens in by_folio.items():
        if len(tokens) < 5:
            continue
        decoded = decoder.decode_folio(tokens, folio_id=folio)
        csp_translations[folio] = decoded
        if ENABLE_GRADUATED_CSP and decoder.medium_candidates:
            folio_medium_candidates[folio] = dict(decoder.medium_candidates)

    # Scaffold
    scaffolded_translations = {}
    for folio, decoded_text in csp_translations.items():
        scaffolded_translations[folio] = scaffolder.scaffold(decoded_text)

    # Solve
    final_translations = {}
    for folio, scaffolded_text in scaffolded_translations.items():
        medium_cands = folio_medium_candidates.get(folio) if ENABLE_GRADUATED_CSP else None
        resolved_text, stats = solver.solve_folio(
            scaffolded_text, folio_id=folio, medium_candidates=medium_cands,
        )
        final_translations[folio] = resolved_text

    # Cross-folio consistency
    if ENABLE_CROSS_FOLIO_CONSISTENCY:
        min_agr = consistency_min_agreement or CROSS_FOLIO_MIN_AGREEMENT
        consistency_engine = CrossFolioConsistencyEngine(
            fuzzy_skel, min_agreement=min_agr,
            min_occurrences=CROSS_FOLIO_MIN_OCCURRENCES,
        )
        for folio, tokens in by_folio.items():
            if folio in final_translations:
                consistency_engine.collect_folio(
                    final_translations[folio], tokens, folio,
                )
        consistency_engine.compute_consistent_mappings()
        for folio, tokens in by_folio.items():
            if folio in final_translations:
                updated_text, _ = consistency_engine.apply_consistency(
                    final_translations[folio], tokens,
                )
                final_translations[folio] = updated_text

        if ENABLE_RELAXED_CONSISTENCY:
            consistency_engine.compute_relaxed_mappings(
                min_occurrences_relaxed=CROSS_FOLIO_MIN_OCCURRENCES_RELAXED,
            )
            for folio, tokens in by_folio.items():
                if folio in final_translations:
                    updated_text, _ = consistency_engine.apply_consistency(
                        final_translations[folio], tokens,
                    )
                    final_translations[folio] = updated_text

    # POS backoff
    if ENABLE_POS_BACKOFF:
        for folio in final_translations:
            updated_text, _ = solver.pos_backoff_pass(
                final_translations[folio], folio_id=folio,
            )
            final_translations[folio] = updated_text

    # Char n-gram fallback
    if ENABLE_CHAR_NGRAM_FALLBACK:
        for folio in final_translations:
            updated_text, _ = solver.char_ngram_pass(
                final_translations[folio], folio_id=folio,
            )
            final_translations[folio] = updated_text

    # Iterative refinement
    if ENABLE_ITERATIVE_REFINEMENT:
        for iteration in range(ITERATIVE_MAX_PASSES):
            total_refined = 0
            for folio in final_translations:
                updated_text, n = solver.refine_pass(
                    final_translations[folio],
                    by_folio.get(folio, []),
                    folio_id=folio,
                )
                final_translations[folio] = updated_text
                total_refined += n

            if ENABLE_CROSS_FOLIO_CONSISTENCY:
                min_agr = consistency_min_agreement or CROSS_FOLIO_MIN_AGREEMENT
                iter_consistency = CrossFolioConsistencyEngine(
                    fuzzy_skel, min_agreement=min_agr,
                    min_occurrences=CROSS_FOLIO_MIN_OCCURRENCES,
                )
                for folio, tokens in by_folio.items():
                    if folio in final_translations:
                        iter_consistency.collect_folio(
                            final_translations[folio], tokens, folio,
                        )
                iter_consistency.compute_consistent_mappings()
                if ENABLE_RELAXED_CONSISTENCY:
                    iter_consistency.compute_relaxed_mappings(
                        min_occurrences_relaxed=CROSS_FOLIO_MIN_OCCURRENCES_RELAXED,
                    )
                for folio, tokens in by_folio.items():
                    if folio in final_translations:
                        updated, cstats = iter_consistency.apply_consistency(
                            final_translations[folio], tokens,
                        )
                        final_translations[folio] = updated
                        total_refined += cstats['applied']

            if ENABLE_POS_BACKOFF:
                for folio in final_translations:
                    t, n = solver.pos_backoff_pass(
                        final_translations[folio], folio_id=folio,
                    )
                    final_translations[folio] = t
                    total_refined += n

            if ENABLE_CHAR_NGRAM_FALLBACK:
                for folio in final_translations:
                    t, n = solver.char_ngram_pass(
                        final_translations[folio], folio_id=folio,
                    )
                    final_translations[folio] = t
                    total_refined += n

            if total_refined < ITERATIVE_MIN_IMPROVEMENT:
                break

    # Compute resolution rate
    total_words = 0
    total_brackets = 0
    lang_a_words = lang_a_brackets = 0
    lang_b_words = lang_b_brackets = 0
    for folio, text in final_translations.items():
        wc = len(text.split())
        bc = _count_brackets(text)
        total_words += wc
        total_brackets += bc
        meta = folio_metadata.get(folio, {})
        if meta.get('language') == 'B':
            lang_b_words += wc
            lang_b_brackets += bc
        else:
            lang_a_words += wc
            lang_a_brackets += bc

    overall_rate = (total_words - total_brackets) / max(1, total_words)
    lang_a_rate = (lang_a_words - lang_a_brackets) / max(1, lang_a_words)
    lang_b_rate = (lang_b_words - lang_b_brackets) / max(1, lang_b_words)

    return {
        'final_translations': final_translations,
        'overall_resolution': overall_rate,
        'lang_a_resolution': lang_a_rate,
        'lang_b_resolution': lang_b_rate,
        'total_words': total_words,
        'total_brackets': total_brackets,
    }


def run_robustness_tests(
    tests: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/robustness',
) -> Dict:
    """Run Tier 1 robustness validation tests.

    Args:
        tests: List of test names to run, or None for all.
               Valid: 'skeleton', 'reversed', 'consistency', 'sensitivity', 'bootstrap'
        verbose: Print progress and results.
        output_dir: Output directory for JSON reports.
    """
    ensure_output_dir(output_dir)
    t0 = time.time()

    all_tests = ['skeleton', 'reversed', 'consistency', 'sensitivity', 'bootstrap']
    run_tests = tests or all_tests

    if verbose:
        print('=' * 70)
        print('TIER 1 ROBUSTNESS VALIDATION TESTS')
        print('=' * 70)

    components = build_pipeline_components(verbose=verbose)

    results = {}

    if 'skeleton' in run_tests:
        if verbose:
            print(f'\n{"=" * 70}')
            print('TEST 3a: Skeleton Length Analysis')
            print('=' * 70)
        from modules.robustness.skeleton_length_analysis import SkeletonLengthAnalysis
        test = SkeletonLengthAnalysis(components, verbose=verbose)
        results['skeleton'] = test.run()
        _save_result(output_dir, 'skeleton_length_analysis.json', results['skeleton'])

    if 'reversed' in run_tests:
        if verbose:
            print(f'\n{"=" * 70}')
            print('TEST 1a: Reversed Voynich Text')
            print('=' * 70)
        from modules.robustness.reversed_text_test import ReversedTextTest
        test = ReversedTextTest(components, verbose=verbose)
        results['reversed'] = test.run()
        _save_result(output_dir, 'reversed_text_test.json', results['reversed'])

    if 'consistency' in run_tests:
        if verbose:
            print(f'\n{"=" * 70}')
            print('TEST 4c: Consistency Significance')
            print('=' * 70)
        from modules.robustness.consistency_significance import ConsistencySignificance
        test = ConsistencySignificance(components, verbose=verbose)
        results['consistency'] = test.run()
        _save_result(output_dir, 'consistency_significance.json', results['consistency'])

    if 'sensitivity' in run_tests:
        if verbose:
            print(f'\n{"=" * 70}')
            print('TEST 5a: Parameter Sensitivity Sweep')
            print('=' * 70)
        from modules.robustness.parameter_sensitivity import ParameterSensitivity
        test = ParameterSensitivity(components, verbose=verbose)
        results['sensitivity'] = test.run()
        _save_result(output_dir, 'parameter_sensitivity.json', results['sensitivity'])

    if 'bootstrap' in run_tests:
        if verbose:
            print(f'\n{"=" * 70}')
            print('TEST 5b: Bootstrap Confidence Intervals')
            print('=' * 70)
        from modules.robustness.bootstrap_confidence import BootstrapConfidence
        test = BootstrapConfidence(components, verbose=verbose)
        results['bootstrap'] = test.run()
        _save_result(output_dir, 'bootstrap_confidence.json', results['bootstrap'])

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 2)

    if verbose:
        print(f'\n{"=" * 70}')
        print(f'All robustness tests complete in {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results


def _save_result(output_dir, filename, data):
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
