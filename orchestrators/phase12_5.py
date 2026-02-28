"""
Phase 12.5 Orchestrator: Adversarial Defense Suite
===================================================
Subjects the Phase 11/12 decoding pipeline to 5 adversarial conditions
to mathematically prove that its success on real Voynich text is genuine.

Tests:
  12.5.1  Unicity Distance     — scrambled/random text baseline
  12.5.2  Domain Swap          — Bible/Legal transition matrix injection
  12.5.3  Polyglot Dictionary  — Italian/Occitan dictionary substitution
  12.5.4  EVA Collapse         — re-tokenized ligature collapse
  12.5.5  Ablation Study       — function word removal + grammar trace

Each test produces a pass/fail verdict with empirical metrics.
The overall "Defense Score" is tests_passed / 5.

February 2026  ·  Voynich Convergence Attack  ·  Phase 12.5
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from orchestrators._utils import save_json, ensure_output_dir
from orchestrators._config import (
    LATIN_CORPUS_TOKENS_LARGE, FOLIO_LIMIT_DEFAULT,
    MIN_CONFIDENCE_RATIO, ADV_UNICITY_TRIALS,
    MIN_CONFIDENCE_RATIO_LONG, LONG_SKELETON_SEGMENTS, ENABLE_LENGTH_SCALED_RATIO,
    ENABLE_BIDIRECTIONAL_SOLVING, MAX_SOLVING_PASSES,
    ENABLE_FUNCTION_WORD_RECOVERY, FUNCTION_WORD_TRIGRAM_THRESHOLD,
    DUAL_CONTEXT_RATIO_FACTOR, DUAL_CONTEXT_MAX_DISTANCE,
    ENABLE_UNIGRAM_BACKOFF, UNIGRAM_BACKOFF_RATIO_FACTOR, UNIGRAM_BACKOFF_MIN_SEGMENTS,
    ENABLE_POS_BACKOFF, POS_BACKOFF_WEIGHT, POS_BACKOFF_MIN_CONFIDENCE,
    ENABLE_CHAR_NGRAM_FALLBACK, CHAR_NGRAM_ORDER, CHAR_NGRAM_SMOOTHING,
    CHAR_NGRAM_MIN_SCORE_GAP, CHAR_NGRAM_MIN_SEGMENTS,
    CHAR_NGRAM_MAX_CONTEXT_DISTANCE, CHAR_NGRAM_REQUIRE_CONTEXT,
    ENABLE_ILLUSTRATION_PRIOR, ILLUSTRATION_BOOSTED_RATIO_FACTOR,
    ENABLE_ADAPTIVE_CONFIDENCE, ADAPTIVE_CONFIDENCE_2_CAND_FACTOR,
    ADAPTIVE_CONFIDENCE_FEW_CAND_FACTOR,
    ENABLE_SINGLE_CAND_CHAR_RESCUE, SINGLE_CAND_MIN_SEGMENTS, SINGLE_CAND_MIN_CHAR_SCORE,
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

from modules.phase12_5.adv_1_unicity_distance import UnicityDistanceTest
from modules.phase12_5.adv_2_domain_swap import DomainSwapTest
from modules.phase12_5.adv_3_polyglot_dict import PolyglotDictTest
from modules.phase12_5.adv_4_eva_collapse import EvaCollapseTest
from modules.phase12_5.adv_5_ablation_study import AblationStudyTest
from modules.phase12_5.proof_of_compositionality import CompositionalityProof

from data.botanical_identifications import PLANT_IDS

def _build_illustration_prior_safe():
    """Build illustration prior dict, returning None if data is unavailable."""
    if not ENABLE_ILLUSTRATION_PRIOR:
        return None
    try:
        from data.folio_illustration_priors import build_illustration_prior
        return build_illustration_prior()
    except (ImportError, FileNotFoundError):
        return None

def run_phase12_5_adversarial(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/phase12_5',
    folio_limit: int = FOLIO_LIMIT_DEFAULT,
) -> Dict:
    """
    Run the Phase 12.5 Adversarial Defense Suite.

    Args:
        phases: Optional list of sub-phases to run. Valid values:
            'unicity', 'domain_swap', 'polyglot', 'eva_collapse', 'ablation',
            'compositionality'
            If None, runs all 5 core tests (compositionality is opt-in).
        verbose: Print progress and results
        output_dir: Output directory for JSON results
        folio_limit: Number of folios to process per test

    Returns:
        Results dict with per-test verdicts and overall defense score
    """
    ensure_output_dir(output_dir)
    all_phases = ['unicity', 'domain_swap', 'polyglot', 'eva_collapse', 'ablation']
    run_phases = set(phases or all_phases)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 12.5')
        print('Adversarial Defense Suite')
        print('=' * 70)

    results = {
        'timestamp': datetime.now().isoformat(),
        'purpose': (
            'Subjects the Phase 11/12 decoding pipeline to adversarial '
            'conditions to prove its success is genuine, not an artifact '
            'of algorithmic overfitting.'
        ),
    }

    if verbose:
        print('\n[0/5] Loading shared pipeline dependencies...')

    ctx = build_morphological_context(
        verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE,
    )
    extractor = ctx.extractor
    v_morph = ctx.voynich_morphemer
    l_tokens = ctx.latin_tokens
    l_corpus = ctx.latin_corpus

    latin_skel = LatinPhoneticSkeletonizer(l_tokens)
    fuzzy_skel = FuzzySkeletonizer(v_morph)
    trans_matrix, trans_vocab = l_corpus.build_transition_matrix()

    budgeted_decoder = BudgetedCSPDecoder(
        latin_skel, fuzzy_skel, l_tokens, PLANT_IDS,
    )
    scaffolder = SyntacticScaffolder(v_morph)

    pos_matrix, pos_vocab, pos_tagger = build_pos_transition_matrix(l_tokens)

    char_ngram_model = None
    if ENABLE_CHAR_NGRAM_FALLBACK:
        char_ngram_model = LatinCharNgramModel(
            order=CHAR_NGRAM_ORDER,
            smoothing=CHAR_NGRAM_SMOOTHING,
        )
        char_ngram_model.train(l_tokens)

    ngram_solver = NgramMaskSolver(
        trans_matrix, trans_vocab, latin_skel, fuzzy_skel,
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
        illustration_prior=_build_illustration_prior_safe(),
        illustration_boosted_ratio_factor=ILLUSTRATION_BOOSTED_RATIO_FACTOR,
        enable_adaptive_confidence=ENABLE_ADAPTIVE_CONFIDENCE,
        adaptive_confidence_2_cand_factor=ADAPTIVE_CONFIDENCE_2_CAND_FACTOR,
        adaptive_confidence_few_cand_factor=ADAPTIVE_CONFIDENCE_FEW_CAND_FACTOR,
        enable_single_cand_char_rescue=ENABLE_SINGLE_CAND_CHAR_RESCUE,
        single_cand_min_segments=SINGLE_CAND_MIN_SEGMENTS,
        single_cand_min_char_score=SINGLE_CAND_MIN_CHAR_SCORE,
    )
    ngram_solver.set_corpus_frequencies(l_tokens)

    by_folio = extractor.extract_lang_a_by_folio()

    if verbose:
        print(f'  → {len(l_tokens)} Latin tokens, '
              f'{len(latin_skel.skeleton_index)} skeletons, '
              f'{trans_matrix.shape[0]}x{trans_matrix.shape[1]} matrix')
        print(f'  → {len(by_folio)} folios available')

    test_results = {}

    if 'unicity' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[1/5] Test 1: Unicity Distance (Procrustean Bed)')
            print('─' * 70)

        test = UnicityDistanceTest(budgeted_decoder, scaffolder, ngram_solver)
        test_results['unicity'] = test.run(
            by_folio, folio_id='f1r',
            n_trials=ADV_UNICITY_TRIALS, verbose=verbose,
        )

    if 'domain_swap' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[2/5] Test 2: Domain Swap (Texas Sharpshooter)')
            print('─' * 70)

        test = DomainSwapTest(
            budgeted_decoder, scaffolder,
            latin_skel, fuzzy_skel,
            trans_matrix, trans_vocab,
        )
        test_results['domain_swap'] = test.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    if 'polyglot' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[3/5] Test 3: Polyglot Dictionary (A Priori Latin Bias)')
            print('─' * 70)

        test = PolyglotDictTest(
            fuzzy_skel, scaffolder,
            trans_matrix, trans_vocab,
            folio_metadata=PLANT_IDS,
        )
        test_results['polyglot'] = test.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    if 'eva_collapse' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[4/5] Test 4: EVA Collapse (Transliteration Immunity)')
            print('─' * 70)

        test = EvaCollapseTest(
            budgeted_decoder, scaffolder, ngram_solver,
            latin_skel, v_morph, l_tokens,
            folio_metadata=PLANT_IDS,
            herbal_matrix=trans_matrix,
            herbal_vocab=trans_vocab,
        )
        test_results['eva_collapse'] = test.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    if 'ablation' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[5/5] Test 5: Ablation Study (Frequency Forcing)')
            print('─' * 70)

        test = AblationStudyTest(
            budgeted_decoder, scaffolder, ngram_solver,
            latin_skel, fuzzy_skel, l_tokens,
            herbal_matrix=trans_matrix,
            herbal_vocab=trans_vocab,
            folio_metadata=PLANT_IDS,
        )
        test_results['ablation'] = test.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    if 'compositionality' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[6] Compositionality Proof (Paleographic Theorem)')
            print('─' * 70)

        test = CompositionalityProof(
            budgeted_decoder, scaffolder, ngram_solver,
            latin_skel, v_morph, l_tokens,
            folio_metadata=PLANT_IDS,
            herbal_matrix=trans_matrix,
            herbal_vocab=trans_vocab,
        )
        test_results['compositionality'] = test.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    if 'dictionary_diagnostic' in run_phases:
        if verbose:
            print('\n' + '─' * 70)
            print('[D] Dictionary Coverage Diagnostic')
            print('─' * 70)

        from modules.phase12_5.dictionary_diagnostic import DictionaryDiagnostic
        diag = DictionaryDiagnostic(
            budgeted_decoder, scaffolder, ngram_solver,
            latin_skel, fuzzy_skel, l_tokens,
        )
        test_results['dictionary_diagnostic'] = diag.run(
            by_folio, folio_limit=folio_limit, verbose=verbose,
        )

    elapsed = time.time() - t0

    adversarial_results = {
        k: v for k, v in test_results.items()
        if k != 'dictionary_diagnostic'
    }
    tests_run = len(adversarial_results)
    tests_passed = sum(1 for r in adversarial_results.values() if r.get('pass'))
    defense_score = tests_passed / max(1, tests_run)

    results['test_results'] = test_results
    results['defense_summary'] = {
        'tests_run': tests_run,
        'tests_passed': tests_passed,
        'defense_score': round(defense_score, 2),
        'verdict': 'PIPELINE VALIDATED' if tests_passed == tests_run else 'ISSUES DETECTED',
    }
    results['elapsed_seconds'] = round(elapsed, 2)

    report_path = os.path.join(output_dir, 'phase12_5_adversarial.json')
    save_json(report_path, results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 12.5: ADVERSARIAL DEFENSE SUMMARY')
        print('=' * 70)
        for name, result in test_results.items():
            status = 'PASS' if result.get('pass') else 'FAIL'
            print(f'  {name:20s} → {status}')
        print(f'\n  Defense Score: {tests_passed}/{tests_run} '
              f'({defense_score:.0%})')
        print(f'  Verdict: {results["defense_summary"]["verdict"]}')
        print(f'\n  Total time: {elapsed:.1f}s')
        print(f'  Report saved: {report_path}')

    return results
