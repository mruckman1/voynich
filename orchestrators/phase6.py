"""
Phase 6 Orchestrator: Three Recovery Paths
=============================================
Entry point for Phase 6 of the Voynich Convergence Attack.

Phase 5's SAA produced zero recognizable Latin phrases. Four root causes:
  1. Surjective mapping (1001 Voynich → 629 Latin, not bijective)
  2. Synthetic corpus mismatch (TTR=0.021, H2 delta=0.345)
  3. Cost function imbalance (matrix distance dominated cribs/rank)
  4. Homophony blindspot (assumed 1:1 mapping)

Phase 6 runs three parallel recovery paths:
  Path A: Fix the SAA (improved corpus + bijection + inverted weights)
  Path B: Homophonic hypothesis (distributional similarity → merge → SAA)
  Path C: Morphological hypothesis (prefix/suffix + boundary analysis)

February 2026  ·  Voynich Convergence Attack  ·  Phase 6
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from orchestrators._utils import save_json, ensure_output_dir
from orchestrators._config import SAA_ITERATIONS_DEFAULT, LATIN_CORPUS_TOKENS_DEFAULT

from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter
from modules.phase5.tier1_matrix_builder import Tier1MatrixBuilder
from modules.phase5.rank_paired_cribs import RankPairedCribs
from modules.phase5.nmf_scaffold import NMFScaffold
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.fixed_saa import FixedSAA
from modules.phase6.homophone_detector import HomophoneDetector
from modules.phase6.homophone_merger import HomophoneMerger
from modules.phase6.reduced_saa import ReducedSAA
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from modules.phase6.boundary_analyzer import BoundaryAnalyzer


# ============================================================================
# PHASE 6 ORCHESTRATOR
# ============================================================================

def run_phase6_attack(
    paths: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/phase6',
    saa_iterations: int = SAA_ITERATIONS_DEFAULT,
    latin_corpus_size: int = LATIN_CORPUS_TOKENS_DEFAULT,
) -> Dict:
    """
    Run Phase 6: Three Recovery Paths.

    Parameters:
        paths:              Which paths to run. Default: all.
        verbose:            Print detailed output.
        output_dir:         Output directory for Phase 6 files.
        saa_iterations:     SAA iteration count.
        latin_corpus_size:  Target tokens for Latin corpus.

    Returns:
        Complete Phase 6 results dict.
    """
    if paths is None:
        paths = ['path_a', 'path_b', 'path_c']

    ensure_output_dir(output_dir)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase5_failure_summary': {
            'saa_phrases_found': 0,
            'crib_satisfaction': '36.9%',
            'corpus_ttr': 0.021,
            'h2_delta': 0.345,
            'mapping_surjective': True,
            'root_causes': [
                'Surjective mapping (1001→629)',
                'Synthetic corpus mismatch',
                'Cost function imbalance',
                'Homophony blindspot',
            ],
        },
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 6')
        print('Three Recovery Paths After SAA Failure')
        print('=' * 70)
        print(f'Paths: {paths}')
        print(f'SAA iterations: {saa_iterations:,}')
        print(f'Latin corpus target: {latin_corpus_size:,} tokens')
        print()

    # ---- Foundation: always needed ----
    if verbose:
        print('=' * 70)
        print('FOUNDATION: Language A + Tier Split + Matrix')
        print('=' * 70)

    extractor = LanguageAExtractor(verbose=verbose)
    splitter = TierSplitter(extractor)
    splitter.split()
    matrix_builder = Tier1MatrixBuilder(splitter)

    if verbose:
        stats = splitter.split()
        print(f'  Tier 1: {stats["tier1_types"]} types, '
              f'{stats["tier1_tokens"]} tokens ({stats["tier1_coverage"]:.1%})')
        print(f'  Tier 2: {stats["tier2_types"]} types, '
              f'{stats["tier2_tokens"]} tokens')

    # Shared improved corpus (needed by Path A and Path B's SAA)
    improved_corpus = None
    rank_cribs = None
    nmf_scaffold = None

    needs_corpus = 'path_a' in paths or 'path_b' in paths
    if needs_corpus:
        if verbose:
            print('\n' + '=' * 70)
            print('IMPROVED LATIN CORPUS')
            print('=' * 70)

        improved_corpus = ImprovedLatinCorpus(
            target_tokens=latin_corpus_size, verbose=verbose
        )
        corpus_results = improved_corpus.run(verbose=verbose)
        results['improved_corpus'] = corpus_results
        save_json(os.path.join(output_dir, 'improved_corpus.json'), corpus_results)

    # Recompute cribs and NMF if Path A is running
    if 'path_a' in paths and improved_corpus is not None:
        if verbose:
            print('\n' + '=' * 70)
            print('RECOMPUTED RANK-PAIRED CRIBS')
            print('=' * 70)

        rank_cribs = RankPairedCribs(splitter, improved_corpus)
        crib_results = rank_cribs.run(verbose=verbose)
        results['rank_cribs'] = crib_results
        save_json(os.path.join(output_dir, 'improved_corpus_cribs.json'),
                   crib_results)

        if verbose:
            print('\n' + '=' * 70)
            print('RECOMPUTED NMF SCAFFOLD')
            print('=' * 70)

        nmf_scaffold = NMFScaffold(extractor, improved_corpus)
        nmf_results = nmf_scaffold.run(verbose=verbose)
        results['nmf_scaffold'] = nmf_results
        save_json(os.path.join(output_dir, 'improved_corpus_nmf.json'),
                   nmf_results)

    # ==================================================================
    # PATH A: Fix the SAA
    # ==================================================================
    if 'path_a' in paths:
        if verbose:
            print('\n' + '=' * 70)
            print('PATH A: FIXED SAA (Bijection + Inverted Weights)')
            print('=' * 70)

        v_matrix, v_vocab = matrix_builder.build_voynich_matrix()
        l_matrix, l_vocab = improved_corpus.build_transition_matrix(
            top_n=len(v_vocab)
        )

        candidate_matrix = rank_cribs.get_candidate_matrix()

        saa = FixedSAA(
            voynich_matrix=v_matrix,
            voynich_vocab=v_vocab,
            latin_matrix=l_matrix,
            latin_vocab=l_vocab,
            rank_pairs=candidate_matrix,
            nmf_scaffold=nmf_scaffold,
            n_locked=50,
            alpha=0.3, beta=2.0, gamma=1.0, delta=0.2,
        )

        if verbose:
            print(f'    Voynich matrix: {len(v_vocab)}×{len(v_vocab)}')
            print(f'    Latin matrix:   {len(l_vocab)}×{len(l_vocab)}')
            print(f'    Cribs:          {len(candidate_matrix)} entries')
            print(f'    Locked:         50 top words')
            print(f'    Weights:        α=0.3 β=2.0 γ=1.0 δ=0.2')
            print(f'    Running SAA ({saa_iterations:,} iterations)...')

        saa_results = saa.run(n_iter=saa_iterations)

        # Validate
        tier1_mapping = saa_results['mapping']
        page_tokens = extractor.extract_lang_a_by_folio()
        validation = saa.validate_mapping(tier1_mapping, page_tokens)
        saa_results['validation'] = validation

        crib_eval = saa.evaluate_crib_satisfaction(tier1_mapping)
        saa_results['crib_evaluation'] = crib_eval

        all_tokens = extractor.extract_lang_a_tokens()
        decoded_sample = saa.decode_text(tier1_mapping, all_tokens[:100])
        saa_results['decoded_sample'] = decoded_sample

        results['path_a'] = saa_results

        save_json(os.path.join(output_dir, 'path_a_saa_results.json'),
                   saa_results)
        save_json(os.path.join(output_dir, 'path_a_mapping.json'),
                   tier1_mapping)

        if verbose:
            print(f'\n  Path A Results:')
            print(f'    Best cost:        {saa_results["best_cost"]:.4f}')
            print(f'    Normalized cost:  {saa_results["normalized_cost"]:.6f}')
            print(f'    Bijective:        {saa_results["is_bijective"]}')
            print(f'    Acceptance rate:  {saa_results["acceptance_rate"]:.1%}')
            print(f'    Crib satisfaction: {crib_eval["rate"]:.1%} '
                  f'({crib_eval["satisfied"]}/{crib_eval["total"]})')
            print(f'    Phrases found:    {validation["best_phrase_count"]} '
                  f'(best page: {validation["best_page"]})')
            print(f'    Decoded sample:   {decoded_sample[:120]}...')
            print(f'    --- Top Mappings ---')
            for v, l in list(tier1_mapping.items())[:10]:
                print(f'      {v} → {l}')

    # ==================================================================
    # PATH B: Homophonic Hypothesis
    # ==================================================================
    if 'path_b' in paths:
        if verbose:
            print('\n' + '=' * 70)
            print('PATH B: HOMOPHONIC HYPOTHESIS')
            print('=' * 70)

        # Step 1: Detect homophones
        if verbose:
            print('\n  --- Step 1: Homophone Detection ---')

        detector = HomophoneDetector(matrix_builder, splitter)
        detect_results = detector.run(verbose=verbose)
        results['path_b_detect'] = detect_results
        save_json(os.path.join(output_dir, 'path_b_homophones.json'),
                   detect_results)

        # Step 2: Merge and re-measure
        groups = detector.cluster_homophones()

        if len(groups) >= 5:
            if verbose:
                print('\n  --- Step 2: Merge and Re-measure ---')

            merger = HomophoneMerger(groups, splitter, extractor)
            merge_results = merger.run(verbose=verbose)
            results['path_b_merge'] = merge_results
            save_json(os.path.join(output_dir, 'path_b_merged_stats.json'),
                       merge_results)

            # Step 3: Reduced SAA (only if significant groups found)
            if len(groups) >= 10 and improved_corpus is not None:
                if verbose:
                    print('\n  --- Step 3: Reduced SAA ---')

                # Build rank pairs from the improved corpus cribs
                rp = {}
                if rank_cribs is not None:
                    rp = rank_cribs.get_candidate_matrix()

                reduced = ReducedSAA(
                    merger=merger,
                    latin_corpus=improved_corpus,
                    rank_pairs=rp,
                )
                reduced_results = reduced.run(
                    verbose=verbose, n_iter=saa_iterations
                )
                results['path_b_saa'] = reduced_results
                save_json(os.path.join(output_dir, 'path_b_saa_results.json'),
                           reduced_results)

                expanded_mapping = reduced.expand_mapping()
                if expanded_mapping:
                    save_json(os.path.join(output_dir, 'path_b_mapping.json'),
                               expanded_mapping)
            else:
                if verbose:
                    print(f'\n  Skipping reduced SAA: only {len(groups)} groups '
                          f'(need ≥ 10)')
        else:
            if verbose:
                print(f'\n  Skipping merge/SAA: only {len(groups)} groups (need ≥ 5)')

    # ==================================================================
    # PATH C: Morphological Hypothesis
    # ==================================================================
    if 'path_c' in paths:
        if verbose:
            print('\n' + '=' * 70)
            print('PATH C: MORPHOLOGICAL HYPOTHESIS')
            print('=' * 70)

        # Step 1: Morpheme analysis
        if verbose:
            print('\n  --- Step 1: Morpheme Analysis ---')

        morpheme = MorphemeAnalyzer(splitter)
        morpheme_results = morpheme.run(verbose=verbose)
        results['path_c_morpheme'] = morpheme_results
        save_json(os.path.join(output_dir, 'path_c_morphemes.json'),
                   morpheme_results)

        # Step 2: Boundary analysis
        if verbose:
            print('\n  --- Step 2: Boundary Analysis ---')

        boundary = BoundaryAnalyzer(extractor, splitter)
        boundary_results = boundary.run(verbose=verbose)
        results['path_c_boundary'] = boundary_results
        save_json(os.path.join(output_dir, 'path_c_boundaries.json'),
                   boundary_results)

    # ==================================================================
    # SYNTHESIS
    # ==================================================================
    elapsed = time.time() - t0
    conclusion = _synthesize_phase6(results)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    save_json(os.path.join(output_dir, 'phase6_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 6 CONCLUSION')
        print('=' * 70)
        for key, value in conclusion.items():
            if isinstance(value, str):
                print(f'  {key}: {value}')
            else:
                print(f'  {key}: {value}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results


# ============================================================================
# SYNTHESIS
# ============================================================================

def _synthesize_phase6(results: Dict) -> Dict:
    """
    Cross-path synthesis and decision tree.

    Evaluates all three paths and determines which model best fits the data.
    """
    conclusion = {}

    # ---- Path A evaluation ----
    path_a = results.get('path_a', {})
    a_validation = path_a.get('validation', {})
    a_phrases = a_validation.get('best_phrase_count', 0)
    a_passes = a_validation.get('passes_threshold', False)
    a_crib = path_a.get('crib_evaluation', {})
    a_crib_rate = a_crib.get('rate', 0)
    a_bijective = path_a.get('is_bijective', False)
    a_cost = path_a.get('normalized_cost', float('inf'))

    corpus_val = results.get('improved_corpus', {}).get('validation', {})
    corpus_ok = corpus_val.get('all_ok', False)

    path_a_success = a_passes and a_crib_rate > 0.6

    conclusion['path_a'] = {
        'ran': bool(path_a),
        'corpus_validation': 'PASS' if corpus_ok else 'FAIL',
        'bijective': a_bijective,
        'phrases_found': a_phrases,
        'crib_rate': f'{a_crib_rate:.1%}',
        'success': path_a_success,
        'verdict': ('SUCCESS — recognizable Latin phrases found'
                    if path_a_success else
                    f'FAIL — {a_phrases} phrases, {a_crib_rate:.1%} crib rate'),
    }

    # ---- Path B evaluation ----
    path_b_detect = results.get('path_b_detect', {})
    b_n_groups = path_b_detect.get('group_statistics', {}).get('n_groups', 0)
    b_homophony = path_b_detect.get('homophony_plausible', False)

    path_b_merge = results.get('path_b_merge', {})
    b_hypothesis = path_b_merge.get('validation', {}).get('hypothesis_supported', False)
    b_after_zipf = path_b_merge.get('comparison', {}).get('after', {}).get('zipf_exponent', 0)

    path_b_saa = results.get('path_b_saa', {})
    b_saa_phrases = path_b_saa.get('synthesis', {}).get('best_phrase_count', 0)

    homophonic_confirmed = b_homophony and b_hypothesis
    homophony_rejected = not b_homophony and b_n_groups < 10

    conclusion['path_b'] = {
        'ran': bool(path_b_detect),
        'n_groups': b_n_groups,
        'homophony_plausible': b_homophony,
        'merge_hypothesis_supported': b_hypothesis,
        'merged_zipf': f'{b_after_zipf:.3f}' if b_after_zipf else 'N/A',
        'reduced_saa_phrases': b_saa_phrases,
        'confirmed': homophonic_confirmed,
        'verdict': ('CONFIRMED — homophonic substitution detected'
                    if homophonic_confirmed else
                    'REJECTED' if homophony_rejected else
                    f'INCONCLUSIVE — {b_n_groups} groups'),
    }

    # ---- Path C evaluation ----
    path_c_morph = results.get('path_c_morpheme', {})
    c_n_affixes = path_c_morph.get('statistics', {}).get('n_productive_prefixes', 0) + \
                  path_c_morph.get('statistics', {}).get('n_productive_suffixes', 0)
    c_paradigm_cov = path_c_morph.get('statistics', {}).get('paradigm_coverage', 0)
    c_n_paradigms = path_c_morph.get('statistics', {}).get('n_paradigms', 0)
    c_morph_confirmed = path_c_morph.get('morphology_confirmed', False)

    path_c_bound = results.get('path_c_boundary', {})
    c_boundaries_artificial = path_c_bound.get('boundaries_artificial', False)

    morphological_confirmed = c_morph_confirmed

    conclusion['path_c'] = {
        'ran': bool(path_c_morph),
        'n_productive_affixes': c_n_affixes,
        'n_paradigms': c_n_paradigms,
        'paradigm_coverage': f'{c_paradigm_cov:.1%}',
        'morphology_confirmed': c_morph_confirmed,
        'boundaries_artificial': c_boundaries_artificial,
        'verdict': ('CONFIRMED — strong morphological structure'
                    if morphological_confirmed else
                    f'NOT CONFIRMED — {c_n_affixes} affixes, '
                    f'{c_paradigm_cov:.1%} paradigm coverage'),
    }

    # ---- Cross-path synthesis ----
    if path_a_success:
        overall = (
            'PATH A SUCCESS — The fixed SAA with bijection enforcement, '
            'inverted cost weights, and improved corpus produced recognizable '
            'Latin phrases. The nomenclator model is viable when the corpus '
            'and optimization are properly configured. '
            'Recommended: refine the mapping with more iterations and '
            'test against authentic medieval Latin herbal texts.'
        )
    elif homophonic_confirmed and b_saa_phrases >= 3:
        overall = (
            'PATH B SUCCESS — Homophonic cipher confirmed. Multiple Voynich '
            'words encode the same Latin word. The reduced-alphabet SAA '
            'produced readable text after merging variant groups. '
            'The original 1:1 assumption was wrong. '
            'Recommended: refine homophone groups with stricter thresholds '
            'and validate against known plant sections.'
        )
    elif homophonic_confirmed:
        overall = (
            'PATH B PARTIAL — Homophonic structure detected and statistically '
            'validated, but the reduced SAA did not yet produce coherent text. '
            'The homophonic model is the most promising direction. '
            'Recommended: Phase 7 should focus on refining homophone groups '
            'and testing with authentic Latin corpus.'
        )
    elif morphological_confirmed:
        overall = (
            'PATH C CONFIRMED — Strong morphological structure detected in '
            'Voynich words. This means word-level codebook substitution is '
            'fundamentally wrong — the cipher operates on sub-word morphemes. '
            'Recommended: Phase 7 should attack at the morpheme level, '
            'decomposing words into stems+affixes before substitution.'
        )
    elif homophony_rejected and not morphological_confirmed:
        overall = (
            'ALL PATHS INCONCLUSIVE — No path produced clear success. '
            'Path A: SAA did not find phrases. '
            'Path B: insufficient homophone groups. '
            'Path C: weak morphological structure. '
            'The nomenclator model may need a different source language, '
            'or the cipher is more complex than any model tested.'
        )
    else:
        overall = (
            'MIXED RESULTS — Partial signals from multiple paths. '
            f'Path A: {a_phrases} phrases, {a_crib_rate:.1%} cribs. '
            f'Path B: {b_n_groups} groups, homophony={b_homophony}. '
            f'Path C: {c_n_affixes} affixes, {c_paradigm_cov:.1%} paradigms. '
            'Recommended: focus on the strongest signal for Phase 7.'
        )

    conclusion['overall'] = overall

    # Ranked recommendations
    recommendations = []
    if homophonic_confirmed:
        recommendations.append('1. Refine homophone groups (Path B)')
    if path_a_success:
        recommendations.append('1. Expand SAA with more iterations (Path A)')
    if morphological_confirmed:
        recommendations.append('2. Morpheme-level attack (Path C)')
    if not recommendations:
        recommendations.append('1. Try authentic Latin corpus from digital editions')
        recommendations.append('2. Test alternative source languages (Italian, German)')
    conclusion['recommendations'] = recommendations

    return conclusion
