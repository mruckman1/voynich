"""
Phase 5 Orchestrator: Splitting the Nomenclator
==================================================
Entry point for Phase 5 of the Voynich Convergence Attack.

Phase 4 revealed Language A is a nomenclator: 74.4% of tokens use a
codebook of ~1,001 frequent words, 25.6% are ~2,761 singleton words
with cipher-like character structure. The Phase 4 SAA collapsed because
it tried to map both tiers through one system.

Phase 5 splits them apart and attacks each with the right tool:
  - Attack A: Constrained SAA on Tier 1 (codebook, ~1,001 types)
  - Attack B: Cipher tier decryption on Tier 2 (singletons, ~2,761 types)
  - Attack C: Cross-validation of combined decryption

Sub-phases:
  tier_split       — Corpus tier separation + validation
  latin_corpus     — Expanded Latin herbal corpus (20K-50K tokens)
  rank_cribs       — Rank-paired crib generation
  nmf_scaffold     — NMF topic extraction for SAA penalty
  attack_a         — Constrained SAA on Tier 1 (HIGHEST priority)
  attack_b         — Cipher tier decryption on Tier 2 (HIGH priority)
  cross_validate   — Full cross-validation (CRITICAL priority)

February 2026  ·  Voynich Convergence Attack  ·  Phase 5
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from voynich.core.utils import save_json, ensure_output_dir
from voynich.core.config import SAA_ITERATIONS_DEFAULT, LATIN_CORPUS_TOKENS_DEFAULT

from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor
from voynich.modules.phase5.tier_splitter import TierSplitter
from voynich.modules.phase5.tier1_matrix_builder import Tier1MatrixBuilder
from voynich.modules.phase5.latin_corpus_expanded import ExpandedLatinHerbalCorpus
from voynich.modules.phase5.rank_paired_cribs import RankPairedCribs
from voynich.modules.phase5.nmf_scaffold import NMFScaffold
from voynich.modules.phase5.constrained_saa import ConstrainedSAA
from voynich.modules.phase5.cipher_tier_attack import CipherTierAttack
from voynich.modules.phase5.cross_validator import CrossValidator

def run_phase5_attack(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './results/phase5',
    saa_iterations: int = SAA_ITERATIONS_DEFAULT,
    latin_corpus_size: int = LATIN_CORPUS_TOKENS_DEFAULT,
) -> Dict:
    """
    Run Phase 5: Splitting the Nomenclator — Two-Tier Decryption.

    Parameters:
        phases:            Sub-phases to run. Default: all.
        verbose:           Print detailed output.
        output_dir:        Directory for Phase 5 output files.
        saa_iterations:    SAA iteration count (100000 default, 1000 for quick).
        latin_corpus_size: Target tokens for Latin corpus (30000 default).

    Returns:
        Complete Phase 5 results dict.
    """
    if phases is None:
        phases = ['tier_split', 'latin_corpus', 'rank_cribs', 'nmf_scaffold',
                  'attack_a', 'attack_b', 'cross_validate']

    ensure_output_dir(output_dir)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase4_summary': {
            'nomenclator_confirmed': True,
            'tier1_codebook_types': '~1,001',
            'tier2_cipher_types': '~2,761',
            'h2_drop_significant': True,
            'latin_source_confirmed': True,
            'key_insight': 'Two-tier nomenclator: codebook + character-level cipher',
        },
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 5')
        print('Splitting the Nomenclator: Two-Tier Decryption')
        print('=' * 70)
        print(f'Sub-phases: {phases}')
        print(f'SAA iterations: {saa_iterations:,}')
        print(f'Latin corpus target: {latin_corpus_size:,} tokens')
        print()

    extractor = None
    splitter = None
    latin_corpus = None
    matrix_builder = None
    rank_cribs = None
    nmf_scaffold = None
    tier1_mapping = None
    tier2_mapping = None

    if verbose:
        print('\n' + '=' * 70)
        print('FOUNDATION: LANGUAGE A EXTRACTION')
        print('=' * 70)

    extractor = LanguageAExtractor(verbose=verbose)

    if 'tier_split' in phases or any(p in phases for p in
            ['rank_cribs', 'nmf_scaffold', 'attack_a', 'attack_b', 'cross_validate']):
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 1: TIER SPLIT — Corpus Tier Separation')
            print('=' * 70)

        splitter = TierSplitter(extractor)
        tier_results = splitter.run(verbose=verbose)
        results['tier_split'] = tier_results

        save_json(os.path.join(output_dir, 'tier_split.json'), tier_results)

        if verbose:
            print('\n  --- Building Tier 1 Transition Matrix ---')
        matrix_builder = Tier1MatrixBuilder(splitter)
        matrix_results = matrix_builder.run(verbose=verbose)
        results['tier1_matrix'] = matrix_results

    if splitter is None:
        splitter = TierSplitter(extractor)
        splitter.split()
    if matrix_builder is None:
        matrix_builder = Tier1MatrixBuilder(splitter)

    if 'latin_corpus' in phases or any(p in phases for p in
            ['rank_cribs', 'nmf_scaffold', 'attack_a']):
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 2: EXPANDED LATIN HERBAL CORPUS')
            print('=' * 70)

        latin_corpus = ExpandedLatinHerbalCorpus(
            target_tokens=latin_corpus_size, verbose=verbose
        )
        latin_results = latin_corpus.run(verbose=verbose)
        results['latin_corpus'] = latin_results

        save_json(os.path.join(output_dir, 'latin_corpus_expanded.json'),
                   latin_results)

    if latin_corpus is None:
        latin_corpus = ExpandedLatinHerbalCorpus(
            target_tokens=latin_corpus_size, verbose=False
        )

    if 'rank_cribs' in phases or 'attack_a' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 3: RANK-PAIRED CRIBS')
            print('=' * 70)

        rank_cribs = RankPairedCribs(splitter, latin_corpus)
        crib_results = rank_cribs.run(verbose=verbose)
        results['rank_cribs'] = crib_results

        save_json(os.path.join(output_dir, 'rank_paired_cribs.json'),
                   crib_results)

    if rank_cribs is None:
        rank_cribs = RankPairedCribs(splitter, latin_corpus)

    if 'nmf_scaffold' in phases or 'attack_a' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 4: NMF TOPIC SCAFFOLD')
            print('=' * 70)

        nmf_scaffold = NMFScaffold(extractor, latin_corpus)
        nmf_results = nmf_scaffold.run(verbose=verbose)
        results['nmf_scaffold'] = nmf_results

        save_json(os.path.join(output_dir, 'nmf_scaffold.json'), nmf_results)

    if 'attack_a' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 5: ATTACK A — CONSTRAINED SAA ON TIER 1 (HIGHEST)')
            print('=' * 70)

        v_matrix, v_vocab = matrix_builder.build_voynich_matrix()
        l_matrix, l_vocab = latin_corpus.build_transition_matrix(
            top_n=len(v_vocab)
        )

        candidate_matrix = rank_cribs.get_candidate_matrix()

        saa = ConstrainedSAA(
            voynich_matrix=v_matrix,
            voynich_vocab=v_vocab,
            latin_matrix=l_matrix,
            latin_vocab=l_vocab,
            rank_pairs=candidate_matrix,
            nmf_scaffold=nmf_scaffold,
            alpha=1.0,
            beta=0.5,
            gamma=0.3,
            delta=0.2,
        )

        if verbose:
            print(f'    Voynich matrix: {len(v_vocab)}x{len(v_vocab)}')
            print(f'    Latin matrix:   {len(l_vocab)}x{len(l_vocab)}')
            print(f'    Cribs:          {len(candidate_matrix)} entries')
            print(f'    Iterations:     {saa_iterations:,}')
            print(f'    Running SAA...')

        saa_results = saa.run(n_iter=saa_iterations)

        tier1_mapping = saa_results['mapping']
        page_tokens = extractor.extract_lang_a_by_folio()
        validation = saa.validate_mapping(tier1_mapping, page_tokens)
        saa_results['validation'] = validation

        crib_eval = saa.evaluate_crib_satisfaction(tier1_mapping)
        saa_results['crib_evaluation'] = crib_eval

        all_tokens = extractor.extract_lang_a_tokens()
        decoded_sample = saa.decode_text(tier1_mapping, all_tokens[:100])
        saa_results['decoded_sample'] = decoded_sample

        results['attack_a'] = saa_results

        save_json(os.path.join(output_dir, 'attack_a_saa_results.json'),
                   saa_results)
        save_json(os.path.join(output_dir, 'attack_a_mapping.json'),
                   tier1_mapping)

        if verbose:
            print(f'\n  Attack A Results:')
            print(f'    Best cost:       {saa_results["best_cost"]:.4f}')
            print(f'    Normalized cost: {saa_results["normalized_cost"]:.6f}')
            print(f'    Crib satisfaction: {crib_eval["rate"]:.1%} '
                  f'({crib_eval["satisfied"]}/{crib_eval["total"]})')
            print(f'    Phrases found:   {validation["best_phrase_count"]} '
                  f'(best page: {validation["best_page"]})')
            print(f'    Passes threshold: {validation["passes_threshold"]}')
            print(f'    --- Decoded Sample (first 100 tokens) ---')
            print(f'    {decoded_sample[:200]}...')
            print(f'    --- Sample Mappings ---')
            for v, l in list(tier1_mapping.items())[:10]:
                print(f'      {v} → {l}')

    if 'attack_b' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 6: ATTACK B — CIPHER TIER DECRYPTION (HIGH)')
            print('=' * 70)

        if tier1_mapping is None:
            mapping_path = os.path.join(output_dir, 'attack_a_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    tier1_mapping = json.load(f)
            else:
                tier1_mapping = {}
                if verbose:
                    print('    WARNING: No Tier 1 mapping available. '
                          'Run attack_a first.')

        cipher_attack = CipherTierAttack(splitter, tier1_mapping, extractor)
        cipher_results = cipher_attack.run(verbose=verbose)
        results['attack_b'] = cipher_results

        tier2_mapping = {}
        pattern_data = cipher_results.get('pattern_matching', {})
        for match in pattern_data.get('sample_matches', []):
            if match.get('top') and match.get('n_candidates', 0) == 1:
                tier2_mapping[match['singleton']] = match['top'][0]

        save_json(os.path.join(output_dir, 'attack_b_cipher_results.json'),
                   cipher_results)
        save_json(os.path.join(output_dir, 'attack_b_mapping.json'),
                   tier2_mapping)

    if 'cross_validate' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 7: CROSS-VALIDATION (CRITICAL)')
            print('=' * 70)

        if tier1_mapping is None:
            mapping_path = os.path.join(output_dir, 'attack_a_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    tier1_mapping = json.load(f)
            else:
                tier1_mapping = {}

        if tier2_mapping is None:
            mapping_path = os.path.join(output_dir, 'attack_b_mapping.json')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    tier2_mapping = json.load(f)
            else:
                tier2_mapping = {}

        validator = CrossValidator(
            tier1_mapping, tier2_mapping, extractor, splitter
        )
        cross_results = validator.run(verbose=verbose)
        results['cross_validation'] = cross_results

        save_json(os.path.join(output_dir, 'cross_validation.json'),
                   cross_results)

    elapsed = time.time() - t0
    conclusion = _synthesize_phase5(results)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    save_json(os.path.join(output_dir, 'phase5_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 5 CONCLUSION')
        print('=' * 70)
        for key, value in conclusion.items():
            print(f'  {key}: {value}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results

def _synthesize_phase5(results: Dict) -> Dict:
    """
    Synthesize Phase 5 findings into a conclusion using the decision tree.

    Decision tree:
    IF attack_a cost < threshold AND cross_validation passes
      → DECRYPTION CANDIDATE
    ELIF attack_a cost < threshold BUT cross_validation fails
      → PARTIAL SUCCESS
    ELIF attack_a cost > threshold AND rank correlation high
      → CORPUS INSUFFICIENT
    ELSE
      → NOMENCLATOR MODEL REJECTED
    """
    conclusion = {}

    tier_split = results.get('tier_split', {})
    tier_synth = tier_split.get('synthesis', {})
    conclusion['tier_split'] = tier_synth.get('conclusion', 'not tested')

    latin = results.get('latin_corpus', {})
    conclusion['latin_corpus'] = (
        f'{latin.get("actual_tokens", "?")} tokens, '
        f'{latin.get("vocabulary_size", "?")} types'
    ) if latin else 'not built'

    attack_a = results.get('attack_a', {})
    a_cost = attack_a.get('normalized_cost', float('inf'))
    a_validation = attack_a.get('validation', {})
    a_phrases = a_validation.get('best_phrase_count', 0)
    a_passes = a_validation.get('passes_threshold', False)
    a_crib = attack_a.get('crib_evaluation', {})
    a_crib_rate = a_crib.get('rate', 0)

    conclusion['attack_a_cost'] = f'{a_cost:.6f}'
    conclusion['attack_a_phrases'] = f'{a_phrases} ({"PASS" if a_passes else "FAIL"})'
    conclusion['attack_a_crib_rate'] = f'{a_crib_rate:.1%}'

    attack_b = results.get('attack_b', {})
    b_synth = attack_b.get('synthesis', {})
    conclusion['attack_b'] = b_synth.get('conclusion', 'not tested')

    cross = results.get('cross_validation', {})
    cross_overall = cross.get('overall', {})
    cross_passes = cross_overall.get('overall_pass', False)
    cross_n = cross_overall.get('n_passed', 0)
    conclusion['cross_validation'] = (
        f'{cross_n}/5 checks passed: '
        f'{"OVERALL PASS" if cross_passes else "OVERALL FAIL"}'
    ) if cross_overall else 'not tested'

    rank = results.get('rank_cribs', {})
    rank_val = rank.get('validation', {})
    rank_corr = rank_val.get('rank_correlation', 0)

    cost_threshold = 0.1
    cost_ok = a_cost < cost_threshold

    if cost_ok and cross_passes and a_passes:
        conclusion['overall'] = (
            'DECRYPTION CANDIDATE — Two-tier nomenclator over Latin herbal text. '
            f'Attack A produced a mapping with normalized cost {a_cost:.6f}, '
            f'{a_phrases} recognizable Latin phrases on the best page, '
            f'and cross-validation passed {cross_n}/5 checks. '
            'The decoded text should be reviewed for full linguistic coherence.'
        )
    elif cost_ok and not cross_passes:
        conclusion['overall'] = (
            'PARTIAL SUCCESS — Tier 1 mapping is plausible '
            f'(cost={a_cost:.6f}) but cross-validation failed '
            f'({cross_n}/5 checks). Refine the Latin corpus, '
            'adjust crib constraints, or increase SAA iterations.'
        )
    elif not cost_ok and rank_corr > 0.8:
        conclusion['overall'] = (
            f'CORPUS INSUFFICIENT — SAA cost ({a_cost:.6f}) exceeds threshold '
            f'but rank correlation is high ({rank_corr:.3f}). The Latin '
            'reference corpus may not be representative of the actual source text. '
            'Try a larger or more varied corpus.'
        )
    else:
        conclusion['overall'] = (
            f'NOMENCLATOR MODEL NEEDS REFINEMENT — SAA cost {a_cost:.6f}, '
            f'rank correlation {rank_corr:.3f}, '
            f'cross-validation {cross_n}/5. '
            'The two-tier nomenclator model may need different parameters, '
            'a different tier boundary, or an alternative source language.'
        )

    return conclusion
