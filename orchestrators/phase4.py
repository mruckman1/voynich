"""
Phase 4 Orchestrator: Language A Decryption Attack
====================================================
Entry point for Phase 4 of the Voynich Convergence Attack.

Phase 3 proved Language B is a 13-word notation system. Language A
(57 words, H2=1.487) remains unsolved. Key insight: deltaH2=0 means
Voynich words are NOT character-level encryptions — they are generated
at the WORD level.

Sub-phases:
  extraction     — Full-corpus Language A isolation + validation
  latin_corpus   — Latin herbal reference corpus construction
  model_a1       — Whole-word codebook test (HIGHEST priority)
  model_a2       — Nomenclator test (HIGH priority)
  model_a3       — Semantic compression test (MEDIUM priority)
  botanical      — Known-plaintext botanical attack
  saa            — Successor alphabet attack (transition matrix matching)
  gradient       — Entropy gradient by page position
  multi_lang     — Multi-language source testing
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

from orchestrators._utils import save_json, ensure_output_dir

from modules.phase4.lang_a_extractor import LanguageAExtractor, LANG_A_FULL_TARGETS
from modules.phase4.latin_herbal_corpus import LatinHerbalCorpus
from modules.phase4.model_a1_codebook import WholeWordCodebook
from modules.phase4.model_a2_nomenclator import NomenclatorModel
from modules.phase4.model_a3_semantic_compression import SemanticCompressionModel
from modules.phase4.known_plaintext_botanical import BotanicalKnownPlaintext
from modules.phase4.successor_alphabet import SuccessorAlphabetAttack
from modules.phase4.entropy_gradient_page import EntropyGradientAnalysis
from modules.phase4.multi_language_source import MultiLanguageSourceTest
from modules.phase3.lang_a_reprofiling import LANG_A_TARGETS

def run_phase4_attack(
    phases: Optional[List[str]] = None,
    verbose: bool = True,
    output_dir: str = './output/phase4',
) -> Dict:
    """
    Run Phase 4: Language A Decryption Attack.

    Parameters:
        phases: Sub-phases to run. Default: all.
        verbose: Print detailed output.
        output_dir: Directory for Phase 4 output files.

    Returns:
        Complete Phase 4 results dict.
    """
    if phases is None:
        phases = ['extraction', 'latin_corpus', 'model_a1', 'model_a2',
                  'model_a3', 'botanical', 'saa', 'gradient', 'multi_lang']

    ensure_output_dir(output_dir)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase3_summary': {
            'lang_b_solved': True,
            'lang_b_mechanism': 'notation_system (13-word Markov chain)',
            'lang_a_H2': LANG_A_TARGETS['H2'],
            'lang_a_vocabulary': LANG_A_TARGETS['vocabulary_size'],
            'key_insight': 'deltaH2=0 => word-level encoding, not character-level',
        },
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 4')
        print('The Last Wall: Language A Decryption Attack')
        print('=' * 70)
        print(f'Sub-phases: {phases}')
        print(f'Language A: {LANG_A_TARGETS["vocabulary_size"]} words, '
              f'H2={LANG_A_TARGETS["H2"]}')
        print(f'Key insight: word-level encoding (deltaH2 = 0)')
        print()

    extractor = None
    latin_corpus = None
    botanical_cribs = None

    if 'extraction' in phases or any(p in phases for p in
            ['model_a1', 'model_a2', 'model_a3', 'botanical', 'saa',
             'gradient', 'multi_lang']):
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 1: LANGUAGE A FULL-CORPUS EXTRACTION')
            print('=' * 70)

        extractor = LanguageAExtractor(verbose=verbose)
        extraction_results = extractor.run(verbose=verbose)
        results['extraction'] = extraction_results

        save_json(os.path.join(output_dir, 'lang_a_extraction.json'),
                   extraction_results)

    if 'latin_corpus' in phases or any(p in phases for p in
            ['model_a1', 'saa', 'multi_lang']):
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 2: LATIN HERBAL REFERENCE CORPUS')
            print('=' * 70)

        latin_corpus = LatinHerbalCorpus(method='auto', verbose=verbose)
        latin_results = latin_corpus.run(verbose=verbose)
        results['latin_corpus'] = latin_results

        save_json(os.path.join(output_dir, 'latin_herbal_corpus.json'),
                   latin_results)

    if extractor is None:
        extractor = LanguageAExtractor(verbose=False)
    if latin_corpus is None:
        latin_corpus = LatinHerbalCorpus(method='auto', verbose=False)

    if 'model_a1' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 3: MODEL A1 — WHOLE-WORD CODEBOOK (HIGHEST PRIORITY)')
            print('=' * 70)

        codebook = WholeWordCodebook(extractor, latin_corpus)
        a1_results = codebook.run(verbose=verbose)
        results['model_a1'] = a1_results

        save_json(os.path.join(output_dir, 'model_a1_codebook_results.json'),
                   a1_results)

    if 'model_a2' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 4: MODEL A2 — NOMENCLATOR (HIGH PRIORITY)')
            print('=' * 70)

        nomenclator = NomenclatorModel(extractor)
        a2_results = nomenclator.run(verbose=verbose)
        results['model_a2'] = a2_results

        save_json(os.path.join(output_dir, 'model_a2_nomenclator_results.json'),
                   a2_results)

    if 'model_a3' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 5: MODEL A3 — SEMANTIC COMPRESSION (MEDIUM PRIORITY)')
            print('=' * 70)

        semantic = SemanticCompressionModel(extractor)
        a3_results = semantic.run(verbose=verbose)
        results['model_a3'] = a3_results

        save_json(os.path.join(output_dir, 'model_a3_semantic_compression_results.json'),
                   a3_results)

    if 'botanical' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 6: BOTANICAL KNOWN-PLAINTEXT ATTACK (APPROACH 1)')
            print('=' * 70)

        botanical = BotanicalKnownPlaintext(extractor)
        botanical_results = botanical.run(verbose=verbose)
        results['botanical'] = botanical_results
        botanical_cribs = botanical_results.get('crib_constraints', [])

        save_json(os.path.join(output_dir, 'botanical_known_plaintext_results.json'),
                   botanical_results)

    if 'saa' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 7: SUCCESSOR ALPHABET ATTACK (APPROACH 2)')
            print('=' * 70)

        if botanical_cribs is None and 'botanical' not in phases:
            botanical = BotanicalKnownPlaintext(extractor)
            botanical_results = botanical.run(verbose=False)
            botanical_cribs = botanical_results.get('crib_constraints', [])

        saa = SuccessorAlphabetAttack(extractor, latin_corpus, botanical_cribs)
        saa_results = saa.run(verbose=verbose)
        results['saa'] = saa_results

        save_json(os.path.join(output_dir, 'saa_results.json'),
                   saa_results)

    if 'gradient' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 8: ENTROPY GRADIENT BY PAGE POSITION (APPROACH 3)')
            print('=' * 70)

        gradient = EntropyGradientAnalysis(extractor)
        gradient_results = gradient.run(verbose=verbose)
        results['gradient'] = gradient_results

        save_json(os.path.join(output_dir, 'entropy_gradient_results.json'),
                   gradient_results)

    if 'multi_lang' in phases:
        if verbose:
            print('\n' + '=' * 70)
            print('SUB-PHASE 9: MULTI-LANGUAGE SOURCE TESTING (APPROACH 4)')
            print('=' * 70)

        multi = MultiLanguageSourceTest(extractor, latin_corpus)
        multi_results = multi.run(verbose=verbose)
        results['multi_lang'] = multi_results

        save_json(os.path.join(output_dir, 'multi_language_results.json'),
                   multi_results)

    elapsed = time.time() - t0
    conclusion = _synthesize_phase4(results)
    results['conclusion'] = conclusion
    results['elapsed_seconds'] = elapsed

    save_json(os.path.join(output_dir, 'phase4_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 4 CONCLUSION')
        print('=' * 70)
        for key, value in conclusion.items():
            print(f'  {key}: {value}')
        print(f'\nTotal time: {elapsed:.1f}s')
        print(f'Results saved to {output_dir}/')

    return results

def _synthesize_phase4(results: Dict) -> Dict:
    """
    Synthesize Phase 4 findings into a conclusion.

    Key questions:
    1. Which model (A1/A2/A3) has the strongest support?
    2. Does the botanical attack produce useful cribs?
    3. Does the SAA find a plausible permutation?
    4. Does entropy gradient support header/body structure?
    5. Which source language best matches?
    6. What is the overall decryption confidence?
    """
    conclusion = {}

    a1 = results.get('model_a1', {})
    a1_synth = a1.get('synthesis', {})
    conclusion['model_a1_codebook'] = a1_synth.get('conclusion', 'not tested')
    a1_supported = a1_synth.get('codebook_supported', False)

    a2 = results.get('model_a2', {})
    a2_synth = a2.get('synthesis', {})
    conclusion['model_a2_nomenclator'] = a2_synth.get('conclusion', 'not tested')
    a2_supported = a2_synth.get('nomenclator_supported', False)

    a3 = results.get('model_a3', {})
    a3_synth = a3.get('synthesis', {})
    conclusion['model_a3_semantic'] = a3_synth.get('conclusion', 'not tested')
    a3_supported = a3_synth.get('semantic_compression_supported', False)

    bot = results.get('botanical', {})
    bot_synth = bot.get('synthesis', {})
    conclusion['botanical_cribs'] = bot_synth.get('conclusion', 'not tested')
    n_useful_cribs = bot_synth.get('useful_cribs', 0)

    saa = results.get('saa', {})
    saa_synth = saa.get('synthesis', {})
    conclusion['saa_attack'] = saa_synth.get('conclusion', 'not tested')
    saa_plausible = saa_synth.get('plausible_decryption', False)

    grad = results.get('gradient', {})
    grad_synth = grad.get('synthesis', {})
    conclusion['entropy_gradient'] = grad_synth.get('conclusion', 'not tested')

    multi = results.get('multi_lang', {})
    multi_synth = multi.get('synthesis', {})
    conclusion['best_source_language'] = multi_synth.get('best_source_language', 'not tested')
    conclusion['language_ranking'] = multi_synth.get('conclusion', 'not tested')

    if a1_supported and saa_plausible:
        conclusion['overall'] = (
            'CONVERGENT DECRYPTION CANDIDATE — Model A1 (whole-word codebook) '
            'is strongly supported AND the SAA found a plausible permutation. '
            'Language A may be a word-substitution cipher over Latin herbal text. '
            'The decoded text should be reviewed for linguistic coherence.'
        )
    elif a1_supported and n_useful_cribs > 0:
        conclusion['overall'] = (
            'CONSTRAINED CODEBOOK — Model A1 is supported and botanical '
            f'cross-referencing produced {n_useful_cribs} useful cribs. '
            'The codebook can be partially constrained but full decryption '
            'requires additional reference corpus data.'
        )
    elif a2_supported:
        conclusion['overall'] = (
            'NOMENCLATOR ARCHITECTURE — Language A appears to use a mixed '
            'encoding: word-level codes for common terms and character-level '
            'substitution for unique content. This requires a different '
            'decryption approach than pure codebook matching.'
        )
    elif a3_supported and not a1_supported:
        conclusion['overall'] = (
            'TEMPLATE SYSTEM — Language A appears to be a semantic compression '
            'system with topic-conditioned word classes. The encoding is not '
            'a simple substitution but a slot-filling template system.'
        )
    else:
        conclusion['overall'] = (
            'MODELS INSUFFICIENT — None of the three models received strong '
            'support. Language A may use an encoding mechanism not yet modeled, '
            'or the reference corpora are not representative of the source text. '
            'Consider: (1) larger reference corpora, (2) additional source '
            'languages, (3) alternative encoding models.'
        )

    return conclusion
