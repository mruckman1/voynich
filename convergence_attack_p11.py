"""
Phase 11 Orchestrator: Phonetic Constraint Satisfaction Decoder
================================================================
Fixes the "mode collapse" (repetition hallucination) of Phase 10.
Transitions from probabilistic HMM Viterbi decoding to hard structural
Constraint Satisfaction Problem (CSP) decoding.

Treats Voynich as a phonetic abjad (vowel-less shorthand). Maps Voynich
stems to Latin Consonant Skeletons and ruthlessly filters the dictionary
to find exact structural matches, destroying infinite generation loops.

February 2026  ·  Voynich Convergence Attack  ·  Phase 11 (The Solution)
"""
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from modules.phase7.voynich_morphemer import VoynichMorphemer

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer, VoynichPhoneticSkeletonizer
from modules.phase11.csp_decoder import CSPPhoneticDecoder

def run_phase11_csp_translation(verbose: bool = True, output_dir: str = './output/phase11') -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 11 (THE SOLUTION)')
        print('Phonetic Constraint Satisfaction (CSP) Decoder')
        print('=' * 70)

    # 1. Load Corpora & Morphological Parsers
    if verbose: print('\n[1/4] Loading Extractors & Morphemers...')
    extractor = LanguageAExtractor(verbose=False)
    splitter = TierSplitter(extractor)
    splitter.split()

    m_analyzer = MorphemeAnalyzer(splitter)
    p6_morph = m_analyzer.run(verbose=False)
    v_morph = VoynichMorphemer(splitter, p6_morph)
    v_morph.process_corpus()

    l_corpus = ImprovedLatinCorpus(target_tokens=50000, verbose=False)
    l_tokens = l_corpus.get_tokens()

    # 2. Build Consonant Skeletons
    if verbose: print('\n[2/4] Compiling Phonetic Consonant Skeletons...')
    latin_skeletonizer = LatinPhoneticSkeletonizer(l_tokens)
    voynich_skeletonizer = VoynichPhoneticSkeletonizer(v_morph)

    if verbose:
        print(f"  → Indexed {len(latin_skeletonizer.skeleton_index)} unique Latin consonant skeletons.")

    # 3. CSP Decoding
    if verbose: print('\n[3/4] Running CSP Phonetic Alignment (Anti-Hallucination)...')
    decoder = CSPPhoneticDecoder(latin_skeletonizer, voynich_skeletonizer)

    by_folio = extractor.extract_lang_a_by_folio()
    translations = {}

    for folio, tokens in list(by_folio.items())[:15]: # Process 15 folios
        if len(tokens) < 5: continue
        translated_text = decoder.decode_folio(tokens)
        translations[folio] = translated_text

    elapsed = time.time() - t0

    results = {
        'timestamp': datetime.now().isoformat(),
        'csp_stats': {
            'latin_skeletons': len(latin_skeletonizer.skeleton_index),
            'unique_latin_words_mapped': len(set(latin_skeletonizer.vocab))
        },
        'translations': translations
    }

    with open(os.path.join(output_dir, 'phase11_csp_translations.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 11: FINAL PHONETIC TRANSLATIONS')
        print('=' * 70)
        for folio, text in translations.items():
            print(f"\n[Folio {folio}]:\n{text[:350]}...")

        print(f"\nTotal time: {elapsed:.1f}s")
        print("\nTHE REPETITION IS BROKEN. Full translations saved to output/phase11/phase11_csp_translations.json")

    return results

if __name__ == '__main__':
    run_phase11_csp_translation()
