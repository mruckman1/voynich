"""
Phase 8 Orchestrator: The Viterbi Translation Engine
======================================================
Transitions from cryptanalysis to applied translation.
Takes the morphological dictionaries generated in Phase 7 and applies
Hidden Markov Model (HMM) Viterbi decoding to fix contextual translation
errors. It then synthesizes the Latin stems and affixes into readable
Medieval Latin text.

February 2026  ·  Voynich Convergence Attack  ·  Phase 8
"""
import sys
import os
import json
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from modules.phase7.voynich_morphemer import VoynichMorphemer
from modules.phase7.latin_morphology import LatinMorphologyParser

from modules.phase8.viterbi_decoder import ViterbiDecoder
from modules.phase8.morphological_synthesizer import MorphologicalSynthesizer
from modules.phase8.folio_translator import FolioTranslator

def load_phase7_data(filepath: str = './output/phase7/phase7_report.json') -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def run_phase8_translation(verbose: bool = True, output_dir: str = './output/phase8') -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 8')
        print('Viterbi Translation & Morphological Synthesis')
        print('=' * 70)

    # 1. Load Phase 7 Dictionaries & Rebuild Corpora
    if verbose: print('\n[1/4] Loading Context and Dictionaries...')
    p7_data = load_phase7_data()
    base_stem_map = p7_data['stem_saa']['best_mapping']
    base_affix_map = p7_data['affix_alignment']['affix_map']

    extractor = LanguageAExtractor(verbose=False)
    splitter = TierSplitter(extractor)
    splitter.split()
    m_analyzer = MorphemeAnalyzer(splitter)
    p6_morph = m_analyzer.run(verbose=False)
    v_morph = VoynichMorphemer(splitter, p6_morph)
    v_morph.process_corpus()

    l_corpus = ImprovedLatinCorpus(target_tokens=30000, verbose=False)
    l_parser = LatinMorphologyParser(l_corpus)
    l_parser.process_corpus()

    # 2. Viterbi Contextual Smoothing
    if verbose: print('\n[2/4] Running HMM Viterbi Decoder for Context Correction...')
    decoder = ViterbiDecoder(base_stem_map, v_morph, l_parser)
    refined_stem_map, viterbi_stats = decoder.run_viterbi_smoothing()

    if verbose:
        corrections = viterbi_stats['total_corrections']
        print(f"  → Viterbi smoothed {corrections} contextual stemming errors")

    # 3. Morphological Synthesis
    if verbose: print('\n[3/4] Initializing Latin Morphological Synthesizer...')
    synthesizer = MorphologicalSynthesizer(base_affix_map)

    # 4. Folio Translation
    if verbose: print('\n[4/4] Translating Complete Folios...')
    translator = FolioTranslator(extractor, v_morph, refined_stem_map, synthesizer)
    folio_translations = translator.translate_all_folios()

    elapsed = time.time() - t0

    results = {
        'viterbi_stats': viterbi_stats,
        'refined_stem_map': refined_stem_map,
        'translations': folio_translations
    }

    with open(os.path.join(output_dir, 'phase8_translations.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 8 TRANSLATION SAMPLES')
        print('=' * 70)
        for folio, text in list(folio_translations.items())[:3]:
            print(f"[{folio}]: {text[:150]}...")
        print(f"\nTotal time: {elapsed:.1f}s")
        print("Full translations saved to output/phase8/phase8_translations.json")

    return results

if __name__ == '__main__':
    run_phase8_translation()
