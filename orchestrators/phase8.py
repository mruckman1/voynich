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
import json
import os
import time
from typing import Dict, List

from orchestrators._utils import ensure_output_dir
from orchestrators._config import LATIN_CORPUS_TOKENS_DEFAULT
from orchestrators._foundation import build_morphological_context

from modules.phase7.latin_morphology import LatinMorphologyParser
from modules.phase8.viterbi_decoder import ViterbiDecoder
from modules.phase8.morphological_synthesizer import MorphologicalSynthesizer
from modules.phase8.folio_translator import FolioTranslator


def load_phase7_data(filepath: str = './output/phase7/phase7_report.json') -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def run_phase8_translation(verbose: bool = True, output_dir: str = './output/phase8') -> Dict:
    ensure_output_dir(output_dir)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 8')
        print('Viterbi Translation & Morphological Synthesis')
        print('=' * 70)

    # 1. Load Phase 7 Dictionaries & Rebuild Corpora
    if verbose: print('\n[1/4] Loading Context and Dictionaries...')
    # Derive Phase 7 path from output_dir (sibling directory)
    parent_dir = os.path.dirname(output_dir)
    p7_path = os.path.join(parent_dir, 'phase7', 'phase7_report.json')
    p7_data = load_phase7_data(p7_path)
    base_stem_map = p7_data['stem_saa']['best_mapping']
    base_affix_map = p7_data['affix_alignment']['affix_map']

    ctx = build_morphological_context(
        verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_DEFAULT
    )

    l_parser = LatinMorphologyParser(ctx.latin_corpus)
    l_parser.process_corpus()

    # 2. Viterbi Contextual Smoothing
    if verbose: print('\n[2/4] Running HMM Viterbi Decoder for Context Correction...')
    decoder = ViterbiDecoder(base_stem_map, ctx.voynich_morphemer, l_parser)
    refined_stem_map, viterbi_stats = decoder.run_viterbi_smoothing()

    if verbose:
        corrections = viterbi_stats['total_corrections']
        print(f"  → Viterbi smoothed {corrections} contextual stemming errors")

    # 3. Morphological Synthesis
    if verbose: print('\n[3/4] Initializing Latin Morphological Synthesizer...')
    synthesizer = MorphologicalSynthesizer(base_affix_map)

    # 4. Folio Translation
    if verbose: print('\n[4/4] Translating Complete Folios...')
    translator = FolioTranslator(ctx.extractor, ctx.voynich_morphemer, refined_stem_map, synthesizer)
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
        print(f"Full translations saved to {os.path.join(output_dir, 'phase8_translations.json')}")

    return results
