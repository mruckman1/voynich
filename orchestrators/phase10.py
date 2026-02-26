"""
Phase 10 Orchestrator: Dictionary-Guided Trigram Decoder
==========================================================
The culminating translation engine.
Fixes the "phonotactic soup" of Phase 9 and the "mode collapse" of Phase 8
by enforcing a strict rule: All decoded Voynich words must map to
VALID Medieval Latin words from our expanded herbal dictionary, guided
by Morphological (Prefix/Suffix) abbreviation rules and Trigram context.

February 2026  ·  Voynich Convergence Attack  ·  Phase 10 (Final Translation)
"""
import os
import json
import time
from datetime import datetime
from typing import Dict

from orchestrators._utils import ensure_output_dir
from orchestrators._config import LATIN_CORPUS_TOKENS_LARGE, FOLIO_LIMIT_DEMO
from orchestrators._foundation import build_morphological_context

from modules.phase10.latin_dictionary_decoder import LatinDictionaryDecoder, viterbi_trigram_decode


def run_phase10_final_translation(verbose: bool = True, output_dir: str = './output/phase10') -> Dict:
    ensure_output_dir(output_dir)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 10 (FINAL)')
        print('Dictionary-Guided Trigram Translation Engine')
        print('=' * 70)

    # 1. Setup Corpora
    if verbose: print('\n[1/3] Loading Corpora & Morphological Parsers...')
    ctx = build_morphological_context(
        verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE
    )

    # 2. Build Dictionary Decoder
    if verbose: print('\n[2/3] Compiling Latin Herbal Dictionary & Trigram Model...')
    decoder = LatinDictionaryDecoder(ctx.latin_tokens)

    if verbose:
        print(f"  → Indexed {len(decoder.vocab)} unique Latin herbal words.")

    # 3. Decode Folios
    if verbose: print('\n[3/3] Trigram Viterbi Decoding of Folios...')
    by_folio = ctx.extractor.extract_lang_a_by_folio()
    translations = {}

    for folio, tokens in list(by_folio.items())[:FOLIO_LIMIT_DEMO]:  # Process first 10 for demonstration
        if len(tokens) < 5: continue
        translated_text = viterbi_trigram_decode(tokens, ctx.voynich_morphemer, decoder)
        translations[folio] = translated_text

    elapsed = time.time() - t0

    results = {
        'timestamp': datetime.now().isoformat(),
        'decoder_stats': {
            'latin_vocabulary_size': len(decoder.vocab),
            'total_latin_tokens': decoder.total_words
        },
        'translations': translations
    }

    with open(os.path.join(output_dir, 'phase10_final_translations.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 10: FINAL DECODED TRANSLATIONS')
        print('=' * 70)
        for folio, text in translations.items():
            print(f"\n[Folio {folio}]:\n{text[:300]}...")

        print(f"\nTotal time: {elapsed:.1f}s")
        print("\nSUCCESS. Full translations saved to output/phase10/phase10_final_translations.json")

    return results
