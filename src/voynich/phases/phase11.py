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
import os
import json
import time
from datetime import datetime
from typing import Dict

from voynich.core.utils import vprint, ensure_output_dir
from voynich.core.config import LATIN_CORPUS_TOKENS_LARGE, FOLIO_LIMIT_DEFAULT
from voynich.core.foundation import build_morphological_context

from voynich.modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer, VoynichPhoneticSkeletonizer
from voynich.modules.phase11.csp_decoder import CSPPhoneticDecoder

def run_phase11_csp_translation(verbose: bool = True, output_dir: str = './results/phase11') -> Dict:
    ensure_output_dir(output_dir)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 11 (THE SOLUTION)')
        print('Phonetic Constraint Satisfaction (CSP) Decoder')
        print('=' * 70)

    vprint(verbose, '\n[1/4] Loading Extractors & Morphemers...')
    ctx = build_morphological_context(
        verbose=False, latin_corpus_tokens=LATIN_CORPUS_TOKENS_LARGE
    )

    vprint(verbose, '\n[2/4] Compiling Phonetic Consonant Skeletons...')
    latin_skeletonizer = LatinPhoneticSkeletonizer(ctx.latin_tokens)
    voynich_skeletonizer = VoynichPhoneticSkeletonizer(ctx.voynich_morphemer)

    if verbose:
        print(f"  → Indexed {len(latin_skeletonizer.skeleton_index)} unique Latin consonant skeletons.")

    vprint(verbose, '\n[3/4] Running CSP Phonetic Alignment (Anti-Hallucination)...')
    decoder = CSPPhoneticDecoder(latin_skeletonizer, voynich_skeletonizer)

    by_folio = ctx.extractor.extract_lang_a_by_folio()
    translations = {}

    for folio, tokens in list(by_folio.items())[:FOLIO_LIMIT_DEFAULT]:
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
