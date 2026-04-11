"""
Phase 9 Orchestrator: The Syllabic & Sigla Translation Engine
===============================================================
Phase 8 proved that space-delimited Voynich tokens are too small to be
semantic roots (collapsing into single characters like 'o', 'a', 'y').
They are phonotactic syllables and scribal abbreviations (Sigla).

This phase breaks the Latin corpus into syllables, maps Voynich prefixes/suffixes
to specific medieval scribal abbreviations, and uses character/syllable-level
Beam Search to prevent the HMM mode-collapse seen in Phase 8.

February 2026  ·  Voynich Convergence Attack  ·  Phase 9
"""
import os
import json
import time
from datetime import datetime
from typing import Dict

from voynich.core.utils import vprint, ensure_output_dir
from voynich.core.config import LATIN_CORPUS_TOKENS_DEFAULT, BEAM_WIDTH_DEFAULT

from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor
from voynich.modules.phase5.tier_splitter import TierSplitter
from voynich.modules.phase6.improved_latin_corpus import ImprovedLatinCorpus

from voynich.modules.phase9.latin_syllabifier import LatinSyllabifier
from voynich.modules.phase9.sigla_mapper import SiglaMapper
from voynich.modules.phase9.beam_search_decoder import SyllabicBeamSearch

def run_phase9_attack(verbose: bool = True, output_dir: str = './results/phase9') -> Dict:
    ensure_output_dir(output_dir)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 9')
        print('Syllabic & Sigla Translation Engine')
        print('=' * 70)

    vprint(verbose, '\n[1/4] Establishing Baseline Corpora...')
    extractor = LanguageAExtractor(verbose=False)
    splitter = TierSplitter(extractor)
    splitter.split()

    v_tokens = splitter.get_tier1_tokens()

    l_corpus = ImprovedLatinCorpus(target_tokens=LATIN_CORPUS_TOKENS_DEFAULT, verbose=False)
    l_tokens = l_corpus.get_tokens()

    vprint(verbose, '\n[2/4] Breaking Latin Corpus into Syllables...')
    syllabifier = LatinSyllabifier()
    l_syllables, syllable_transitions = syllabifier.process_corpus(l_tokens)

    if verbose:
        print(f"  → Extracted {len(l_syllables)} unique Latin syllables.")

    vprint(verbose, '\n[3/4] Mapping Voynich Affixes to Medieval Sigla...')
    sigla_mapper = SiglaMapper(v_tokens, l_syllables)
    sigla_constraints = sigla_mapper.generate_mappings()

    if verbose:
        print(f"  → Mapped {len(sigla_constraints)} rigid scribal abbreviations.")

    vprint(verbose, '\n[4/4] Running Beam Search Decoding (Anti-Mode-Collapse)...')
    decoder = SyllabicBeamSearch(
        voynich_tokens=v_tokens,
        syllable_transitions=syllable_transitions,
        sigla_constraints=sigla_constraints,
        beam_width=BEAM_WIDTH_DEFAULT
    )

    decoded_text, search_stats = decoder.decode(max_tokens=200)

    elapsed = time.time() - t0

    results = {
        'timestamp': datetime.now().isoformat(),
        'latin_syllable_stats': {
            'unique_syllables': len(l_syllables),
            'top_syllables': l_syllables[:20]
        },
        'sigla_constraints': sigla_constraints,
        'beam_search_stats': search_stats,
        'decoded_sample': decoded_text
    }

    with open(os.path.join(output_dir, 'phase9_report.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 9 DECODED OUTPUT (First 200 Syllables)')
        print('=' * 70)
        print(f"\n{decoded_text[:1000]}")
        print(f"\nTotal time: {elapsed:.1f}s")
        print("Report saved to output/phase9/phase9_report.json")

    return results
