"""
Phase 7 Orchestrator: The Morphological Sub-Word Attack
=========================================================
Phase 6 proved that Voynich words are highly compositional (81% paradigm
coverage, 123 productive affixes). Whole-word codebooks failed because
"calida" and "calidam" were being treated as entirely different cipher
tokens, destroying the transition matrix structure.

Phase 7 executes a sub-word attack:
  1. voynich_morphemer  — Strips productive affixes, isolating semantic stems.
  2. latin_morphology   — Parses expanded Latin corpus into (Stem + Inflection).
  3. stem_saa           — Runs the matrix matching strictly on the STEMS.
  4. affix_aligner      — Maps V-affixes to L-affixes via bipartite matching.

February 2026  ·  Voynich Convergence Attack  ·  Phase 7
"""
import time
from datetime import datetime
from typing import Dict

from voynich.core.utils import vprint, save_json, ensure_output_dir
from voynich.core.config import LATIN_CORPUS_TOKENS_DEFAULT

from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor
from voynich.modules.phase5.tier_splitter import TierSplitter
from voynich.modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from voynich.modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from voynich.modules.phase7.voynich_morphemer import VoynichMorphemer
from voynich.modules.phase7.latin_morphology import LatinMorphologyParser
from voynich.modules.phase7.stem_saa import StemSAA
from voynich.modules.phase7.affix_aligner import AffixAligner

def run_phase7_attack(verbose: bool = True, output_dir: str = './results/phase7') -> Dict:
    ensure_output_dir(output_dir)
    t0 = time.time()

    results = {
        'timestamp': datetime.now().isoformat(),
        'phase6_synthesis': {
            'conclusion': 'Word-level mapping abandoned. Proceeding with sub-word morphological mapping.',
            'paradigm_coverage': 0.81
        }
    }

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 7')
        print('The Morphological Sub-Word Attack')
        print('=' * 70)

    extractor = LanguageAExtractor(verbose=False)
    splitter = TierSplitter(extractor)
    splitter.split()

    vprint(verbose, '\n[1/4] Isolating Voynich Semantic Stems...')
    m_analyzer = MorphemeAnalyzer(splitter)
    p6_morph_results = m_analyzer.run(verbose=False)

    v_morphemer = VoynichMorphemer(splitter, p6_morph_results)
    v_morph_stats = v_morphemer.process_corpus()
    results['voynich_morphology'] = v_morph_stats

    if verbose:
        print(f"  → Reduced {v_morph_stats['original_vocab_size']} word types to "
              f"{v_morph_stats['unique_stems']} pure semantic stems.")

    vprint(verbose, '\n[2/4] Parsing Latin Herbal Morphology...')
    l_corpus = ImprovedLatinCorpus(target_tokens=LATIN_CORPUS_TOKENS_DEFAULT, verbose=False)
    l_parser = LatinMorphologyParser(l_corpus)
    l_morph_stats = l_parser.process_corpus()
    results['latin_morphology'] = l_morph_stats

    if verbose:
        print(f"  → Reduced {l_morph_stats['original_vocab_size']} Latin word forms to "
              f"{l_morph_stats['unique_stems']} semantic stems.")

    vprint(verbose, '\n[3/4] Running SAA on Isolated Stems (Semantic Core)...')
    stem_saa = StemSAA(v_morphemer, l_parser)
    saa_results = stem_saa.run(n_iter=50000, verbose=verbose)
    results['stem_saa'] = saa_results

    vprint(verbose, '\n[4/4] Aligning Grammatical Affixes...')
    aligner = AffixAligner(v_morphemer, l_parser, saa_results['best_mapping'])
    affix_results = aligner.run()
    results['affix_alignment'] = affix_results

    if verbose:
        print(f"  → Mapped {affix_results['n_voynich_suffixes_mapped']} "
              f"Voynich suffixes to Latin inflections.")

    elapsed = time.time() - t0

    import os
    save_json(os.path.join(output_dir, 'phase7_report.json'), results)

    if verbose:
        print('\n' + '=' * 70)
        print('PHASE 7 CONCLUSION')
        print('=' * 70)
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Stem Mapping Quality (Normalized Cost): {saa_results['normalized_cost']:.4f}")
        print(f"  Decoded Sample:\n    {affix_results['decoded_sample']}")

    return results
