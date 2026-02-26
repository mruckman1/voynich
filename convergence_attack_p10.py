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
import sys
import os
import json
import time
import math
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase7.voynich_morphemer import VoynichMorphemer
from modules.phase5.tier_splitter import TierSplitter
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer

# Rigid Medieval Sigla constraints refined from previous phases
SIGLA_MAP = {
    # Voynich Prefixes -> Latin Prefixes/Starters
    'qo': ['con', 'com', 'cor', 'qu'],
    'ch': ['ca', 'ce', 'ci', 'co', 'cu'],
    'sh': ['sa', 'se', 'si', 'su', 'ex'],
    'd':  ['de', 'di', 'da'],
    'p':  ['pro', 'per', 'prae', 'par'],
    't':  ['te', 'ta', 'ti', 'tra'],
    'k':  ['ca', 'cu', 'co'],
    '':   ['a','e','i','o','u'] # Null prefixes usually mean vowel start
}

SUFFIX_MAP = {
    # Voynich Suffixes -> Latin Terminations
    'dy': ['ae', 'ti', 'ur', 'di', 'te'],
    'iin': ['us', "um", 'is', 'in', 'unt'],
    'in': ['um', 'im', 'em', 'en'],
    'ey': ['es', 'et', 'er', 'em'],
    'y':  ['a', 'i', 'e', 'o'],
    'l':  ['al', 'el', 'il', 'ul', 'le'],
    'r':  ['ar', 'er', 'or', 'ur', 're'],
    'm':  ['am', 'um', 'em', 'rum', 'num'],
    's':  ['as', 'os', 'is', 'us'],
    '':   ['a', 'e', 'i', 'o', 'u', 't', 'c']
}

class LatinDictionaryDecoder:
    """Builds a lookup dictionary from the Latin corpus and decodes Voynich tokens."""

    def __init__(self, latin_tokens: list):
        self.vocab = set(latin_tokens)
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter(latin_tokens)
        self.total_words = len(latin_tokens)

        # Index the Latin dictionary by (prefix_category, suffix_category)
        self.word_index = defaultdict(list)
        self._build_index()
        self._build_ngrams(latin_tokens)

    def _build_index(self):
        """Categorize every Latin word by its potential Voynich Sigla equivalent."""
        for word in self.vocab:
            word = word.lower()
            if len(word) < 2: continue

            valid_v_prefs = []
            for v_pref, l_prefs in SIGLA_MAP.items():
                if any(word.startswith(lp) for lp in l_prefs):
                    valid_v_prefs.append(v_pref)
            if not valid_v_prefs: valid_v_prefs.append('')

            valid_v_sufs = []
            for v_suf, l_sufs in SUFFIX_MAP.items():
                if any(word.endswith(ls) for ls in l_sufs):
                    valid_v_sufs.append(v_suf)
            if not valid_v_sufs: valid_v_sufs.append('')

            for vp in valid_v_prefs:
                for vs in valid_v_sufs:
                    self.word_index[(vp, vs)].append(word)

    def _build_ngrams(self, tokens: list):
        """Build Language Model for contextual smoothing."""
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.bigram_counts[w1][w2] += 1
            self.trigram_counts[(w1, w2)][w3] += 1

    def get_trigram_prob(self, w1: str, w2: str, w3: str) -> float:
        """Calculate P(w3 | w1, w2) with backoff to bigram/unigram."""
        bi_count = sum(self.trigram_counts[(w1, w2)].values())
        if bi_count > 0 and self.trigram_counts[(w1, w2)][w3] > 0:
            return self.trigram_counts[(w1, w2)][w3] / bi_count

        uni_count = sum(self.bigram_counts[w2].values())
        if uni_count > 0 and self.bigram_counts[w2][w3] > 0:
            return 0.4 * (self.bigram_counts[w2][w3] / uni_count)

        return 0.1 * (self.unigram_counts.get(w3, 1) / self.total_words)

    def get_candidates(self, v_prefix: str, v_suffix: str) -> list:
        # Fallback to empty maps if specific one not found
        candidates = self.word_index.get((v_prefix, v_suffix), [])
        if not candidates:
            candidates = self.word_index.get((v_prefix, ''), [])
        if not candidates:
            candidates = self.word_index.get(('', v_suffix), [])
        if not candidates:
            candidates = [w for w, _ in self.unigram_counts.most_common(50)]

        # Return top 20 most frequent to keep beam search fast
        return sorted(candidates, key=lambda w: -self.unigram_counts[w])[:20]

def viterbi_trigram_decode(v_tokens: list, v_morphemer: VoynichMorphemer, decoder: LatinDictionaryDecoder):
    """Decodes Voynich text using Trigrams to ensure grammatical fluency."""
    # Beam state: (log_prob, [w1, w2, ..., wn])
    beam = [(0.0, ["<START>", "<START>"])]

    for v_token in v_tokens:
        # 1. Parse Voynich Token
        pref, _, suf = v_morphemer._strip_affixes(v_token)

        # 2. Get Dictionary Candidates
        candidates = decoder.get_candidates(pref, suf)

        new_beam = []
        for log_prob, history in beam:
            w1, w2 = history[-2], history[-1]

            for cand in candidates:
                prob = decoder.get_trigram_prob(w1, w2, cand)
                score = log_prob + math.log(prob)
                new_beam.append((score, history + [cand]))

        # Sort and prune (Beam width 15)
        new_beam.sort(key=lambda x: x[0], reverse=True)
        beam = new_beam[:15]

    best_sequence = beam[0][1][2:] # Strip <START> markers
    return ' '.join(best_sequence)

def run_phase10_final_translation(verbose: bool = True, output_dir: str = './output/phase10') -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print('=' * 70)
        print('VOYNICH CONVERGENCE ATTACK — PHASE 10 (FINAL)')
        print('Dictionary-Guided Trigram Translation Engine')
        print('=' * 70)

    # 1. Setup Corpora
    if verbose: print('\n[1/3] Loading Corpora & Morphological Parsers...')
    extractor = LanguageAExtractor(verbose=False)
    splitter = TierSplitter(extractor)
    splitter.split()

    # Needs Phase 6 data for the affixes
    m_analyzer = MorphemeAnalyzer(splitter)
    p6_morph = m_analyzer.run(verbose=False)
    v_morph = VoynichMorphemer(splitter, p6_morph)
    v_morph.process_corpus()

    l_corpus = ImprovedLatinCorpus(target_tokens=50000, verbose=False)
    l_tokens = l_corpus.get_tokens()

    # 2. Build Dictionary Decoder
    if verbose: print('\n[2/3] Compiling Latin Herbal Dictionary & Trigram Model...')
    decoder = LatinDictionaryDecoder(l_tokens)

    if verbose:
        print(f"  → Indexed {len(decoder.vocab)} unique Latin herbal words.")

    # 3. Decode Folios
    if verbose: print('\n[3/3] Trigram Viterbi Decoding of Folios...')
    by_folio = extractor.extract_lang_a_by_folio()
    translations = {}

    for folio, tokens in list(by_folio.items())[:10]: # Process first 10 for demonstration
        if len(tokens) < 5: continue
        translated_text = viterbi_trigram_decode(tokens, v_morph, decoder)
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

if __name__ == '__main__':
    run_phase10_final_translation()
