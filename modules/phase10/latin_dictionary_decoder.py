"""
Latin Dictionary Decoder for Phase 10.

Builds a lookup dictionary from the Latin corpus and decodes Voynich tokens
using trigram-guided beam search constrained to valid Medieval Latin words.

Extracted from convergence_attack_p10.py — logic is identical.
"""
import math
from collections import defaultdict, Counter

from orchestrators._config import SIGLA_PREFIX_MAP, SIGLA_SUFFIX_MAP, BEAM_WIDTH_TRIGRAM

SIGLA_MAP = SIGLA_PREFIX_MAP
SUFFIX_MAP = SIGLA_SUFFIX_MAP

class LatinDictionaryDecoder:
    """Builds a lookup dictionary from the Latin corpus and decodes Voynich tokens."""

    def __init__(self, latin_tokens: list):
        self.vocab = set(latin_tokens)
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter(latin_tokens)
        self.total_words = len(latin_tokens)

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
        candidates = self.word_index.get((v_prefix, v_suffix), [])
        if not candidates:
            candidates = self.word_index.get((v_prefix, ''), [])
        if not candidates:
            candidates = self.word_index.get(('', v_suffix), [])
        if not candidates:
            candidates = [w for w, _ in self.unigram_counts.most_common(50)]

        return sorted(candidates, key=lambda w: -self.unigram_counts[w])[:20]

def viterbi_trigram_decode(v_tokens: list, v_morphemer, decoder: LatinDictionaryDecoder):
    """Decodes Voynich text using Trigrams to ensure grammatical fluency."""
    beam = [(0.0, ["<START>", "<START>"])]

    for v_token in v_tokens:
        pref, _, suf = v_morphemer._strip_affixes(v_token)

        candidates = decoder.get_candidates(pref, suf)

        new_beam = []
        for log_prob, history in beam:
            w1, w2 = history[-2], history[-1]

            for cand in candidates:
                prob = decoder.get_trigram_prob(w1, w2, cand)
                score = log_prob + math.log(prob)
                new_beam.append((score, history + [cand]))

        new_beam.sort(key=lambda x: (-x[0], x[1]))
        beam = new_beam[:BEAM_WIDTH_TRIGRAM]

    best_sequence = beam[0][1][2:]
    return ' '.join(best_sequence)
