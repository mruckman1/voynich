from collections import Counter
from voynich.modules.phase6.improved_latin_corpus import ImprovedLatinCorpus

LATIN_SUFFIXES = [
    'ibus', 'arum', 'orum', 'atum', 'ium', 'ius', 'bus', 'que',
    'am', 'em', 'um', 'os', 'as', 'es', 'is', 'us', 'ae', 'oe',
    'a', 'e', 'i', 'o', 'u', 'm', 's', 't'
]

LATIN_PREFIXES = ['con', 'per', 'sub', 'pro', 'in', 'ex', 'de', 'ad', 're']

class LatinMorphologyParser:
    def __init__(self, corpus: ImprovedLatinCorpus):
        self.tokens = corpus.get_tokens()
        self.parsed_tokens = []
        self.stem_sequence = []

    def process_corpus(self) -> dict:
        for word in self.tokens:
            w_lower = word.lower()
            p, s, suf = '', w_lower, ''

            for pref in LATIN_PREFIXES:
                if s.startswith(pref) and len(s) > len(pref) + 2:
                    p = pref
                    s = s[len(pref):]
                    break

            for suffix in LATIN_SUFFIXES:
                if s.endswith(suffix) and len(s) > len(suffix) + 2:
                    suf = suffix
                    s = s[:-len(suffix)]
                    break

            self.parsed_tokens.append({'word': word, 'prefix': p, 'stem': s, 'suffix': suf})
            self.stem_sequence.append(s)

        stem_counts = Counter(self.stem_sequence)
        return {
            'original_vocab_size': len(set(self.tokens)),
            'unique_stems': len(stem_counts),
            'top_stems': stem_counts.most_common(20)
        }
