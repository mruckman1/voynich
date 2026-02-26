import numpy as np
from collections import Counter
from modules.phase5.tier_splitter import TierSplitter

class VoynichMorphemer:
    """Strips words down to stems based on Phase 6 productive affixes."""

    def __init__(self, splitter: TierSplitter, p6_morph_results: dict):
        self.splitter = splitter
        self.raw_tokens = splitter.get_tier1_tokens()

        affixes = p6_morph_results['productive_affixes']
        self.valid_prefixes = {a['prefix'] for a in affixes.get('productive_prefixes', [])}
        self.valid_suffixes = {a['suffix'] for a in affixes.get('productive_suffixes', [])}

        self.parsed_tokens = []
        self.stem_sequence = []

    def _strip_affixes(self, word: str):
        best_pref, best_suff = '', ''

        for p_len in range(min(5, len(word)-1), 0, -1):
            if word[:p_len] in self.valid_prefixes:
                best_pref = word[:p_len]
                break

        remainder = word[len(best_pref):]

        for s_len in range(min(5, len(remainder)-1), 0, -1):
            if remainder[-s_len:] in self.valid_suffixes:
                best_suff = remainder[-s_len:]
                break

        stem = remainder[:-len(best_suff)] if best_suff else remainder
        return best_pref, stem, best_suff

    def process_corpus(self) -> dict:
        for t in self.raw_tokens:
            p, s, suf = self._strip_affixes(t)
            self.parsed_tokens.append({'word': t, 'prefix': p, 'stem': s, 'suffix': suf})
            if s: self.stem_sequence.append(s)

        stem_counts = Counter(self.stem_sequence)

        return {
            'original_vocab_size': len(set(self.raw_tokens)),
            'unique_stems': len(stem_counts),
            'top_stems': stem_counts.most_common(20),
            'parsed_sample': self.parsed_tokens[:10]
        }
