import re
from collections import defaultdict, Counter

LATIN_CONSONANT_CLASSES = {
    'c': 'K', 'k': 'K', 'q': 'K', 'g': 'K',
    't': 'T', 'd': 'T',
    'p': 'P', 'b': 'P', 'v': 'P', 'f': 'P',
    's': 'S', 'z': 'S', 'x': 'S',
    'l': 'L', 'r': 'R', 'm': 'M', 'n': 'N'
}

VOYNICH_CONSONANT_CLASSES = {
    'k': 'K', 't': 'T', 'p': 'P', 'f': 'P',
    'ch': 'K', 'c': 'K', 'sh': 'S', 's': 'S',
    'l': 'L', 'r': 'R', 'm': 'M', 'n': 'N', 'in': 'N', 'iin': 'N',
    'd': 'T', 'ck': 'K', 'ct': 'T'
}

class LatinPhoneticSkeletonizer:
    def __init__(self, latin_tokens: list):
        self.vocab = sorted(set(latin_tokens))
        self.unigram_counts = Counter(latin_tokens)
        self.skeleton_index = defaultdict(list)
        self._build_index()

    def get_skeleton(self, word: str) -> str:
        word = word.lower()
        skeleton = []
        last_char = ''
        for char in word:
            if char in LATIN_CONSONANT_CLASSES:
                mapped = LATIN_CONSONANT_CLASSES[char]
                if mapped != last_char:
                    skeleton.append(mapped)
                    last_char = mapped
        return '-'.join(skeleton)

    def _build_index(self):
        """Index all Latin words by their phonetic consonant skeleton."""
        sorted_vocab = sorted(self.vocab, key=lambda w: (-self.unigram_counts[w], w))
        for word in sorted_vocab:
            skel = self.get_skeleton(word)
            if skel:
                self.skeleton_index[skel].append(word)

class VoynichPhoneticSkeletonizer:
    def __init__(self, v_morphemer):
        self.v_morphemer = v_morphemer

    def get_skeleton(self, v_stem: str) -> str:
        skeleton = []
        i = 0
        while i < len(v_stem):
            if i < len(v_stem) - 1 and v_stem[i:i+2] in VOYNICH_CONSONANT_CLASSES:
                skeleton.append(VOYNICH_CONSONANT_CLASSES[v_stem[i:i+2]])
                i += 2
            elif v_stem[i] in VOYNICH_CONSONANT_CLASSES:
                skeleton.append(VOYNICH_CONSONANT_CLASSES[v_stem[i]])
                i += 1
            else:
                i += 1
        return '-'.join(skeleton)
