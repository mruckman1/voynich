import Levenshtein
from collections import Counter, deque

SIGLA_MAP = {
    'qo': ['con', 'com', 'cor', 'qu'], 'ch': ['ca', 'ce', 'ci', 'co', 'cu'],
    'sh': ['sa', 'se', 'si', 'su', 'ex'], 'd':  ['de', 'di', 'da'],
    'p':  ['pro', 'per', 'prae', 'par'], 't':  ['te', 'ta', 'ti', 'tra']
}

SUFFIX_MAP = {
    'dy': ['ae', 'ti', 'ur', 'di', 'te'], 'iin': ['us', "um", 'is', 'in', 'unt'],
    'in': ['um', 'im', 'em', 'en'], 'ey': ['es', 'et', 'er', 'em'],
    'y':  ['a', 'i', 'e', 'o'], 'l':  ['al', 'el', 'il', 'ul', 'le'],
    'r':  ['ar', 'er', 'or', 'ur', 're'], 'm':  ['am', 'um', 'em', 'rum', 'num'],
    's':  ['as', 'os', 'is', 'us']
}

FUNCTION_WORDS = {
    'o': 'et', 'y': 'in', 'a': 'ad', 's': 'sed', 'd': 'est', 'l': 'vel', 'r': 'per'
}

class CSPPhoneticDecoder:
    def __init__(self, l_skeletonizer, v_skeletonizer):
        self.l_skel = l_skeletonizer
        self.v_skel = v_skeletonizer
        self.l_unigrams = l_skeletonizer.unigram_counts

    def find_best_match(self, v_token: str) -> str:
        if v_token in FUNCTION_WORDS:
            return FUNCTION_WORDS[v_token]

        pref, stem, suf = self.v_skel.v_morphemer._strip_affixes(v_token)

        target_skel = self.v_skel.get_skeleton(stem)
        if not target_skel:
            return f"[{v_token}]"

        valid_latin_words = self.l_skel.skeleton_index.get(target_skel, [])

        if not valid_latin_words:
            closest_skel = None
            min_dist = float('inf')
            for l_skel in self.l_skel.skeleton_index.keys():
                dist = Levenshtein.distance(target_skel, l_skel)
                if dist <= 2 and (dist < min_dist or (dist == min_dist and (closest_skel is None or l_skel < closest_skel))):
                    min_dist = dist
                    closest_skel = l_skel

            if closest_skel:
                valid_latin_words = self.l_skel.skeleton_index.get(closest_skel, [])
            else:
                return f"<{v_token}>"

        scored_candidates = []
        l_prefixes = SIGLA_MAP.get(pref, [''])
        l_suffixes = SUFFIX_MAP.get(suf, [''])

        for l_word in valid_latin_words:
            score = 0

            if pref and any(l_word.startswith(lp) for lp in l_prefixes):
                score += 10
            if suf and any(l_word.endswith(ls) for ls in l_suffixes):
                score += 10

            score += (self.l_unigrams.get(l_word, 0) / 50000)

            scored_candidates.append((score, l_word))

        if scored_candidates:
            scored_candidates.sort(key=lambda x: (-x[0], x[1]))
            best_word = scored_candidates[0][1]
            return best_word

        return f"<{v_token}>"

    def decode_folio(self, tokens: list) -> str:
        decoded_words = []
        recent_words = deque(maxlen=6)

        for token in tokens:
            word = self.find_best_match(token)

            if word in recent_words and word not in FUNCTION_WORDS.values():
                pass

            decoded_words.append(word)
            recent_words.append(word)

        return ' '.join(decoded_words)
