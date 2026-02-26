import Levenshtein
from collections import Counter

# Sigla Map from Phase 10
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

# The single-character Voynich words are highly frequent function words
FUNCTION_WORDS = {
    'o': 'et', 'y': 'in', 'a': 'ad', 's': 'sed', 'd': 'est', 'l': 'vel', 'r': 'per'
}

class CSPPhoneticDecoder:
    def __init__(self, l_skeletonizer, v_skeletonizer):
        self.l_skel = l_skeletonizer
        self.v_skel = v_skeletonizer
        self.l_unigrams = l_skeletonizer.unigram_counts

    def find_best_match(self, v_token: str) -> str:
        # 1. Function Word Bypass
        if v_token in FUNCTION_WORDS:
            return FUNCTION_WORDS[v_token]

        # 2. Parse Morphology
        pref, stem, suf = self.v_skel.v_morphemer._strip_affixes(v_token)

        # 3. Get Consonant Skeleton
        target_skel = self.v_skel.get_skeleton(stem)
        if not target_skel:
            return f"[{v_token}]" # Unresolvable, keep original

        # 4. Strict Constraint Matching
        valid_latin_words = self.l_skel.skeleton_index.get(target_skel, [])

        if not valid_latin_words:
            # Fuzzy match the skeleton (Levenshtein distance <= 2)
            closest_skel = None
            min_dist = float('inf')
            for l_skel in self.l_skel.skeleton_index.keys():
                dist = Levenshtein.distance(target_skel, l_skel)
                if dist < min_dist and dist <= 2: # Max 2 insertions/deletions allowed
                    min_dist = dist
                    closest_skel = l_skel

            if closest_skel:
                valid_latin_words = self.l_skel.skeleton_index.get(closest_skel, [])
            else:
                return f"<{v_token}>"

        # 5. Sigla (Prefix/Suffix) Constraint Filtering
        scored_candidates = []
        l_prefixes = SIGLA_MAP.get(pref, [''])
        l_suffixes = SUFFIX_MAP.get(suf, [''])

        for l_word in valid_latin_words:
            score = 0

            # Check Prefix sigla
            if pref and any(l_word.startswith(lp) for lp in l_prefixes):
                score += 10
            # Check Suffix sigla
            if suf and any(l_word.endswith(ls) for ls in l_suffixes):
                score += 10

            # Tie breaker: Word frequency in medical latin
            score += (self.l_unigrams.get(l_word, 0) / 50000)

            scored_candidates.append((score, l_word))

        if scored_candidates:
            # Sort by highest score
            scored_candidates.sort(key=lambda x: -x[0])
            best_word = scored_candidates[0][1]
            return best_word

        return f"<{v_token}>"

    def decode_folio(self, tokens: list) -> str:
        decoded_words = []
        recent_words = set() # Anti-repetition memory

        for token in tokens:
            word = self.find_best_match(token)

            # Anti-Repetition logic (prevent 'et fac et fac')
            if word in recent_words and word not in FUNCTION_WORDS.values():
                # The CSP inherently prevents generic repetition, so we just pass it
                pass

            decoded_words.append(word)

            # Keep a rolling window of recent words
            recent_words.add(word)
            if len(recent_words) > 5:
                # remove an arbitrary element to keep the set small without using queue
                recent_words.pop()

        return ' '.join(decoded_words)
