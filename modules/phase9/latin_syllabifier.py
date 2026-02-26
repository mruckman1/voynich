import re
from collections import Counter, defaultdict

class LatinSyllabifier:
    """Uses phonetic rules to break Latin words into constituent syllables."""

    def __init__(self):
        self.vowels = set("aeiouy")
        self.diphthongs = {"ae", "au", "ei", "eu", "oe", "ui"}

    def syllabify(self, word: str) -> list:
        """Rule-based pseudo-syllabification for Latin."""
        word = word.lower()
        syllables = []
        current = ""

        for c in word:
            current += c
            # Break after a vowel unless it's forming a diphthong or end of word
            if c in self.vowels:
                if len(current) >= 2 and current[-2:] in self.diphthongs:
                    continue
                syllables.append(current)
                current = ""

        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)

        return syllables

    def process_corpus(self, tokens: list):
        all_syllables = []
        transitions = defaultdict(lambda: defaultdict(int))

        for word in tokens:
            syls = self.syllabify(word)
            all_syllables.extend(syls)

            # Map valid syllable continuations inside words
            for i in range(len(syls) - 1):
                transitions[syls[i]][syls[i+1]] += 1

            # Map word-to-word transition (space as syllable barrier)
            if len(syls) > 0:
                transitions[syls[-1]]["<SPACE>"] += 1
                transitions["<SPACE>"][syls[0]] += 1

        counts = Counter(all_syllables)

        # Convert to probabilities
        prob_transitions = defaultdict(dict)
        for s1, next_syls in transitions.items():
            total = sum(next_syls.values())
            for s2, count in next_syls.items():
                prob_transitions[s1][s2] = count / total

        return [s for s, c in counts.most_common()], prob_transitions
