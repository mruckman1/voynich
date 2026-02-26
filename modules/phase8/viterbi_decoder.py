import numpy as np
from collections import Counter, defaultdict

class ViterbiDecoder:
    """Uses Hidden Markov Modeling to correct SAA mappings based on word context."""

    def __init__(self, base_map: dict, v_morph, l_parser):
        self.base_map = base_map
        self.v_seq = v_morph.stem_sequence
        self.l_seq = l_parser.stem_sequence
        self.l_vocab = sorted(set(self.l_seq))

        # Transition Probabilities P(Stem_B | Stem_A)
        self.transitions = defaultdict(lambda: defaultdict(float))
        self._build_transitions()

    def _build_transitions(self):
        bigrams = list(zip(self.l_seq[:-1], self.l_seq[1:]))
        counts = Counter(bigrams)
        unigrams = Counter(self.l_seq)

        for (w1, w2), count in counts.items():
            self.transitions[w1][w2] = count / unigrams[w1]

    def run_viterbi_smoothing(self):
        refined_map = dict(self.base_map)
        corrections = 0

        # Look for suspicious trigrams in Voynich sequence
        for i in range(1, len(self.v_seq) - 1):
            v_prev, v_curr, v_next = self.v_seq[i-1], self.v_seq[i], self.v_seq[i+1]

            l_prev = self.base_map.get(v_prev)
            l_curr = self.base_map.get(v_curr)
            l_next = self.base_map.get(v_next)

            if not (l_prev and l_curr and l_next):
                continue

            # If P(curr | prev) is 0 and P(next | curr) is 0, the SAA likely guessed wrong
            if self.transitions[l_prev][l_curr] < 0.001 and self.transitions[l_curr][l_next] < 0.001:
                # Search for a better Latin stem that fits the context
                best_replacement = l_curr
                best_prob = 0

                for candidate in self.l_vocab:
                    prob = self.transitions[l_prev][candidate] * self.transitions[candidate][l_next]
                    if prob > best_prob:
                        best_prob = prob
                        best_replacement = candidate

                if best_prob > 0.05 and best_replacement != l_curr:
                    refined_map[v_curr] = best_replacement
                    corrections += 1

        return refined_map, {'total_corrections': corrections}
