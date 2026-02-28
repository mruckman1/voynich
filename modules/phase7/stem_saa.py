import numpy as np
from collections import Counter
import random
import math
from modules.phase7.voynich_morphemer import VoynichMorphemer
from modules.phase7.latin_morphology import LatinMorphologyParser

class StemSAA:
    """Transition matrix matching applied only to semantic stems."""

    def __init__(self, v_morphemer: VoynichMorphemer, l_parser: LatinMorphologyParser):
        self.v_seq = v_morphemer.stem_sequence
        self.l_seq = l_parser.stem_sequence

    def _build_matrix(self, sequence: list, top_n: int = 150):
        vocab = [w for w, _ in Counter(sequence).most_common(top_n)]
        w2i = {w: i for i, w in enumerate(vocab)}

        matrix = np.zeros((top_n, top_n))
        for i in range(len(sequence)-1):
            w1, w2 = sequence[i], sequence[i+1]
            if w1 in w2i and w2 in w2i:
                matrix[w2i[w1]][w2i[w2]] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return matrix / row_sums, vocab

    def run(self, n_iter=50000, seed=42, verbose=True):
        v_mat, v_voc = self._build_matrix(self.v_seq, 150)
        l_mat, l_voc = self._build_matrix(self.l_seq, 150)

        n_v, n_l = len(v_voc), len(l_voc)
        rng = random.Random(seed)

        current = [i % n_l for i in range(n_v)]

        def calc_cost(perm):
            p_arr = np.array(perm)
            l_perm = l_mat[p_arr][:, p_arr]
            return float(np.sum((v_mat - l_perm)**2))

        best_cost = calc_cost(current)
        best = current[:]

        T = 1.0
        T_min = 0.001

        for k in range(n_iter):
            t = T * (T_min / T) ** (k / n_iter)
            i, j = rng.sample(range(n_v), 2)

            current[i], current[j] = current[j], current[i]
            cost = calc_cost(current)

            if cost < best_cost or rng.random() < math.exp((best_cost - cost) / t):
                best_cost = cost
                best = current[:]
            else:
                current[i], current[j] = current[j], current[i]

        mapping = {v_voc[i]: l_voc[best[i]] for i in range(n_v)}

        return {
            'best_mapping': mapping,
            'normalized_cost': best_cost / (n_v * n_v),
            'sample_mappings': list(mapping.items())[:10]
        }
