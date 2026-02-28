"""
Homophone Detector — Path B Step 1
====================================
Tests whether the flat Zipf exponent (0.743) is caused by homophonic
substitution — multiple Voynich words encoding the same Latin word.

Algorithm:
  1. Get 1001×1001 transition matrix from Tier1MatrixBuilder
  2. Normalize rows to unit vectors
  3. Compute all-pairs cosine similarity: sim = normed @ normed.T
  4. Find pairs with similarity > threshold (default 0.7)
  5. Cluster via connected components (BFS)
  6. Each component = homophone group; canonical = most frequent member

Phase 6  ·  Voynich Convergence Attack
"""

import numpy as np
from collections import Counter, deque
from typing import Dict, List, Tuple, Set

from modules.phase5.tier1_matrix_builder import Tier1MatrixBuilder
from modules.phase5.tier_splitter import TierSplitter

class HomophoneDetector:
    """
    Detect candidate homophone groups via distributional similarity
    of word transition vectors.

    Words that appear in identical contexts (high cosine similarity
    of their transition matrix rows) are candidate homophones — they
    may encode the same plaintext word, with the choice of variant
    depending on position or context.
    """

    def __init__(self, matrix_builder: Tier1MatrixBuilder,
                 splitter: TierSplitter,
                 similarity_threshold: float = 0.7):
        self.matrix_builder = matrix_builder
        self.splitter = splitter
        self.threshold = similarity_threshold
        self._sim_matrix = None
        self._groups = None

    def compute_cosine_similarity_matrix(self) -> np.ndarray:
        """
        Compute all-pairs cosine similarity of transition row vectors.

        Each row of the transition matrix is a 1001-dimensional distribution
        P(next_word | word_i). Cosine similarity between rows i and j
        measures how similar their right-context distributions are.

        Returns 1001×1001 symmetric similarity matrix.
        """
        if self._sim_matrix is not None:
            return self._sim_matrix

        matrix, vocab = self.matrix_builder.build_voynich_matrix()
        n = len(vocab)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = matrix / norms

        self._sim_matrix = normed @ normed.T

        return self._sim_matrix

    def find_homophone_candidates(self) -> List[Dict]:
        """
        Find all word pairs with cosine similarity above threshold.

        Returns list of {word_a, word_b, similarity, idx_a, idx_b}.
        """
        sim = self.compute_cosine_similarity_matrix()
        _, vocab = self.matrix_builder.build_voynich_matrix()
        n = len(vocab)

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                s = sim[i, j]
                if s >= self.threshold:
                    pairs.append({
                        'word_a': vocab[i],
                        'word_b': vocab[j],
                        'similarity': float(s),
                        'idx_a': i,
                        'idx_b': j,
                    })

        pairs.sort(key=lambda x: -x['similarity'])
        return pairs

    def cluster_homophones(self) -> Dict[str, List[str]]:
        """
        Cluster candidate homophones using connected components.

        Two words are in the same group if their cosine similarity
        exceeds the threshold. Groups are found via BFS on the
        adjacency graph.

        Returns {canonical_word: [variant1, variant2, ...]}
        where canonical is the most frequent member.
        """
        if self._groups is not None:
            return self._groups

        sim = self.compute_cosine_similarity_matrix()
        _, vocab = self.matrix_builder.build_voynich_matrix()
        n = len(vocab)

        word_freqs = Counter(self.splitter.get_tier1_tokens())

        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        visited = set()
        components = []

        for start in range(n):
            if start in visited:
                continue
            if not adj[start]:
                continue

            component = []
            queue = deque([start])
            visited.add(start)

            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) >= 2:
                components.append(component)

        groups = {}
        for comp in components:
            words = [(vocab[idx], word_freqs.get(vocab[idx], 0)) for idx in comp]
            words.sort(key=lambda x: -x[1])
            canonical = words[0][0]
            variants = [w for w, _ in words]
            groups[canonical] = variants

        self._groups = groups
        return groups

    def compute_group_statistics(self) -> Dict:
        """Compute statistics about the detected homophone groups."""
        groups = self.cluster_homophones()
        _, vocab = self.matrix_builder.build_voynich_matrix()
        word_freqs = Counter(self.splitter.get_tier1_tokens())

        if not groups:
            return {
                'n_groups': 0,
                'total_words_in_groups': 0,
                'coverage': 0.0,
                'mean_group_size': 0,
                'max_group_size': 0,
                'groups_by_size': {},
            }

        sizes = [len(v) for v in groups.values()]
        total_words = sum(sizes)

        group_freqs = []
        for canonical, variants in groups.items():
            combined_freq = sum(word_freqs.get(w, 0) for w in variants)
            group_freqs.append({
                'canonical': canonical,
                'n_variants': len(variants),
                'combined_frequency': combined_freq,
                'variants': variants[:5],
            })
        group_freqs.sort(key=lambda x: -x['combined_frequency'])

        size_dist = Counter(sizes)

        return {
            'n_groups': len(groups),
            'total_words_in_groups': total_words,
            'coverage': total_words / max(1, len(vocab)),
            'mean_group_size': float(np.mean(sizes)),
            'median_group_size': float(np.median(sizes)),
            'max_group_size': max(sizes),
            'size_distribution': dict(sorted(size_dist.items())),
            'top_groups': group_freqs[:20],
            'ungrouped_words': len(vocab) - total_words,
        }

    def compute_similarity_distribution(self) -> Dict:
        """Compute statistics about the overall similarity distribution."""
        sim = self.compute_cosine_similarity_matrix()
        n = sim.shape[0]

        upper = []
        for i in range(n):
            for j in range(i + 1, n):
                upper.append(sim[i, j])
        upper = np.array(upper)

        return {
            'mean_similarity': float(np.mean(upper)),
            'median_similarity': float(np.median(upper)),
            'std_similarity': float(np.std(upper)),
            'min_similarity': float(np.min(upper)),
            'max_similarity': float(np.max(upper)),
            'n_above_threshold': int(np.sum(upper >= self.threshold)),
            'n_above_0.5': int(np.sum(upper >= 0.5)),
            'n_above_0.8': int(np.sum(upper >= 0.8)),
            'n_above_0.9': int(np.sum(upper >= 0.9)),
            'n_total_pairs': len(upper),
            'pct_above_threshold': float(np.sum(upper >= self.threshold) / max(1, len(upper))),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run homophone detection and return full results."""
        sim_stats = self.compute_similarity_distribution()
        groups = self.cluster_homophones()
        group_stats = self.compute_group_statistics()

        n_groups = group_stats['n_groups']
        coverage = group_stats['coverage']

        if n_groups > 50 and coverage > 0.3:
            interpretation = (
                f'STRONG homophonic signal: {n_groups} groups covering '
                f'{coverage:.1%} of vocabulary. Multiple Voynich words '
                f'share distributional context, suggesting they encode '
                f'the same plaintext word.'
            )
            homophony_plausible = True
        elif n_groups > 10:
            interpretation = (
                f'MODERATE homophonic signal: {n_groups} groups covering '
                f'{coverage:.1%}. Some distributional overlap exists but '
                f'may reflect syntactic similarity rather than homophony.'
            )
            homophony_plausible = True
        else:
            interpretation = (
                f'WEAK/NO homophonic signal: only {n_groups} groups. '
                f'Most Voynich words have distinct distributional profiles. '
                f'Homophonic substitution is unlikely.'
            )
            homophony_plausible = False

        results = {
            'threshold': self.threshold,
            'similarity_distribution': sim_stats,
            'group_statistics': group_stats,
            'homophony_plausible': homophony_plausible,
            'interpretation': interpretation,
            'synthesis': {
                'n_groups': n_groups,
                'coverage': coverage,
                'homophony_plausible': homophony_plausible,
                'conclusion': interpretation,
            },
        }

        if verbose:
            print(f'\n  Homophone Detector (threshold={self.threshold}):')
            print(f'    Similarity distribution:')
            print(f'      Mean:   {sim_stats["mean_similarity"]:.4f}')
            print(f'      Median: {sim_stats["median_similarity"]:.4f}')
            print(f'      Pairs above threshold: {sim_stats["n_above_threshold"]}')
            print(f'    Groups found: {n_groups}')
            print(f'    Words in groups: {group_stats["total_words_in_groups"]} '
                  f'({coverage:.1%} coverage)')
            if n_groups > 0:
                print(f'    Mean group size: {group_stats["mean_group_size"]:.1f}')
                print(f'    Max group size:  {group_stats["max_group_size"]}')
                print(f'    --- Top Groups ---')
                for g in group_stats['top_groups'][:5]:
                    print(f'      {g["canonical"]} ({g["n_variants"]} variants, '
                          f'freq={g["combined_frequency"]}): '
                          f'{", ".join(g["variants"])}')
            print(f'    Interpretation: {interpretation}')

        return results
