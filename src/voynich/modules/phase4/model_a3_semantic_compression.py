"""
Model A3: Semantic Compression (Topic-Conditioned Generation)
===============================================================
Tests the hypothesis that Language A words are slot markers in
topic-conditioned templates. Words cluster into 5-8 semantic classes
corresponding to herbal formula elements.

Priority: MEDIUM

Critical test: Apply spectral clustering or NMF to the word
co-occurrence matrix. If 5-8 coherent classes emerge with
interpretable transition patterns, the model is supported.
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from voynich.core.stats import word_conditional_entropy
from voynich.modules.nmf_analysis import simple_nmf
from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor

class SemanticCompressionModel:
    """
    Test whether Language A words are slot markers in topic templates.

    Expected herbal formula slots:
    [plant name] [quality] [body part] [action] [preparation]

    If Language A words cluster into ~5-8 classes corresponding to
    these slots, the semantic compression model is supported.
    """

    def __init__(self, extractor: LanguageAExtractor):
        self.extractor = extractor

    MAX_VOCAB = 150

    def build_cooccurrence_matrix(self, window: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Build word co-occurrence matrix using a sliding window.
        Restricted to top MAX_VOCAB most-frequent words for tractability.

        Returns:
            (matrix, vocabulary) where matrix[i][j] = count of word_i
            and word_j appearing within 'window' positions of each other.
        """
        tokens = self.extractor.extract_lang_a_tokens()
        freqs = Counter(tokens)
        vocab = [w for w, _ in freqs.most_common(self.MAX_VOCAB)]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        matrix = np.zeros((n, n), dtype=float)

        for i, token in enumerate(tokens):
            if token not in word_to_idx:
                continue
            idx_i = word_to_idx[token]
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            for j in range(start, end):
                if i != j and tokens[j] in word_to_idx:
                    idx_j = word_to_idx[tokens[j]]
                    matrix[idx_i][idx_j] += 1

        return matrix, vocab

    def apply_spectral_clustering(self, n_clusters_range: Tuple[int, int] = (3, 10)) -> Dict:
        """
        Apply spectral clustering to the co-occurrence matrix.

        Tests cluster counts from n_clusters_range[0] to n_clusters_range[1]
        and selects the best by silhouette score.
        """
        matrix, vocab = self.build_cooccurrence_matrix()

        if len(vocab) < 5:
            return {
                'error': 'Vocabulary too small for clustering',
                'vocabulary_size': len(vocab),
            }

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        similarity = matrix / row_sums

        similarity = (similarity + similarity.T) / 2
        np.fill_diagonal(similarity, 1.0)

        best_score = -1
        best_k = 3
        best_labels = None
        all_results = []

        for k in range(n_clusters_range[0], min(n_clusters_range[1] + 1, len(vocab))):
            labels = self._spectral_cluster(similarity, k)
            if labels is None:
                continue

            score = self._silhouette_score(similarity, labels)
            all_results.append({'k': k, 'silhouette': score})

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        clusters = defaultdict(list)
        if best_labels is not None:
            for i, label in enumerate(best_labels):
                clusters[int(label)].append(vocab[i])

        return {
            'best_k': best_k,
            'best_silhouette': best_score,
            'all_k_scores': all_results,
            'clusters': dict(clusters),
            'cluster_sizes': {k: len(v) for k, v in clusters.items()},
            'in_target_range': 5 <= best_k <= 8,
        }

    def _spectral_cluster(self, similarity: np.ndarray, k: int) -> list:
        """Simple spectral clustering using eigendecomposition."""
        n = similarity.shape[0]
        if k >= n:
            return None

        degree = np.diag(similarity.sum(axis=1))
        laplacian = degree - similarity

        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(degree), 1e-10)))
        L_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
        except np.linalg.LinAlgError:
            return None

        features = eigenvectors[:, :k]

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms

        labels = self._simple_kmeans(features, k)
        return labels

    def _simple_kmeans(self, X: np.ndarray, k: int,
                       max_iter: int = 100, seed: int = 42) -> list:
        """Simple k-means clustering."""
        rng = np.random.RandomState(seed)
        n = X.shape[0]

        indices = rng.choice(n, size=k, replace=False)
        centroids = X[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            for i in range(n):
                dists = np.linalg.norm(X[i] - centroids, axis=1)
                labels[i] = np.argmin(dists)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]

            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids

        return labels.tolist()

    def _silhouette_score(self, similarity: np.ndarray, labels: list) -> float:
        """Compute average silhouette score from similarity matrix."""
        n = len(labels)
        if n < 2:
            return 0.0

        distance = 1.0 - similarity / (similarity.max() + 1e-10)

        scores = []
        for i in range(n):
            same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
            if not same_cluster:
                scores.append(0.0)
                continue
            a_i = np.mean([distance[i][j] for j in same_cluster])

            other_clusters = set(labels) - {labels[i]}
            if not other_clusters:
                scores.append(0.0)
                continue

            b_i = min(
                np.mean([distance[i][j] for j in range(n) if labels[j] == c])
                for c in other_clusters
                if any(labels[j] == c for j in range(n))
            )

            s_i = (b_i - a_i) / max(a_i, b_i, 1e-10)
            scores.append(s_i)

        return float(np.mean(scores))

    def apply_nmf_topic_model(self, n_topics_range: Tuple[int, int] = (3, 10)) -> Dict:
        """
        Apply NMF (Non-negative Matrix Factorization) for topic extraction.

        NMF factorizes the co-occurrence matrix into W * H, where:
        - W: word-topic matrix (which topic each word belongs to)
        - H: topic-context matrix (how topics relate to contexts)
        """
        matrix, vocab = self.build_cooccurrence_matrix()

        matrix = np.maximum(matrix, 0)

        best_error = float('inf')
        best_k = 3
        best_W = None
        all_results = []

        for k in range(n_topics_range[0], min(n_topics_range[1] + 1, len(vocab))):
            W, H, error = simple_nmf(matrix, k)
            all_results.append({'k': k, 'reconstruction_error': error})

            if error < best_error:
                best_error = error
                best_k = k
                best_W = W

        topics = defaultdict(list)
        if best_W is not None:
            assignments = np.argmax(best_W, axis=1)
            for i, topic in enumerate(assignments):
                topics[int(topic)].append((vocab[i], float(best_W[i, topic])))

        for topic in topics:
            topics[topic].sort(key=lambda x: -x[1])

        return {
            'best_k': best_k,
            'best_error': best_error,
            'all_k_errors': all_results,
            'topics': {k: [(w, round(s, 3)) for w, s in v]
                       for k, v in topics.items()},
            'topic_sizes': {k: len(v) for k, v in topics.items()},
            'in_target_range': 5 <= best_k <= 8,
        }

    def evaluate_cluster_coherence(self, clusters: Dict[int, List[str]]) -> Dict:
        """
        Evaluate semantic coherence of discovered clusters.

        Measures within-cluster co-occurrence density vs between-cluster
        co-occurrence density. Higher ratio = more coherent clusters.
        """
        matrix, vocab = self.build_cooccurrence_matrix()
        word_to_idx = {w: i for i, w in enumerate(vocab)}

        within_total = 0.0
        within_count = 0
        between_total = 0.0
        between_count = 0

        for c1, words1 in clusters.items():
            indices1 = [word_to_idx[w] for w in words1 if w in word_to_idx]
            for c2, words2 in clusters.items():
                indices2 = [word_to_idx[w] for w in words2 if w in word_to_idx]
                for i in indices1:
                    for j in indices2:
                        if i != j:
                            if c1 == c2:
                                within_total += matrix[i][j]
                                within_count += 1
                            else:
                                between_total += matrix[i][j]
                                between_count += 1

        within_density = within_total / max(1, within_count)
        between_density = between_total / max(1, between_count)
        ratio = within_density / max(between_density, 1e-10)

        return {
            'within_density': within_density,
            'between_density': between_density,
            'coherence_ratio': ratio,
            'coherent': ratio > 1.5,
        }

    def test_transition_structure(self, clusters: Dict[int, List[str]]) -> Dict:
        """
        Test whether cluster transitions follow interpretable patterns.

        E.g., class 1 (plant names) always precedes class 2 (qualities).
        """
        tokens = self.extractor.extract_lang_a_tokens()

        word_to_cluster = {}
        for c, words in clusters.items():
            for w in words:
                word_to_cluster[w] = c

        n_clusters = max(clusters.keys()) + 1 if clusters else 0
        trans = np.zeros((n_clusters, n_clusters), dtype=float)

        for i in range(len(tokens) - 1):
            c1 = word_to_cluster.get(tokens[i], -1)
            c2 = word_to_cluster.get(tokens[i + 1], -1)
            if c1 >= 0 and c2 >= 0:
                trans[c1][c2] += 1

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans / row_sums

        dominant_transitions = []
        for i in range(n_clusters):
            for j in range(n_clusters):
                if trans_prob[i][j] > 0.4:
                    dominant_transitions.append({
                        'from': i, 'to': j,
                        'probability': float(trans_prob[i][j]),
                    })

        return {
            'transition_matrix': trans_prob.tolist(),
            'n_clusters': n_clusters,
            'dominant_transitions': dominant_transitions,
            'has_structure': len(dominant_transitions) > 0,
        }

    def _synthesize(self, spectral: Dict, nmf: Dict,
                    coherence: Dict, transitions: Dict) -> Dict:
        """Combine all results into a synthesis."""
        signals = sum([
            spectral.get('in_target_range', False),
            nmf.get('in_target_range', False),
            coherence.get('coherent', False),
            transitions.get('has_structure', False),
        ])

        if signals >= 3:
            confidence = 'MODERATE'
            supported = True
        elif signals >= 2:
            confidence = 'WEAK'
            supported = False
        else:
            confidence = 'UNSUPPORTED'
            supported = False

        return {
            'semantic_compression_supported': supported,
            'confidence': confidence,
            'signals': signals,
            'signals_total': 4,
            'spectral_in_range': spectral.get('in_target_range', False),
            'nmf_in_range': nmf.get('in_target_range', False),
            'clusters_coherent': coherence.get('coherent', False),
            'transitions_structured': transitions.get('has_structure', False),
            'conclusion': (
                f'Semantic compression model: {confidence}. '
                f'{signals}/4 signals detected. '
                f'Best spectral k={spectral.get("best_k", "?")}, '
                f'best NMF k={nmf.get("best_k", "?")}.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all semantic compression tests."""
        spectral = self.apply_spectral_clustering()
        nmf = self.apply_nmf_topic_model()

        clusters = spectral.get('clusters', {})
        coherence = self.evaluate_cluster_coherence(clusters) if clusters else {'coherent': False}
        transitions = self.test_transition_structure(clusters) if clusters else {'has_structure': False}

        synthesis = self._synthesize(spectral, nmf, coherence, transitions)

        results = {
            'spectral_clustering': spectral,
            'nmf_topic_model': nmf,
            'cluster_coherence': coherence,
            'transition_structure': transitions,
            'synthesis': synthesis,
        }

        if verbose:
            print(f'\n  Model A3: Semantic Compression Results')
            print(f'    --- Spectral Clustering ---')
            print(f'    Best k: {spectral.get("best_k", "?")} '
                  f'(silhouette={spectral.get("best_silhouette", 0):.3f})')
            print(f'    In target range [5,8]: {spectral.get("in_target_range", False)}')
            print(f'    --- NMF Topic Model ---')
            print(f'    Best k: {nmf.get("best_k", "?")} '
                  f'(error={nmf.get("best_error", 0):.1f})')
            print(f'    In target range [5,8]: {nmf.get("in_target_range", False)}')
            print(f'    --- Coherence ---')
            print(f'    Coherent: {coherence.get("coherent", False)} '
                  f'(ratio={coherence.get("coherence_ratio", 0):.2f})')
            print(f'    --- Transitions ---')
            print(f'    Structured: {transitions.get("has_structure", False)}')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
