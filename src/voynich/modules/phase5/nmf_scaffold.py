"""
NMF Topic Coherence Scaffold
===============================
Extracts NMF topics from both the Voynich corpus and the Latin reference
corpus, then builds a topic coherence penalty matrix for the constrained SAA.

Words that cluster in the same NMF topic should map to Latin words that
co-occur in the same medical contexts. This adds soft semantic structure
to the SAA cost function.

Phase 5  ·  Voynich Convergence Attack
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor
from voynich.modules.phase5.latin_corpus_expanded import ExpandedLatinHerbalCorpus
from voynich.modules.nmf_analysis import simple_nmf

class NMFScaffold:
    """
    Extract NMF topics from Voynich and Latin corpora, then build a
    topic coherence penalty matrix for the SAA.
    """

    def __init__(self, extractor: LanguageAExtractor,
                 latin_corpus: ExpandedLatinHerbalCorpus,
                 n_topics: int = 10,
                 max_vocab: int = 150,
                 window: int = 5):
        self.extractor = extractor
        self.latin_corpus = latin_corpus
        self.n_topics = n_topics
        self.max_vocab = max_vocab
        self.window = window
        self._voynich_topics = None
        self._latin_topics = None
        self._voynich_word_topic = None
        self._latin_cooccurrence = None

    def _build_cooccurrence_matrix(self, tokens: List[str],
                                    top_n: int = 150) -> Tuple[np.ndarray, List[str]]:
        """Build word co-occurrence matrix using a sliding window."""
        freqs = Counter(tokens)
        vocab = [w for w, _ in freqs.most_common(top_n)]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        matrix = np.zeros((n, n), dtype=float)
        for i, token in enumerate(tokens):
            if token not in word_to_idx:
                continue
            idx_i = word_to_idx[token]
            start = max(0, i - self.window)
            end = min(len(tokens), i + self.window + 1)
            for j in range(start, end):
                if i != j and tokens[j] in word_to_idx:
                    idx_j = word_to_idx[tokens[j]]
                    matrix[idx_i][idx_j] += 1

        return matrix, vocab

    def extract_voynich_topics(self) -> Dict:
        """
        Extract NMF topics from the Voynich Language A corpus.

        Returns dict with:
            topics: {topic_id: [(word, weight), ...]},
            word_topic: {word: topic_id},
            W: word-topic weight matrix,
            reconstruction_error: float
        """
        if self._voynich_topics is not None:
            return self._voynich_topics

        tokens = self.extractor.extract_lang_a_tokens()
        matrix, vocab = self._build_cooccurrence_matrix(tokens, self.max_vocab)
        matrix = np.maximum(matrix, 0)

        k = min(self.n_topics, len(vocab) - 1)
        if k < 2:
            self._voynich_topics = {
                'topics': {}, 'word_topic': {}, 'W': None,
                'reconstruction_error': float('inf'),
            }
            return self._voynich_topics

        W, H, error = simple_nmf(matrix, k)

        assignments = np.argmax(W, axis=1)
        topics = defaultdict(list)
        word_topic = {}

        for i, topic in enumerate(assignments):
            topics[int(topic)].append((vocab[i], float(W[i, topic])))
            word_topic[vocab[i]] = int(topic)

        for topic in topics:
            topics[topic].sort(key=lambda x: -x[1])

        self._voynich_word_topic = word_topic
        self._voynich_topics = {
            'topics': dict(topics),
            'word_topic': word_topic,
            'W': W,
            'reconstruction_error': error,
            'n_topics': k,
            'vocabulary_size': len(vocab),
        }
        return self._voynich_topics

    def extract_latin_topics(self) -> Dict:
        """
        Extract NMF topics from the Latin reference corpus.

        Returns same structure as extract_voynich_topics().
        """
        if self._latin_topics is not None:
            return self._latin_topics

        tokens = self.latin_corpus.get_tokens()
        matrix, vocab = self._build_cooccurrence_matrix(tokens, self.max_vocab)
        matrix = np.maximum(matrix, 0)

        k = min(self.n_topics, len(vocab) - 1)
        if k < 2:
            self._latin_topics = {
                'topics': {}, 'word_topic': {}, 'W': None,
                'reconstruction_error': float('inf'),
            }
            return self._latin_topics

        W, H, error = simple_nmf(matrix, k)

        assignments = np.argmax(W, axis=1)
        topics = defaultdict(list)
        word_topic = {}

        for i, topic in enumerate(assignments):
            topics[int(topic)].append((vocab[i], float(W[i, topic])))
            word_topic[vocab[i]] = int(topic)

        for topic in topics:
            topics[topic].sort(key=lambda x: -x[1])

        self._latin_topics = {
            'topics': dict(topics),
            'word_topic': word_topic,
            'W': W,
            'reconstruction_error': error,
            'n_topics': k,
            'vocabulary_size': len(vocab),
        }
        return self._latin_topics

    def build_latin_cooccurrence_set(self) -> Dict[str, set]:
        """
        Build a co-occurrence lookup for the Latin corpus.
        For each Latin word, stores the set of words that co-occur
        within the sliding window.
        """
        if self._latin_cooccurrence is not None:
            return self._latin_cooccurrence

        tokens = self.latin_corpus.get_tokens()
        cooccurrence = defaultdict(set)

        for i, token in enumerate(tokens):
            start = max(0, i - self.window)
            end = min(len(tokens), i + self.window + 1)
            for j in range(start, end):
                if i != j:
                    cooccurrence[token].add(tokens[j])

        self._latin_cooccurrence = dict(cooccurrence)
        return self._latin_cooccurrence

    def compute_topic_penalty(self, voynich_word1: str, voynich_word2: str,
                               latin1: str, latin2: str) -> float:
        """
        Compute topic coherence penalty for a pair of word mappings.

        If voynich_word1 and voynich_word2 share an NMF topic, their
        Latin mappings (latin1, latin2) should co-occur in the Latin corpus.

        Returns:
            0.0 if they share a topic AND Latin words co-occur
            1.0 if they share a topic BUT Latin words DON'T co-occur
            0.0 if they don't share a topic (no penalty)
        """
        voynich_data = self.extract_voynich_topics()
        word_topic = voynich_data.get('word_topic', {})

        topic1 = word_topic.get(voynich_word1, -1)
        topic2 = word_topic.get(voynich_word2, -1)

        if topic1 < 0 or topic2 < 0 or topic1 != topic2:
            return 0.0

        latin_cooc = self.build_latin_cooccurrence_set()
        if latin1 in latin_cooc and latin2 in latin_cooc[latin1]:
            return 0.0

        return 1.0

    def build_topic_coherence_matrix(self, voynich_vocab: List[str]) -> np.ndarray:
        """
        Build an NxN matrix where entry [i][j] = 1 if voynich_vocab[i] and
        voynich_vocab[j] share an NMF topic, 0 otherwise.

        This matrix is used by the SAA to know which word pairs need
        topic-coherent Latin mappings.
        """
        voynich_data = self.extract_voynich_topics()
        word_topic = voynich_data.get('word_topic', {})

        n = len(voynich_vocab)
        shared_topic = np.zeros((n, n), dtype=float)

        for i in range(n):
            t_i = word_topic.get(voynich_vocab[i], -1)
            if t_i < 0:
                continue
            for j in range(i + 1, n):
                t_j = word_topic.get(voynich_vocab[j], -1)
                if t_i == t_j:
                    shared_topic[i][j] = 1.0
                    shared_topic[j][i] = 1.0

        return shared_topic

    def run(self, verbose: bool = True) -> Dict:
        """Extract topics from both corpora and build coherence scaffold."""
        voynich_data = self.extract_voynich_topics()
        latin_data = self.extract_latin_topics()

        v_topic_summary = {
            k: [(w, round(s, 3)) for w, s in v[:5]]
            for k, v in voynich_data['topics'].items()
        }
        l_topic_summary = {
            k: [(w, round(s, 3)) for w, s in v[:5]]
            for k, v in latin_data['topics'].items()
        }

        results = {
            'voynich_topics': {
                'n_topics': voynich_data['n_topics'],
                'vocabulary_size': voynich_data['vocabulary_size'],
                'reconstruction_error': voynich_data['reconstruction_error'],
                'topic_summary': v_topic_summary,
                'n_words_assigned': len(voynich_data['word_topic']),
            },
            'latin_topics': {
                'n_topics': latin_data['n_topics'],
                'vocabulary_size': latin_data['vocabulary_size'],
                'reconstruction_error': latin_data['reconstruction_error'],
                'topic_summary': l_topic_summary,
                'n_words_assigned': len(latin_data['word_topic']),
            },
            'synthesis': {
                'conclusion': (
                    f'NMF scaffold: {voynich_data["n_topics"]} Voynich topics, '
                    f'{latin_data["n_topics"]} Latin topics. '
                    f'{len(voynich_data["word_topic"])} Voynich words assigned, '
                    f'{len(latin_data["word_topic"])} Latin words assigned.'
                ),
            },
        }

        if verbose:
            print(f'\n  NMF Topic Scaffold:')
            print(f'    --- Voynich ---')
            print(f'    Topics: {voynich_data["n_topics"]}, '
                  f'Vocab: {voynich_data["vocabulary_size"]}, '
                  f'Error: {voynich_data["reconstruction_error"]:.1f}')
            for tid, words in list(v_topic_summary.items())[:3]:
                top_words = [w for w, _ in words[:3]]
                print(f'      Topic {tid}: {top_words}')
            print(f'    --- Latin ---')
            print(f'    Topics: {latin_data["n_topics"]}, '
                  f'Vocab: {latin_data["vocabulary_size"]}, '
                  f'Error: {latin_data["reconstruction_error"]:.1f}')
            for tid, words in list(l_topic_summary.items())[:3]:
                top_words = [w for w, _ in words[:3]]
                print(f'      Topic {tid}: {top_words}')

        return results
