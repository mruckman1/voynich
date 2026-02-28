"""
Language A Full-Corpus Extractor
=================================
Loads the full IVTFF corpus and isolates Language A text (Currier Language A,
herbal_a section, Scribe 1). This is the foundational module that all
Phase 4 analyses depend on.

Falls back to the SAMPLE_CORPUS via LanguageABSplitter if full IVTFF
files are not available.
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

from modules.statistical_analysis import (
    full_statistical_profile, compute_all_entropy, zipf_analysis,
    word_conditional_entropy, word_transition_matrix,
)
from modules.phase3.lang_a_reprofiling import LANG_A_TARGETS

LANG_A_FULL_TARGETS = {
    'H2_char': 1.487,
    'H2_word': None,
    'zipf_exponent': 0.931,
    'type_token_ratio': 0.305,
    'vocabulary_size': 57,
    'total_tokens': 187,
    'mean_word_length': 4.43,
}

class LanguageAExtractor:
    """
    Extract and profile Language A from the best available corpus source.

    Tries full IVTFF corpus first (via data/ivtff_parser.py), then falls
    back to SAMPLE_CORPUS (via LanguageABSplitter).
    """

    def __init__(self, corpus_dir: str = 'data/corpus', verbose: bool = True):
        self.verbose = verbose
        self.corpus_dir = corpus_dir
        self._corpus = None
        self._lang_a_text = None
        self._lang_a_tokens = None
        self._lang_a_by_folio = None
        self._word_freqs = None
        self._profile = None
        self._source = None

        self._load_corpus()

    def _load_corpus(self):
        """Try full IVTFF corpus, fall back to SAMPLE_CORPUS."""
        try:
            from data.ivtff_parser import load_corpus
            self._corpus = load_corpus(self.corpus_dir, verbose=self.verbose)
            self._source = 'ivtff'
            if self.verbose:
                print(f'  Language A Extractor: loaded IVTFF corpus')
        except (FileNotFoundError, Exception) as e:
            if self.verbose:
                print(f'  IVTFF not available ({e}), falling back to SAMPLE_CORPUS')
            self._source = 'sample'

    def extract_lang_a_text(self) -> str:
        """Get all Language A text as a single string."""
        if self._lang_a_text is not None:
            return self._lang_a_text

        if self._source == 'ivtff' and self._corpus is not None:
            self._lang_a_text = self._corpus.get_text(
                language='A', paragraph_only=True
            )
            if not self._lang_a_text.strip():
                self._lang_a_text = self._corpus.get_text(
                    hand=1, paragraph_only=True
                )
        else:
            from modules.phase2.cross_cutting import LanguageABSplitter
            splitter = LanguageABSplitter()
            self._lang_a_text, _ = splitter.split_corpus()

        if self.verbose:
            tokens = self._lang_a_text.split()
            print(f'  Language A text: {len(tokens)} tokens, '
                  f'{len(set(tokens))} types, source={self._source}')

        return self._lang_a_text

    def extract_lang_a_tokens(self) -> List[str]:
        """Get all Language A word tokens."""
        if self._lang_a_tokens is None:
            text = self.extract_lang_a_text()
            self._lang_a_tokens = [t for t in text.split() if t]
        return self._lang_a_tokens

    def extract_lang_a_by_folio(self) -> Dict[str, List[str]]:
        """
        Get Language A tokens organized by folio.
        Returns: {folio_id: [token, token, ...]}
        """
        if self._lang_a_by_folio is not None:
            return self._lang_a_by_folio

        self._lang_a_by_folio = {}

        if self._source == 'ivtff' and self._corpus is not None:
            for folio, page in self._corpus.pages.items():
                if page.language != 'A':
                    continue
                tokens = page.paragraph_text.split()
                if tokens:
                    self._lang_a_by_folio[folio] = tokens
        else:
            from data.voynich_corpus import SAMPLE_CORPUS
            for folio, data in SAMPLE_CORPUS.items():
                if data.get('lang') == 'A':
                    text = ' '.join(data.get('text', []))
                    tokens = text.split()
                    if tokens:
                        self._lang_a_by_folio[folio] = tokens

        if self.verbose:
            print(f'  Language A folios: {len(self._lang_a_by_folio)}')

        return self._lang_a_by_folio

    def compute_word_frequencies(self) -> Counter:
        """Compute word frequency distribution for Language A."""
        if self._word_freqs is None:
            self._word_freqs = Counter(self.extract_lang_a_tokens())
        return self._word_freqs

    def compute_full_profile(self) -> Dict:
        """Compute comprehensive statistical profile of Language A."""
        if self._profile is None:
            text = self.extract_lang_a_text()
            tokens = self.extract_lang_a_tokens()

            self._profile = full_statistical_profile(text, 'language_a_full')

            self._profile['word_entropy'] = {
                'H2_word': word_conditional_entropy(tokens, order=1),
                'H3_word': word_conditional_entropy(tokens, order=2),
            }
            self._profile['word_level_zipf'] = zipf_analysis(tokens)

        return self._profile

    def build_word_bigram_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build the 57x57 (or NxN) word-level transition probability matrix.

        Returns:
            (matrix, vocabulary) where matrix[i][j] = P(word_j | word_i)
        """
        return word_transition_matrix(self.extract_lang_a_tokens())

    def separate_frequent_and_singleton(self, threshold: int = 2) -> Dict:
        """
        Split vocabulary into frequent (count >= threshold) and singleton words.

        Returns dict with:
            frequent_types: list of word types appearing >= threshold times
            singleton_types: list of word types appearing < threshold times
            frequent_tokens: subsequence of tokens (only frequent words)
            singleton_tokens: subsequence of tokens (only singletons)
            frequent_text: joined text of frequent-word subsequence
        """
        freqs = self.compute_word_frequencies()
        tokens = self.extract_lang_a_tokens()

        frequent_types = sorted(w for w, c in freqs.items() if c >= threshold)
        singleton_types = sorted(w for w, c in freqs.items() if c < threshold)

        frequent_set = set(frequent_types)
        singleton_set = set(singleton_types)

        frequent_tokens = [t for t in tokens if t in frequent_set]
        singleton_tokens = [t for t in tokens if t in singleton_set]
        frequent_text = ' '.join(frequent_tokens)

        return {
            'frequent_types': frequent_types,
            'singleton_types': singleton_types,
            'n_frequent_types': len(frequent_types),
            'n_singleton_types': len(singleton_types),
            'frequent_tokens': frequent_tokens,
            'singleton_tokens': singleton_tokens,
            'n_frequent_tokens': len(frequent_tokens),
            'n_singleton_tokens': len(singleton_tokens),
            'frequent_text': frequent_text,
        }

    def validate_against_phase3(self) -> Dict:
        """
        Compare full-corpus Language A metrics against Phase 3 targets.
        Reports whether the full corpus confirms the sample-based estimates.
        """
        profile = self.compute_full_profile()
        tokens = self.extract_lang_a_tokens()
        freqs = self.compute_word_frequencies()

        measured = {
            'H2_char': profile['entropy']['H2'],
            'vocabulary_size': len(freqs),
            'total_tokens': len(tokens),
            'type_token_ratio': len(freqs) / max(1, len(tokens)),
            'zipf_exponent': profile['zipf']['zipf_exponent'],
            'mean_word_length': float(profile.get('mean_word_length', 0)),
            'H2_word': profile['word_entropy']['H2_word'],
        }

        deltas = {}
        for key in ['H2_char', 'vocabulary_size', 'type_token_ratio', 'zipf_exponent']:
            target = LANG_A_TARGETS.get(
                key if key != 'H2_char' else 'H2',
                LANG_A_FULL_TARGETS.get(key)
            )
            if target is not None:
                deltas[key] = measured[key] - target

        return {
            'source': self._source,
            'measured': measured,
            'phase3_targets': dict(LANG_A_TARGETS),
            'deltas': deltas,
            'full_corpus_confirms_sample': all(
                abs(v) < 0.3 for v in deltas.values()
                if isinstance(v, float)
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run full extraction and validation."""
        tokens = self.extract_lang_a_tokens()
        by_folio = self.extract_lang_a_by_folio()
        freqs = self.compute_word_frequencies()
        profile = self.compute_full_profile()
        validation = self.validate_against_phase3()
        split = self.separate_frequent_and_singleton()

        results = {
            'source': self._source,
            'total_tokens': len(tokens),
            'vocabulary_size': len(freqs),
            'type_token_ratio': len(freqs) / max(1, len(tokens)),
            'n_folios': len(by_folio),
            'entropy': profile['entropy'],
            'word_entropy': profile['word_entropy'],
            'zipf': {
                'zipf_exponent': profile['zipf']['zipf_exponent'],
                'r_squared': profile['zipf']['r_squared'],
            },
            'top_20_words': freqs.most_common(20),
            'vocabulary_split': {
                'frequent_types': split['n_frequent_types'],
                'singleton_types': split['n_singleton_types'],
                'frequent_tokens': split['n_frequent_tokens'],
                'singleton_tokens': split['n_singleton_tokens'],
            },
            'validation': validation,
        }

        if verbose:
            print(f'\n  Language A Extraction Results:')
            print(f'    Source: {self._source}')
            print(f'    Tokens: {len(tokens)}, Types: {len(freqs)}')
            print(f'    H2 (char): {profile["entropy"]["H2"]:.3f}')
            print(f'    H2 (word): {profile["word_entropy"]["H2_word"]:.3f}')
            print(f'    Zipf exponent: {profile["zipf"]["zipf_exponent"]:.3f}')
            print(f'    TTR: {len(freqs)/max(1,len(tokens)):.3f}')
            print(f'    Folios: {len(by_folio)}')
            print(f'    Frequent/Singleton split: '
                  f'{split["n_frequent_types"]}/{split["n_singleton_types"]} types')
            print(f'    Validates Phase 3: {validation["full_corpus_confirms_sample"]}')

        return results
