"""
Model A1: Whole-Word Codebook
================================
Tests the hypothesis that each of 57 Language A Voynich words maps 1:1
to a plaintext word. Under this model, the character-level H2 of the
ciphertext equals the word-bigram entropy of the plaintext.

Priority: HIGHEST

Critical prediction: If Language A is a whole-word codebook over Latin,
then the word-bigram H2 of a medieval Latin herbal should match
H2 = 1.49 ± 0.2. This is a clean, falsifiable test.
"""

import os
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from modules.statistical_analysis import (
    word_conditional_entropy, zipf_analysis, compute_all_entropy,
)
from modules.phase4.lang_a_extractor import LanguageAExtractor, LANG_A_FULL_TARGETS
from modules.phase4.latin_herbal_corpus import LatinHerbalCorpus

class WholeWordCodebook:
    """
    Test the whole-word codebook hypothesis for Language A.

    If each Voynich word maps to exactly one plaintext word (57-entry
    codebook), then:
    - The word-bigram H2 of the plaintext source equals the
      character-level H2 of Language A (~1.487)
    - The Zipf distribution of Voynich word frequencies matches
      the Zipf of source-language word frequencies
    - The TTR of Language A matches the source text TTR
    - 57 codebook entries is plausible for a herbal vocabulary
    """

    def __init__(self, extractor: LanguageAExtractor,
                 latin_corpus: LatinHerbalCorpus):
        self.extractor = extractor
        self.latin_corpus = latin_corpus

    def test_h2_match(self) -> Dict:
        """
        THE critical test: does word-bigram H2 of the Latin herbal
        match Language A's character-level H2?

        Pass criterion: |H2_latin_word_bigram - 1.49| < 0.2
        """
        latin_h2 = self.latin_corpus.compute_word_bigram_h2()
        voynich_h2 = self.extractor.compute_full_profile()['entropy']['H2']

        delta = abs(latin_h2 - voynich_h2)
        passes = delta < 0.2

        latin_h3 = self.latin_corpus.compute_word_trigram_h3()
        voynich_h3 = self.extractor.compute_full_profile()['entropy']['H3']

        interpretation = self._interpret_h2_result(latin_h2, voynich_h2, delta, passes)

        return {
            'latin_word_bigram_h2': latin_h2,
            'voynich_lang_a_char_h2': voynich_h2,
            'delta_h2': delta,
            'passes_h2': passes,
            'latin_word_trigram_h3': latin_h3,
            'voynich_lang_a_char_h3': voynich_h3,
            'delta_h3': abs(latin_h3 - voynich_h3),
            'pass_criterion': '|delta| < 0.2',
            'interpretation': interpretation,
        }

    def test_zipf_compatibility(self) -> Dict:
        """
        Compare Zipf distribution of Language A words to Zipf of
        Latin herbal word frequencies.

        Under the codebook model, the rank-frequency distribution of
        Voynich words mirrors the source text, because the 1:1 mapping
        preserves frequency ordering.
        """
        voynich_tokens = self.extractor.extract_lang_a_tokens()
        latin_tokens = self.latin_corpus.get_tokens()

        voynich_zipf = zipf_analysis(voynich_tokens)
        latin_zipf = zipf_analysis(latin_tokens)

        delta_exp = abs(voynich_zipf['zipf_exponent'] - latin_zipf['zipf_exponent'])

        return {
            'voynich_zipf_exponent': voynich_zipf['zipf_exponent'],
            'latin_zipf_exponent': latin_zipf['zipf_exponent'],
            'delta_exponent': delta_exp,
            'compatible': delta_exp < 0.3,
            'voynich_r_squared': voynich_zipf['r_squared'],
            'latin_r_squared': latin_zipf['r_squared'],
        }

    def test_ttr_compatibility(self) -> Dict:
        """
        Compare type-token ratios.

        The TTR of Language A should be close to the TTR of the source
        text, since a 1:1 codebook preserves vocabulary size relative
        to text length.
        """
        voynich_tokens = self.extractor.extract_lang_a_tokens()
        latin_tokens = self.latin_corpus.get_tokens()

        voynich_ttr = len(set(voynich_tokens)) / max(1, len(voynich_tokens))
        latin_ttr = len(set(latin_tokens)) / max(1, len(latin_tokens))

        delta = abs(voynich_ttr - latin_ttr)

        return {
            'voynich_ttr': voynich_ttr,
            'latin_ttr': latin_ttr,
            'delta_ttr': delta,
            'compatible': delta < 0.15,
            'note': ('TTR comparison is approximate — text lengths differ '
                     'significantly, which naturally affects TTR'),
        }

    def estimate_codebook_size(self) -> Dict:
        """
        Analyze whether a 57-entry codebook is plausible for encoding
        a Latin herbal text.

        Compare Language A's 57 types to the vocabulary structure of
        the Latin herbal.
        """
        voynich_freqs = self.extractor.compute_word_frequencies()
        latin_tokens = self.latin_corpus.get_tokens()
        latin_freqs = Counter(latin_tokens)

        latin_vocab = len(latin_freqs)

        top_57_latin = latin_freqs.most_common(57)
        top_57_count = sum(c for _, c in top_57_latin)
        total_latin = len(latin_tokens)
        coverage = top_57_count / max(1, total_latin)

        top_10_count = sum(c for _, c in latin_freqs.most_common(10))
        concentration = top_10_count / max(1, total_latin)

        return {
            'voynich_vocabulary_size': len(voynich_freqs),
            'latin_vocabulary_size': latin_vocab,
            'top_57_latin_coverage': coverage,
            'top_10_latin_concentration': concentration,
            'plausible': coverage > 0.7,
            'interpretation': (
                f'Top 57 Latin words cover {coverage:.1%} of the text. '
                f'A 57-entry codebook is {"plausible" if coverage > 0.7 else "insufficient"} '
                f'for encoding this text.'
            ),
        }

    def test_frequency_rank_correlation(self) -> Dict:
        """
        Under the codebook model, the most frequent Voynich word should
        map to the most frequent Latin word, etc. Test whether the
        frequency rank distributions are compatible.
        """
        voynich_freqs = self.extractor.compute_word_frequencies()
        latin_freqs = Counter(self.latin_corpus.get_tokens())

        voynich_ranked = sorted(voynich_freqs.values(), reverse=True)
        latin_ranked = sorted(latin_freqs.values(), reverse=True)

        n = min(len(voynich_ranked), len(latin_ranked), 20)
        voynich_top = np.array(voynich_ranked[:n], dtype=float)
        latin_top = np.array(latin_ranked[:n], dtype=float)

        voynich_prop = voynich_top / voynich_top.sum()
        latin_prop = latin_top / latin_top.sum()

        if n >= 3:
            correlation = float(np.corrcoef(voynich_prop, latin_prop)[0, 1])
        else:
            correlation = 0.0

        return {
            'n_compared': n,
            'rank_correlation': correlation,
            'compatible': correlation > 0.7,
            'voynich_top_5_proportions': voynich_prop[:5].tolist(),
            'latin_top_5_proportions': latin_prop[:5].tolist(),
        }

    def _interpret_h2_result(self, latin_h2: float, voynich_h2: float,
                              delta: float, passes: bool) -> str:
        """Generate interpretation string for H2 match result."""
        if passes and delta < 0.1:
            return (
                f'STRONG MATCH: Latin word-bigram H2 ({latin_h2:.3f}) is very '
                f'close to Voynich Language A char H2 ({voynich_h2:.3f}). '
                f'Delta={delta:.3f}. The whole-word codebook model is '
                f'strongly supported.'
            )
        elif passes:
            return (
                f'MODERATE MATCH: Latin word-bigram H2 ({latin_h2:.3f}) is within '
                f'tolerance of Voynich Language A char H2 ({voynich_h2:.3f}). '
                f'Delta={delta:.3f}. The codebook model is plausibly supported.'
            )
        elif delta < 0.3:
            return (
                f'MARGINAL: Latin word-bigram H2 ({latin_h2:.3f}) is outside '
                f'strict tolerance but close to Voynich H2 ({voynich_h2:.3f}). '
                f'Delta={delta:.3f}. Codebook model is weakly supported — '
                f'corpus size or text type may need adjustment.'
            )
        else:
            return (
                f'NO MATCH: Latin word-bigram H2 ({latin_h2:.3f}) diverges from '
                f'Voynich H2 ({voynich_h2:.3f}). Delta={delta:.3f}. '
                f'The codebook model is not supported for this source language, '
                f'or the reference corpus is not representative.'
            )

    def _synthesize(self, h2_result: Dict, zipf_result: Dict,
                    ttr_result: Dict, size_result: Dict,
                    rank_result: Dict) -> Dict:
        """Combine all test results into a synthesis."""
        tests_passed = sum([
            h2_result['passes_h2'],
            zipf_result['compatible'],
            ttr_result['compatible'],
            size_result['plausible'],
            rank_result['compatible'],
        ])

        if tests_passed >= 4:
            confidence = 'STRONG'
            supported = True
        elif tests_passed >= 3:
            confidence = 'MODERATE'
            supported = True
        elif tests_passed >= 2:
            confidence = 'WEAK'
            supported = False
        else:
            confidence = 'UNSUPPORTED'
            supported = False

        return {
            'codebook_supported': supported,
            'confidence': confidence,
            'tests_passed': tests_passed,
            'tests_total': 5,
            'h2_passes': h2_result['passes_h2'],
            'zipf_passes': zipf_result['compatible'],
            'ttr_passes': ttr_result['compatible'],
            'size_passes': size_result['plausible'],
            'rank_passes': rank_result['compatible'],
            'conclusion': (
                f'Whole-word codebook model: {confidence}. '
                f'{tests_passed}/5 tests passed. '
                f'{"The codebook hypothesis is viable — proceed to SAA." if supported else "The codebook hypothesis is not well-supported."}'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all codebook model tests."""
        h2_result = self.test_h2_match()
        zipf_result = self.test_zipf_compatibility()
        ttr_result = self.test_ttr_compatibility()
        size_result = self.estimate_codebook_size()
        rank_result = self.test_frequency_rank_correlation()

        synthesis = self._synthesize(
            h2_result, zipf_result, ttr_result, size_result, rank_result
        )

        results = {
            'h2_match_test': h2_result,
            'zipf_compatibility': zipf_result,
            'ttr_compatibility': ttr_result,
            'codebook_size_analysis': size_result,
            'frequency_rank_correlation': rank_result,
            'synthesis': synthesis,
        }

        if verbose:
            print(f'\n  Model A1: Whole-Word Codebook Results')
            print(f'    --- H2 Match (Critical Test) ---')
            print(f'    Latin word-bigram H2: {h2_result["latin_word_bigram_h2"]:.3f}')
            print(f'    Voynich Lang A H2:    {h2_result["voynich_lang_a_char_h2"]:.3f}')
            print(f'    Delta:                {h2_result["delta_h2"]:.3f}')
            print(f'    PASSES:               {h2_result["passes_h2"]}')
            print(f'    --- Zipf ---')
            print(f'    Compatible: {zipf_result["compatible"]} '
                  f'(delta={zipf_result["delta_exponent"]:.3f})')
            print(f'    --- TTR ---')
            print(f'    Compatible: {ttr_result["compatible"]} '
                  f'(delta={ttr_result["delta_ttr"]:.3f})')
            print(f'    --- Codebook Size ---')
            print(f'    Plausible: {size_result["plausible"]} '
                  f'(top-57 coverage={size_result["top_57_latin_coverage"]:.1%})')
            print(f'    --- Rank Correlation ---')
            print(f'    Compatible: {rank_result["compatible"]} '
                  f'(r={rank_result["rank_correlation"]:.3f})')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
