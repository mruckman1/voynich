"""
Approach 3: Entropy Gradient by Page Position
================================================
Tests whether H2 varies systematically between the first and last
quarters of each page. A gradient would segment the codebook into
header words (formulaic openers) vs body words (content-bearing).

Prediction: Under the codebook model, the first words on a herbal
page should be formulaic (plant name, genus label → low H2) while
the middle and end should be more variable (symptoms, preparations
→ higher H2).
"""

import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List

from voynich.core.stats import (
    conditional_entropy, first_order_entropy, word_conditional_entropy,
)
from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor

class EntropyGradientAnalysis:
    """
    Test whether H2 varies by position within each page.

    Segments each folio's tokens into quartiles and compares
    statistical properties between early (Q1) and late (Q4) text.
    """

    def __init__(self, extractor: LanguageAExtractor):
        self.extractor = extractor

    def compute_position_quartiles(self) -> Dict:
        """
        For each Language A folio, split tokens into Q1 (first 25%)
        and Q4 (last 25%). Pool across all folios.

        Returns quartile token lists and text.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()

        q1_tokens = []
        q4_tokens = []
        q2q3_tokens = []

        for folio, tokens in by_folio.items():
            n = len(tokens)
            if n < 4:
                continue

            q_size = max(1, n // 4)

            q1_tokens.extend(tokens[:q_size])
            q4_tokens.extend(tokens[-q_size:])
            q2q3_tokens.extend(tokens[q_size:-q_size])

        return {
            'q1_tokens': q1_tokens,
            'q4_tokens': q4_tokens,
            'q2q3_tokens': q2q3_tokens,
            'q1_text': ' '.join(q1_tokens),
            'q4_text': ' '.join(q4_tokens),
            'q2q3_text': ' '.join(q2q3_tokens),
            'n_folios': len(by_folio),
            'n_q1_tokens': len(q1_tokens),
            'n_q4_tokens': len(q4_tokens),
        }

    def compute_h2_by_quartile(self) -> Dict:
        """
        Compute character-level H2 separately for Q1 and Q4.

        If Q1 H2 < Q4 H2, the beginning of pages is more formulaic.
        """
        quartiles = self.compute_position_quartiles()

        q1_h2 = conditional_entropy(quartiles['q1_text'], order=1) if quartiles['q1_text'] else 0
        q4_h2 = conditional_entropy(quartiles['q4_text'], order=1) if quartiles['q4_text'] else 0
        full_h2 = conditional_entropy(self.extractor.extract_lang_a_text(), order=1)

        q1_h1 = first_order_entropy(quartiles['q1_text']) if quartiles['q1_text'] else 0
        q4_h1 = first_order_entropy(quartiles['q4_text']) if quartiles['q4_text'] else 0

        gradient = q4_h2 - q1_h2

        return {
            'q1_char_h2': q1_h2,
            'q4_char_h2': q4_h2,
            'full_char_h2': full_h2,
            'gradient_h2': gradient,
            'q1_char_h1': q1_h1,
            'q4_char_h1': q4_h1,
            'gradient_h1': q4_h1 - q1_h1,
            'q1_tokens': quartiles['n_q1_tokens'],
            'q4_tokens': quartiles['n_q4_tokens'],
        }

    def compute_word_frequency_by_quartile(self) -> Dict:
        """
        Compare word frequency distributions between Q1 and Q4.

        Under the header/body hypothesis, certain words should be
        concentrated at page starts (header words like plant names)
        and others in the middle/end (body words like properties).
        """
        quartiles = self.compute_position_quartiles()

        q1_freqs = Counter(quartiles['q1_tokens'])
        q4_freqs = Counter(quartiles['q4_tokens'])

        all_words = set(q1_freqs.keys()) | set(q4_freqs.keys())

        q1_total = max(1, sum(q1_freqs.values()))
        q4_total = max(1, sum(q4_freqs.values()))

        enrichment = {}
        for word in all_words:
            q1_rate = q1_freqs.get(word, 0) / q1_total
            q4_rate = q4_freqs.get(word, 0) / q4_total
            ratio = (q1_rate + 1e-6) / (q4_rate + 1e-6)
            enrichment[word] = {
                'q1_count': q1_freqs.get(word, 0),
                'q4_count': q4_freqs.get(word, 0),
                'q1_rate': q1_rate,
                'q4_rate': q4_rate,
                'q1_enrichment': ratio,
            }

        q1_enriched = sorted(enrichment.items(),
                             key=lambda x: -x[1]['q1_enrichment'])
        q4_enriched = sorted(enrichment.items(),
                             key=lambda x: x[1]['q1_enrichment'])

        return {
            'q1_enriched_words': [(w, d['q1_enrichment'])
                                   for w, d in q1_enriched[:10]],
            'q4_enriched_words': [(w, d['q1_enrichment'])
                                   for w, d in q4_enriched[:10]],
            'full_enrichment': enrichment,
            'q1_vocabulary_size': len(q1_freqs),
            'q4_vocabulary_size': len(q4_freqs),
        }

    def test_gradient_significance(self) -> Dict:
        """
        Test whether the H2 gradient is statistically significant
        using a bootstrap procedure.

        Permutes token assignments to positions and recomputes the
        gradient many times to build a null distribution.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()

        h2_data = self.compute_h2_by_quartile()
        observed_gradient = h2_data['gradient_h2']

        rng = np.random.RandomState(42)
        n_bootstrap = 100
        null_gradients = []

        all_tokens = self.extractor.extract_lang_a_tokens()

        for _ in range(n_bootstrap):
            shuffled = list(all_tokens)
            rng.shuffle(shuffled)

            n = len(shuffled)
            q_size = n // 4
            q1_text = ' '.join(shuffled[:q_size])
            q4_text = ' '.join(shuffled[-q_size:])

            q1_h2 = conditional_entropy(q1_text, order=1) if q1_text else 0
            q4_h2 = conditional_entropy(q4_text, order=1) if q4_text else 0

            null_gradients.append(q4_h2 - q1_h2)

        null_gradients = np.array(null_gradients)
        null_mean = float(np.mean(null_gradients))
        null_std = float(np.std(null_gradients))

        if observed_gradient > 0:
            p_value = float(np.mean(null_gradients >= observed_gradient))
        else:
            p_value = float(np.mean(null_gradients <= observed_gradient))

        significant = p_value < 0.05

        return {
            'observed_gradient': observed_gradient,
            'null_mean': null_mean,
            'null_std': null_std,
            'p_value': p_value,
            'significant': significant,
            'n_bootstrap': n_bootstrap,
            'interpretation': (
                f'Gradient={observed_gradient:.4f}, p={p_value:.3f}. '
                f'{"Significant — page position affects entropy." if significant else "Not significant — no positional structure detected."}'
            ),
        }

    def identify_positional_words(self) -> Dict:
        """
        Find words that appear predominantly at the start or end
        of pages. These are candidate header/footer words.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()

        first_words = Counter()
        last_words = Counter()
        all_words = Counter()

        for folio, tokens in by_folio.items():
            if len(tokens) < 2:
                continue
            first_words[tokens[0]] += 1
            last_words[tokens[-1]] += 1
            all_words.update(tokens)

        n_folios = len(by_folio)
        total = sum(all_words.values())

        header_candidates = []
        for word, count in first_words.most_common(20):
            expected = all_words[word] / max(1, total) * n_folios
            if expected > 0:
                enrichment = count / expected
                header_candidates.append({
                    'word': word,
                    'first_count': count,
                    'total_count': all_words[word],
                    'enrichment': enrichment,
                })

        footer_candidates = []
        for word, count in last_words.most_common(20):
            expected = all_words[word] / max(1, total) * n_folios
            if expected > 0:
                enrichment = count / expected
                footer_candidates.append({
                    'word': word,
                    'last_count': count,
                    'total_count': all_words[word],
                    'enrichment': enrichment,
                })

        return {
            'header_candidates': header_candidates[:10],
            'footer_candidates': footer_candidates[:10],
            'n_folios': n_folios,
        }

    def _synthesize(self, h2_data: Dict, significance: Dict,
                    freq_data: Dict, positional: Dict) -> Dict:
        """Combine all results."""
        gradient_exists = significance.get('significant', False)
        has_header_words = any(
            h.get('enrichment', 0) > 2.0
            for h in positional.get('header_candidates', [])
        )

        return {
            'gradient_detected': gradient_exists,
            'header_words_detected': has_header_words,
            'supports_codebook_structure': gradient_exists or has_header_words,
            'conclusion': (
                f'Entropy gradient: {"significant" if gradient_exists else "not significant"} '
                f'(p={significance.get("p_value", 1.0):.3f}). '
                f'Header words: {"detected" if has_header_words else "not detected"}. '
                f'{"Page structure supports header/body vocabulary split." if gradient_exists or has_header_words else "No clear positional structure."}'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run entropy gradient analysis."""
        h2_data = self.compute_h2_by_quartile()
        freq_data = self.compute_word_frequency_by_quartile()
        significance = self.test_gradient_significance()
        positional = self.identify_positional_words()

        synthesis = self._synthesize(h2_data, significance, freq_data, positional)

        results = {
            'h2_by_quartile': h2_data,
            'word_frequency_by_quartile': {
                'q1_enriched': freq_data['q1_enriched_words'],
                'q4_enriched': freq_data['q4_enriched_words'],
                'q1_vocab_size': freq_data['q1_vocabulary_size'],
                'q4_vocab_size': freq_data['q4_vocabulary_size'],
            },
            'significance_test': significance,
            'positional_words': positional,
            'synthesis': synthesis,
        }

        if verbose:
            print(f'\n  Approach 3: Entropy Gradient by Page Position')
            print(f'    Q1 (first 25%) H2: {h2_data["q1_char_h2"]:.3f}')
            print(f'    Q4 (last 25%) H2:  {h2_data["q4_char_h2"]:.3f}')
            print(f'    Gradient:          {h2_data["gradient_h2"]:.4f}')
            print(f'    P-value:           {significance["p_value"]:.3f}')
            print(f'    Significant:       {significance["significant"]}')
            print(f'    Header candidates: {len(positional["header_candidates"])}')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
