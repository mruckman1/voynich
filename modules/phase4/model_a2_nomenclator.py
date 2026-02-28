"""
Model A2: Nomenclator with Word-Level Codes
=============================================
Tests the hypothesis that Language A uses a mixed system: word-level
codes for the top ~26 frequent words, and character-level substitutions
for the remaining ~31 singleton/rare words.

Priority: HIGH

Critical test: Extract frequent-word-only subsequences from Language A.
If H2 drops significantly below 1.487, the nomenclator model is supported
(the codebook tier has lower entropy than the cipher tier).
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List

from modules.statistical_analysis import (
    compute_all_entropy, zipf_analysis, word_conditional_entropy,
    first_order_entropy, conditional_entropy,
)
from modules.phase4.lang_a_extractor import LanguageAExtractor

class NomenclatorModel:
    """
    Test the nomenclator hypothesis for Language A.

    A nomenclator uses:
    - Word-level codes for common terms (the codebook tier)
    - Character-level substitution for unique content (the cipher tier)

    If Language A is a nomenclator, the frequent words (codebook tier)
    should have different statistical properties than singleton words
    (cipher tier).
    """

    def __init__(self, extractor: LanguageAExtractor):
        self.extractor = extractor

    def compute_frequent_only_h2(self) -> Dict:
        """
        THE critical test: compute H2 of the frequent-word-only subsequence.

        If H2 drops significantly below 1.487, the codebook tier has
        lower entropy (more structured transitions), supporting the
        nomenclator model.
        """
        split = self.extractor.separate_frequent_and_singleton()
        all_tokens = self.extractor.extract_lang_a_tokens()

        full_text = self.extractor.extract_lang_a_text()
        full_h2 = conditional_entropy(full_text, order=1)

        frequent_text = split['frequent_text']
        frequent_h2 = conditional_entropy(frequent_text, order=1) if frequent_text else 0.0

        freq_tokens = split['frequent_tokens']
        word_h2_frequent = word_conditional_entropy(freq_tokens, order=1) if len(freq_tokens) > 2 else 0.0

        word_h2_full = word_conditional_entropy(all_tokens, order=1)

        h2_drop = full_h2 - frequent_h2
        significant_drop = h2_drop > 0.15

        return {
            'full_char_h2': full_h2,
            'frequent_only_char_h2': frequent_h2,
            'h2_drop': h2_drop,
            'significant_drop': significant_drop,
            'word_h2_frequent': word_h2_frequent,
            'word_h2_full': word_h2_full,
            'n_frequent_tokens': len(freq_tokens),
            'n_frequent_types': split['n_frequent_types'],
            'pass_criterion': 'h2_drop > 0.15',
            'interpretation': (
                f'Frequent-word H2 ({frequent_h2:.3f}) vs full H2 ({full_h2:.3f}): '
                f'drop={h2_drop:.3f}. '
                f'{"Significant — nomenclator tier structure detected." if significant_drop else "Not significant — vocabulary is not clearly two-tiered."}'
            ),
        }

    def test_singleton_character_structure(self) -> Dict:
        """
        Test whether singleton words have internal character-level
        patterns consistent with a substitution cipher.

        Singleton words (appearing once) should have:
        - Higher within-word character entropy (more diverse characters)
        - Different character frequency distribution from frequent words
        """
        split = self.extractor.separate_frequent_and_singleton()

        freq_chars = Counter()
        for token in split['frequent_tokens']:
            freq_chars.update(token)

        sing_chars = Counter()
        for token in split['singleton_tokens']:
            sing_chars.update(token)

        def char_entropy(counter):
            total = sum(counter.values())
            if total == 0:
                return 0.0
            return -sum((c / total) * math.log2(c / total)
                        for c in counter.values() if c > 0)

        freq_h1 = char_entropy(freq_chars)
        sing_h1 = char_entropy(sing_chars)

        freq_lengths = [len(t) for t in split['frequent_tokens']]
        sing_lengths = [len(t) for t in split['singleton_tokens']]

        freq_mean_len = np.mean(freq_lengths) if freq_lengths else 0
        sing_mean_len = np.mean(sing_lengths) if sing_lengths else 0

        freq_alphabet = len(freq_chars)
        sing_alphabet = len(sing_chars)

        return {
            'frequent_char_h1': freq_h1,
            'singleton_char_h1': sing_h1,
            'h1_difference': sing_h1 - freq_h1,
            'frequent_mean_word_length': float(freq_mean_len),
            'singleton_mean_word_length': float(sing_mean_len),
            'frequent_alphabet_size': freq_alphabet,
            'singleton_alphabet_size': sing_alphabet,
            'cipher_like_singletons': (
                sing_h1 > freq_h1 and sing_mean_len > freq_mean_len
            ),
            'interpretation': (
                f'Singleton char H1={sing_h1:.3f} vs frequent H1={freq_h1:.3f}. '
                f'Singleton mean length={sing_mean_len:.1f} vs frequent={freq_mean_len:.1f}. '
                f'{"Singletons show cipher-like character diversity." if sing_h1 > freq_h1 else "No clear cipher signal in singletons."}'
            ),
        }

    def estimate_plaintext_split(self) -> Dict:
        """
        Estimate what fraction of plaintext is codebook vs cipher encoded.
        """
        split = self.extractor.separate_frequent_and_singleton()

        total_tokens = split['n_frequent_tokens'] + split['n_singleton_tokens']
        codebook_fraction = split['n_frequent_tokens'] / max(1, total_tokens)
        cipher_fraction = split['n_singleton_tokens'] / max(1, total_tokens)

        return {
            'codebook_fraction': codebook_fraction,
            'cipher_fraction': cipher_fraction,
            'codebook_types': split['n_frequent_types'],
            'cipher_types': split['n_singleton_types'],
            'total_types': split['n_frequent_types'] + split['n_singleton_types'],
            'interpretation': (
                f'{codebook_fraction:.1%} of tokens are codebook-encoded '
                f'({split["n_frequent_types"]} types), '
                f'{cipher_fraction:.1%} are cipher-encoded '
                f'({split["n_singleton_types"]} types).'
            ),
        }

    def test_transition_patterns(self) -> Dict:
        """
        Test whether transitions between frequent and singleton words
        have distinct patterns.

        In a nomenclator, codebook words should have different transition
        probabilities than cipher words.
        """
        tokens = self.extractor.extract_lang_a_tokens()
        split = self.extractor.separate_frequent_and_singleton()
        frequent_set = set(split['frequent_types'])

        ff_count = 0
        fs_count = 0
        sf_count = 0
        ss_count = 0

        for i in range(len(tokens) - 1):
            curr_freq = tokens[i] in frequent_set
            next_freq = tokens[i + 1] in frequent_set
            if curr_freq and next_freq:
                ff_count += 1
            elif curr_freq and not next_freq:
                fs_count += 1
            elif not curr_freq and next_freq:
                sf_count += 1
            else:
                ss_count += 1

        total = ff_count + fs_count + sf_count + ss_count
        if total == 0:
            return {'error': 'No transitions found'}

        p_freq = split['n_frequent_tokens'] / max(1, len(tokens))
        p_sing = 1 - p_freq

        expected_ff = p_freq * p_freq * total
        expected_fs = p_freq * p_sing * total
        expected_sf = p_sing * p_freq * total
        expected_ss = p_sing * p_sing * total

        return {
            'observed': {
                'freq_to_freq': ff_count,
                'freq_to_sing': fs_count,
                'sing_to_freq': sf_count,
                'sing_to_sing': ss_count,
            },
            'expected_random': {
                'freq_to_freq': round(expected_ff, 1),
                'freq_to_sing': round(expected_fs, 1),
                'sing_to_freq': round(expected_sf, 1),
                'sing_to_sing': round(expected_ss, 1),
            },
            'ff_enrichment': ff_count / max(1, expected_ff),
            'ss_enrichment': ss_count / max(1, expected_ss),
            'non_random_transitions': (
                abs(ff_count - expected_ff) > 2 * math.sqrt(max(1, expected_ff))
            ),
        }

    def _synthesize(self, h2_test: Dict, char_test: Dict,
                    split_test: Dict, trans_test: Dict) -> Dict:
        """Combine all test results into a synthesis."""
        supported_signals = sum([
            h2_test.get('significant_drop', False),
            char_test.get('cipher_like_singletons', False),
            trans_test.get('non_random_transitions', False),
        ])

        if supported_signals >= 2:
            confidence = 'MODERATE'
            supported = True
        elif supported_signals >= 1:
            confidence = 'WEAK'
            supported = False
        else:
            confidence = 'UNSUPPORTED'
            supported = False

        return {
            'nomenclator_supported': supported,
            'confidence': confidence,
            'supporting_signals': supported_signals,
            'signals_total': 3,
            'h2_drop_significant': h2_test.get('significant_drop', False),
            'cipher_like_singletons': char_test.get('cipher_like_singletons', False),
            'non_random_transitions': trans_test.get('non_random_transitions', False),
            'conclusion': (
                f'Nomenclator model: {confidence}. '
                f'{supported_signals}/3 signals detected. '
                f'{"Two-tier encoding structure present." if supported else "Vocabulary is not clearly two-tiered."}'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all nomenclator model tests."""
        h2_test = self.compute_frequent_only_h2()
        char_test = self.test_singleton_character_structure()
        split_test = self.estimate_plaintext_split()
        trans_test = self.test_transition_patterns()

        synthesis = self._synthesize(h2_test, char_test, split_test, trans_test)

        results = {
            'frequent_only_h2': h2_test,
            'singleton_character_structure': char_test,
            'plaintext_split': split_test,
            'transition_patterns': trans_test,
            'synthesis': synthesis,
        }

        if verbose:
            print(f'\n  Model A2: Nomenclator Results')
            print(f'    --- Frequent-Only H2 (Critical Test) ---')
            print(f'    Full H2:         {h2_test["full_char_h2"]:.3f}')
            print(f'    Frequent-only:   {h2_test["frequent_only_char_h2"]:.3f}')
            print(f'    H2 drop:         {h2_test["h2_drop"]:.3f}')
            print(f'    Significant:     {h2_test["significant_drop"]}')
            print(f'    --- Singleton Character Structure ---')
            print(f'    Cipher-like:     {char_test["cipher_like_singletons"]}')
            print(f'    --- Plaintext Split ---')
            print(f'    Codebook:        {split_test["codebook_fraction"]:.1%}')
            print(f'    Cipher:          {split_test["cipher_fraction"]:.1%}')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
