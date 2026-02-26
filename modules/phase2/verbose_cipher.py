"""
Model 1: Verbose Cipher (Plaintext Letter → Ciphertext Word)
==============================================================
Each plaintext letter maps to one of several ciphertext *words*.
A plaintext word like "herba" (5 letters) becomes 5 Voynich words.

The ciphertext vocabulary is small (20-40 word types, one or more per
plaintext letter). Homophones exist: the letter 'e' might map to any
of {chedy, shedy, otedy}, selected by context or random choice.

Historical plausibility: HIGH
Predicted H2: 1.2–1.8
Priority: HIGHEST

Critical test: H2 within 0.1 of 1.41 AND TTR within 0.02 of 0.164
AND Zipf within 0.15 of 1.24, all simultaneously.
"""

import sys
import os
import random
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS, TRIPLE_THRESHOLDS
from modules.naibbe_cipher import PREFIX_GLYPHS, MEDIAL_GLYPHS, SUFFIX_GLYPHS, ANY_GLYPHS


# ============================================================================
# LATIN LETTER FREQUENCIES (medieval medical Latin)
# ============================================================================

# Approximate letter frequencies for medieval Latin medical text
LATIN_LETTER_FREQ = {
    'a': 0.085, 'b': 0.015, 'c': 0.035, 'd': 0.030,
    'e': 0.120, 'f': 0.008, 'g': 0.012, 'h': 0.010,
    'i': 0.095, 'l': 0.035, 'm': 0.040, 'n': 0.065,
    'o': 0.055, 'p': 0.025, 'q': 0.008, 'r': 0.065,
    's': 0.060, 't': 0.070, 'u': 0.055, 'v': 0.010,
    'x': 0.003,
}

# Sort letters by frequency for homophone allocation
_LETTERS_BY_FREQ = sorted(LATIN_LETTER_FREQ.items(), key=lambda x: -x[1])
LATIN_ALPHABET = [l for l, _ in _LETTERS_BY_FREQ]


class VerboseCipher(Phase2GenerativeModel):
    """
    Model 1: Verbose Cipher.

    Maps each plaintext letter to one of several complete ciphertext words.
    The output vocabulary is deliberately small (20-40 types) to produce
    the Voynich's low TTR. Character transitions within each fixed word
    are deterministic, producing low H2.
    """

    MODEL_NAME = 'verbose_cipher'
    MODEL_PRIORITY = 'HIGHEST'

    def __init__(self, vocab_size: int = 30, homophones_per_letter: int = 3,
                 word_length_min: int = 3, word_length_max: int = 7,
                 positional_bias: float = 0.5, seed: int = 42, **kwargs):
        params = {
            'vocab_size': vocab_size,
            'homophones_per_letter': homophones_per_letter,
            'word_length_min': word_length_min,
            'word_length_max': word_length_max,
            'positional_bias': positional_bias,
            'seed': seed,
        }
        super().__init__(**params)

        self.vocab_size = vocab_size
        self.homophones_per_letter = homophones_per_letter
        self.word_length_min = word_length_min
        self.word_length_max = word_length_max
        self.positional_bias = positional_bias

        self.mapping_table = {}  # letter -> [word1, word2, ...]
        self._build_mapping_table()

    def _build_mapping_table(self):
        """
        Build the letter-to-word-list mapping.

        Higher-frequency letters get more homophones. Each word is assembled
        from EVA glyphs respecting positional constraints.
        """
        # Calculate homophone allocation based on letter frequency
        total_freq = sum(LATIN_LETTER_FREQ.values())
        words_used = 0
        remaining_vocab = self.vocab_size

        for letter, freq in _LETTERS_BY_FREQ:
            if remaining_vocab <= 0:
                # Assign at least 1 word to remaining letters
                self.mapping_table[letter] = [self._generate_word()]
                continue

            # Allocate homophones proportional to frequency
            relative_freq = freq / total_freq
            n_homophones = max(1, round(
                relative_freq * self.vocab_size * self.homophones_per_letter / len(LATIN_ALPHABET)
            ))
            n_homophones = min(n_homophones, self.homophones_per_letter, remaining_vocab)

            words = []
            for _ in range(n_homophones):
                words.append(self._generate_word())
            self.mapping_table[letter] = words
            remaining_vocab -= n_homophones

    def _generate_word(self) -> str:
        """
        Generate a single Voynich-like word respecting positional constraints.

        Structure: [optional_prefix] + medial_body + [optional_suffix]
        """
        length = self.rng.randint(self.word_length_min, self.word_length_max)
        chars = []

        # Optional prefix (gallows/initial glyph)
        if self.rng.random() < self.positional_bias and length >= 3:
            chars.append(self.rng.choice(PREFIX_GLYPHS))
            length -= 1

        # Body (medial glyphs)
        body_length = max(1, length - 1)  # Reserve 1 for suffix
        for _ in range(body_length):
            if self.rng.random() < 0.7:
                chars.append(self.rng.choice(MEDIAL_GLYPHS))
            else:
                chars.append(self.rng.choice(ANY_GLYPHS))

        # Optional suffix
        if self.rng.random() < self.positional_bias:
            chars.append(self.rng.choice(SUFFIX_GLYPHS))
        elif len(chars) < self.word_length_min:
            chars.append(self.rng.choice(MEDIAL_GLYPHS))

        return ''.join(chars)

    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Generate verbose-cipher text from plaintext.

        Each plaintext letter produces one ciphertext word.
        If no plaintext is provided, generates random Latin-frequency letters.
        """
        if plaintext:
            letters = [c.lower() for c in plaintext if c.isalpha()]
        else:
            # Generate random letters with Latin frequency distribution
            letters_list = list(LATIN_LETTER_FREQ.keys())
            weights = list(LATIN_LETTER_FREQ.values())
            letters = self.rng.choices(letters_list, weights=weights, k=n_words)

        # Limit to n_words
        letters = letters[:n_words]

        output_words = []
        for letter in letters:
            if letter in self.mapping_table:
                word = self.rng.choice(self.mapping_table[letter])
            else:
                # Unknown letter — use a random mapping
                word = self.rng.choice(self.mapping_table.get('e', ['chedy']))
            output_words.append(word)

        return ' '.join(output_words)

    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """Generate parameter sweep grid."""
        if resolution == 'coarse':
            vocab_sizes = [20, 30, 40]
            homophones = [2, 3, 4]
            wl_ranges = [(3, 6), (3, 8)]
            biases = [0.3, 0.5, 0.7]
        elif resolution == 'medium':
            vocab_sizes = [18, 22, 26, 30, 34, 38, 42]
            homophones = [1, 2, 3, 4, 5]
            wl_ranges = [(2, 5), (3, 6), (3, 7), (3, 8), (4, 8)]
            biases = [0.2, 0.4, 0.6, 0.8]
        else:  # fine
            vocab_sizes = list(range(16, 46, 2))
            homophones = [1, 2, 3, 4, 5, 6]
            wl_ranges = [(2, 5), (3, 5), (3, 6), (3, 7), (3, 8), (4, 7), (4, 8), (4, 9)]
            biases = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        grid = []
        for vs in vocab_sizes:
            for hp in homophones:
                for wl_min, wl_max in wl_ranges:
                    for bias in biases:
                        grid.append({
                            'vocab_size': vs,
                            'homophones_per_letter': hp,
                            'word_length_min': wl_min,
                            'word_length_max': wl_max,
                            'positional_bias': bias,
                            'seed': 42,
                        })
        return grid

    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Critical test: simultaneous H2/TTR/Zipf triple match.

        All three must match simultaneously — matching any one is easy,
        matching all three is the real constraint.
        """
        entropy = generated_profile.get('entropy', {})
        zipf = generated_profile.get('zipf', {})

        h2 = entropy.get('H2', 0.0)
        ttr = zipf.get('type_token_ratio', 0.0)
        zipf_exp = zipf.get('zipf_exponent', 0.0)

        h2_match = abs(h2 - VOYNICH_TARGETS['H2']) < TRIPLE_THRESHOLDS['H2']
        ttr_match = abs(ttr - VOYNICH_TARGETS['type_token_ratio']) < TRIPLE_THRESHOLDS['TTR']
        zipf_match = abs(zipf_exp - VOYNICH_TARGETS['zipf_exponent']) < TRIPLE_THRESHOLDS['zipf_exponent']

        passes = h2_match and ttr_match and zipf_match

        return {
            'passes': passes,
            'description': (
                f'Triple test: H2={h2:.4f} ({"PASS" if h2_match else "FAIL"}), '
                f'TTR={ttr:.4f} ({"PASS" if ttr_match else "FAIL"}), '
                f'Zipf={zipf_exp:.4f} ({"PASS" if zipf_match else "FAIL"})'
            ),
            'details': {
                'H2': h2,
                'H2_target': VOYNICH_TARGETS['H2'],
                'H2_delta': abs(h2 - VOYNICH_TARGETS['H2']),
                'H2_match': h2_match,
                'TTR': ttr,
                'TTR_target': VOYNICH_TARGETS['type_token_ratio'],
                'TTR_delta': abs(ttr - VOYNICH_TARGETS['type_token_ratio']),
                'TTR_match': ttr_match,
                'zipf_exponent': zipf_exp,
                'zipf_target': VOYNICH_TARGETS['zipf_exponent'],
                'zipf_delta': abs(zipf_exp - VOYNICH_TARGETS['zipf_exponent']),
                'zipf_match': zipf_match,
                'all_match': passes,
            },
        }

    def get_mapping_summary(self) -> Dict:
        """Return summary of the letter-to-word mapping table."""
        summary = {}
        for letter, words in sorted(self.mapping_table.items()):
            summary[letter] = {
                'n_homophones': len(words),
                'words': words,
                'avg_length': sum(len(w) for w in words) / max(len(words), 1),
            }
        return {
            'total_unique_words': len(set(
                w for words in self.mapping_table.values() for w in words
            )),
            'letters_covered': len(self.mapping_table),
            'per_letter': summary,
        }
