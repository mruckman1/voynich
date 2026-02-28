"""
Model 2: Syllabary Code (Plaintext Syllable → Ciphertext Word)
================================================================
Each plaintext syllable maps to one Voynich word. Medieval Latin has
~200-400 common syllables. With homophones, this requires 300-600
ciphertext word types.

Historical plausibility: HIGH
Predicted H2: 1.0–1.8
Priority: HIGH

Critical test: NMF effective rank ~12 decomposes into onset × nucleus × coda
classes, indicating the encoding reflects syllable structure.
"""

import random
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

from modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS, TRIPLE_THRESHOLDS
from modules.naibbe_cipher import PREFIX_GLYPHS, MEDIAL_GLYPHS, SUFFIX_GLYPHS, ANY_GLYPHS
from data.latin_syllables import (
    LATIN_SYLLABLES, syllabify, get_syllable_structure,
    get_top_syllables, ONSETS, NUCLEI, CODAS
)

class SyllabaryCode(Phase2GenerativeModel):
    """
    Model 2: Syllabary Code.

    Maps each plaintext syllable to one of several ciphertext words.
    The vocabulary size (~200-600) is determined by the number of
    distinct syllables in the source language × homophones per syllable.
    """

    MODEL_NAME = 'syllabary_code'
    MODEL_PRIORITY = 'HIGH'

    def __init__(self, n_syllable_types: int = 300,
                 homophones_per_syllable: int = 2,
                 word_length_min: int = 3, word_length_max: int = 8,
                 onset_coding: bool = True,
                 positional_bias: float = 0.5,
                 seed: int = 42, **kwargs):
        params = {
            'n_syllable_types': n_syllable_types,
            'homophones_per_syllable': homophones_per_syllable,
            'word_length_min': word_length_min,
            'word_length_max': word_length_max,
            'onset_coding': onset_coding,
            'positional_bias': positional_bias,
            'seed': seed,
        }
        super().__init__(**params)

        self.n_syllable_types = n_syllable_types
        self.homophones_per_syllable = homophones_per_syllable
        self.word_length_min = word_length_min
        self.word_length_max = word_length_max
        self.onset_coding = onset_coding
        self.positional_bias = positional_bias

        self.syllable_table = {}
        self._build_syllable_table()

    def _build_syllable_table(self):
        """
        Build mapping from Latin syllables to ciphertext words.

        If onset_coding is True, words with the same onset share a prefix,
        words with the same nucleus share a medial pattern, etc.
        This structural encoding is the key prediction of Model 2.
        """
        top_syllables = get_top_syllables(self.n_syllable_types)

        if self.onset_coding:
            self._build_structured_table(top_syllables)
        else:
            self._build_random_table(top_syllables)

    def _build_structured_table(self, syllable_freq_pairs: List[Tuple[str, float]]):
        """
        Build a structurally-coded table where syllable components map
        to word components:
        - onset -> word prefix
        - nucleus -> word medial
        - coda -> word suffix
        """
        onset_map = self._build_component_map(ONSETS[:15], PREFIX_GLYPHS + [''])
        nucleus_map = self._build_component_map(NUCLEI[:8], MEDIAL_GLYPHS)
        coda_map = self._build_component_map(CODAS[:12], SUFFIX_GLYPHS + ANY_GLYPHS[:3] + [''])

        for syllable, freq in syllable_freq_pairs:
            structure = get_syllable_structure(syllable)
            onset = structure['onset']
            nucleus = structure['nucleus']
            coda = structure['coda']

            words = []
            for _ in range(self.homophones_per_syllable):
                prefix = onset_map.get(onset, self.rng.choice(PREFIX_GLYPHS + ['']))
                middle = nucleus_map.get(nucleus, self.rng.choice(MEDIAL_GLYPHS))
                suffix = coda_map.get(coda, self.rng.choice(SUFFIX_GLYPHS + ['']))

                filler_len = self.rng.randint(0, 2)
                filler = ''.join(self.rng.choice(MEDIAL_GLYPHS) for _ in range(filler_len))

                word = prefix + middle + filler + suffix
                if len(word) < self.word_length_min:
                    word += self.rng.choice(MEDIAL_GLYPHS)
                if len(word) > self.word_length_max:
                    word = word[:self.word_length_max]

                words.append(word)

            self.syllable_table[syllable] = words

    def _build_random_table(self, syllable_freq_pairs: List[Tuple[str, float]]):
        """Build a random (non-structured) mapping table."""
        for syllable, freq in syllable_freq_pairs:
            words = []
            for _ in range(self.homophones_per_syllable):
                word = self._generate_random_word()
                words.append(word)
            self.syllable_table[syllable] = words

    def _build_component_map(self, components: List[str],
                             glyph_pool: list) -> Dict[str, str]:
        """Map phonological components to glyph sequences."""
        mapping = {}
        available = list(glyph_pool)
        self.rng.shuffle(available)

        for i, comp in enumerate(components):
            if i < len(available):
                mapping[comp] = available[i]
            else:
                mapping[comp] = self.rng.choice(glyph_pool) + self.rng.choice(MEDIAL_GLYPHS)
        return mapping

    def _generate_random_word(self) -> str:
        """Generate a random Voynich-like word."""
        length = self.rng.randint(self.word_length_min, self.word_length_max)
        chars = []

        if self.rng.random() < self.positional_bias:
            chars.append(self.rng.choice(PREFIX_GLYPHS))

        for _ in range(length - len(chars) - 1):
            chars.append(self.rng.choice(MEDIAL_GLYPHS))

        if self.rng.random() < self.positional_bias:
            chars.append(self.rng.choice(SUFFIX_GLYPHS))
        else:
            chars.append(self.rng.choice(MEDIAL_GLYPHS))

        return ''.join(chars)

    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Generate syllabary-coded text from plaintext.

        Each plaintext syllable produces one ciphertext word.
        """
        if plaintext:
            words = plaintext.lower().split()
            all_syllables = []
            for word in words:
                word_clean = ''.join(c for c in word if c.isalpha())
                if word_clean:
                    syls = syllabify(word_clean)
                    all_syllables.extend(syls)
        else:
            syllables = list(LATIN_SYLLABLES.keys())
            weights = list(LATIN_SYLLABLES.values())
            all_syllables = self.rng.choices(syllables, weights=weights, k=n_words)

        all_syllables = all_syllables[:n_words]

        output_words = []
        for syl in all_syllables:
            if syl in self.syllable_table:
                word = self.rng.choice(self.syllable_table[syl])
            else:
                word = self._fallback_encode(syl)
            output_words.append(word)

        return ' '.join(output_words)

    def _fallback_encode(self, syllable: str) -> str:
        """Encode an unknown syllable using component matching."""
        structure = get_syllable_structure(syllable)
        onset = structure['onset']

        for known_syl, words in self.syllable_table.items():
            known_struct = get_syllable_structure(known_syl)
            if known_struct['onset'] == onset:
                return self.rng.choice(words)

        return self._generate_random_word()

    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """Generate parameter sweep grid."""
        if resolution == 'coarse':
            n_types = [200, 300, 400]
            homophones = [1, 2, 3]
            onset_codings = [True, False]
            biases = [0.3, 0.5]
        elif resolution == 'medium':
            n_types = [150, 200, 250, 300, 400, 500]
            homophones = [1, 2, 3]
            onset_codings = [True, False]
            biases = [0.2, 0.4, 0.6]
        else:
            n_types = [100, 150, 200, 250, 300, 350, 400, 500, 600]
            homophones = [1, 2, 3, 4]
            onset_codings = [True, False]
            biases = [0.1, 0.3, 0.5, 0.7]

        grid = []
        for nt in n_types:
            for hp in homophones:
                for oc in onset_codings:
                    for bias in biases:
                        grid.append({
                            'n_syllable_types': nt,
                            'homophones_per_syllable': hp,
                            'onset_coding': oc,
                            'positional_bias': bias,
                            'seed': 42,
                        })
        return grid

    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Critical test: NMF rank and triple match.

        The syllabary model predicts that the NMF effective rank (~12)
        corresponds to the number of distinct syllable structure types
        (onset types x nucleus types x coda types). If rank 12 decomposes
        into 3-4 onset classes x 2-3 nucleus classes x 2 coda classes,
        the syllabary model is strongly supported.
        """
        entropy = generated_profile.get('entropy', {})
        zipf = generated_profile.get('zipf', {})

        h2 = entropy.get('H2', 0.0)
        ttr = zipf.get('type_token_ratio', 0.0)
        zipf_exp = zipf.get('zipf_exponent', 0.0)

        h2_match = abs(h2 - VOYNICH_TARGETS['H2']) < TRIPLE_THRESHOLDS['H2']
        ttr_match = abs(ttr - VOYNICH_TARGETS['type_token_ratio']) < TRIPLE_THRESHOLDS['TTR']
        zipf_match = abs(zipf_exp - VOYNICH_TARGETS['zipf_exponent']) < TRIPLE_THRESHOLDS['zipf_exponent']
        triple_match = h2_match and ttr_match and zipf_match

        return {
            'passes': triple_match,
            'description': (
                f'Triple: H2={h2:.4f} ({"PASS" if h2_match else "FAIL"}), '
                f'TTR={ttr:.4f} ({"PASS" if ttr_match else "FAIL"}), '
                f'Zipf={zipf_exp:.4f} ({"PASS" if zipf_match else "FAIL"}). '
                f'NMF decomposition test requires separate analysis.'
            ),
            'details': {
                'H2': h2, 'TTR': ttr, 'zipf_exponent': zipf_exp,
                'triple_match': triple_match,
                'nmf_test_pending': True,
            },
        }

    def run_nmf_test(self, text: str) -> Dict:
        """
        Run NMF decomposition test on generated text.

        Checks whether the effective rank decomposes into syllable
        structure components (onset x nucleus x coda).
        """
        from modules.nmf_analysis import BigramNMF

        nmf = BigramNMF()
        nmf.build_bigram_matrix(text)
        rank = nmf.optimal_rank()
        W, H, error = nmf.factorize(rank=rank)
        components = nmf.interpret_components(W, H)

        rank_match = abs(rank - VOYNICH_TARGETS['nmf_effective_rank']) <= 2

        return {
            'effective_rank': rank,
            'target_rank': VOYNICH_TARGETS['nmf_effective_rank'],
            'rank_match': rank_match,
            'reconstruction_error': error,
            'components': components,
        }
