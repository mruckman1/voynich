"""
Expanded Latin Herbal Corpus Builder (20,000–50,000 tokens)
=============================================================
Provides a much larger medieval Latin herbal reference corpus for
Phase 5's constrained SAA.

Sources:
  1. Hardcoded Circa Instans excerpts (from Phase 4)
  2. Additional entries modeled on Macer Floridus, Herbarius Latinus,
     and Pseudo-Apuleius Herbarius
  3. Expanded synthetic generation with 8-10 template structures

Phase 5  ·  Voynich Convergence Attack
"""

import os
import json
import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

from modules.statistical_analysis import (
    word_conditional_entropy, zipf_analysis, full_statistical_profile,
    word_transition_matrix,
)
from modules.phase4.latin_herbal_corpus import CIRCA_INSTANS_ENTRIES

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'json')
with open(os.path.join(_DATA_DIR, 'macer_floridus.json')) as _f:
    _mf_data = json.load(_f)

EXPANDED_PLANT_NAMES = _mf_data['EXPANDED_PLANT_NAMES']
EXPANDED_BODY_WORDS = _mf_data['EXPANDED_BODY_WORDS']
CONDITION_WORDS = _mf_data['CONDITION_WORDS']
PREPARATION_WORDS = _mf_data['PREPARATION_WORDS']
DELIVERY_WORDS = _mf_data['DELIVERY_WORDS']
EXPANDED_SUBSTANCE_WORDS = _mf_data['EXPANDED_SUBSTANCE_WORDS']
EXPANDED_PROPERTY_WORDS = _mf_data['EXPANDED_PROPERTY_WORDS']
DOSAGE_WORDS = _mf_data['DOSAGE_WORDS']
TIME_WORDS = _mf_data['TIME_WORDS']
EXPANDED_CLOSING_PHRASES = _mf_data['EXPANDED_CLOSING_PHRASES']
TRANSITIONAL_PHRASES = _mf_data['TRANSITIONAL_PHRASES']
MACER_FLORIDUS_ENTRIES = _mf_data['MACER_FLORIDUS_ENTRIES']

del _mf_data, _f

def _template_standard(rng, plant, quality, moisture, degree):
    """Standard Circa Instans entry: name-quality-properties-remedy-closing."""
    props = rng.sample(EXPANDED_PROPERTY_WORDS, k=min(2, len(EXPANDED_PROPERTY_WORDS)))
    bodies = rng.sample(EXPANDED_BODY_WORDS, k=min(2, len(EXPANDED_BODY_WORDS)))
    condition = rng.choice(CONDITION_WORDS)
    prep = rng.choice(PREPARATION_WORDS)
    delivery = rng.choice(DELIVERY_WORDS)
    closing = rng.choice(EXPANDED_CLOSING_PHRASES)
    parts = [
        f'{plant} est {quality} et {moisture} in {degree} gradu',
        f'habet virtutem {props[0]}',
    ]
    if len(props) > 1:
        parts.append(f'et {props[1]}')
    parts.append(f'valet contra {condition} {bodies[0]}')
    if len(bodies) > 1:
        parts.append(f'et contra {rng.choice(CONDITION_WORDS)} {bodies[1]}')
    parts.append(f'recipe {plant} et {prep}')
    parts.append(delivery)
    parts.append(closing)
    return ' '.join(parts)

def _template_multi_remedy(rng, plant, quality, moisture, degree):
    """Entry with multiple remedies separated by 'item' or 'praeterea'."""
    props = rng.sample(EXPANDED_PROPERTY_WORDS, k=min(3, len(EXPANDED_PROPERTY_WORDS)))
    condition1 = rng.choice(CONDITION_WORDS)
    condition2 = rng.choice(CONDITION_WORDS)
    body1 = rng.choice(EXPANDED_BODY_WORDS)
    body2 = rng.choice(EXPANDED_BODY_WORDS)
    parts = [
        f'{plant} est {quality} et {moisture} in {degree} gradu',
        f'habet virtutem {props[0]} et {props[1]}',
        f'valet contra {condition1} {body1} et contra {condition2} {body2}',
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)}',
        rng.choice(DELIVERY_WORDS),
        rng.choice(EXPANDED_CLOSING_PHRASES),
        rng.choice(TRANSITIONAL_PHRASES),
        f'{plant} {rng.choice(PREPARATION_WORDS)}',
        f'valet contra {rng.choice(CONDITION_WORDS)}',
        rng.choice(DELIVERY_WORDS),
        rng.choice(EXPANDED_CLOSING_PHRASES),
    ]
    return ' '.join(parts)

def _template_short_warning(rng, plant, quality, moisture, degree):
    """Short entry with danger warning (for toxic plants)."""
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'sed est periculosus recipe {plant} cum magna cautela '
        f'et in parva dosi {rng.choice(DELIVERY_WORDS)} '
        f'cave ne des {rng.choice(["gravidis", "infantibus", "senibus", "debilibus"])}'
    )

def _template_compound_preparation(rng, plant, quality, moisture, degree):
    """Entry with compound preparation involving multiple ingredients."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'et {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {plant2} ana partes aequales '
        f'et contere et misce cum {substance} '
        f'et da {dosage} {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(TIME_WORDS)} {rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_external_application(rng, plant, quality, moisture, degree):
    """Entry focused on external application (poultice, ointment, bath)."""
    application = rng.choice([
        f'fac emplastrum cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} et pone super locum dolentem',
        f'fac unguentum cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} et unge locum',
        f'fac cataplasma et applica super {rng.choice(EXPANDED_BODY_WORDS)}',
        f'fac balneum et immitte membrum in aqua',
        f'fac fomentum et pone super locum dolentem',
    ])
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'et {application} '
        f'{rng.choice(TIME_WORDS)} et {rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_detailed_dosage(rng, plant, quality, moisture, degree):
    """Entry with detailed dosage instructions."""
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'et {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)} '
        f'et {rng.choice(PREPARATION_WORDS)} '
        f'et da {rng.choice(DOSAGE_WORDS)} '
        f'{rng.choice(TIME_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_comparative(rng, plant, quality, moisture, degree):
    """Entry comparing properties to another plant."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'similiter ut {plant2} sed est fortior '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_seasonal(rng, plant, quality, moisture, degree):
    """Entry with seasonal harvesting instructions."""
    season = rng.choice([
        'in vere collige', 'in aestate collige', 'in autumno collige',
        'ante florem collige', 'post florem collige',
        'in plenilunio collige',
    ])
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'{season} {plant} et {rng.choice(["sicca in umbra", "sicca in sole", "conserva in loco sicco"])} '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

TEMPLATE_GENERATORS = [
    _template_standard,
    _template_multi_remedy,
    _template_short_warning,
    _template_compound_preparation,
    _template_external_application,
    _template_detailed_dosage,
    _template_comparative,
    _template_seasonal,
]

class ExpandedLatinHerbalCorpus:
    """
    Provides an expanded medieval Latin herbal reference corpus for
    Phase 5's constrained SAA.

    Target: 20,000–50,000 tokens with rich template variety.
    """

    def __init__(self, target_tokens: int = 30000, seed: int = 42,
                 verbose: bool = True):
        self.target_tokens = target_tokens
        self.seed = seed
        self.verbose = verbose
        self._corpus_text = None
        self._tokens = None
        self._profile = None

    def build_corpus(self) -> str:
        """Build the expanded corpus from three sources."""
        if self._corpus_text is not None:
            return self._corpus_text

        parts = []
        parts.extend(CIRCA_INSTANS_ENTRIES)
        parts.extend(MACER_FLORIDUS_ENTRIES)

        current_text = ' '.join(parts)
        current_tokens = len(current_text.split())

        rng = random.Random(self.seed)
        remaining = self.target_tokens - current_tokens

        while remaining > 0:
            plant = rng.choice(EXPANDED_PLANT_NAMES)
            quality = rng.choice(['calida', 'frigida', 'calidus', 'frigidus',
                                  'calidum', 'frigidum'])
            moisture = rng.choice(['sicca', 'humida', 'siccus', 'humidus',
                                   'siccum', 'humidum'])
            degree = rng.choice(['primo', 'secundo', 'tertio', 'quarto'])
            template = rng.choice(TEMPLATE_GENERATORS)
            entry = template(rng, plant, quality, moisture, degree)
            parts.append(entry)
            remaining -= len(entry.split())

        self._corpus_text = ' '.join(parts)

        if self.verbose:
            tokens = self._corpus_text.split()
            print(f'  Expanded Latin corpus: {len(tokens)} tokens, '
                  f'{len(set(tokens))} types')

        return self._corpus_text

    def get_corpus(self) -> str:
        """Return the corpus text (alias for build_corpus)."""
        return self.build_corpus()

    def get_tokens(self) -> List[str]:
        """Get tokenized corpus."""
        if self._tokens is None:
            self._tokens = [t for t in self.build_corpus().split() if t]
        return self._tokens

    def get_top_n_words(self, n: int = 1000) -> List[Tuple[str, int]]:
        """Return the top N most frequent words with their counts."""
        return Counter(self.get_tokens()).most_common(n)

    def build_transition_matrix(self, top_n: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """Build word-level transition matrix restricted to the top N most frequent words."""
        return word_transition_matrix(self.get_tokens(), top_n=top_n)

    def compute_word_bigram_h2(self) -> float:
        """Compute word-level bigram conditional entropy."""
        return word_conditional_entropy(self.get_tokens(), order=1)

    def compute_full_profile(self) -> Dict:
        """Compute comprehensive statistical profile."""
        if self._profile is None:
            text = self.get_corpus()
            tokens = self.get_tokens()
            self._profile = full_statistical_profile(text, 'latin_herbal_expanded')
            self._profile['word_entropy'] = {
                'H2_word': self.compute_word_bigram_h2(),
                'H3_word': word_conditional_entropy(tokens, order=2),
            }
            self._profile['word_level_zipf'] = zipf_analysis(tokens)
        return self._profile

    def run(self, verbose: bool = True) -> Dict:
        """Build corpus and compute all metrics."""
        tokens = self.get_tokens()
        h2_word = self.compute_word_bigram_h2()
        h3_word = word_conditional_entropy(tokens, order=2)
        profile = self.compute_full_profile()
        zipf = zipf_analysis(tokens)

        results = {
            'target_tokens': self.target_tokens,
            'actual_tokens': len(tokens),
            'vocabulary_size': len(set(tokens)),
            'type_token_ratio': len(set(tokens)) / max(1, len(tokens)),
            'word_bigram_h2': h2_word,
            'word_trigram_h3': h3_word,
            'char_entropy': profile['entropy'],
            'zipf': {
                'zipf_exponent': zipf['zipf_exponent'],
                'r_squared': zipf['r_squared'],
            },
            'top_30_words': Counter(tokens).most_common(30),
            'voynich_comparison': {
                'latin_word_bigram_h2': h2_word,
                'voynich_h2': 2.385,
                'delta': abs(h2_word - 2.385),
                'within_range': abs(h2_word - 2.385) < 0.3,
            },
        }

        if verbose:
            print(f'\n  Expanded Latin Herbal Corpus:')
            print(f'    Target:  {self.target_tokens} tokens')
            print(f'    Actual:  {len(tokens)} tokens, {len(set(tokens))} types')
            print(f'    TTR:     {len(set(tokens))/max(1,len(tokens)):.3f}')
            print(f'    Word H2: {h2_word:.3f}')
            print(f'    Word H3: {h3_word:.3f}')
            print(f'    Char H2: {profile["entropy"]["H2"]:.3f}')
            print(f'    Zipf:    {zipf["zipf_exponent"]:.3f}')
            print(f'    --- Voynich Comparison ---')
            print(f'    Latin word H2:   {h2_word:.3f}')
            print(f'    Voynich H2:      2.385')
            print(f'    Delta:           {abs(h2_word - 2.385):.3f}')
            print(f'    Within range:    {abs(h2_word - 2.385) < 0.3}')

        return results
