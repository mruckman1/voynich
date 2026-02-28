"""
Latin Herbal Reference Corpus Builder
=======================================
Provides a medieval Latin herbal text for Model A1 testing and SAA matching.

Three-tier strategy:
  1. Hardcoded Circa Instans excerpts (~2000 words)
  2. Synthetic generation from existing vocabulary/templates
  3. HuggingFace datasets (attempt to load Latin corpus)
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
)
from data.voynich_corpus import HARTLIEB_MEDICAL_VOCAB, LATIN_RECIPE_FORMULAS
from data.botanical_identifications import (
    PLANT_IDS, PLANT_PART_TERMS, HUMORAL_LABEL_TERMS, DEGREE_TERMS,
    HUMORAL_QUALITIES,
)
from data.medieval_text_templates import (
    OPENING_FORMULAS, CLOSING_FORMULAS, PARAGRAPH_STATS,
)

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'json')
with open(os.path.join(_DATA_DIR, 'circa_instans.json')) as _f:
    _ci_data = json.load(_f)

CIRCA_INSTANS_ENTRIES = _ci_data['CIRCA_INSTANS_ENTRIES']
LATIN_FUNCTION_WORDS = _ci_data['LATIN_FUNCTION_WORDS']
LATIN_QUALITY_WORDS = _ci_data['LATIN_QUALITY_WORDS']
LATIN_DEGREE_WORDS = _ci_data['LATIN_DEGREE_WORDS']
LATIN_PROPERTY_WORDS = _ci_data['LATIN_PROPERTY_WORDS']
LATIN_BODY_WORDS = _ci_data['LATIN_BODY_WORDS']
LATIN_SUBSTANCE_WORDS = _ci_data['LATIN_SUBSTANCE_WORDS']
LATIN_PLANT_NAMES = _ci_data['LATIN_PLANT_NAMES']
LATIN_CLOSING_PHRASES = _ci_data['LATIN_CLOSING_PHRASES']

del _ci_data, _f

def _generate_synthetic_entry(rng: random.Random) -> str:
    """Generate one synthetic herbal entry following Circa Instans structure."""
    plant = rng.choice(LATIN_PLANT_NAMES)
    quality = rng.choice(['calida', 'frigida'])
    moisture = rng.choice(['sicca', 'humida'])
    degree = rng.choice(['primo', 'secundo', 'tertio'])

    props = rng.sample(LATIN_PROPERTY_WORDS, k=min(2, len(LATIN_PROPERTY_WORDS)))
    bodies = rng.sample(LATIN_BODY_WORDS, k=min(2, len(LATIN_BODY_WORDS)))
    substance = rng.choice(LATIN_SUBSTANCE_WORDS)

    parts = [
        f'{plant} est {quality} et {moisture} in {degree} gradu',
        f'habet virtutem {props[0]}',
    ]
    if len(props) > 1:
        parts.append(f'et {props[1]}')

    parts.append(f'valet contra {rng.choice(["dolorem", "inflammationem", "fluxum"])} '
                 f'{bodies[0]}')
    if len(bodies) > 1:
        parts.append(f'et contra {bodies[1]}')

    action = rng.choice(['recipe', 'accipe'])
    prep = rng.choice(['contere', 'coque in aqua', 'misce cum vino',
                        'fac emplastrum', 'destilla per alembicum'])
    parts.append(f'{action} {plant} et {prep}')

    if rng.random() < 0.5:
        parts.append(f'cum {substance}')

    delivery = rng.choice(['et da bibere', 'et pone super locum',
                           'et unge', 'et lava', 'et bibe'])
    parts.append(delivery)
    parts.append(rng.choice(LATIN_CLOSING_PHRASES))

    return ' '.join(parts)

class LatinHerbalCorpus:
    """
    Provides a reference medieval Latin herbal corpus for Phase 4 analysis.

    The critical deliverable: compute word-bigram H2 of realistic Latin
    herbal text. If H2 ≈ 1.49 ± 0.2, the codebook model is supported.
    """

    def __init__(self, method: str = 'auto', seed: int = 42,
                 verbose: bool = True):
        self.method = method
        self.seed = seed
        self.verbose = verbose
        self._corpus_text = None
        self._tokens = None
        self._profile = None

    def _get_circa_instans_text(self) -> str:
        """Return the hardcoded Circa Instans excerpts."""
        return ' '.join(CIRCA_INSTANS_ENTRIES)

    def _generate_synthetic_text(self, n_entries: int = 30) -> str:
        """Generate synthetic Latin herbal entries."""
        rng = random.Random(self.seed)
        entries = [_generate_synthetic_entry(rng) for _ in range(n_entries)]
        return ' '.join(entries)

    def _try_huggingface(self) -> Optional[str]:
        """Attempt to load Latin text from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            ds = load_dataset('latin_library', split='train', trust_remote_code=True)
            texts = []
            for item in ds:
                text = item.get('text', '')
                if any(kw in text.lower() for kw in
                       ['herba', 'radix', 'recipe', 'contra', 'calidus']):
                    texts.append(text[:2000])
                if len(texts) >= 10:
                    break
            if texts:
                return ' '.join(texts)
        except Exception:
            pass
        return None

    def get_corpus(self) -> str:
        """Return the best available Latin herbal corpus text."""
        if self._corpus_text is not None:
            return self._corpus_text

        if self.method == 'circa_instans':
            self._corpus_text = self._get_circa_instans_text()
        elif self.method == 'synthetic':
            self._corpus_text = self._generate_synthetic_text(n_entries=50)
        elif self.method == 'combined':
            circa = self._get_circa_instans_text()
            synth = self._generate_synthetic_text(n_entries=20)
            self._corpus_text = circa + ' ' + synth
        elif self.method == 'auto':
            circa = self._get_circa_instans_text()
            synth = self._generate_synthetic_text(n_entries=20)
            self._corpus_text = circa + ' ' + synth
        else:
            self._corpus_text = self._get_circa_instans_text()

        if self.verbose:
            tokens = self._corpus_text.split()
            print(f'  Latin herbal corpus: {len(tokens)} tokens, '
                  f'{len(set(tokens))} types, method={self.method}')

        return self._corpus_text

    def get_tokens(self) -> List[str]:
        """Get tokenized corpus."""
        if self._tokens is None:
            self._tokens = [t for t in self.get_corpus().split() if t]
        return self._tokens

    def compute_word_bigram_h2(self) -> float:
        """THE CRITICAL METRIC: word-level bigram conditional entropy."""
        tokens = self.get_tokens()
        return word_conditional_entropy(tokens, order=1)

    def compute_word_trigram_h3(self) -> float:
        """Word-level trigram conditional entropy."""
        return word_conditional_entropy(self.get_tokens(), order=2)

    def compute_full_profile(self) -> Dict:
        """Compute comprehensive profile of the Latin herbal corpus."""
        if self._profile is None:
            text = self.get_corpus()
            tokens = self.get_tokens()
            self._profile = full_statistical_profile(text, 'latin_herbal')
            self._profile['word_entropy'] = {
                'H2_word': self.compute_word_bigram_h2(),
                'H3_word': self.compute_word_trigram_h3(),
            }
            self._profile['word_level_zipf'] = zipf_analysis(tokens)
        return self._profile

    def build_word_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Build word-level transition probability matrix."""
        tokens = self.get_tokens()
        vocab = sorted(set(tokens))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)
        counts = np.zeros((n, n), dtype=float)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            counts[word_to_idx[w1]][word_to_idx[w2]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = counts / row_sums
        return matrix, vocab

    def run(self, verbose: bool = True) -> Dict:
        """Run corpus construction and compute all metrics."""
        tokens = self.get_tokens()
        h2_word = self.compute_word_bigram_h2()
        h3_word = self.compute_word_trigram_h3()
        profile = self.compute_full_profile()
        zipf = zipf_analysis(tokens)

        results = {
            'method': self.method,
            'total_tokens': len(tokens),
            'vocabulary_size': len(set(tokens)),
            'type_token_ratio': len(set(tokens)) / max(1, len(tokens)),
            'word_bigram_h2': h2_word,
            'word_trigram_h3': h3_word,
            'char_entropy': profile['entropy'],
            'zipf': {
                'zipf_exponent': zipf['zipf_exponent'],
                'r_squared': zipf['r_squared'],
            },
            'top_20_words': Counter(tokens).most_common(20),
            'codebook_h2_test': {
                'latin_word_bigram_h2': h2_word,
                'voynich_lang_a_h2': 1.487,
                'delta': abs(h2_word - 1.487),
                'within_range': abs(h2_word - 1.487) < 0.2,
                'range': [1.29, 1.69],
            },
        }

        if verbose:
            print(f'\n  Latin Herbal Corpus Results:')
            print(f'    Method: {self.method}')
            print(f'    Tokens: {len(tokens)}, Types: {len(set(tokens))}')
            print(f'    Word-bigram H2: {h2_word:.3f}')
            print(f'    Word-trigram H3: {h3_word:.3f}')
            print(f'    Char H2: {profile["entropy"]["H2"]:.3f}')
            print(f'    Zipf exponent: {zipf["zipf_exponent"]:.3f}')
            print(f'    TTR: {len(set(tokens))/max(1,len(tokens)):.3f}')
            print(f'    --- Codebook H2 Test ---')
            print(f'    Latin word-bigram H2: {h2_word:.3f}')
            print(f'    Voynich Lang A H2:    1.487')
            print(f'    Delta:                {abs(h2_word - 1.487):.3f}')
            print(f'    Within [1.29, 1.69]:  {abs(h2_word - 1.487) < 0.2}')

        return results
