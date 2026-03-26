"""
Medieval Text Templates and Reference Statistics
==================================================
Reference paragraph structures, formulaic patterns, and expected word counts
for medieval Latin text types relevant to the Voynich Manuscript content.
"""

import json
import os

from voynich.core._paths import data_dir, json_dir
_DIR = str(data_dir())
with open(os.path.join(_DIR, 'json', 'medieval_text_templates.json')) as _f:
    _data = json.load(_f)

PARAGRAPH_STATS = _data['PARAGRAPH_STATS']
SECTION_TEXT_TYPE_MAP = _data['SECTION_TEXT_TYPE_MAP']
OPENING_FORMULAS = _data['OPENING_FORMULAS']
CLOSING_FORMULAS = _data['CLOSING_FORMULAS']
ITALIAN_MEDICAL_VOCAB = _data['ITALIAN_MEDICAL_VOCAB']
GERMAN_MEDICAL_VOCAB = _data['GERMAN_MEDICAL_VOCAB']

del _data, _f

def get_text_type_for_section(section: str) -> dict:
    """Get the expected text type statistics for a Voynich section."""
    text_type = SECTION_TEXT_TYPE_MAP.get(section, 'herbal_entry')
    return PARAGRAPH_STATS[text_type]

def get_opening_formulas(text_type: str) -> list:
    """Get formulaic opening patterns for a text type."""
    return OPENING_FORMULAS.get(text_type, [])

def get_closing_formulas(text_type: str) -> list:
    """Get formulaic closing patterns for a text type."""
    return CLOSING_FORMULAS.get(text_type, [])

def generate_italian_text(n_words: int = 300, seed: int = 42) -> str:
    """Generate synthetic Italian medical text for null distribution testing."""
    import random
    rng = random.Random(seed)
    high = ITALIAN_MEDICAL_VOCAB['high_freq']
    med = ITALIAN_MEDICAL_VOCAB['medium_freq']
    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))
    return ' '.join(words)

def generate_german_text(n_words: int = 300, seed: int = 42) -> str:
    """Generate synthetic Middle High German medical text for null distribution testing."""
    import random
    rng = random.Random(seed)
    high = GERMAN_MEDICAL_VOCAB['high_freq']
    med = GERMAN_MEDICAL_VOCAB['medium_freq']
    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))
    return ' '.join(words)
