"""
Voynich Manuscript Corpus Data Module
======================================
Contains EVA transliterations, scribe assignments, section mappings,
and known metadata derived from published research.
"""

import json
import re
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

_DIR = os.path.dirname(__file__)
with open(os.path.join(_DIR, 'json', 'voynich_corpus.json')) as _f:
    _data = json.load(_f)

EVA_GLYPHS = _data['EVA_GLYPHS']
EVA_LIGATURES = _data['EVA_LIGATURES']
SECTIONS = _data['SECTIONS']
SCRIBES = {int(k): v for k, v in _data['SCRIBES'].items()}
ZODIAC_LABELS = _data['ZODIAC_LABELS']
SAMPLE_CORPUS = _data['SAMPLE_CORPUS']
HARTLIEB_MEDICAL_VOCAB = _data['HARTLIEB_MEDICAL_VOCAB']
LATIN_RECIPE_FORMULAS = _data['LATIN_RECIPE_FORMULAS']

del _data, _f

def tokenize(text: str) -> List[str]:
    """Split EVA text into tokens (words)."""
    return [w for w in text.strip().split() if w]

def get_all_tokens(section: Optional[str] = None,
                   scribe: Optional[int] = None,
                   lang: Optional[str] = None) -> List[str]:
    """Extract all tokens from the corpus, optionally filtered."""
    tokens = []
    for folio, data in SAMPLE_CORPUS.items():
        if section and data['section'] != section:
            continue
        if scribe and data['scribe'] != scribe:
            continue
        if lang and data['lang'] != lang:
            continue
        for line in data['text']:
            tokens.extend(tokenize(line))
    return tokens

def get_folio_text(folio: str) -> str:
    """Get full text of a folio as a single string."""
    if folio in SAMPLE_CORPUS:
        return ' '.join(SAMPLE_CORPUS[folio]['text'])
    return ''

def get_section_text(section: str) -> str:
    """Get all text from a section."""
    lines = []
    for folio, data in SAMPLE_CORPUS.items():
        if data['section'] == section:
            lines.extend(data['text'])
    return ' '.join(lines)

def get_scribe_text(scribe_id: int) -> str:
    """Get all text attributed to a specific scribe."""
    lines = []
    for folio, data in SAMPLE_CORPUS.items():
        if data['scribe'] == scribe_id:
            lines.extend(data['text'])
    return ' '.join(lines)

def get_scribe_transition_pairs() -> List[Tuple[str, str, int, int]]:
    """Find adjacent folios where scribe changes occur."""
    sorted_folios = sorted(SAMPLE_CORPUS.keys(),
                           key=lambda f: (int(re.search(r'\d+', f).group()), f))
    transitions = []
    for i in range(len(sorted_folios) - 1):
        f1, f2 = sorted_folios[i], sorted_folios[i + 1]
        s1 = SAMPLE_CORPUS[f1]['scribe']
        s2 = SAMPLE_CORPUS[f2]['scribe']
        if s1 != s2:
            transitions.append((f1, f2, s1, s2))
    return transitions

def load_full_corpus(filepath: str) -> Dict:
    """Load a complete EVA transliteration file (e.g., Takahashi format)."""
    corpus = {}
    if not os.path.exists(filepath):
        print(f"[WARN] Full corpus file not found: {filepath}")
        print("       Using built-in sample corpus instead.")
        return SAMPLE_CORPUS

    current_folio = None
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.match(r'<(f\d+[rv]\d?)[\.,]', line)
            if match:
                folio = match.group(1)
                text_part = re.sub(r'<[^>]+>', '', line).strip()
                text_part = text_part.replace('.', ' ')
                if folio not in corpus:
                    corpus[folio] = {
                        'scribe': _infer_scribe(folio),
                        'lang': _infer_language(folio),
                        'section': _infer_section(folio),
                        'text': [],
                    }
                if text_part:
                    corpus[folio]['text'].append(text_part)

    print(f"[INFO] Loaded {len(corpus)} folios from {filepath}")
    return corpus

def _infer_scribe(folio: str) -> int:
    """Infer scribe from folio number based on quire assignments."""
    try:
        num = int(re.search(r'\d+', folio).group())
    except (AttributeError, ValueError):
        return 0
    if num <= 56:
        return 1
    elif 87 <= num <= 102:
        return 2
    elif 67 <= num <= 74:
        return 3
    elif 75 <= num <= 84:
        return 4
    elif num >= 103:
        return 5
    return 0

def _infer_language(folio: str) -> str:
    """Infer Currier language from folio."""
    scribe = _infer_scribe(folio)
    return 'A' if scribe == 1 else 'B'

def _infer_section(folio: str) -> str:
    """Infer section from folio number."""
    try:
        num = int(re.search(r'\d+', folio).group())
    except (AttributeError, ValueError):
        return 'unknown'
    if num <= 56:
        return 'herbal_a'
    elif 67 <= num <= 74:
        return 'astronomical'
    elif 75 <= num <= 84:
        return 'biological'
    elif 85 <= num <= 86:
        return 'cosmological'
    elif num == 87:
        return 'herbal_b'
    elif 88 <= num <= 102:
        return 'pharmaceutical'
    elif num >= 103:
        return 'recipes'
    return 'unknown'
