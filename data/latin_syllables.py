"""
Latin Syllable Inventory
=========================
Syllable inventory and syllabification functions for Model 2 (Syllabary Code).
Based on medieval Latin phonological rules.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

_DIR = os.path.dirname(__file__)
with open(os.path.join(_DIR, 'json', 'latin_syllables.json')) as _f:
    _data = json.load(_f)

ONSETS = _data['ONSETS']
NUCLEI = _data['NUCLEI']
CODAS = _data['CODAS']
LATIN_SYLLABLES: Dict[str, float] = _data['LATIN_SYLLABLES']

del _data, _f

_CONSONANTS = set('bcdfghjklmnpqrstvxz')
_VOWELS = set('aeiou')
_DIPHTHONGS = ['ae', 'oe', 'au', 'eu']
_VALID_ONSETS = set(ONSETS)

def syllabify(word: str) -> List[str]:
    """
    Syllabify a Latin word using the maximum onset principle.

    Parameters:
        word: a Latin word (lowercase)

    Returns:
        List of syllables, e.g., 'herba' -> ['her', 'ba']
    """
    word = word.lower().strip()
    if not word:
        return []

    word = word.replace('qu', 'Q')
    vowel_positions = _find_vowel_positions(word)

    if not vowel_positions:
        return [word.replace('Q', 'qu')]

    if len(vowel_positions) == 1:
        return [word.replace('Q', 'qu')]

    syllables = []
    prev_end = 0

    for idx in range(len(vowel_positions) - 1):
        v1_start, v1_end = vowel_positions[idx]
        v2_start, v2_end = vowel_positions[idx + 1]
        cluster = word[v1_end:v2_start]

        if len(cluster) == 0:
            syllables.append(word[prev_end:v1_end])
            prev_end = v1_end
        elif len(cluster) == 1:
            syllables.append(word[prev_end:v1_end])
            prev_end = v1_end
        else:
            split_point = _find_onset_split(cluster)
            syllables.append(word[prev_end:v1_end + split_point])
            prev_end = v1_end + split_point

    syllables.append(word[prev_end:])
    return [s.replace('Q', 'qu') for s in syllables]

def _find_vowel_positions(word: str) -> List[Tuple[int, int]]:
    """Find (start, end) positions of all vowel nuclei including diphthongs."""
    positions = []
    i = 0
    while i < len(word):
        if word[i] in _VOWELS or word[i] == 'Q':
            start = i
            if i + 1 < len(word) and word[i:i+2].replace('Q', 'qu') in _DIPHTHONGS:
                i += 2
            else:
                i += 1
            positions.append((start, i))
        else:
            i += 1
    return positions

def _find_onset_split(cluster: str) -> int:
    """
    Find where to split a consonant cluster between two vowels.
    Returns the number of consonants that stay with the preceding vowel (coda).
    """
    for coda_len in range(len(cluster)):
        onset = cluster[coda_len:]
        onset_check = onset.replace('Q', 'qu')
        if onset_check in _VALID_ONSETS or len(onset_check) == 1:
            return coda_len
    return 1

def get_syllable_frequencies() -> Dict[str, float]:
    """Return the syllable frequency table."""
    return dict(LATIN_SYLLABLES)

def get_top_syllables(n: int = 50) -> List[Tuple[str, float]]:
    """Return the N most frequent syllables."""
    return sorted(LATIN_SYLLABLES.items(), key=lambda x: -x[1])[:n]

def get_syllable_structure(syllable: str) -> Dict[str, str]:
    """Decompose a syllable into onset, nucleus, coda."""
    s = syllable.lower()
    onset = ''
    nucleus = ''
    coda = ''
    i = 0
    while i < len(s) and s[i] in _CONSONANTS:
        onset += s[i]
        i += 1
        if onset.endswith('q') and i < len(s) and s[i] == 'u':
            onset += 'u'
            i += 1
            break
    while i < len(s) and (s[i] in _VOWELS):
        nucleus += s[i]
        i += 1
        if len(nucleus) == 2 and nucleus in ['ae', 'oe', 'au', 'eu']:
            break
    coda = s[i:]
    return {'onset': onset, 'nucleus': nucleus, 'coda': coda}

def classify_syllable_type(syllable: str) -> str:
    """Classify syllable as 'open' (no coda) or 'closed' (has coda)."""
    structure = get_syllable_structure(syllable)
    return 'open' if not structure['coda'] else 'closed'
