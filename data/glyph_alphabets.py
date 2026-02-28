"""
Alternative Voynich Alphabetizations
======================================
Defines multiple alternative glyph-boundary interpretations for the Voynich
manuscript, with a retranscription engine.

Sources: EVA (Zandbergen & Landini), Currier (1976), Stolfi, Bennett.
"""

import json
import os
from typing import Dict, List, Tuple
import re

_DIR = os.path.dirname(__file__)
with open(os.path.join(_DIR, 'json', 'glyph_alphabets.json')) as _f:
    _data = json.load(_f)

EVA_STANDARD = _data['EVA_STANDARD']
LIGATURE_MERGED = _data['LIGATURE_MERGED']
GALLOWS_SPLIT = _data['GALLOWS_SPLIT']
CURRIER_ALPHABET = _data['CURRIER_ALPHABET']
STOLFI_FUNCTIONAL = _data['STOLFI_FUNCTIONAL']
BENNETT_ATOMS = _data['BENNETT_ATOMS']

ALTERNATIVE_ALPHABETS = {
    'eva_standard':     EVA_STANDARD,
    'ligature_merged':  LIGATURE_MERGED,
    'gallows_split':    GALLOWS_SPLIT,
    'currier':          CURRIER_ALPHABET,
    'stolfi_functional': STOLFI_FUNCTIONAL,
    'bennett_atoms':    BENNETT_ATOMS,
}

del _data, _f

def retranscribe(eva_text: str, alphabet_name: str) -> str:
    """
    Re-transcribe EVA text into an alternative alphabet.

    For multi-character mappings (ligature_merged, currier, stolfi),
    applies longest-match-first substitution to avoid partial matches.
    """
    if alphabet_name not in ALTERNATIVE_ALPHABETS:
        raise ValueError(f'Unknown alphabet: {alphabet_name}. '
                         f'Available: {list(ALTERNATIVE_ALPHABETS.keys())}')

    mapping = ALTERNATIVE_ALPHABETS[alphabet_name]

    if alphabet_name == 'eva_standard':
        return eva_text

    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    if not sorted_keys:
        return eva_text

    tokens = eva_text.split()
    result_tokens = []

    for token in tokens:
        result = _apply_mapping(token, sorted_keys, mapping)
        result_tokens.append(result)

    return ' '.join(result_tokens)

def _apply_mapping(token: str, sorted_keys: List[str], mapping: Dict[str, str]) -> str:
    """Apply longest-match-first substitution to a single token."""
    result = []
    i = 0
    while i < len(token):
        matched = False
        for key in sorted_keys:
            if token[i:i+len(key)] == key:
                result.append(mapping[key])
                i += len(key)
                matched = True
                break
        if not matched:
            result.append(token[i])
            i += 1
    return ''.join(result)

def get_alphabet_info(alphabet_name: str) -> Dict:
    """Return metadata about an alternative alphabet."""
    mapping = ALTERNATIVE_ALPHABETS.get(alphabet_name, {})
    return {
        'name': alphabet_name,
        'n_mappings': len(mapping),
        'max_source_length': max((len(k) for k in mapping), default=0),
        'unique_targets': len(set(mapping.values())),
        'mappings': dict(mapping),
    }

def list_alphabets() -> List[str]:
    """Return list of available alphabet names."""
    return list(ALTERNATIVE_ALPHABETS.keys())
