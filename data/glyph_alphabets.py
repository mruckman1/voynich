"""
Alternative Voynich Alphabetizations
======================================
Defines multiple alternative glyph-boundary interpretations for the Voynich
manuscript. The standard EVA transcription treats certain stroke combinations
as separate characters, but scholars have proposed different decompositions.

Re-analyzing with corrected alphabets could change all entropy calculations.

Sources:
- EVA: Zandbergen & Landini (standard)
- Currier: Prescott Currier (1976) — different glyph boundaries
- Stolfi: Jorge Stolfi — functional decomposition
- Bennett: William Bennett — stroke-atom decomposition
"""

from typing import Dict, List, Tuple
import re


# ============================================================================
# ALPHABET DEFINITIONS
# ============================================================================

# Standard EVA alphabet (baseline — identity mapping)
EVA_STANDARD = {c: c for c in 'oainedylrsktchpfqmgx'}

# Ligature-merged: treat common multi-glyph units as single characters.
# This REDUCES the effective alphabet size and may INCREASE H2 by
# removing predictable internal transitions (e.g., 's' always followed by 'h').
LIGATURE_MERGED = {
    'sh':   'Š',    # sh-bench combination -> single unit
    'ch':   'Č',    # ch-bench combination -> single unit
    'cth':  'Ŧ',    # gallows bench combo
    'ckh':  'Ǩ',    # gallows bench combo
    'cph':  'Ꝑ',    # gallows bench combo
    'cfh':  'Ꝼ',    # gallows bench combo
    'iin':  'Ɨ',    # minim sequence -> single unit
    'iiin': 'Ɲ',    # extended minim sequence
    'aiin': 'Ǡ',    # a + minim sequence
    'qo':   'Ꝗ',    # q-initial combo (always appears together)
    'dy':   'Đ',    # terminal d-y combo
    'ey':   'Ɛ',    # terminal e-y combo
    'ol':   'Ø',    # bench-loop combo
    'or':   'Ɵ',    # bench-flourish combo
    'al':   'Ǻ',    # bench-loop combo
    'ar':   'Ɑ',    # bench-flourish combo
    'in':   'Ɩ',    # minim-terminal
    'an':   'Ã',    # a-terminal
    'ai':   'Æ',    # a-minim
    'ee':   'Ə',    # doubled bench
}

# Gallows-split: decompose gallows letters into base + modifier.
# If gallows are composite (base stroke + plume/crossbar), splitting them
# increases alphabet granularity and may change positional statistics.
GALLOWS_SPLIT = {
    'k': 'ɡ̂',      # k = base-gallows + hat modifier
    't': 'ɡ̃',      # t = base-gallows + tilde modifier
    'p': 'ɡ̂ʰ',     # p = base-gallows + hat + plume
    'f': 'ɡ̃ʰ',     # f = base-gallows + tilde + plume
}

# Currier alphabet: Prescott Currier's original analysis used different
# character boundaries. Key differences from EVA:
# - bench sequences (ch, sh) treated as single characters
# - minim sequences (ii, iii) as single characters
# - gallows left as-is
CURRIER_ALPHABET = {
    'sh':  'S',
    'ch':  'C',
    'cth': 'U',
    'ckh': 'V',
    'cph': 'W',
    'cfh': 'X',
    'ii':  'J',     # double minim
    'iii': 'Z',     # triple minim
    'qo':  'Q',     # q+o as unit
    'ee':  'E',     # double bench-e
}

# Stolfi functional alphabet: groups glyphs by morphological function.
# Jorge Stolfi proposed that Voynich words have a strict structure:
#   [gallows?] [q?] [o/a] [l/r?] [bench-seq] [e?] [terminal]
# This mapping groups characters by their functional slot.
STOLFI_FUNCTIONAL = {
    # Gallows group (all mapped to one super-character)
    'k': 'G',
    't': 'G',
    'p': 'G',
    'f': 'G',
    # Bench-sequence characters merged
    'ch': 'B',
    'sh': 'B',
    # Loop characters merged
    'ol': 'L',
    'or': 'L',
    'al': 'L',
    'ar': 'L',
    # Minim runs merged
    'ii':  'I',
    'iii': 'I',
    'iiii': 'I',
}

# Bennett stroke-atom decomposition: each EVA character decomposed into
# fundamental pen strokes. This MAXIMIZES alphabet granularity.
BENNETT_ATOMS = {
    'o': '⌒',           # single loop
    'a': '⌒·',          # loop + connection
    'i': '|',            # single minim stroke
    'n': '|·',           # minim + terminal
    'e': '⌒⌒',          # double loop (bench-e)
    'd': '↑|',           # tall stroke + minim
    'y': '|↓',           # minim + descender
    'l': '⌒|',          # loop + stroke
    'r': '⌒~',          # loop + flourish
    's': '~',            # initial flourish
    'k': '↑↑',          # double tall (gallows-k)
    't': '↑↑·',         # double tall + bar (gallows-t)
    'c': '⌒',           # same as o (bench-c ≈ bench-o in strokes)
    'h': '⌒|·',         # loop + stroke + connection
    'p': '↑↑↓',         # double tall + plume
    'f': '↑↑·↓',        # double tall + bar + plume
    'q': '↑↓',          # tall + descender
    'm': '|·|',          # minim + terminal + minim
    'g': '|↑',           # minim + tall (rare)
    'x': '×',            # cross
}


# ============================================================================
# ALPHABET REGISTRY
# ============================================================================

ALTERNATIVE_ALPHABETS = {
    'eva_standard':     EVA_STANDARD,
    'ligature_merged':  LIGATURE_MERGED,
    'gallows_split':    GALLOWS_SPLIT,
    'currier':          CURRIER_ALPHABET,
    'stolfi_functional': STOLFI_FUNCTIONAL,
    'bennett_atoms':    BENNETT_ATOMS,
}


# ============================================================================
# RETRANSCRIPTION ENGINE
# ============================================================================

def retranscribe(eva_text: str, alphabet_name: str) -> str:
    """
    Re-transcribe EVA text into an alternative alphabet.

    For multi-character mappings (ligature_merged, currier, stolfi),
    applies longest-match-first substitution to avoid partial matches.

    Parameters:
        eva_text: space-separated EVA tokens
        alphabet_name: key in ALTERNATIVE_ALPHABETS

    Returns:
        Re-transcribed text with the same word boundaries.
    """
    if alphabet_name not in ALTERNATIVE_ALPHABETS:
        raise ValueError(f'Unknown alphabet: {alphabet_name}. '
                         f'Available: {list(ALTERNATIVE_ALPHABETS.keys())}')

    mapping = ALTERNATIVE_ALPHABETS[alphabet_name]

    if alphabet_name == 'eva_standard':
        return eva_text

    # Sort mapping keys by length (longest first) for greedy matching
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    # Build regex pattern for simultaneous replacement
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
