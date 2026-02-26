"""
Latin Syllable Inventory
=========================
Provides the syllable inventory and syllabification functions needed for
Model 2 (Syllabary Code). Based on medieval Latin phonological rules.

The syllable inventory covers the ~250 most common syllables in medieval
medical/scientific Latin, with frequency weights derived from typical
distributions in texts like Circa Instans, Macer Floridus, and similar
pharmaceutical/herbal treatises.

Sources:
- Latin phonotactics: Allen, Vox Latina (1965)
- Medieval Latin peculiarities: Norberg, Manual pratique de latin médiéval (1968)
- Frequency estimates: derived from analysis of Hartlieb-era medical vocabulary
"""

from typing import Dict, List, Tuple, Optional
import re


# ============================================================================
# SYLLABLE STRUCTURE COMPONENTS
# ============================================================================

# Valid Latin onsets (initial consonant clusters)
ONSETS = [
    '',                                          # vowel-initial
    'b', 'c', 'd', 'f', 'g', 'h', 'l', 'm',   # single consonants
    'n', 'p', 'q', 'r', 's', 't', 'v', 'x',
    'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr',  # consonant + liquid
    'gl', 'gr', 'pl', 'pr', 'tr',
    'sc', 'sp', 'st', 'str', 'spr',             # s-clusters
    'qu',                                         # qu digraph (always together)
    'gn', 'mn', 'pn', 'ps', 'pt',               # learned clusters
]

# Latin vowel nuclei
NUCLEI = [
    'a', 'e', 'i', 'o', 'u',                    # short vowels
    'ae', 'oe', 'au',                             # diphthongs
]

# Valid Latin codas (final consonant clusters)
CODAS = [
    '',                                           # open syllable
    'b', 'c', 'd', 'l', 'm', 'n', 'p',         # single consonants
    'r', 's', 't', 'x',
    'ns', 'nt', 'rs', 'rt', 'st',               # common clusters
    'ct', 'pt', 'mn', 'bs', 'ps',               # learned clusters
    'nx', 'nd', 'mp', 'rb', 'rc', 'rd',         # additional clusters
    'rn', 'rm', 'lt', 'lc', 'lm',
]


# ============================================================================
# SYLLABLE FREQUENCY TABLE
# ============================================================================

# ~250 most common Latin syllables in medieval medical/herbal text,
# with approximate relative frequency weights.
# Organized by structure: open CV syllables are most common.

LATIN_SYLLABLES: Dict[str, float] = {
    # --- High-frequency open syllables (CV) ---
    'ra':  0.028, 'ri':  0.026, 'ta':  0.025, 'ti':  0.024,
    'de':  0.023, 're':  0.022, 'na':  0.021, 'ma':  0.020,
    'la':  0.019, 'te':  0.018, 'ca':  0.017, 'li':  0.016,
    'tu':  0.015, 'si':  0.015, 'ne':  0.014, 'ro':  0.014,
    'mi':  0.013, 'ni':  0.013, 'co':  0.012, 'di':  0.012,
    'sa':  0.011, 'se':  0.011, 'no':  0.010, 'lo':  0.010,
    'pa':  0.010, 'da':  0.009, 'bi':  0.009, 'fi':  0.009,
    'vi':  0.009, 'pi':  0.008, 'cu':  0.008, 'su':  0.008,
    'me':  0.008, 'le':  0.007, 'mo':  0.007, 'po':  0.007,
    'ba':  0.007, 'va':  0.006, 'fa':  0.006, 'to':  0.006,
    'do':  0.006, 'bo':  0.005, 'so':  0.005, 'go':  0.005,
    'fu':  0.005, 'ga':  0.005, 'ge':  0.005, 'gi':  0.005,
    'gu':  0.004, 'du':  0.004, 'fe':  0.004, 'fo':  0.004,
    'bu':  0.004, 'ha':  0.004, 'he':  0.003, 'hi':  0.003,
    'ho':  0.003, 'hu':  0.003, 'lu':  0.003, 'mu':  0.003,
    'nu':  0.003, 'pu':  0.003, 'ru':  0.003, 'vo':  0.003,
    've':  0.003, 'vu':  0.002,

    # --- Medium-frequency closed syllables (CVC) ---
    'con': 0.012, 'ter': 0.011, 'men': 0.010, 'cum': 0.009,
    'per': 0.009, 'ver': 0.008, 'tur': 0.008, 'rum': 0.008,
    'est': 0.008, 'ber': 0.007, 'rem': 0.007, 'bus': 0.007,
    'tis': 0.006, 'mus': 0.006, 'nis': 0.006, 'ris': 0.006,
    'lem': 0.006, 'lem': 0.006, 'dum': 0.005, 'num': 0.005,
    'lum': 0.005, 'sum': 0.005, 'tem': 0.005, 'dem': 0.005,
    'pus': 0.005, 'tum': 0.005, 'dis': 0.004, 'mis': 0.004,
    'lis': 0.004, 'ros': 0.004, 'cis': 0.004, 'tes': 0.004,
    'nes': 0.004, 'res': 0.004, 'pes': 0.004, 'des': 0.003,
    'les': 0.003, 'mes': 0.003, 'ves': 0.003, 'bes': 0.003,
    'cos': 0.003, 'pos': 0.003, 'nos': 0.003, 'dos': 0.003,
    'fis': 0.003, 'bis': 0.003, 'vis': 0.003, 'ras': 0.003,
    'las': 0.003, 'mas': 0.002, 'nas': 0.002, 'tas': 0.002,

    # --- Onset cluster syllables ---
    'pra': 0.006, 'tra': 0.006, 'pro': 0.005, 'tri': 0.005,
    'pre': 0.005, 'pri': 0.005, 'cri': 0.004, 'gra': 0.004,
    'bra': 0.004, 'flo': 0.004, 'pla': 0.003, 'cla': 0.003,
    'stra': 0.003, 'spi': 0.003, 'sta': 0.003, 'sti': 0.003,
    'fri': 0.003, 'cra': 0.003, 'gri': 0.002, 'bre': 0.002,
    'dra': 0.002, 'fra': 0.002, 'glo': 0.002, 'pli': 0.002,
    'cli': 0.002, 'tre': 0.002, 'gre': 0.002, 'ble': 0.002,
    'fle': 0.002, 'spe': 0.002, 'ste': 0.002, 'scri': 0.002,

    # --- Vowel-initial syllables ---
    'a':   0.015, 'e':   0.012, 'i':   0.010, 'o':   0.008,
    'u':   0.006, 'ae':  0.004, 'au':  0.003,
    'ad':  0.006, 'ab':  0.005, 'ac':  0.004, 'al':  0.004,
    'am':  0.004, 'an':  0.007, 'ap':  0.003, 'ar':  0.005,
    'as':  0.004, 'at':  0.004, 'ax':  0.002,
    'em':  0.004, 'en':  0.005, 'er':  0.006, 'es':  0.005,
    'et':  0.005, 'ex':  0.004,
    'im':  0.003, 'in':  0.008, 'is':  0.005, 'it':  0.004,
    'ob':  0.003, 'om':  0.003, 'on':  0.003, 'or':  0.004,
    'os':  0.003, 'um':  0.007, 'un':  0.004, 'ur':  0.003,
    'us':  0.006, 'ut':  0.004,

    # --- Diphthong syllables ---
    'quae': 0.003, 'que': 0.008, 'qui': 0.005, 'quo': 0.003,
}


# ============================================================================
# SYLLABIFICATION ENGINE
# ============================================================================

# Latin consonant set (for syllabification rules)
_CONSONANTS = set('bcdfghjklmnpqrstvxz')
_VOWELS = set('aeiou')
_DIPHTHONGS = ['ae', 'oe', 'au', 'eu']

# Valid onset clusters (used for maximum onset principle)
_VALID_ONSETS = set(ONSETS)


def syllabify(word: str) -> List[str]:
    """
    Syllabify a Latin word using the maximum onset principle.

    Rules:
    1. Find vowel nuclei (including diphthongs)
    2. Assign inter-vocalic consonants using maximum onset:
       - Give as many consonants as possible to the following syllable
       - But the onset must be a valid Latin onset cluster
    3. Handle qu as inseparable unit

    Parameters:
        word: a Latin word (lowercase)

    Returns:
        List of syllables, e.g., 'herba' -> ['her', 'ba']
    """
    word = word.lower().strip()
    if not word:
        return []

    # Handle 'qu' as single unit by temporarily replacing
    word = word.replace('qu', 'Q')

    # Find vowel positions (including diphthongs)
    vowel_positions = _find_vowel_positions(word)

    if not vowel_positions:
        # No vowels — treat whole word as one syllable
        return [word.replace('Q', 'qu')]

    if len(vowel_positions) == 1:
        return [word.replace('Q', 'qu')]

    # Build syllable breaks
    syllables = []
    prev_end = 0

    for idx in range(len(vowel_positions) - 1):
        v1_start, v1_end = vowel_positions[idx]
        v2_start, v2_end = vowel_positions[idx + 1]

        # Consonant cluster between the two vowels
        cluster = word[v1_end:v2_start]

        if len(cluster) == 0:
            # Adjacent vowels (hiatus): break between them
            syllables.append(word[prev_end:v1_end])
            prev_end = v1_end
        elif len(cluster) == 1:
            # Single consonant -> onset of next syllable
            syllables.append(word[prev_end:v1_end])
            prev_end = v1_end
        else:
            # Multiple consonants: apply maximum onset principle
            split_point = _find_onset_split(cluster)
            syllables.append(word[prev_end:v1_end + split_point])
            prev_end = v1_end + split_point

    # Last syllable
    syllables.append(word[prev_end:])

    # Restore 'qu'
    return [s.replace('Q', 'qu') for s in syllables]


def _find_vowel_positions(word: str) -> List[Tuple[int, int]]:
    """Find (start, end) positions of all vowel nuclei including diphthongs."""
    positions = []
    i = 0
    while i < len(word):
        if word[i] in _VOWELS or word[i] == 'Q':
            start = i
            # Check for diphthong
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

    Uses maximum onset principle: give as many consonants as possible to
    the following syllable, provided they form a valid Latin onset.
    """
    # Try giving all consonants to the next syllable
    for coda_len in range(len(cluster)):
        onset = cluster[coda_len:]
        # Restore qu for onset checking
        onset_check = onset.replace('Q', 'qu')
        if onset_check in _VALID_ONSETS or len(onset_check) == 1:
            return coda_len

    # Fallback: split after first consonant
    return 1


def get_syllable_frequencies() -> Dict[str, float]:
    """Return the syllable frequency table."""
    return dict(LATIN_SYLLABLES)


def get_top_syllables(n: int = 50) -> List[Tuple[str, float]]:
    """Return the N most frequent syllables."""
    return sorted(LATIN_SYLLABLES.items(), key=lambda x: -x[1])[:n]


def get_syllable_structure(syllable: str) -> Dict[str, str]:
    """
    Decompose a syllable into onset, nucleus, coda.

    Returns:
        {'onset': str, 'nucleus': str, 'coda': str}
    """
    s = syllable.lower()
    onset = ''
    nucleus = ''
    coda = ''

    i = 0
    # Extract onset (leading consonants)
    while i < len(s) and s[i] in _CONSONANTS:
        onset += s[i]
        i += 1
        # Handle 'qu'
        if onset.endswith('q') and i < len(s) and s[i] == 'u':
            onset += 'u'
            i += 1
            break

    # Extract nucleus (vowels/diphthongs)
    while i < len(s) and (s[i] in _VOWELS):
        nucleus += s[i]
        i += 1
        # Check if we formed a diphthong
        if len(nucleus) == 2 and nucleus in ['ae', 'oe', 'au', 'eu']:
            break

    # Rest is coda
    coda = s[i:]

    return {'onset': onset, 'nucleus': nucleus, 'coda': coda}


def classify_syllable_type(syllable: str) -> str:
    """Classify syllable as 'open' (no coda) or 'closed' (has coda)."""
    structure = get_syllable_structure(syllable)
    return 'open' if not structure['coda'] else 'closed'
