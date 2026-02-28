"""Centralized configuration constants for all phases.

Every hardcoded value previously scattered across 12 files is now
defined in one place. Individual phases import what they need.
"""

import re

UNRESOLVED_RE = re.compile(r'\[([^_\]]+)_UNRESOLVED\]|<([^_>]+)_UNRESOLVED>')
TAGGED_BRACKET_RE = re.compile(r'\[([^_\]]+)_([A-Z_]+)\]|<([^_>]+)_([A-Z_]+)>')

SAA_ITERATIONS_DEFAULT = 100_000
SAA_ITERATIONS_QUICK = 1_000

LATIN_CORPUS_TOKENS_DEFAULT = 30_000
LATIN_CORPUS_TOKENS_LARGE = 50_000
LATIN_CORPUS_TOKENS_QUICK = 10_000

BEAM_WIDTH_DEFAULT = 25
BEAM_WIDTH_TRIGRAM = 15

MIN_CONFIDENCE_RATIO = 5.0

MIN_CONFIDENCE_RATIO_LONG = 3.0
LONG_SKELETON_SEGMENTS = 5
ENABLE_LENGTH_SCALED_RATIO = True

ENABLE_BIDIRECTIONAL_SOLVING = True
MAX_SOLVING_PASSES = 4

ENABLE_FUNCTION_WORD_RECOVERY = True
FUNCTION_WORD_TRIGRAM_THRESHOLD = 0.01

DUAL_CONTEXT_RATIO_FACTOR = 0.6
DUAL_CONTEXT_MAX_DISTANCE = 3

ENABLE_UNIGRAM_BACKOFF = True
UNIGRAM_BACKOFF_RATIO_FACTOR = 1.5
UNIGRAM_BACKOFF_MIN_SEGMENTS = 3

ENABLE_POS_BACKOFF = True
POS_BACKOFF_WEIGHT = 0.1
POS_BACKOFF_MIN_CONFIDENCE = 5.0

ENABLE_CHAR_NGRAM_FALLBACK = True
CHAR_NGRAM_ORDER = 3
CHAR_NGRAM_SMOOTHING = 0.01
CHAR_NGRAM_MIN_SCORE_GAP = 0.5
CHAR_NGRAM_MIN_SEGMENTS = 3
CHAR_NGRAM_MAX_CONTEXT_DISTANCE = 4
CHAR_NGRAM_REQUIRE_CONTEXT = True

ENABLE_ILLUSTRATION_PRIOR = True
ILLUSTRATION_TIER1_BOOST = 2.0
ILLUSTRATION_TIER2_BOOST = 1.3
ILLUSTRATION_TIER3_BOOST = 1.1
ILLUSTRATION_BOOSTED_RATIO_FACTOR = 0.5

ENABLE_ADAPTIVE_CONFIDENCE = True
ADAPTIVE_CONFIDENCE_2_CAND_FACTOR = 0.75
ADAPTIVE_CONFIDENCE_FEW_CAND_FACTOR = 0.9

ENABLE_SINGLE_CAND_CHAR_RESCUE = True
SINGLE_CAND_MIN_SEGMENTS = 4
SINGLE_CAND_MIN_CHAR_SCORE = -6.0

ENABLE_CROSS_FOLIO_CONSISTENCY = True
CROSS_FOLIO_MIN_AGREEMENT = 0.6
CROSS_FOLIO_MIN_OCCURRENCES = 3

ENABLE_RELAXED_CONSISTENCY = True
CROSS_FOLIO_MIN_OCCURRENCES_RELAXED = 2

ENABLE_ITERATIVE_REFINEMENT = True
ITERATIVE_MAX_PASSES = 3
ITERATIVE_MIN_IMPROVEMENT = 10

ENABLE_GRADUATED_CSP = True
CSP_HIGH_CONFIDENCE_THRESHOLD = 20.0
CSP_MEDIUM_CONFIDENCE_THRESHOLD = 10.0

ENABLE_SELECTIVE_FUNCTION_WORDS = True
FUNCTION_WORD_MAX_DENSITY = 1.5
FUNCTION_WORD_WINDOW_SIZE = 20

FOLIO_LIMIT_DEFAULT = 15
FOLIO_LIMIT_DEMO = 10

PHASE12_FOLIO_LIMIT = None

VOYNICH_SECTIONS = {
    'herbal_a':       {'currier_lang': 'A', 'primary_scribe': 1},
    'herbal_b':       {'currier_lang': 'B', 'primary_scribe': 2},
    'astronomical':   {'currier_lang': 'B', 'primary_scribe': 3},
    'biological':     {'currier_lang': 'B', 'primary_scribe': 4},
    'cosmological':   {'currier_lang': 'B', 'primary_scribe': 3},
    'pharmaceutical': {'currier_lang': 'B', 'primary_scribe': 5},
    'recipes':        {'currier_lang': 'B', 'primary_scribe': 5},
}

ENABLE_SECTION_SOLVERS = True
SECTION_CORPUS_FRACTION = 0.25

ADV_UNICITY_TRIALS = 10
ADV_RANDOM_BASELINE_THRESHOLD = 0.15
ADV_FOLIO_LIMIT = 15

OUTPUT_ROOT = './output'

def phase_output_dir(phase_num: int) -> str:
    """Return the default output directory for a given phase number."""
    if phase_num == 1:
        return OUTPUT_ROOT
    return f'{OUTPUT_ROOT}/phase{phase_num}'

SIGLA_PREFIX_MAP = {
    'qo': ['con', 'com', 'cor', 'qu'],
    'ch': ['ca', 'ce', 'ci', 'co', 'cu'],
    'sh': ['sa', 'se', 'si', 'su', 'ex'],
    'd':  ['de', 'di', 'da'],
    'p':  ['pro', 'per', 'prae', 'par'],
    't':  ['te', 'ta', 'ti', 'tra'],
    'k':  ['ca', 'cu', 'co'],
    '':   ['a', 'e', 'i', 'o', 'u'],
}

SIGLA_SUFFIX_MAP = {
    'dy': ['ae', 'ti', 'ur', 'di', 'te'],
    'iin': ['us', "um", 'is', 'in', 'unt'],
    'in': ['um', 'im', 'em', 'en'],
    'ey': ['es', 'et', 'er', 'em'],
    'y':  ['a', 'i', 'e', 'o'],
    'l':  ['al', 'el', 'il', 'ul', 'le'],
    'r':  ['ar', 'er', 'or', 'ur', 're'],
    'm':  ['am', 'um', 'em', 'rum', 'num'],
    's':  ['as', 'os', 'is', 'us'],
    '':   ['a', 'e', 'i', 'o', 'u', 't', 'c'],
}
