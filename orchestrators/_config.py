"""Centralized configuration constants for all phases.

Every hardcoded value previously scattered across 12 files is now
defined in one place. Individual phases import what they need.
"""

# ── SAA (Simulated Annealing Attack) ──────────────────────────────
SAA_ITERATIONS_DEFAULT = 100_000
SAA_ITERATIONS_QUICK = 1_000

# ── Latin Corpus ──────────────────────────────────────────────────
LATIN_CORPUS_TOKENS_DEFAULT = 30_000
LATIN_CORPUS_TOKENS_LARGE = 50_000
LATIN_CORPUS_TOKENS_QUICK = 10_000

# ── Beam Search ───────────────────────────────────────────────────
BEAM_WIDTH_DEFAULT = 25       # Phase 9
BEAM_WIDTH_TRIGRAM = 15       # Phase 10

# ── N-Gram Mask Solver ────────────────────────────────────────────
MIN_CONFIDENCE_RATIO = 3.0    # Phase 12

# ── Folio Processing Limits ───────────────────────────────────────
FOLIO_LIMIT_DEFAULT = 15      # Phase 11, 12
FOLIO_LIMIT_DEMO = 10         # Phase 10

# ── Phase 13: Scholarly Synthesis ────────────────────────────────
HITL_OVERRIDES_FILE = 'hitl_overrides.json'
HITL_MAX_CANDIDATES = 10
WHITEPAPER_CHART_DPI = 150

# ── Output Directories ────────────────────────────────────────────
OUTPUT_ROOT = './output'


def phase_output_dir(phase_num: int) -> str:
    """Return the default output directory for a given phase number."""
    if phase_num == 1:
        return OUTPUT_ROOT
    return f'{OUTPUT_ROOT}/phase{phase_num}'


# ── Phase 10 Sigla Maps ──────────────────────────────────────────
# Verbatim from convergence_attack_p10.py lines 29-53

SIGLA_PREFIX_MAP = {
    # Voynich Prefixes -> Latin Prefixes/Starters
    'qo': ['con', 'com', 'cor', 'qu'],
    'ch': ['ca', 'ce', 'ci', 'co', 'cu'],
    'sh': ['sa', 'se', 'si', 'su', 'ex'],
    'd':  ['de', 'di', 'da'],
    'p':  ['pro', 'per', 'prae', 'par'],
    't':  ['te', 'ta', 'ti', 'tra'],
    'k':  ['ca', 'cu', 'co'],
    '':   ['a', 'e', 'i', 'o', 'u'],  # Null prefixes usually mean vowel start
}

SIGLA_SUFFIX_MAP = {
    # Voynich Suffixes -> Latin Terminations
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
