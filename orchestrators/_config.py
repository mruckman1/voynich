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
MIN_CONFIDENCE_RATIO = 5.0    # Phase 12 — raised from 3.0 for Academic Fortification

# Improvement 3: Length-scaled confidence ratio
MIN_CONFIDENCE_RATIO_LONG = 3.0   # For skeletons >= LONG_SKELETON_SEGMENTS
LONG_SKELETON_SEGMENTS = 5
ENABLE_LENGTH_SCALED_RATIO = True

# Improvement 2: Bidirectional solving
ENABLE_BIDIRECTIONAL_SOLVING = True
MAX_SOLVING_PASSES = 4

# Improvement 1: Contextual function word recovery
ENABLE_FUNCTION_WORD_RECOVERY = True
FUNCTION_WORD_TRIGRAM_THRESHOLD = 0.01  # Min P(fw|w_prev) + P(w_next|fw)

# Improvement 4: Dual-context confidence reduction
# When both neighbors are resolved within max_distance, reduce the
# confidence ratio threshold by this factor. Two-sided trigram evidence
# is more discriminative, so the threshold can safely be lowered.
DUAL_CONTEXT_RATIO_FACTOR = 0.6      # 5.0x × 0.6 = 3.0x for dual-context
DUAL_CONTEXT_MAX_DISTANCE = 3        # Max positions to nearest resolved neighbor

# Improvement 5: Unigram frequency backoff
# When bigram transition probabilities are all zero for a token's candidates,
# fall back to corpus unigram frequency as a weaker discriminator.
ENABLE_UNIGRAM_BACKOFF = True
UNIGRAM_BACKOFF_RATIO_FACTOR = 1.5   # 5.0x × 1.5 = 7.5x threshold for unigram-only
UNIGRAM_BACKOFF_MIN_SEGMENTS = 3     # Skip very short skeletons (unicity protection)

# ── Cross-Folio Consistency Engine ────────────────────────────────
ENABLE_CROSS_FOLIO_CONSISTENCY = True
CROSS_FOLIO_MIN_AGREEMENT = 0.6     # Minimum fraction of folios agreeing on mapping
CROSS_FOLIO_MIN_OCCURRENCES = 3     # Skeleton must appear resolved in 3+ folios

# ── Graduated CSP Scoring ────────────────────────────────────────
ENABLE_GRADUATED_CSP = True
CSP_HIGH_CONFIDENCE_THRESHOLD = 20.0    # Resolve directly (existing behavior)
CSP_MEDIUM_CONFIDENCE_THRESHOLD = 10.0  # Pass candidates to n-gram solver

# ── Selective Function Word Reintroduction ───────────────────────
ENABLE_SELECTIVE_FUNCTION_WORDS = True
FUNCTION_WORD_MAX_DENSITY = 1.5    # Max density relative to corpus frequency
FUNCTION_WORD_WINDOW_SIZE = 20     # Window for density calculation

# ── Folio Processing Limits ───────────────────────────────────────
FOLIO_LIMIT_DEFAULT = 15      # Phase 11
FOLIO_LIMIT_DEMO = 10         # Phase 10

# Phase 12 full-manuscript mode: None = all folios, int = limit for testing
PHASE12_FOLIO_LIMIT = None

# ── Voynich Manuscript Sections ──────────────────────────────────
VOYNICH_SECTIONS = {
    'herbal_a':       {'currier_lang': 'A', 'primary_scribe': 1},
    'herbal_b':       {'currier_lang': 'B', 'primary_scribe': 2},
    'astronomical':   {'currier_lang': 'B', 'primary_scribe': 3},
    'biological':     {'currier_lang': 'B', 'primary_scribe': 4},
    'cosmological':   {'currier_lang': 'B', 'primary_scribe': 3},
    'pharmaceutical': {'currier_lang': 'B', 'primary_scribe': 5},
    'recipes':        {'currier_lang': 'B', 'primary_scribe': 5},
}

# ── Phase 13: Scholarly Synthesis ────────────────────────────────
HITL_OVERRIDES_FILE = 'hitl_overrides.json'
HITL_MAX_CANDIDATES = 10
WHITEPAPER_CHART_DPI = 150

# ── Phase 12.5: Adversarial Defense Suite ────────────────────────
ADV_UNICITY_TRIALS = 10            # Number of randomization trials
ADV_RANDOM_BASELINE_THRESHOLD = 0.15  # Max resolution for random controls
ADV_FOLIO_LIMIT = 15               # Folios per adversarial test

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
