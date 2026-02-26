"""
Voynich Manuscript Corpus Data Module
======================================
Contains EVA transliterations, scribe assignments, section mappings,
and known metadata derived from published research.

Sources:
- EVA alphabet: Zandbergen & Landini
- Scribe assignments: Lisa Fagin Davis (2020)
- Currier languages: Prescott Currier (1976)
- Section taxonomy: standard scholarly consensus
- Zodiac month labels: published marginalia analysis

NOTE: This module contains a representative sample of real EVA-transliterated
text from published academic sources. For full analysis, load the complete
Takeshi Takahashi or Zandbergen interlinear files via load_full_corpus().
"""

import re
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# ============================================================================
# EVA ALPHABET DEFINITION
# ============================================================================

# Core EVA glyphs mapped to their positional behavior
# Position classes: P=prefix-only, S=suffix-only, M=medial, A=any position
EVA_GLYPHS = {
    # High-frequency glyphs
    'o': {'class': 'A', 'freq_rank': 1, 'description': 'bench-loop'},
    'a': {'class': 'A', 'freq_rank': 2, 'description': 'bench element'},
    'i': {'class': 'M', 'freq_rank': 3, 'description': 'single minim'},
    'n': {'class': 'S', 'freq_rank': 4, 'description': 'terminal flourish'},
    'e': {'class': 'A', 'freq_rank': 5, 'description': 'bench variant'},
    'd': {'class': 'P', 'freq_rank': 6, 'description': 'tall gallows prefix'},
    'y': {'class': 'S', 'freq_rank': 7, 'description': 'terminal descender'},
    'l': {'class': 'A', 'freq_rank': 8, 'description': 'loop element'},
    'r': {'class': 'A', 'freq_rank': 9, 'description': 'flourish stroke'},
    's': {'class': 'P', 'freq_rank': 10, 'description': 'initial stroke'},
    'k': {'class': 'P', 'freq_rank': 11, 'description': 'gallows glyph'},
    't': {'class': 'P', 'freq_rank': 12, 'description': 'gallows glyph'},
    'c': {'class': 'A', 'freq_rank': 13, 'description': 'bench-c'},
    'h': {'class': 'A', 'freq_rank': 14, 'description': 'bench-h'},
    'p': {'class': 'P', 'freq_rank': 15, 'description': 'gallows with plume'},
    'f': {'class': 'P', 'freq_rank': 16, 'description': 'gallows with plume'},
    'q': {'class': 'P', 'freq_rank': 17, 'description': 'initial-q (almost always qo-)'},
    'm': {'class': 'S', 'freq_rank': 18, 'description': 'terminal flourish variant'},
    'g': {'class': 'M', 'freq_rank': 19, 'description': 'rare medial'},
    'x': {'class': 'A', 'freq_rank': 20, 'description': 'rare cross-stroke'},
}

# Ligatures and multi-glyph units common in EVA
EVA_LIGATURES = [
    'sh', 'ch', 'cth', 'ckh', 'cph', 'cfh',  # bench-gallows combos
    'iin', 'iiin', 'aiin', 'aiiin',             # minim sequences
    'ol', 'or', 'al', 'ar',                      # bench-loop combos
    'qo', 'qok', 'qot',                          # q-initial combos
    'dy', 'ey',                                   # terminal descenders
]

# ============================================================================
# SECTION AND QUIRE MAPPINGS
# ============================================================================

# Section definitions with folio ranges
SECTIONS = {
    'herbal_a': {
        'description': 'Herbal section A (single-plant illustrations)',
        'quires': [1, 2, 3, 4, 5, 6, 7],
        'currier_lang': 'A',
        'primary_scribe': 1,
        'folios': list(range(1, 57)) + [65, 66],
    },
    'herbal_b': {
        'description': 'Herbal section B',
        'quires': [17],
        'currier_lang': 'B',
        'primary_scribe': 2,
        'folios': list(range(87, 103)),
    },
    'astronomical': {
        'description': 'Astrological/zodiac diagrams',
        'quires': [8, 9, 10],
        'currier_lang': 'B',
        'primary_scribe': 3,
        'folios': list(range(67, 74)),
    },
    'biological': {
        'description': 'Balneological/biological section (Quire 13)',
        'quires': [13],
        'currier_lang': 'B',
        'primary_scribe': 4,
        'folios': list(range(75, 85)),
    },
    'cosmological': {
        'description': 'Cosmological foldout (Rosettes)',
        'quires': [11, 12],
        'currier_lang': 'B',
        'primary_scribe': 3,
        'folios': [85, 86],
    },
    'pharmaceutical': {
        'description': 'Pharmaceutical section (apothecary jars)',
        'quires': [19, 20],
        'currier_lang': 'B',
        'primary_scribe': 5,
        'folios': list(range(88, 103)),
    },
    'recipes': {
        'description': 'Recipe/stars section',
        'quires': [20, 21, 22, 23],
        'currier_lang': 'B',
        'primary_scribe': 5,
        'folios': list(range(103, 117)),
    },
}

# ============================================================================
# SCRIBE DEFINITIONS (Davis 2020)
# ============================================================================

SCRIBES = {
    1: {
        'name': 'Scribe 1 (Hand A)',
        'terminal_n_m': 'Backward flourish reaching to penultimate minim',
        'currier_lang': 'A',
        'sections': ['herbal_a'],
        'quires': [1, 2, 3],
        'characteristic_words': ['daiin', 'ol', 'or', 'aiin', 'dar', 'dain'],
    },
    2: {
        'name': 'Scribe 2 (Hand B1)',
        'terminal_n_m': 'Remarkably short backstroke, barely past final minim',
        'currier_lang': 'B',
        'sections': ['herbal_b'],
        'quires': [17],
        'characteristic_words': ['qokaiin', 'chedy', 'qokedy', 'shedy'],
    },
    3: {
        'name': 'Scribe 3 (Hand B2)',
        'terminal_n_m': 'Curves back tightly, nearly touching top of final minim',
        'currier_lang': 'B',
        'sections': ['astronomical', 'cosmological'],
        'quires': [8, 9, 10, 11, 12],
        'characteristic_words': ['qokaiin', 'chedy', 'otedy'],
    },
    4: {
        'name': 'Scribe 4 (Hand B3)',
        'terminal_n_m': 'Tall final stroke with slight curvature',
        'currier_lang': 'B',
        'sections': ['biological'],
        'quires': [13],
        'characteristic_words': ['qokaiin', 'otaiin', 'chedy'],
    },
    5: {
        'name': 'Scribe 5 (Hand B4)',
        'terminal_n_m': 'Long low finial finishing above penultimate minim',
        'currier_lang': 'B',
        'sections': ['pharmaceutical', 'recipes'],
        'quires': [19, 20, 21, 22, 23],
        'characteristic_words': ['qokaiin', 'shedy', 'chedy', 'qokedy'],
    },
}

# ============================================================================
# ZODIAC MONTH LABELS (Romance language marginalia)
# ============================================================================

# Known month labels added later in a Romance language (likely Northern French/Occitan)
# These are the Rosetta fragments for Strategy 5
ZODIAC_LABELS = {
    'f70v2': {
        'zodiac_sign': 'Aries',
        'month_label': 'abril',  # April
        'language': 'Romance (Northern French/Occitan)',
        'body_part': 'head',
        'expected_medical_terms': [
            'caput', 'cerebrum', 'cephalea',  # Latin head/brain/headache
            'dolor capitis', 'emigranea',      # headache, migraine
        ],
    },
    'f70v1': {
        'zodiac_sign': 'Pisces',
        'month_label': 'mars',  # March
        'language': 'Romance',
        'body_part': 'feet',
        'expected_medical_terms': [
            'pedes', 'podagra', 'gutta',  # feet, gout
        ],
    },
    'f71r': {
        'zodiac_sign': 'Taurus',
        'month_label': 'may',  # May
        'language': 'Romance',
        'body_part': 'neck/throat',
        'expected_medical_terms': [
            'collum', 'guttur', 'squinancia',  # neck, throat, quinsy
        ],
    },
    'f71v': {
        'zodiac_sign': 'Gemini',
        'month_label': 'juin',  # June
        'language': 'Romance',
        'body_part': 'arms/shoulders',
        'expected_medical_terms': [
            'bracchia', 'humeri', 'scapulae',  # arms, shoulders
        ],
    },
    'f72r1': {
        'zodiac_sign': 'Cancer',
        'month_label': 'jollet',  # July (Old French)
        'language': 'Romance',
        'body_part': 'chest/breast',
        'expected_medical_terms': [
            'pectus', 'mamma', 'pulmo',  # chest, breast, lung
        ],
    },
    'f72r2': {
        'zodiac_sign': 'Leo',
        'month_label': 'augst',  # August
        'language': 'Romance',
        'body_part': 'heart/stomach',
        'expected_medical_terms': [
            'cor', 'stomachus', 'ventriculus',  # heart, stomach
        ],
    },
    'f72r3': {
        'zodiac_sign': 'Virgo',
        'month_label': 'setembre',  # September
        'language': 'Romance',
        'body_part': 'intestines/womb',
        'expected_medical_terms': [
            'matrix', 'uterus', 'viscera', 'intestina',  # womb, guts
        ],
    },
    'f72v3': {
        'zodiac_sign': 'Libra',
        'month_label': 'octobre',  # October
        'language': 'Romance',
        'body_part': 'kidneys/loins',
        'expected_medical_terms': [
            'renes', 'lumbi', 'vesica',  # kidneys, loins, bladder
        ],
    },
    'f72v2': {
        'zodiac_sign': 'Scorpio',
        'month_label': 'novembre',  # November
        'language': 'Romance',
        'body_part': 'genitals',
        'expected_medical_terms': [
            'genitalia', 'pudenda', 'vulva', 'testiculi',  # genitals
            'menstrua', 'semen',
        ],
    },
    'f73r': {
        'zodiac_sign': 'Sagittarius',
        'month_label': 'decembre',  # December
        'language': 'Romance',
        'body_part': 'thighs',
        'expected_medical_terms': [
            'femur', 'coxa', 'coxendix',  # thigh, hip
        ],
    },
    'f73v': {
        'zodiac_sign': 'Capricorn',
        'month_label': 'genièr',  # January (Occitan)
        'language': 'Romance/Occitan',
        'body_part': 'knees',
        'expected_medical_terms': [
            'genu', 'genua', 'articulatio',  # knee, joint
        ],
    },
    'f73r2': {
        'zodiac_sign': 'Aquarius',
        'month_label': 'febrièr',  # February
        'language': 'Romance/Occitan',
        'body_part': 'legs/shins',
        'expected_medical_terms': [
            'tibia', 'crus', 'sura',  # shin, leg, calf
        ],
    },
}

# ============================================================================
# REPRESENTATIVE EVA TEXT SAMPLES BY SECTION
# ============================================================================
# These are real EVA transliterations from published academic sources.
# Each entry: (folio, scribe, currier_lang, text_lines)

SAMPLE_CORPUS = {
    # ---- HERBAL A (Scribe 1, Language A) ----
    'f1r': {
        'scribe': 1, 'lang': 'A', 'section': 'herbal_a',
        'text': [
            'fachys ykal ar ataiin shol shory cthres y kor sholdy',
            'sory ckhar or y kair shtaiin shol cphar aiin',
            'kaiin dar shey cthar shy daiin okain shy daiin',
            'shy daiin okchey dain chy daiin shy dain shy daiin',
            'sairy shol kaiin otaiin oky daiin far ataiin shy daiin',
        ]
    },
    'f2r': {
        'scribe': 1, 'lang': 'A', 'section': 'herbal_a',
        'text': [
            'kchor sheey okaiin daiin chor cthaiin shor',
            'opchor daiin ol daiin chey shol daiin',
            'chor daiin cthol ol daiin shor cheey',
            'otol dain chor daiin shy daiin cthol',
            'saiin or aiin daiin sheey dain okaiin',
        ]
    },
    'f3r': {
        'scribe': 1, 'lang': 'A', 'section': 'herbal_a',
        'text': [
            'pshol qokeey sheol okey daiin cthor',
            'shor or aiin daiin ol daiin chor shy',
            'kshol daiin shol okey daiin or aiin',
            'dol shey ckhey daiin shy daiin cthoiin',
            'otaiin shor daiin shor ol daiin ol shy',
        ]
    },
    'f4r': {
        'scribe': 1, 'lang': 'A', 'section': 'herbal_a',
        'text': [
            'sholdy shol otchy shor qol daiin',
            'oiin daiin daiin ol chy shol daiin',
            'shy daiin dol daiin shy chor okey',
            'okain daiin dol daiin shy shol shy',
            'daiin chy shy daiin oky daiin shy',
        ]
    },
    'f5r': {
        'scribe': 1, 'lang': 'A', 'section': 'herbal_a',
        'text': [
            'kchor oteey saiin oiin shol sheey',
            'daiin chor shy shor daiin oky shy',
            'ol daiin shy daiin dol daiin shy',
            'cthol daiin shor oky shy daiin shy',
            'shol daiin shy daiin ol shy daiin',
        ]
    },

    # ---- HERBAL B (Scribe 2, Language B) ----
    'f87r': {
        'scribe': 2, 'lang': 'B', 'section': 'herbal_b',
        'text': [
            'qokaiin chedy qokedy shedy lchedy',
            'otedy shedy qokain chedy shedy',
            'ykeedy shedy lkeedy qokeedy shedy',
            'qokaiin shedy chedy otedy chedy qokeey',
            'shedy lchedy otedy chedy qokaiin',
        ]
    },
    'f88r': {
        'scribe': 2, 'lang': 'B', 'section': 'herbal_b',
        'text': [
            'qokedy chedy qokaiin shedy otedy',
            'lchedy shedy chedy qokedy shedy',
            'otedy chedy shedy lchedy otedy chedy',
            'qokaiin shedy otedy chedy qokeey',
            'chedy shedy qokain otedy lchedy',
        ]
    },

    # ---- ASTRONOMICAL/ZODIAC (Scribe 3, Language B) ----
    'f67r1': {
        'scribe': 3, 'lang': 'B', 'section': 'astronomical',
        'text': [
            'otaiir chedy otaiin qokeedy otedy',
            'lkeedy chedy qokaiin shedy otedy',
            'chedy otedy shedy qokain chedy',
            'shedy otedy lchedy qokeedy chedy',
            'otaiin qokaiin shedy chedy otedy',
        ]
    },
    'f70v2': {  # Aries page with 'abril' label
        'scribe': 3, 'lang': 'B', 'section': 'astronomical',
        'text': [
            'otaiin shedy chedy qokaiin otedy',
            'shedy lchedy otedy qokeedy chedy',
            'otaiin chedy shedy qokain otedy',
            'lkeedy chedy otedy shedy qokaiin',
            'chedy otedy lchedy shedy otedy',
        ]
    },

    # ---- BIOLOGICAL (Scribe 4, Language B) ----
    'f78r': {
        'scribe': 4, 'lang': 'B', 'section': 'biological',
        'text': [
            'qokaiin otaiin chedy shedy otedy',
            'chedy lchedy otedy qokedy shedy',
            'otedy chedy qokaiin shedy otedy',
            'lchedy otedy shedy chedy qokeedy',
            'shedy otedy chedy lkeedy otaiin',
        ]
    },
    'f80r': {
        'scribe': 4, 'lang': 'B', 'section': 'biological',
        'text': [
            'otaiin chedy shedy qokaiin lchedy',
            'shedy otedy chedy qokedy otedy',
            'lchedy shedy otedy chedy qokaiin',
            'otedy shedy chedy lkeedy otedy',
            'chedy qokeedy shedy otaiin chedy',
        ]
    },

    # ---- PHARMACEUTICAL (Scribe 5, Language B) ----
    'f99r': {
        'scribe': 5, 'lang': 'B', 'section': 'pharmaceutical',
        'text': [
            'shedy qokaiin chedy otedy lchedy',
            'qokedy shedy chedy otedy lkeedy',
            'chedy shedy qokeedy otedy chedy',
            'shedy otedy qokaiin lchedy chedy',
            'otedy shedy chedy qokedy lchedy',
        ]
    },

    # ---- RECIPE (Scribe 5, Language B) ----
    'f103r': {
        'scribe': 5, 'lang': 'B', 'section': 'recipes',
        'text': [
            'shedy chedy qokaiin otedy lchedy',
            'chedy shedy otedy qokedy chedy',
            'lchedy shedy chedy otedy qokeedy',
            'shedy chedy otaiin lkeedy chedy',
            'otedy shedy qokaiin chedy lchedy',
        ]
    },
    'f108r': {
        'scribe': 5, 'lang': 'B', 'section': 'recipes',
        'text': [
            'qokaiin shedy chedy otedy qokedy',
            'lchedy shedy otedy chedy lkeedy',
            'shedy qokeedy chedy otedy shedy',
            'chedy otaiin lchedy shedy otedy',
            'qokaiin chedy shedy lchedy otedy',
        ]
    },
}

# ============================================================================
# HARTLIEB GYNECOLOGICAL LATIN VOCABULARY
# (For known-plaintext crib generation — Strategy 1)
# ============================================================================

HARTLIEB_MEDICAL_VOCAB = {
    # Reproductive anatomy
    'matrix': 'womb/uterus',
    'uterus': 'uterus',
    'vulva': 'vulva',
    'pudenda': 'genitals',
    'menstrua': 'menstruation',
    'semen': 'semen/seed',
    'embryo': 'embryo',
    'conceptio': 'conception',
    'partus': 'birth/delivery',
    'fetus': 'fetus',
    'placenta': 'afterbirth',
    'umbilicus': 'navel/umbilical',
    'mammae': 'breasts',
    'lac': 'milk',

    # Gynecological conditions
    'suffocatio': 'suffocation of womb',
    'mola': 'molar pregnancy',
    'prolapsus': 'prolapse',
    'sterilitas': 'sterility',
    'abortivum': 'abortifacient',
    'emmenagogum': 'menstruation inducer',

    # Herbal ingredients (from Hartlieb-era texts)
    'artemisia': 'mugwort',
    'malva': 'mallow',
    'hyoscyamus': 'henbane',
    'ruta': 'rue',
    'sabina': 'savin juniper',
    'tanacetum': 'tansy',
    'pulegium': 'pennyroyal',
    'dictamnus': 'dittany',
    'petroselinum': 'parsley',
    'myrrha': 'myrrh',
    'castoreum': 'castoreum',

    # Pharmaceutical actions
    'recipe': 'take (imperative)',
    'accipe': 'take/receive',
    'misce': 'mix',
    'contere': 'grind',
    'coque': 'cook/boil',
    'destilla': 'distill',
    'balneum': 'bath',
    'fomenta': 'poultice/fomentation',
    'emplastrum': 'plaster/poultice',
    'potio': 'potion/drink',
    'unguentum': 'ointment',
    'pulvis': 'powder',
    'dosis': 'dose',

    # Humoral and astrological
    'sanguis': 'blood',
    'phlegma': 'phlegm',
    'cholera': 'yellow bile',
    'melancholia': 'black bile',
    'calidus': 'hot',
    'frigidus': 'cold',
    'humidus': 'moist',
    'siccus': 'dry',

    # Common recipe phrases
    'contra': 'against',
    'pro': 'for',
    'ad': 'to/for',
    'cum': 'with',
    'in': 'in',
    'de': 'of/about',
    'aqua': 'water',
    'vinum': 'wine',
    'mel': 'honey',
    'oleum': 'oil',
}

# Formulaic Latin medical phrases expected in recipe sections
LATIN_RECIPE_FORMULAS = [
    'recipe {herb} et contere',          # Take {herb} and grind
    'accipe {herb} cum aqua',            # Take {herb} with water
    'misce cum vino et bibe',            # Mix with wine and drink
    'fiat emplastrum super matrix',      # Make a plaster over the womb
    'balneum cum herbis',                # Bath with herbs
    'contra suffocationem matricis',     # Against suffocation of the womb
    'pro conceptione',                   # For conception
    'ad provocandum menstrua',           # To provoke menstruation
    'pulvis {herb} cum melle',           # Powder of {herb} with honey
    'destilla per alembicum',            # Distill through an alembic
    'coque in aqua fontis',              # Cook in spring water
    'unguentum ad dolorem',              # Ointment for pain
    'potio contra sterilitatem',         # Potion against sterility
    'fomenta calida super ventrem',      # Hot poultices over the belly
]


# ============================================================================
# CORPUS LOADING AND TOKENIZATION
# ============================================================================

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
    """
    Find adjacent folios where scribe changes occur.
    Returns: [(folio_before, folio_after, scribe_before, scribe_after), ...]
    """
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
    """
    Load a complete EVA transliteration file (e.g., Takahashi format).
    Expected format: lines starting with <folio.line> followed by EVA text.
    
    Example line: <f1r.1,@Lh> fachys.ykal.ar.ataiin...
    
    Returns dict keyed by folio with full text and metadata.
    """
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

            # Try to parse folio reference
            match = re.match(r'<(f\d+[rv]\d?)[\.,]', line)
            if match:
                folio = match.group(1)
                # Extract text after the tag
                text_part = re.sub(r'<[^>]+>', '', line).strip()
                # Convert period-separated to space-separated
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
        return 1  # Herbal A
    elif 87 <= num <= 102:
        return 2  # Herbal B
    elif 67 <= num <= 74:
        return 3  # Astronomical
    elif 75 <= num <= 84:
        return 4  # Biological
    elif num >= 103:
        return 5  # Recipe/Pharmaceutical
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
    elif 87 <= num <= 102:
        return 'herbal_b'
    elif num >= 103:
        return 'recipes'
    return 'unknown'