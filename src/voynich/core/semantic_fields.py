"""
Semantic field classifications for medieval medical Latin vocabulary.

Each word is assigned to one or more semantic fields based on its role
in medical recipe structures (Circa Instans, Antidotarium Nicolai,
Macer Floridus patterns).

Fields:
  PLANT       — plant names and botanical terms
  PREPARATION — verbs describing preparation methods
  MEDIUM      — carrier substances (water, wine, oil)
  APPLICATION — how the remedy is applied (drink, anoint)
  BODY_PART   — anatomical targets
  INDICATION  — diseases and symptoms
  DOSAGE      — quantity and measurement terms
  HUMORAL     — Galenic humoral theory terms
  TEMPORAL    — timing and duration
  CONNECTIVE  — structural function words
  INGREDIENT  — animal/mineral ingredients (non-plant)
  PROPERTY    — therapeutic property adjectives
"""

import json
import os
import re
from functools import lru_cache

from voynich.core._paths import data_dir, json_dir
_DIR = str(data_dir())

# ── Load source vocabularies from JSON ──────────────────────────────
with open(os.path.join(_DIR, 'json', 'macer_floridus.json')) as _f:
    _mf = json.load(_f)

with open(os.path.join(_DIR, 'json', 'medical_vocabulary.json')) as _f:
    _med = json.load(_f)

with open(os.path.join(_DIR, 'json', 'phase6_vocabulary.json')) as _f:
    _p6 = json.load(_f)

with open(os.path.join(_DIR, 'json', 'botanical_identifications.json')) as _f:
    _bot = json.load(_f)


def _flatten_inflected(category_dict):
    """Flatten {lemma: [form1, form2, ...]} → set of all forms."""
    forms = set()
    for lemma, inflections in category_dict.items():
        forms.add(lemma.lower())
        for f in inflections:
            forms.add(f.lower())
    return forms


def _split_phrases(word_list):
    """Split multi-word phrases into individual words."""
    words = set()
    for entry in word_list:
        for w in entry.lower().split():
            if len(w) >= 2:
                words.add(w)
    return words


# ── PLANT field ─────────────────────────────────────────────────────
_PLANT_WORDS = set()
# Source 1: EXPANDED_PLANT_NAMES (106 names from macer_floridus.json)
for name in _mf['EXPANDED_PLANT_NAMES']:
    _PLANT_WORDS.add(name.lower())
# Source 2: MEDICAL_PLANT_NAMES (44 lemmas + inflections)
_PLANT_WORDS |= _flatten_inflected(_med['MEDICAL_PLANT_NAMES'])
# Source 3: PLANT_IDS candidates from botanical identifications
for folio_data in _bot.get('PLANT_IDS', {}).values():
    for candidate in folio_data.get('candidates', []):
        for w in candidate.lower().split():
            if len(w) >= 3:
                _PLANT_WORDS.add(w)
# Source 4: plant part terms
for term in _bot.get('PLANT_PART_TERMS', {}).keys():
    _PLANT_WORDS.add(term.lower())
# Core plant-part words always included
_PLANT_WORDS |= {
    'herba', 'radix', 'folium', 'flos', 'semen', 'cortex',
    'fructus', 'succus', 'resina', 'gummi', 'bacca',
    'folia', 'radicis', 'florem', 'seminis', 'corticis',
    'herbae', 'herbam', 'radicem', 'flores', 'semina',
    'pulvis', 'pulverem',
}

# ── PREPARATION field ───────────────────────────────────────────────
_PREPARATION_WORDS = set()
# Source 1: MEDICAL_PROCESS_VERBS (43 lemmas + inflections)
_PREPARATION_WORDS |= _flatten_inflected(_med['MEDICAL_PROCESS_VERBS'])
# Source 2: PREPARATION_WORDS phrases → individual words
_PREPARATION_WORDS |= _split_phrases(_mf['PREPARATION_WORDS'])
# Source 3: EXTRA_VERBS
for v in _p6['EXTRA_VERBS']:
    _PREPARATION_WORDS.add(v.lower())
# Core preparation verbs
_PREPARATION_WORDS |= {
    'recipe', 'accipe', 'coque', 'tere', 'misce', 'distilla',
    'solve', 'cola', 'exprime', 'infunde', 'macera', 'decoque',
    'combure', 'contunde', 'incide', 'pista', 'cribra',
    'incorpora', 'liquefac', 'confice', 'fac', 'adde',
    'pone', 'impone', 'extrahe', 'sicca', 'pulveriza',
    'nota', 'collige', 'contere', 'lava',
}

# ── MEDIUM field ────────────────────────────────────────────────────
_MEDIUM_WORDS = set()
# Source 1: EXPANDED_SUBSTANCE_WORDS
_MEDIUM_WORDS |= _split_phrases(_mf['EXPANDED_SUBSTANCE_WORDS'])
# Core medium words
_MEDIUM_WORDS |= {
    'aqua', 'aquae', 'aquam', 'vinum', 'vini', 'vino',
    'oleum', 'olei', 'oleo', 'mel', 'melle', 'mellis',
    'acetum', 'aceti', 'aceto', 'lac', 'lacte', 'lactis',
    'butyrum', 'butyrm', 'axungia', 'cera', 'cerae', 'ceram',
    'serum', 'mustum', 'syrupum', 'sirupum',
    'unguentum', 'emplastrum', 'balneum', 'fomenta', 'potio',
    'lardo', 'pice', 'sapone', 'sale', 'sevo', 'succo',
}

# ── APPLICATION field ───────────────────────────────────────────────
_APPLICATION_WORDS = set()
# Source 1: DELIVERY_WORDS phrases → individual words
_APPLICATION_WORDS |= _split_phrases(_mf['DELIVERY_WORDS'])
# Source 2: MEDICAL_PHARMACEUTICAL_TERMS (product forms)
_APPLICATION_WORDS |= _flatten_inflected(_med['MEDICAL_PHARMACEUTICAL_TERMS'])
# Core application verbs/nouns
_APPLICATION_WORDS |= {
    'bibe', 'bibat', 'bibere', 'applica', 'unge', 'ungatur',
    'gargarisma', 'gargarizet', 'fomentum', 'cataplasma',
    'collyrium', 'pessarium', 'suffumiga', 'fumiga',
    'inspira', 'insufla', 'superpone', 'ablue',
    'instilla', 'liga', 'potu', 'commisce', 'cibo',
}

# ── BODY_PART field ─────────────────────────────────────────────────
_BODY_PART_WORDS = set()
# Source 1: EXPANDED_BODY_WORDS
for w in _mf['EXPANDED_BODY_WORDS']:
    _BODY_PART_WORDS.add(w.lower())
# Source 2: MEDICAL_ANATOMICAL_TERMS (50 lemmas + inflections)
_BODY_PART_WORDS |= _flatten_inflected(_med['MEDICAL_ANATOMICAL_TERMS'])
# Source 3: EXTRA_NOUNS (anatomical subset)
_ANAT_NOUNS = {
    'cerebrum', 'cerebellum', 'medulla', 'diaphragma', 'peritoneum',
    'mesenterium', 'omentum', 'pancreas', 'glandula', 'cartilago',
    'ligamentum', 'tendon', 'musculus', 'arteria', 'vena', 'nervus',
    'cutis', 'epidermis', 'dermis', 'bronchus', 'larynx', 'pharynx',
    'oesophagus', 'pylorus', 'duodenum', 'colon', 'rectum', 'anus',
    'vesica', 'ureter', 'urethra', 'uterus', 'ovarium', 'testis',
    'prostata',
}
_BODY_PART_WORDS |= _ANAT_NOUNS
# Core body parts
_BODY_PART_WORDS |= {
    'caput', 'capitis', 'oculus', 'oculi', 'oculorum',
    'auris', 'aures', 'aurium', 'nasus', 'nasi',
    'os', 'oris', 'dens', 'dentium', 'guttur',
    'pectus', 'pectoris', 'pulmo', 'cor', 'cordis',
    'stomachi', 'stomachus', 'hepar', 'hepatis',
    'ren', 'renis', 'renum', 'vulnus', 'vulneris',
    'ossa', 'ossium', 'sanguis', 'sanguinis',
    'manus', 'manuum', 'pes', 'pedis', 'pedum',
    'dorsi', 'dorsum', 'lumborum',
}

# ── INDICATION field ────────────────────────────────────────────────
_INDICATION_WORDS = set()
# Source 1: CONDITION_WORDS
for w in _mf['CONDITION_WORDS']:
    _INDICATION_WORDS.add(w.lower())
# Source 2: MEDICAL_DISEASE_TERMS (45 lemmas + inflections)
_INDICATION_WORDS |= _flatten_inflected(_med['MEDICAL_DISEASE_TERMS'])
# Source 3: DIAGNOSTIC_TERMS
for w in _p6['DIAGNOSTIC_TERMS']:
    _INDICATION_WORDS.add(w.lower())
# Core indication words
_INDICATION_WORDS |= {
    'febris', 'febrem', 'dolor', 'dolorem', 'tussis', 'tussim',
    'asthma', 'dysenteria', 'hydrops', 'epilepsia', 'epilepsiam',
    'paralysis', 'paralysim', 'lepra', 'lepram', 'scabies', 'scabiem',
    'podagra', 'podagram', 'arthritis', 'calculis',
    'pestilentia', 'pestem', 'vermis', 'vermes',
    'fluxus', 'fluxum', 'inflammatio', 'inflammationem',
    'apostema', 'ulcus', 'ulcera', 'morsus', 'venenum',
    'fractura', 'fracturas', 'spasmus', 'morbus', 'morbi', 'morbum',
    'cephalgia', 'pleuritis', 'colica', 'colicam',
    'nausea', 'vomitus', 'diarrhea', 'obstipatio',
    'insomnia', 'melancholia', 'melancholiam', 'mania',
    'putrefactionem', 'convulsiones', 'tremorem',
    'quartana', 'tertiana', 'quotidiana', 'continua',
    'acuta', 'chronica',
}

# ── DOSAGE field ────────────────────────────────────────────────────
_DOSAGE_WORDS = set()
# Source 1: DOSAGE_WORDS phrases → individual words
_DOSAGE_WORDS |= _split_phrases(_mf['DOSAGE_WORDS'])
# Source 2: MEDICAL_DOSAGE_TERMS (12 lemmas + inflections)
_DOSAGE_WORDS |= _flatten_inflected(_med['MEDICAL_DOSAGE_TERMS'])
# Core dosage words
_DOSAGE_WORDS |= {
    'drachma', 'drachmae', 'drachmam', 'uncia', 'unciae', 'unciam',
    'libra', 'scrupulus', 'scrupulum', 'obolus',
    'cochleare', 'cochlear', 'pugillum', 'pugillus',
    'manipulus', 'manipulum', 'fasciculus',
    'quantum', 'parum', 'multum', 'modicum', 'sufficiens',
    'partes', 'aequales', 'ana', 'dosi', 'dosis',
}

# ── HUMORAL field ───────────────────────────────────────────────────
_HUMORAL_WORDS = set()
# Source 1: GALENIC_TERMS
for w in _p6['GALENIC_TERMS']:
    _HUMORAL_WORDS.add(w.lower())
# Source 2: HUMORAL_LABEL_TERMS
for term in _bot.get('HUMORAL_LABEL_TERMS', {}).keys():
    _HUMORAL_WORDS.add(term.lower())
# Source 3: DEGREE_TERMS
for term in _bot.get('DEGREE_TERMS', {}).keys():
    _HUMORAL_WORDS.add(term.lower())
# Core humoral words
_HUMORAL_WORDS |= {
    'calida', 'frigida', 'humida', 'sicca',
    'calidum', 'frigidum', 'humidum', 'siccum',
    'calidus', 'frigidus', 'humidus', 'siccus',
    'gradu', 'primo', 'secundo', 'tertio', 'quarto',
    'temperata', 'complexio', 'natura', 'naturalis',
    'humor', 'sanguis', 'phlegma', 'cholera',
    'primus', 'secundus', 'tertius', 'quartus',
}

# ── TEMPORAL field ──────────────────────────────────────────────────
_TEMPORAL_WORDS = set()
# Source 1: TIME_WORDS
_TEMPORAL_WORDS |= _split_phrases(_mf['TIME_WORDS'])
# Core temporal words
_TEMPORAL_WORDS |= {
    'mane', 'sero', 'vespere', 'nocte', 'hora', 'horis',
    'dies', 'die', 'diebus', 'jejunus', 'ieiunus',
    'post', 'ante', 'prandium', 'cenam', 'cibum',
    'continuo', 'quotidie', 'semel', 'bis', 'ter',
    'vere', 'aestate', 'autumno', 'hieme',
}

# ── CONNECTIVE field ───────────────────────────────────────────────
_CONNECTIVE_WORDS = {
    'et', 'in', 'cum', 'ad', 'per', 'contra', 'vel',
    'aut', 'sed', 'non', 'est', 'sunt', 'sit',
    'si', 'quod', 'quia', 'item', 'similiter',
    'nam', 'enim', 'ergo', 'igitur', 'autem',
    'da', 'de', 'ex', 'ab', 'super', 'sub',
    'valet', 'habet', 'quae', 'qui', 'tunc',
    'praeterea', 'insuper', 'quoque', 'aliter',
    'ut', 'nec', 'sic', 'tam', 'ita',
    'probatum', 'verum', 'certum', 'expertum',
    'sanabitur', 'curabitur', 'deo', 'volente',
}

# ── INGREDIENT field (animal/mineral) ──────────────────────────────
_INGREDIENT_WORDS = set()
for w in _p6['ANIMAL_INGREDIENTS']:
    _INGREDIENT_WORDS.add(w.lower())
for w in _p6['MINERAL_INGREDIENTS']:
    _INGREDIENT_WORDS.add(w.lower())

# ── PROPERTY field (therapeutic adjectives) ─────────────────────────
_PROPERTY_WORDS = set()
for w in _mf['EXPANDED_PROPERTY_WORDS']:
    _PROPERTY_WORDS.add(w.lower())
for w in _p6['EXTRA_ADJECTIVES']:
    _PROPERTY_WORDS.add(w.lower())
_PROPERTY_WORDS |= {
    'utilis', 'efficax', 'fortior', 'melius',
    'periculosus', 'viva', 'viridis',
}


# ══ Assembled field dictionary ══════════════════════════════════════

SEMANTIC_FIELDS = {
    'PLANT':       _PLANT_WORDS,
    'PREPARATION': _PREPARATION_WORDS,
    'MEDIUM':      _MEDIUM_WORDS,
    'APPLICATION':  _APPLICATION_WORDS,
    'BODY_PART':   _BODY_PART_WORDS,
    'INDICATION':  _INDICATION_WORDS,
    'DOSAGE':      _DOSAGE_WORDS,
    'HUMORAL':     _HUMORAL_WORDS,
    'TEMPORAL':    _TEMPORAL_WORDS,
    'CONNECTIVE':  _CONNECTIVE_WORDS,
    'INGREDIENT':  _INGREDIENT_WORDS,
    'PROPERTY':    _PROPERTY_WORDS,
}

# Fields that are part of medical recipes (everything except CONNECTIVE)
MEDICAL_FIELDS = frozenset({
    'PLANT', 'PREPARATION', 'MEDIUM', 'APPLICATION', 'BODY_PART',
    'INDICATION', 'DOSAGE', 'HUMORAL', 'TEMPORAL', 'INGREDIENT', 'PROPERTY',
})

# Union of all medical field words
MEDICAL_VOCABULARY = set()
for field_name in MEDICAL_FIELDS:
    MEDICAL_VOCABULARY |= SEMANTIC_FIELDS[field_name]

# Reverse lookup: word → frozenset of fields
WORD_TO_FIELDS = {}
for field_name, words in SEMANTIC_FIELDS.items():
    for word in words:
        if word not in WORD_TO_FIELDS:
            WORD_TO_FIELDS[word] = set()
        WORD_TO_FIELDS[word].add(field_name)
# Freeze all values for hashability (enables lru_cache on get_fields)
WORD_TO_FIELDS = {w: frozenset(fs) for w, fs in WORD_TO_FIELDS.items()}


# ── Latin stemmer ───────────────────────────────────────────────────

# Common Latin suffixes ordered longest-first for greedy stripping
_SUFFIXES = [
    'ibus', 'arum', 'orum', 'ibus', 'ione', 'onis', 'inem',
    'atis', 'atus', 'endo', 'ando',
    'um', 'am', 'em', 'is', 'ae', 'os', 'as', 'es',
    'us', 'im', 'ui', 'ei', 'ii', 'io',
    'i', 'a', 'e', 'o',
]

def latin_stem(word):
    """Simple Latin stemmer: strip common inflectional suffixes.

    Returns stem of minimum length 3, or the word itself if too short.
    """
    w = word.lower()
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


# Pre-computed stem → fields index for O(1) stem lookup.
# Built once at module load; replaces the O(N) linear scan in get_fields().
_STEM_TO_FIELDS = {}
for _w, _fs in WORD_TO_FIELDS.items():
    _st = latin_stem(_w)
    if len(_st) >= 3 and _st not in _STEM_TO_FIELDS:
        _STEM_TO_FIELDS[_st] = _fs

_UNKNOWN_FIELDS = frozenset({'UNKNOWN'})


@lru_cache(maxsize=8192)
def get_fields(word):
    """Look up semantic fields for a word, handling inflection.

    Returns a frozenset of field names. Returns frozenset({'UNKNOWN'})
    if no match found. Results are cached for performance.
    """
    w = word.lower()

    # Exact match
    if w in WORD_TO_FIELDS:
        return WORD_TO_FIELDS[w]

    # Stem match via pre-computed index (O(1) instead of O(N))
    stem = latin_stem(w)
    if len(stem) >= 3 and stem in _STEM_TO_FIELDS:
        return _STEM_TO_FIELDS[stem]

    return _UNKNOWN_FIELDS


# Priority order for primary field selection: most specific first.
# Content-specific fields (what it IS) > action fields (what you DO)
# > descriptive fields > structural/function words.
_FIELD_PRIORITY = {
    'PLANT': 0, 'BODY_PART': 1, 'INDICATION': 2, 'INGREDIENT': 3,
    'DOSAGE': 4, 'MEDIUM': 5,
    'PREPARATION': 6, 'APPLICATION': 7,
    'HUMORAL': 8, 'TEMPORAL': 9, 'PROPERTY': 10,
    'CONNECTIVE': 11, 'UNKNOWN': 12,
}


@lru_cache(maxsize=8192)
def get_primary_field(word):
    """Return the single most specific field for a word.

    Uses specificity-based priority: content fields (PLANT, BODY_PART,
    INDICATION) outrank action fields (PREPARATION, APPLICATION),
    which outrank descriptive and structural fields. Cached for performance.
    """
    fields = get_fields(word)
    if fields == _UNKNOWN_FIELDS:
        return 'UNKNOWN'

    return min(fields, key=lambda f: _FIELD_PRIORITY.get(f, 99))


# Clean up module namespace
del _mf, _med, _p6, _bot, _f
del _PLANT_WORDS, _PREPARATION_WORDS, _MEDIUM_WORDS, _APPLICATION_WORDS
del _BODY_PART_WORDS, _INDICATION_WORDS, _DOSAGE_WORDS, _HUMORAL_WORDS
del _TEMPORAL_WORDS, _CONNECTIVE_WORDS, _INGREDIENT_WORDS, _PROPERTY_WORDS
del _ANAT_NOUNS, _w, _fs, _st
