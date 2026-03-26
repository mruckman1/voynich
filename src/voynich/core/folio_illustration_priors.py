"""
Folio Illustration Priors: Per-folio botanical word boost tables
================================================================
Builds a {folio_id: {latin_word: boost_factor}} dictionary that the
NgramMaskSolver uses to prefer candidates semantically related to the
plant depicted on each folio's illustration.
"""

import json
import os
from typing import Dict, List, Set

from voynich.core.config import (
    ILLUSTRATION_TIER1_BOOST,
    ILLUSTRATION_TIER2_BOOST,
    ILLUSTRATION_TIER3_BOOST,
)
from voynich.core.botanical_name_mapping import build_folio_name_map
from voynich.core.botanical_identifications import (
    PLANT_IDS, PLANT_PART_TERMS, HUMORAL_LABEL_TERMS,
)

from voynich.core._paths import data_dir, json_dir
_DIR = str(data_dir())
with open(os.path.join(_DIR, 'json', 'folio_illustration_priors.json')) as _f:
    _data = json.load(_f)

_LATIN_SUFFIXES = _data['LATIN_SUFFIXES']
_LATIN_ENDINGS = _data['LATIN_ENDINGS']
PROPERTY_LATIN_WORDS: Dict[str, List[str]] = _data['PROPERTY_LATIN_WORDS']
_HUMORAL_QUALITY_WORDS: Dict[str, List[str]] = _data['HUMORAL_QUALITY_WORDS']

del _data, _f

_MIN_STEM = 3

GENERIC_BOTANICAL_WORDS: Set[str] = (
    set(PLANT_PART_TERMS.keys())
    | set(HUMORAL_LABEL_TERMS.keys())
    | {
        'gradu', 'primo', 'secundo', 'tertio', 'quarto',
        'herba', 'planta', 'folium', 'fructus',
        'decoctio', 'infusio', 'emplastrum', 'unguentum',
        'pulvis', 'syrupum', 'potio', 'dosis',
    }
)

def _latin_inflections(word: str) -> Set[str]:
    """Generate common Latin inflections of a word."""
    forms = {word}
    stem = word

    for suf in _LATIN_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= _MIN_STEM:
            stem = word[:-len(suf)]
            break

    for ending in _LATIN_ENDINGS:
        form = stem + ending
        if len(form) >= 4:
            forms.add(form)

    return forms

def build_illustration_prior() -> Dict[str, Dict[str, float]]:
    """Build per-folio illustration prior for NgramMaskSolver.

    Returns:
        {folio_id: {latin_word: boost_factor}}
    """
    folio_map = build_folio_name_map()
    prior: Dict[str, Dict[str, float]] = {}

    for folio_id, entry in folio_map.items():
        if not entry['testable']:
            continue
        if folio_id not in PLANT_IDS:
            continue

        word_boosts: Dict[str, float] = {}

        for word in GENERIC_BOTANICAL_WORDS:
            word_boosts[word] = ILLUSTRATION_TIER3_BOOST

        plant_data = PLANT_IDS[folio_id]
        for prop in plant_data.get('properties', []):
            prop_lower = prop.lower().strip()
            if prop_lower in PROPERTY_LATIN_WORDS:
                for word in PROPERTY_LATIN_WORDS[prop_lower]:
                    word_boosts[word] = ILLUSTRATION_TIER2_BOOST

        humoral_cat = plant_data.get('humoral', '')
        if humoral_cat:
            for part in humoral_cat.split('_'):
                if part in _HUMORAL_QUALITY_WORDS:
                    for word in _HUMORAL_QUALITY_WORDS[part]:
                        word_boosts[word] = ILLUSTRATION_TIER2_BOOST

        for name in entry['single_word_names']:
            for form in _latin_inflections(name):
                word_boosts[form] = ILLUSTRATION_TIER1_BOOST

        prior[folio_id] = word_boosts

    return prior
