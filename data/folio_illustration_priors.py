"""
Folio Illustration Priors: Per-folio botanical word boost tables
================================================================
Builds a {folio_id: {latin_word: boost_factor}} dictionary that the
NgramMaskSolver uses to prefer candidates semantically related to the
plant depicted on each folio's illustration.

Three tiers of association (highest wins, no stacking):
  Tier 1: Exact plant names + inflected forms
  Tier 2: Semantic associates (medicinal properties, humoral terms)
  Tier 3: Generic botanical vocabulary (plant parts, degree terms)

Boost factors are multiplicative on the transition probability score.
0 × boost = 0, so the prior alone cannot create resolutions — it only
disambiguates when multiple candidates are competitive.

Phase 13b  ·  Voynich Convergence Attack
"""

from typing import Dict, List, Set

from orchestrators._config import (
    ILLUSTRATION_TIER1_BOOST,
    ILLUSTRATION_TIER2_BOOST,
    ILLUSTRATION_TIER3_BOOST,
)
from data.botanical_name_mapping import build_folio_name_map
from data.botanical_identifications import (
    PLANT_IDS, PLANT_PART_TERMS, HUMORAL_LABEL_TERMS,
)

# ── Common Latin inflectional suffixes (longest first) ──────────
_LATIN_SUFFIXES = [
    'ibus', 'orum', 'arum',
    'ae', 'am', 'um', 'em', 'is', 'us', 'os', 'es', 'as',
    'i', 'o', 'a', 'e',
]

# ── Suffixes to re-add when generating inflected forms ──────────
_LATIN_ENDINGS = [
    'a', 'ae', 'am', 'arum', 'as',
    'e', 'em', 'es',
    'i', 'is', 'ibus',
    'o', 'orum', 'os',
    'um', 'us',
]

_MIN_STEM = 3


def _latin_inflections(word: str) -> Set[str]:
    """Generate common Latin inflections of a word.

    Strips known suffixes to find the stem, then re-adds common endings.
    E.g. "ruta" → {ruta, rutae, rutam, rutarum, rutas, ruti, rutis, ...}
    """
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


# ── Tier 2: Property→Latin word mapping ─────────────────────────
# Maps medicinal property strings from PLANT_IDS to Latin vocabulary
# that would appear in a herbal text discussing that property.
PROPERTY_LATIN_WORDS: Dict[str, List[str]] = {
    'vulnerary': ['vulnus', 'vulnera', 'vulneribus'],
    'anti-inflammatory': ['inflammatio', 'inflammationis'],
    'purgative': ['purgat', 'purgatio'],
    'emmenagogue': ['menstrua', 'menstruis', 'matricis'],
    'vermifuge': ['vermes', 'vermibus'],
    'sedative': ['soporem', 'somnium'],
    'expectorant': ['pectoris', 'tussim'],
    'laxative': ['ventris', 'laxat'],
    'carminative': ['ventris', 'flatulentiam'],
    'digestive': ['stomachi', 'digestionem'],
    'diuretic': ['renum', 'urinam'],
    'astringent': ['constringit', 'adstringit'],
    'anti-septic': ['putredinem'],
    'stimulant': ['excitat', 'confortativam'],
    'analgesic': ['dolorem', 'dolor'],
    'soporific': ['soporem', 'dormire', 'somnium'],
    'anti-tussive': ['tussim', 'pectoris'],
    'anti-venom': ['venenum', 'veneni', 'antidotum', 'theriaca'],
    'cholagogue': ['bilem', 'hepatis'],
    'emollient': ['emollit', 'mollificat'],
    'styptic': ['sanguinem', 'constringit'],
    'sudorific': ['sudorem', 'sudoris'],
    'cordial': ['cor', 'cordis'],
    'anti-pyretic': ['febrem', 'febris'],
    'galactagogue': ['lac', 'lactis'],
    'demulcent': ['emollientem'],
    'anaphrodisiac': ['libidinem'],
    'anti-spasmodic': ['spasmos', 'convulsiones'],
    'anti-depressant': ['melancholiam'],
    'eye wash': ['oculorum', 'oculos', 'collyrium'],
    'eye treatment': ['oculorum', 'oculos', 'collyrium'],
    'wart removal': ['verrucas', 'verrucis'],
    'counter-irritant': ['rubefacientem'],
    'poison': ['venenum', 'toxicum'],
}

# ── Humoral quality→Latin inflected forms ───────────────────────
_HUMORAL_QUALITY_WORDS: Dict[str, List[str]] = {
    'hot': ['calidus', 'calida', 'calidum', 'calidi', 'calidae'],
    'cold': ['frigidus', 'frigida', 'frigidum', 'frigidi', 'frigidae'],
    'dry': ['siccus', 'sicca', 'siccum', 'sicci', 'siccae'],
    'wet': ['humidus', 'humida', 'humidum', 'humidi', 'humidae'],
}

# ── Tier 3: Generic botanical vocabulary ────────────────────────
# These terms appear on any herbal folio regardless of plant species.
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


def build_illustration_prior() -> Dict[str, Dict[str, float]]:
    """Build per-folio illustration prior for NgramMaskSolver.

    Returns:
        {folio_id: {latin_word: boost_factor}}
        Only includes folios with testable botanical identifications.
        Words are lowercase. Highest tier wins (no stacking).
    """
    folio_map = build_folio_name_map()
    prior: Dict[str, Dict[str, float]] = {}

    for folio_id, entry in folio_map.items():
        if not entry['testable']:
            continue
        if folio_id not in PLANT_IDS:
            continue

        word_boosts: Dict[str, float] = {}

        # Tier 3 first (lowest priority, overwritten by higher tiers)
        for word in GENERIC_BOTANICAL_WORDS:
            word_boosts[word] = ILLUSTRATION_TIER3_BOOST

        # Tier 2: property-specific vocabulary
        plant_data = PLANT_IDS[folio_id]
        for prop in plant_data.get('properties', []):
            prop_lower = prop.lower().strip()
            if prop_lower in PROPERTY_LATIN_WORDS:
                for word in PROPERTY_LATIN_WORDS[prop_lower]:
                    word_boosts[word] = ILLUSTRATION_TIER2_BOOST

        # Tier 2: humoral quality terms for this plant's category
        humoral_cat = plant_data.get('humoral', '')
        if humoral_cat:
            for part in humoral_cat.split('_'):
                if part in _HUMORAL_QUALITY_WORDS:
                    for word in _HUMORAL_QUALITY_WORDS[part]:
                        word_boosts[word] = ILLUSTRATION_TIER2_BOOST

        # Tier 1: exact plant names and inflected forms (highest priority)
        for name in entry['single_word_names']:
            for form in _latin_inflections(name):
                word_boosts[form] = ILLUSTRATION_TIER1_BOOST

        prior[folio_id] = word_boosts

    return prior
