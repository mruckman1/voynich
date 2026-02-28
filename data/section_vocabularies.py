"""
Section-specific vocabulary profiles for the Voynich manuscript.

Each section gets supplementary vocabulary, frequency boosts, and template
functions that produce section-characteristic bigrams.  The base generic
corpus (~50K tokens) remains unchanged; these profiles generate an addendum
of ~12,500 tokens appended to the base.

Template functions follow the existing signature:
    (rng, plant, quality, moisture, degree) -> str
"""

from modules.phase5.latin_corpus_expanded import (
    EXPANDED_PLANT_NAMES, EXPANDED_BODY_WORDS, CONDITION_WORDS,
    PREPARATION_WORDS, DELIVERY_WORDS, EXPANDED_SUBSTANCE_WORDS,
    EXPANDED_PROPERTY_WORDS, DOSAGE_WORDS, TIME_WORDS,
    EXPANDED_CLOSING_PHRASES, TRANSITIONAL_PHRASES,
)

# ── Shared vocabulary pools for section templates ──────────────────────

ZODIAC_SIGNS = [
    'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
    'libra', 'scorpio', 'sagittarius', 'capricornus', 'aquarius', 'pisces',
]

PLANETS = ['sol', 'luna', 'mars', 'venus', 'mercurius', 'jupiter', 'saturnus']

ELEMENTS = ['ignis', 'aer', 'aqua', 'terra']

DIRECTIONS = ['oriens', 'occidens', 'meridies', 'septentrio']

MONTHS = [
    'januario', 'februario', 'martio', 'aprili', 'maio', 'junio',
    'julio', 'augusto', 'septembri', 'octobri', 'novembri', 'decembri',
]

HUMORAL_QUALITIES = ['calida', 'frigida', 'humida', 'sicca']

CONTAINER_WORDS = ['vas', 'ampulla', 'phiala', 'pyxis', 'olla']

COMPOUND_FORMS = [
    'electuarium', 'pilulae', 'trochiscus', 'unguentum',
    'emplastrum', 'collyrium', 'linimentum', 'cataplasma',
]

ANATOMICAL_EXTENDED = [
    'corpus', 'membrum', 'vena', 'arteria', 'nervus',
    'musculus', 'cutis', 'matrix', 'uterus', 'mamma',
]

HUMORAL_TERMS = [
    'sanguis', 'cholera', 'melancholia', 'phlegma',
    'complexio', 'temperamentum',
]

BATHING_WORDS = ['balneum', 'lavacrum', 'thermae', 'immersio', 'ablutio']

RECIPE_CONNECTIVES = [
    'item', 'similiter', 'praeterea', 'aliud', 'alterum',
    'probatum', 'expertum', 'optimum',
]


# ── Pharmaceutical templates ──────────────────────────────────────────

def _tpl_pharma_compound(rng, plant, quality, moisture, degree):
    form = rng.choice(COMPOUND_FORMS)
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'{form} de {plant} recipe {plant} {dosage} '
        f'et {plant2} {rng.choice(DOSAGE_WORDS)} misce'
    )

def _tpl_pharma_container(rng, plant, quality, moisture, degree):
    container = rng.choice(CONTAINER_WORDS)
    medium = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'solve {plant} in {medium} et cola '
        f'reserva in {container} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_pharma_dosage(rng, plant, quality, moisture, degree):
    dosage = rng.choice(DOSAGE_WORDS)
    medium = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    form = rng.choice(COMPOUND_FORMS)
    return (
        f'da {dosage} {form} de {plant} cum {medium} '
        f'mane et vespere'
    )

def _tpl_pharma_multi_ingredient(rng, plant, quality, moisture, degree):
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    plant3 = rng.choice(EXPANDED_PLANT_NAMES)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'recipe {plant} {dosage} {plant2} {dosage} '
        f'{plant3} {dosage} ana partes aequales '
        f'tere et cribra et fac {rng.choice(COMPOUND_FORMS)}'
    )

def _tpl_pharma_preparation(rng, plant, quality, moisture, degree):
    medium = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    prep = rng.choice(PREPARATION_WORDS)
    return (
        f'accipe {plant} et {prep} cum {medium} '
        f'inspissa super ignem et {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_pharma_ointment(rng, plant, quality, moisture, degree):
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'unguentum de {plant} recipe {plant} et {plant2} '
        f'cum oleo et cera unge {body}'
    )

PHARMA_TEMPLATES = [
    _tpl_pharma_compound, _tpl_pharma_container, _tpl_pharma_dosage,
    _tpl_pharma_multi_ingredient, _tpl_pharma_preparation,
    _tpl_pharma_ointment,
]


# ── Biological templates ──────────────────────────────────────────────

def _tpl_bio_bathing(rng, plant, quality, moisture, degree):
    bath = rng.choice(BATHING_WORDS)
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    return (
        f'{bath} de {plant} coque {plant} et {plant2} '
        f'in aqua et lava corpus'
    )

def _tpl_bio_humoral(rng, plant, quality, moisture, degree):
    humor = rng.choice(HUMORAL_TERMS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'{humor} abundat in {body} '
        f'complexio {quality} et {moisture} in {degree} gradu '
        f'recipe {plant} contra {rng.choice(CONDITION_WORDS)}'
    )

def _tpl_bio_anatomy(rng, plant, quality, moisture, degree):
    part = rng.choice(ANATOMICAL_EXTENDED)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'{part} est {quality} et {moisture} '
        f'{body} lavetur cum aqua {plant} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_bio_purgation(rng, plant, quality, moisture, degree):
    body = rng.choice(EXPANDED_BODY_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'purgatio {body} recipe {plant} {dosage} '
        f'cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} '
        f'evacuatio per {rng.choice(DELIVERY_WORDS)}'
    )

def _tpl_bio_unction(rng, plant, quality, moisture, degree):
    part = rng.choice(ANATOMICAL_EXTENDED)
    return (
        f'post balneum unge {part} cum oleo {plant} '
        f'est {quality} et {moisture} in {degree} gradu '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_bio_complexion(rng, plant, quality, moisture, degree):
    humor = rng.choice(HUMORAL_TERMS)
    return (
        f'complexio {quality} et {moisture} in {degree} gradu '
        f'{humor} regit per venas et arterias '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)}'
    )

BIO_TEMPLATES = [
    _tpl_bio_bathing, _tpl_bio_humoral, _tpl_bio_anatomy,
    _tpl_bio_purgation, _tpl_bio_unction, _tpl_bio_complexion,
]


# ── Astronomical / Zodiac templates ───────────────────────────────────

def _tpl_astro_melothesia(rng, plant, quality, moisture, degree):
    sign = rng.choice(ZODIAC_SIGNS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'{sign} regit {body} '
        f'quando sol est in {sign} non tange {body} '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)}'
    )

def _tpl_astro_luna(rng, plant, quality, moisture, degree):
    sign = rng.choice(ZODIAC_SIGNS)
    condition = rng.choice(CONDITION_WORDS)
    return (
        f'luna in {sign} bonum est contra {condition} '
        f'recipe {plant} cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)}'
    )

def _tpl_astro_timing(rng, plant, quality, moisture, degree):
    sign = rng.choice(ZODIAC_SIGNS)
    planet = rng.choice(PLANETS)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'quando {planet} est in {sign} '
        f'accipe {plant} {dosage} '
        f'hora {planet} est {quality}'
    )

def _tpl_astro_collection(rng, plant, quality, moisture, degree):
    month = rng.choice(MONTHS)
    sign = rng.choice(ZODIAC_SIGNS)
    return (
        f'mense {month} collige {plant} '
        f'sub signo {sign} '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)}'
    )

def _tpl_astro_planet_herb(rng, plant, quality, moisture, degree):
    planet = rng.choice(PLANETS)
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    return (
        f'{plant} est herba {planet} '
        f'et {plant2} est herba {rng.choice(PLANETS)} '
        f'stella {rng.choice(ZODIAC_SIGNS)} {quality}'
    )

def _tpl_astro_plenilunium(rng, plant, quality, moisture, degree):
    condition = rng.choice(CONDITION_WORDS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'in plenilunio recipe {plant} contra {condition} {body} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

ASTRO_TEMPLATES = [
    _tpl_astro_melothesia, _tpl_astro_luna, _tpl_astro_timing,
    _tpl_astro_collection, _tpl_astro_planet_herb, _tpl_astro_plenilunium,
]


# ── Cosmological templates ───────────────────────────────────────────

def _tpl_cosmo_element(rng, plant, quality, moisture, degree):
    elem = rng.choice(ELEMENTS)
    q = rng.choice(HUMORAL_QUALITIES)
    m = rng.choice(HUMORAL_QUALITIES)
    return (
        f'elementum {elem} est {q} et {m} '
        f'{plant} habet naturam {elem} '
        f'in {degree} gradu'
    )

def _tpl_cosmo_direction(rng, plant, quality, moisture, degree):
    direction = rng.choice(DIRECTIONS)
    elem = rng.choice(ELEMENTS)
    return (
        f'in {direction} dominatur {elem} '
        f'{plant} collige in {direction} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_cosmo_sphere(rng, plant, quality, moisture, degree):
    planet = rng.choice(PLANETS)
    sign = rng.choice(ZODIAC_SIGNS)
    return (
        f'circulus {planet} continet signum {sign} '
        f'sphaera {quality} et {moisture} '
        f'orbis {rng.choice(["primus", "secundus", "tertius", "quartus"])}'
    )

def _tpl_cosmo_mundus(rng, plant, quality, moisture, degree):
    elem = rng.choice(ELEMENTS)
    return (
        f'mundus constat ex {elem} et {rng.choice(ELEMENTS)} '
        f'caelum est {quality} '
        f'terra est {rng.choice(HUMORAL_QUALITIES)} '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)}'
    )

def _tpl_cosmo_elemental_plant(rng, plant, quality, moisture, degree):
    elem = rng.choice(ELEMENTS)
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    return (
        f'{plant} est de natura {elem} '
        f'et {plant2} est de natura {rng.choice(ELEMENTS)} '
        f'centrum mundi est {rng.choice(HUMORAL_QUALITIES)}'
    )

COSMO_TEMPLATES = [
    _tpl_cosmo_element, _tpl_cosmo_direction, _tpl_cosmo_sphere,
    _tpl_cosmo_mundus, _tpl_cosmo_elemental_plant,
]


# ── Recipe templates ──────────────────────────────────────────────────

def _tpl_recipe_item(rng, plant, quality, moisture, degree):
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    medium = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'item recipe {plant} et {plant2} '
        f'coque cum {medium} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_recipe_aliud(rng, plant, quality, moisture, degree):
    condition = rng.choice(CONDITION_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'aliud remedium contra {condition} '
        f'recipe {plant} {dosage} '
        f'{rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)}'
    )

def _tpl_recipe_probatum(rng, plant, quality, moisture, degree):
    condition = rng.choice(CONDITION_WORDS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'probatum est contra {condition} {body} '
        f'accipe {plant} {rng.choice(DOSAGE_WORDS)} '
        f'cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)}'
    )

def _tpl_recipe_antidotum(rng, plant, quality, moisture, degree):
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'antidotum de {plant} recipe {plant} {dosage} '
        f'et {plant2} {rng.choice(DOSAGE_WORDS)} '
        f'misce et da {rng.choice(DELIVERY_WORDS)}'
    )

def _tpl_recipe_decoction(rng, plant, quality, moisture, degree):
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'decoctio de {plant} coque in aqua '
        f'et cola bibe {dosage} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _tpl_recipe_potio(rng, plant, quality, moisture, degree):
    condition = rng.choice(CONDITION_WORDS)
    medium = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'potio contra {condition} recipe {plant} in {medium} '
        f'da {rng.choice(DOSAGE_WORDS)} '
        f'mane et vespere'
    )

RECIPE_TEMPLATES = [
    _tpl_recipe_item, _tpl_recipe_aliud, _tpl_recipe_probatum,
    _tpl_recipe_antidotum, _tpl_recipe_decoction, _tpl_recipe_potio,
]


# ── Section profiles ──────────────────────────────────────────────────

SECTION_PROFILES = {
    'herbal_a': {
        'additional_vocabulary': [],
        'frequency_boosts': {},
        'template_functions': [],
    },
    'herbal_b': {
        'additional_vocabulary': [],
        'frequency_boosts': {},
        'template_functions': [],
    },
    'pharmaceutical': {
        'additional_vocabulary': [
            'electuarium', 'electuarii', 'confectio', 'confectionis',
            'pilulae', 'pilularum', 'trochiscus', 'trochisci',
            'unguentum', 'unguenti', 'emplastrum', 'emplastri',
            'collyrium', 'collyrii', 'linimentum', 'linimenti',
            'cataplasma', 'cataplasmate',
            'vas', 'vasis', 'ampulla', 'ampullae',
            'phiala', 'phialae', 'pyxis', 'pyxidis', 'olla', 'ollae',
            'incorpora', 'incorporare', 'liquefac', 'liquefacere',
            'inspissa', 'inspissare', 'cola', 'colare',
            'ana', 'partes', 'aequales', 'quantum', 'sufficit',
        ],
        'frequency_boosts': {
            'recipe': 2.0, 'misce': 2.0, 'ana': 2.0,
            'tere': 2.0, 'cribra': 2.0, 'fac': 2.0,
        },
        'template_functions': PHARMA_TEMPLATES,
    },
    'biological': {
        'additional_vocabulary': [
            'corpus', 'corporis', 'membrum', 'membri',
            'vena', 'venae', 'arteria', 'arteriae',
            'nervus', 'nervi', 'musculus', 'musculi',
            'cutis', 'matrix', 'matricis', 'uterus', 'uteri',
            'mamma', 'mammae', 'sanguis', 'sanguinis',
            'humor', 'humoris', 'spiritus',
            'cholera', 'cholerae', 'melancholia', 'melancholiae',
            'phlegma', 'phlegmatis', 'complexio', 'complexionis',
            'temperamentum', 'temperamenti',
            'balneum', 'balnei', 'lavacrum', 'lavacri',
            'thermae', 'thermarum', 'immersio', 'immersionis',
            'ablutio', 'ablutionis',
            'digestio', 'digestionis', 'evacuatio', 'evacuationis',
            'purgatio', 'purgationis', 'menstruum', 'menstrui',
            'conceptio', 'conceptionis', 'nutritio', 'nutritionis',
        ],
        'frequency_boosts': {
            'corpus': 2.0, 'sanguis': 2.0, 'balneum': 2.0,
            'complexio': 2.0, 'humor': 2.0, 'lava': 2.0,
        },
        'template_functions': BIO_TEMPLATES,
    },
    'astronomical': {
        'additional_vocabulary': [
            'aries', 'arietis', 'taurus', 'tauri',
            'gemini', 'geminorum', 'cancer', 'cancri',
            'leo', 'leonis', 'virgo', 'virginis',
            'libra', 'librae', 'scorpio', 'scorpionis',
            'sagittarius', 'sagittarii', 'capricornus', 'capricorni',
            'aquarius', 'aquarii', 'pisces', 'piscium',
            'sol', 'solis', 'luna', 'lunae',
            'stella', 'stellae', 'planeta', 'planetae',
            'mars', 'martis', 'venus', 'veneris',
            'mercurius', 'mercurii', 'jupiter', 'jovis',
            'saturnus', 'saturni',
            'signum', 'signi', 'gradus', 'aspectus',
            'conjunctio', 'conjunctionis',
            'eclipsis', 'nativitas', 'nativitatis',
            'mensis', 'annus', 'anni', 'dies', 'diei',
            'hora', 'horae', 'nox', 'noctis',
            'decanus', 'decani', 'influentia', 'influentiae',
        ],
        'frequency_boosts': {
            'luna': 2.0, 'sol': 2.0, 'stella': 2.0,
            'signum': 2.0, 'hora': 2.0, 'regit': 2.0,
        },
        'template_functions': ASTRO_TEMPLATES,
    },
    'cosmological': {
        'additional_vocabulary': [
            'ignis', 'aer', 'aeris', 'terra', 'terrae',
            'elementum', 'elementi',
            'oriens', 'orientis', 'occidens', 'occidentis',
            'meridies', 'meridiei', 'septentrio', 'septentrionis',
            'mundus', 'mundi', 'caelum', 'caeli',
            'orbis', 'sphaera', 'sphaerae',
            'centrum', 'centri', 'circulus', 'circuli',
            'motus', 'rotatio', 'rotationis',
        ],
        'frequency_boosts': {
            'elementum': 2.0, 'circulus': 2.0,
            'mundus': 2.0, 'sphaera': 2.0,
        },
        'template_functions': COSMO_TEMPLATES,
    },
    'recipes': {
        'additional_vocabulary': [
            'item', 'similiter', 'praeterea',
            'aliud', 'alterum',
            'probatum', 'expertum', 'optimum',
            'secretum', 'mirabile',
            'antidotum', 'antidoti', 'remedium', 'remedii',
            'medicamen', 'medicaminis',
            'potio', 'potionis', 'decoctio', 'decoctionis',
            'infusio', 'infusionis',
        ],
        'frequency_boosts': {
            'item': 2.0, 'aliud': 2.0, 'probatum': 2.0,
            'recipe': 2.0, 'contra': 2.0,
        },
        'template_functions': RECIPE_TEMPLATES,
    },
}
