"""
Botanical Identifications Database
====================================
Published plant identifications for Voynich Manuscript herbal illustrations.
Sources: Tucker & Talbert (2013), Bax (2014), Sherwood & Sherwood (2008),
         O'Neill (1944), Brumbaugh (1976), and consensus candidates.

Each entry maps a folio ID to:
  - candidates: list of proposed plant species
  - humoral: Galenic humoral quality (hot/cold, wet/dry)
  - properties: medicinal properties in medieval tradition
  - source: primary identification source
  - confidence: HIGH/MODERATE/LOW based on scholarly consensus
"""

# Humoral quality categories used in medieval medicine
HUMORAL_QUALITIES = {
    'hot_dry': {'latin': 'calidus et siccus', 'element': 'ignis', 'humor': 'cholera'},
    'hot_wet': {'latin': 'calidus et humidus', 'element': 'aer', 'humor': 'sanguis'},
    'cold_dry': {'latin': 'frigidus et siccus', 'element': 'terra', 'humor': 'melancholia'},
    'cold_wet': {'latin': 'frigidus et humidus', 'element': 'aqua', 'humor': 'phlegma'},
}

# Published plant identifications mapped to folios
# These are the more widely accepted or discussed identifications
PLANT_IDS = {
    # ---- Herbal A (Quires 1-7, Hand 1, Language A) ----
    'f1v': {
        'candidates': ['Centaurea cyanus', 'Centaurea scabiosa'],
        'common': 'cornflower / knapweed',
        'humoral': 'cold_dry',
        'properties': ['vulnerary', 'anti-inflammatory', 'eye wash'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f2r': {
        'candidates': ['Capsicum annuum'],
        'common': 'pepper',
        'humoral': 'hot_dry',
        'properties': ['stimulant', 'digestive', 'counter-irritant'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f2v': {
        'candidates': ['Helleborus niger'],
        'common': 'Christmas rose / black hellebore',
        'humoral': 'hot_dry',
        'properties': ['purgative', 'emmenagogue', 'vermifuge'],
        'confidence': 'MODERATE',
        'source': 'multiple',
    },
    'f3r': {
        'candidates': ['Nymphaea alba', 'Nymphaea lotus'],
        'common': 'water lily',
        'humoral': 'cold_wet',
        'properties': ['sedative', 'anaphrodisiac', 'anti-inflammatory'],
        'confidence': 'MODERATE',
        'source': 'ONeill',
    },
    'f4r': {
        'candidates': ['Viola odorata'],
        'common': 'sweet violet',
        'humoral': 'cold_wet',
        'properties': ['expectorant', 'laxative', 'anti-inflammatory'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f4v': {
        'candidates': ['Coriandrum sativum'],
        'common': 'coriander',
        'humoral': 'cold_dry',
        'properties': ['carminative', 'digestive', 'anti-spasmodic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f5r': {
        'candidates': ['Borage officinalis'],
        'common': 'borage',
        'humoral': 'hot_wet',
        'properties': ['sudorific', 'diuretic', 'cordial'],
        'confidence': 'LOW',
        'source': 'Sherwood',
    },
    'f6r': {
        'candidates': ['Aristolochia clematitis'],
        'common': 'birthwort',
        'humoral': 'hot_dry',
        'properties': ['emmenagogue', 'vulnerary', 'anti-venom'],
        'confidence': 'MODERATE',
        'source': 'multiple',
    },
    'f6v': {
        'candidates': ['Chelidonium majus'],
        'common': 'greater celandine',
        'humoral': 'hot_dry',
        'properties': ['cholagogue', 'eye treatment', 'wart removal'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f9v': {
        'candidates': ['Calendula officinalis'],
        'common': 'marigold',
        'humoral': 'hot_dry',
        'properties': ['vulnerary', 'anti-inflammatory', 'emmenagogue'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f13r': {
        'candidates': ['Dracunculus vulgaris'],
        'common': 'dragon arum',
        'humoral': 'hot_dry',
        'properties': ['expectorant', 'anti-venom', 'emmenagogue'],
        'confidence': 'MODERATE',
        'source': 'multiple',
    },
    'f15v': {
        'candidates': ['Plantago major'],
        'common': 'plantain',
        'humoral': 'cold_dry',
        'properties': ['vulnerary', 'anti-inflammatory', 'astringent'],
        'confidence': 'LOW',
        'source': 'Sherwood',
    },
    'f17r': {
        'candidates': ['Ricinus communis'],
        'common': 'castor oil plant',
        'humoral': 'hot_wet',
        'properties': ['purgative', 'emollient', 'anti-inflammatory'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f22r': {
        'candidates': ['Nigella sativa'],
        'common': 'black cumin',
        'humoral': 'hot_dry',
        'properties': ['carminative', 'diuretic', 'galactagogue'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f25v': {
        'candidates': ['Aconitum napellus'],
        'common': 'monkshood / wolfsbane',
        'humoral': 'cold_dry',
        'properties': ['analgesic', 'anti-pyretic', 'poison'],
        'confidence': 'LOW',
        'source': 'Brumbaugh',
    },
    'f33v': {
        'candidates': ['Helianthus annuus'],
        'common': 'sunflower',
        'humoral': 'hot_dry',
        'properties': ['diuretic', 'expectorant', 'anti-pyretic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
        'note': 'Controversial - sunflower is New World, predates MS if correct dating',
    },
    'f34r': {
        'candidates': ['Salvia officinalis'],
        'common': 'sage',
        'humoral': 'hot_dry',
        'properties': ['astringent', 'anti-septic', 'digestive'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f41v': {
        'candidates': ['Artemisia vulgaris'],
        'common': 'mugwort',
        'humoral': 'hot_dry',
        'properties': ['emmenagogue', 'digestive', 'anti-spasmodic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f49v': {
        'candidates': ['Papaver somniferum'],
        'common': 'opium poppy',
        'humoral': 'cold_wet',
        'properties': ['analgesic', 'soporific', 'anti-tussive'],
        'confidence': 'LOW',
        'source': 'Brumbaugh',
    },
    'f56r': {
        'candidates': ['Rosmarinus officinalis'],
        'common': 'rosemary',
        'humoral': 'hot_dry',
        'properties': ['stimulant', 'carminative', 'anti-spasmodic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },

    # ---- Herbal B (Quire 17, Hand 2, Language B) ----
    'f87r': {
        'candidates': ['Malva sylvestris'],
        'common': 'common mallow',
        'humoral': 'cold_wet',
        'properties': ['emollient', 'laxative', 'anti-inflammatory'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f89r1': {
        'candidates': ['Verbascum thapsus'],
        'common': 'great mullein',
        'humoral': 'hot_dry',
        'properties': ['expectorant', 'demulcent', 'vulnerary'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f90r1': {
        'candidates': ['Achillea millefolium'],
        'common': 'yarrow',
        'humoral': 'hot_dry',
        'properties': ['vulnerary', 'styptic', 'anti-pyretic'],
        'confidence': 'LOW',
        'source': 'Sherwood',
    },
    'f93r': {
        'candidates': ['Hypericum perforatum'],
        'common': "St. John's wort",
        'humoral': 'hot_dry',
        'properties': ['vulnerary', 'anti-depressant', 'anti-inflammatory'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f94r': {
        'candidates': ['Mentha pulegium'],
        'common': 'pennyroyal',
        'humoral': 'hot_dry',
        'properties': ['emmenagogue', 'carminative', 'anti-spasmodic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f96v': {
        'candidates': ['Ruta graveolens'],
        'common': 'rue',
        'humoral': 'hot_dry',
        'properties': ['emmenagogue', 'anti-spasmodic', 'anti-venom'],
        'confidence': 'MODERATE',
        'source': 'multiple',
    },

    # ---- Pharmaceutical section plants (Quires 19-20) ----
    'f99r': {
        'candidates': ['Matricaria chamomilla'],
        'common': 'chamomile',
        'humoral': 'hot_dry',
        'properties': ['anti-spasmodic', 'anti-inflammatory', 'sedative'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
    'f100r': {
        'candidates': ['Lavandula angustifolia'],
        'common': 'lavender',
        'humoral': 'hot_dry',
        'properties': ['sedative', 'anti-spasmodic', 'anti-septic'],
        'confidence': 'LOW',
        'source': 'Tucker_Talbert',
    },
}

# Common plant-part terms in medieval Latin herbals
# These are expected label candidates
PLANT_PART_TERMS = {
    'radix': 'root',
    'folia': 'leaves',
    'flos': 'flower',
    'semen': 'seed',
    'cortex': 'bark',
    'herba': 'herb (whole plant)',
    'succus': 'juice',
    'pulvis': 'powder',
    'oleum': 'oil',
    'aqua': 'water/distillate',
}

# Humoral quality terms expected in labels or short descriptions
HUMORAL_LABEL_TERMS = {
    'calidus': 'hot',
    'frigidus': 'cold',
    'siccus': 'dry',
    'humidus': 'wet/moist',
    'temperatus': 'temperate',
}

# Degree system (1st-4th degree hot/cold/dry/wet)
DEGREE_TERMS = {
    'in primo gradu': '1st degree',
    'in secundo gradu': '2nd degree',
    'in tertio gradu': '3rd degree',
    'in quarto gradu': '4th degree',
}


def get_plants_by_humoral(quality: str):
    """Get all plants with a given humoral quality."""
    return {folio: data for folio, data in PLANT_IDS.items()
            if data.get('humoral') == quality}


def get_plants_by_section(section: str):
    """Get plants from a specific manuscript section."""
    section_folios = {
        'herbal_a': [f for f in PLANT_IDS if not f.startswith('f8') and
                     not f.startswith('f9') and not f.startswith('f10')
                     and int(''.join(c for c in f[1:] if c.isdigit()) or '0') < 57],
        'herbal_b': [f for f in PLANT_IDS if f.startswith(('f87', 'f88', 'f89',
                     'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96'))],
        'pharmaceutical': [f for f in PLANT_IDS if f.startswith(('f99', 'f100',
                           'f101', 'f102'))],
    }
    folios = section_folios.get(section, [])
    return {f: PLANT_IDS[f] for f in folios if f in PLANT_IDS}


def get_high_confidence_ids():
    """Get only MODERATE or HIGH confidence identifications."""
    return {folio: data for folio, data in PLANT_IDS.items()
            if data.get('confidence') in ('MODERATE', 'HIGH')}
