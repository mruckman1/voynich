"""
Medieval Text Templates and Reference Statistics
==================================================
Reference paragraph structures, formulaic patterns, and expected word counts
for medieval Latin text types relevant to the Voynich Manuscript content.

Sources: Circa Instans, Macer Floridus, Trotula, Balneis Puteolanis,
         Hartlieb's Kräuterbuch, Secretum Secretorum, Tacuinum Sanitatis.
"""

# ============================================================================
# PARAGRAPH LENGTH STATISTICS BY TEXT TYPE
# ============================================================================

# Expected word counts per entry/paragraph in medieval text types
PARAGRAPH_STATS = {
    'herbal_entry': {
        'description': 'Single plant entry in a Latin herbal (Circa Instans style)',
        'mean_words': 55,
        'std_words': 18,
        'min_words': 20,
        'max_words': 120,
        'structure': 'name + quality + properties + preparations + indications',
        'examples': ['Circa Instans', 'Macer Floridus', 'Dioscorides Latin'],
    },
    'recipe_entry': {
        'description': 'Single recipe in a medical recipe collection',
        'mean_words': 30,
        'std_words': 12,
        'min_words': 10,
        'max_words': 60,
        'structure': 'Recipe + ingredients + preparation + application',
        'examples': ['Antidotarium Nicolai', 'Trotula recipes'],
    },
    'zodiac_medical': {
        'description': 'Zodiac-body part medical text (iatromathematics)',
        'mean_words': 45,
        'std_words': 15,
        'min_words': 20,
        'max_words': 80,
        'structure': 'sign + body_part + conditions + bloodletting_advice',
        'examples': ['Secretum Secretorum zodiac', 'Guild-book zodiac medicine'],
    },
    'balneological': {
        'description': 'Bath/spa description (balneological treatise)',
        'mean_words': 50,
        'std_words': 20,
        'min_words': 15,
        'max_words': 100,
        'structure': 'bath_name + water_quality + conditions_treated + regimen',
        'examples': ['De Balneis Puteolanis', 'Petrus de Ebulo'],
    },
    'anatomical': {
        'description': 'Anatomical description with humoral theory',
        'mean_words': 60,
        'std_words': 20,
        'min_words': 25,
        'max_words': 120,
        'structure': 'body_part + complexion + function + diseases + treatments',
        'examples': ['Anatomia porci', 'Mundinus anatomy'],
    },
    'cosmological': {
        'description': 'Cosmological/astronomical text with medical implications',
        'mean_words': 40,
        'std_words': 15,
        'min_words': 15,
        'max_words': 80,
        'structure': 'celestial_body + influence + season + medical_advice',
        'examples': ['Tacuinum Sanitatis', 'Secretum Secretorum'],
    },
}

# Mapping from Voynich sections to expected text types
SECTION_TEXT_TYPE_MAP = {
    'herbal_a': 'herbal_entry',
    'herbal_b': 'herbal_entry',
    'pharmaceutical': 'recipe_entry',
    'recipes': 'recipe_entry',
    'astronomical': 'zodiac_medical',
    'biological': 'balneological',
    'cosmological': 'cosmological',
}


# ============================================================================
# FORMULAIC OPENING AND CLOSING PATTERNS
# ============================================================================

# Common opening formulas in medieval medical texts
# These are candidates for known-plaintext matching at paragraph starts
OPENING_FORMULAS = {
    'herbal_entry': [
        '{plant_name} est {quality} in {degree} gradu',
        '{plant_name} habet virtutem {property}',
        'De {plant_name}. {plant_name} est herba {quality}',
        '{plant_name} nascitur in locis {habitat}',
        'Nota quod {plant_name} est {quality}',
    ],
    'recipe_entry': [
        'Recipe {ingredient} et {ingredient}',
        'Accipe {ingredient} {quantity}',
        'Ad {condition} recipe {ingredient}',
        'Contra {condition} fac sic',
        'Item ad {condition}',
    ],
    'zodiac_medical': [
        '{sign} regit {body_part}',
        'Quando sol est in {sign}',
        'In signo {sign} bonum est {action}',
        '{sign} est signum {quality}',
        'Sub {sign} nascuntur {type}',
    ],
    'balneological': [
        'Balneum {name} est {quality}',
        'Aqua {name} valet contra {condition}',
        'De balneo {name}',
        'Hoc balneum est {quality} et {quality}',
    ],
}

# Common closing formulas
CLOSING_FORMULAS = {
    'herbal_entry': [
        'et est probatum',
        'et hoc est verum',
        'et sanabitur',
        'et curabitur deo volente',
    ],
    'recipe_entry': [
        'et sanabitur',
        'et est probatum',
        'et curabitur',
        'fiat ut supra',
        'et cetera',
    ],
    'zodiac_medical': [
        'et cetera',
        'et hoc per totum mensem',
        'cave tibi',
        'et sic de aliis',
    ],
}


# ============================================================================
# ITALIAN AND GERMAN MEDICAL VOCABULARY
# ============================================================================

# Northern Italian (Venetian/Paduan) medical vocabulary for plaintext generation
ITALIAN_MEDICAL_VOCAB = {
    'high_freq': [
        'herba', 'radice', 'foglia', 'fiore', 'seme', 'acqua', 'olio',
        'polvere', 'calda', 'fredda', 'secca', 'umida', 'corpo', 'sangue',
        'medicina', 'rimedio', 'virtude', 'natura', 'male', 'dolore',
        'testa', 'ventre', 'petto', 'stomaco', 'fegato', 'rene',
        'prendi', 'mescola', 'cuoci', 'bevi', 'metti', 'lava',
        'mattina', 'sera', 'notte', 'giorno', 'luna', 'sole',
    ],
    'medium_freq': [
        'artemisia', 'malva', 'ruta', 'salvia', 'menta', 'camomilla',
        'rosmarino', 'lavanda', 'assenzio', 'borragine', 'piantaggine',
        'ferita', 'febbre', 'tosse', 'veleno', 'peste', 'apostema',
        'unguento', 'sciroppo', 'decotto', 'impiastro', 'suffumigio',
        'caldo', 'freddo', 'secco', 'umido', 'temperato',
        'primo', 'secondo', 'terzo', 'quarto', 'grado',
    ],
    'formulas': [
        'prendi {erba} e mescola con {ingrediente}',
        'questa erba e {qualita}',
        'contra {condizione} prendi {erba}',
        'la virtu de questa herba e {proprieta}',
        'fa bollire {erba} in acqua',
    ],
}

# Middle High German medical vocabulary
GERMAN_MEDICAL_VOCAB = {
    'high_freq': [
        'krut', 'wurzel', 'blat', 'blume', 'sam', 'wasser', 'oel',
        'pulver', 'warm', 'kalt', 'trucken', 'fucht', 'lip', 'blut',
        'arzenie', 'helfe', 'kraft', 'natur', 'siechtum', 'smerze',
        'houbet', 'buch', 'brust', 'mage', 'leber', 'niere',
        'nim', 'mische', 'siude', 'trinke', 'lege', 'wasche',
        'morgen', 'abent', 'naht', 'tac', 'mane', 'sunne',
    ],
    'medium_freq': [
        'byfuoz', 'pappeln', 'raute', 'salbei', 'minze', 'kamillen',
        'rosmarin', 'lavendel', 'wermut', 'borretsch', 'wegerich',
        'wunde', 'fieber', 'huoste', 'gift', 'pestilenz', 'geswulst',
        'salbe', 'latwerge', 'getranc', 'pflaster', 'rouch',
        'heiz', 'kalt', 'durre', 'feuht', 'getempert',
        'erste', 'ander', 'dritte', 'vierde', 'grat',
    ],
    'formulas': [
        'nim {krut} und mische mit {zutat}',
        'diz krut ist {eigenschaft}',
        'wider {krankheit} nim {krut}',
        'die kraft von disem krut ist {eigenschaft}',
        'siude {krut} in wasser',
    ],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text_type_for_section(section: str) -> dict:
    """Get the expected text type statistics for a Voynich section."""
    text_type = SECTION_TEXT_TYPE_MAP.get(section, 'herbal_entry')
    return PARAGRAPH_STATS[text_type]


def get_opening_formulas(text_type: str) -> list:
    """Get formulaic opening patterns for a text type."""
    return OPENING_FORMULAS.get(text_type, [])


def get_closing_formulas(text_type: str) -> list:
    """Get formulaic closing patterns for a text type."""
    return CLOSING_FORMULAS.get(text_type, [])


def generate_italian_text(n_words: int = 300, seed: int = 42) -> str:
    """Generate synthetic Italian medical text for null distribution testing."""
    import random
    rng = random.Random(seed)

    vocab = ITALIAN_MEDICAL_VOCAB
    high = vocab['high_freq']
    med = vocab['medium_freq']

    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))

    return ' '.join(words)


def generate_german_text(n_words: int = 300, seed: int = 42) -> str:
    """Generate synthetic Middle High German medical text for null distribution testing."""
    import random
    rng = random.Random(seed)

    vocab = GERMAN_MEDICAL_VOCAB
    high = vocab['high_freq']
    med = vocab['medium_freq']

    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))

    return ' '.join(words)
