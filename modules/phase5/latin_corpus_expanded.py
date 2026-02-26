"""
Expanded Latin Herbal Corpus Builder (20,000–50,000 tokens)
=============================================================
Provides a much larger medieval Latin herbal reference corpus for
Phase 5's constrained SAA. Phase 4's corpus was only ~1,604 tokens
with 238 types — too small to build reliable word-bigram statistics.

Sources:
  1. Hardcoded Circa Instans excerpts (from Phase 4)
  2. Additional entries modeled on Macer Floridus, Herbarius Latinus,
     and Pseudo-Apuleius Herbarius
  3. Expanded synthetic generation with 8-10 template structures,
     80+ plant names, 30+ body parts, 20+ preparation methods

Target: 20,000–50,000 tokens to stabilize bigram statistics for a
~1,000-word transition matrix.

Phase 5  ·  Voynich Convergence Attack
"""

import sys
import os
import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.statistical_analysis import (
    word_conditional_entropy, zipf_analysis, full_statistical_profile,
)
from modules.phase4.latin_herbal_corpus import (
    CIRCA_INSTANS_ENTRIES,
    LATIN_FUNCTION_WORDS, LATIN_QUALITY_WORDS, LATIN_DEGREE_WORDS,
    LATIN_PROPERTY_WORDS, LATIN_BODY_WORDS, LATIN_SUBSTANCE_WORDS,
    LATIN_PLANT_NAMES, LATIN_CLOSING_PHRASES,
)


# ============================================================================
# EXPANDED VOCABULARY
# ============================================================================

# Additional plant names (beyond Phase 4's 30)
EXPANDED_PLANT_NAMES = LATIN_PLANT_NAMES + [
    # Common medieval herbal plants
    'absinthium', 'agrimonia', 'allium', 'aloe', 'ambrosia',
    'anethum', 'angelica', 'apium', 'asarum', 'basilicum',
    'betonica', 'bryonia', 'buglossa', 'cannabis', 'capparis',
    'cardamomum', 'centaurea', 'chelidonium', 'cicuta', 'cinnamomum',
    'consolida', 'crocus', 'cuminum', 'cyclamen', 'daucus',
    'ebulus', 'eupatorium', 'euphrasia', 'feniculum', 'filipendula',
    'galanga', 'gentiana', 'grana', 'hedera', 'hermodactylus',
    'hyssopus', 'imperatoria', 'iris', 'juniperus', 'lactuca',
    'lilium', 'lupinus', 'mandragora', 'marrubium', 'melissa',
    'millefolium', 'morus', 'nasturtium', 'origanum', 'paeonia',
    'peucedanum', 'pimpinella', 'polypodium', 'portulaca', 'pyrethrum',
    'raphanus', 'rhabarbarum', 'sambucus', 'satureia', 'scabiosa',
    'sempervivum', 'senecio', 'serpyllum', 'solanum', 'stachys',
    'succisa', 'symphytum', 'thymus', 'tormentilla', 'urtica',
    'valeriana', 'verbena', 'veronica', 'vinca', 'zingiber',
]

# Expanded body parts and conditions
EXPANDED_BODY_WORDS = LATIN_BODY_WORDS + [
    'hepatis', 'splenis', 'pulmonum', 'vesicae', 'nervorum',
    'ossium', 'cutis', 'oris', 'linguae', 'aurium',
    'nasi', 'gutturis', 'colli', 'dorsi', 'lumborum',
    'pedum', 'manuum', 'digitorum', 'genu', 'brachii',
    'membri', 'corporis', 'oculi', 'frontis', 'temporum',
]

# Expanded conditions/diseases
CONDITION_WORDS = [
    'dolorem', 'inflammationem', 'fluxum', 'tumorem', 'ulcera',
    'apostema', 'pustulas', 'scabiem', 'lepram', 'paralysim',
    'epilepsiam', 'podagram', 'colicam', 'dysenteria', 'hydropisis',
    'tussim', 'asthma', 'febrem', 'pestem', 'venenum',
    'morsus', 'vulnera', 'fracturas', 'luxationes', 'vermes',
    'calculis', 'stranguriam', 'ictericiam', 'melancholiam',
    'insomnia', 'tremorem', 'convulsiones', 'putrefactionem',
    'sanguinis', 'menstruorum', 'partum', 'lactis', 'sterilitatem',
]

# Expanded preparation methods
PREPARATION_WORDS = [
    'contere', 'coque in aqua', 'misce cum vino', 'fac emplastrum',
    'destilla per alembicum', 'tere in pulverem', 'dissolve in aqua',
    'infunde in oleo', 'decoque cum melle', 'fac unguentum',
    'exprime succum', 'sicca in umbra', 'macera in aceto',
    'combure et fac cinerem', 'fac cataplasma', 'cola per pannum',
    'fac pilulas', 'fac electuarium', 'fac syrupum', 'fac collyrium',
]

# Expanded delivery/administration
DELIVERY_WORDS = [
    'et da bibere', 'et pone super locum', 'et unge', 'et lava',
    'et bibe', 'et insuffla in nares', 'et fac gargarismum',
    'et instilla in aures', 'et applica super oculos',
    'et fac fomentum', 'et fac balneum', 'et liga super vulnus',
    'et da in potu', 'et commisce cum cibo', 'et fac suffumigium',
]

# Expanded substances and vehicles
EXPANDED_SUBSTANCE_WORDS = LATIN_SUBSTANCE_WORDS + [
    'aceto', 'acetum', 'butyrum', 'cera', 'lardo',
    'pice', 'resina', 'sapone', 'sale', 'sevo',
    'succo', 'lacte', 'aqua rosarum', 'aqua plantaginis',
    'vino albo', 'vino rubeo', 'melle rosato', 'oleo olivarum',
    'oleo amygdalarum', 'adipe anserino',
]

# Expanded properties/virtues
EXPANDED_PROPERTY_WORDS = LATIN_PROPERTY_WORDS + [
    'aperitivam', 'astringentem', 'attractivam', 'calefacientem',
    'consolidativam', 'consumptivam', 'corroborativam', 'dessicativam',
    'dissolutivam', 'expulsivam', 'incisivam', 'lenitivam',
    'maturativam', 'mitigativam', 'mollificativam', 'mundificativam',
    'nutritivam', 'penetrativam', 'provocativam', 'resolutivam',
    'restrictivam', 'roborativam', 'sedativam', 'stupefactivam',
    'subtiliativam', 'sudorificam', 'suppurativam',
]

# Dosage and measurement terms
DOSAGE_WORDS = [
    'uncia', 'drachma', 'scrupulum', 'manipulum', 'pugillum',
    'cochlear', 'cyathum', 'libra', 'quantum sufficit',
    'in parva dosi', 'in magna dosi', 'mane et sero',
    'ter in die', 'bis in die', 'semel in die',
    'ante cibum', 'post cibum', 'ieiunus',
]

# Time and frequency terms
TIME_WORDS = [
    'mane', 'sero', 'nocte', 'diebus', 'horis',
    'per tres dies', 'per septem dies', 'per mensem',
    'quotidie', 'alternis diebus', 'usque ad sanitatem',
]

# Expanded closing phrases
EXPANDED_CLOSING_PHRASES = LATIN_CLOSING_PHRASES + [
    'et est medicina probata', 'et sanabitur deo volente',
    'et liberabitur', 'et curabitur infallibiliter',
    'et est experimentum verum', 'et est secretum magnum',
    'et hoc est certum', 'et proficiet mirabiliter',
    'et est probatissimum', 'et non fallit',
]

# Transitional phrases for longer entries
TRANSITIONAL_PHRASES = [
    'item', 'praeterea', 'similiter', 'insuper', 'quoque',
    'alio modo', 'aliter', 'vel', 'aut', 'sive',
    'nota quod', 'sciendum est quod', 'unde', 'ideo',
]


# ============================================================================
# ADDITIONAL HARDCODED ENTRIES (Macer Floridus style)
# ============================================================================

MACER_FLORIDUS_ENTRIES = [
    # Absinthium (Wormwood)
    'absinthium est calidum et siccum in secundo gradu habet virtutem '
    'aperitivam et digestivam et mundificativam valet contra vermes '
    'intestinorum et contra dolorem stomachi et contra ictericiam '
    'recipe absinthium et coque in vino et cola et da bibere ieiunus '
    'per tres dies et expellet vermes et mundificabit stomachum '
    'et est probatum',

    # Agrimonia (Agrimony)
    'agrimonia est calida et sicca in primo gradu habet virtutem '
    'aperitivam et mundificativam et consolidativam valet contra '
    'dolorem hepatis et contra obstructiones splenis et contra '
    'vulnera recentia recipe agrimoniam et contere cum vino et da '
    'bibere et pone super vulnus et consolidabitur et est probatum',

    # Aloe
    'aloe est calida et sicca in secundo gradu habet virtutem '
    'purgativam et mundificativam et consolidativam valet contra '
    'constipationem et contra putrefactionem et contra vulnera '
    'recipe aloe et dissolve in aqua rosarum et da bibere in parva '
    'dosi et purgabit ventrem sine dolore item aloe posita super '
    'vulnus mundificabit et consolidabit et est medicina probata',

    # Angelica
    'angelica est calida et sicca in tertio gradu habet virtutem '
    'contra venenum et contra pestem et sudorificam valet contra '
    'morsus serpentis et contra pestilentiam et contra ventositatem '
    'recipe angelicam et tere in pulverem et misce cum vino et bibe '
    'statim et expellet venenum per sudorem et est secretum magnum',

    # Betonica (Betony)
    'betonica est calida et sicca in secundo gradu habet virtutem '
    'aperitivam et incisivam et expectorantem valet contra dolorem '
    'capitis et contra epilepsiam et contra tussim antiquam '
    'recipe betonicam et coque in aqua cum melle et bibe mane et '
    'sero et curabit dolorem capitis et est probatissimum item '
    'betonica in pulvere data cum vino valet contra epilepsiam',

    # Buglossa (Bugloss/Borage family)
    'buglossa est calida et humida in primo gradu habet virtutem '
    'cordialem et laetificantem et mundificativam sanguinis valet '
    'contra melancholiam et contra tristitiam cordis et contra '
    'febrem recipe buglossam et coque cum vino et da bibere et '
    'laetificabit cor et purificabit sanguinem et est verum',

    # Cannabis
    'cannabis est calida et sicca in tertio gradu habet virtutem '
    'anodynam et resolutivam et maturativam valet contra dolorem '
    'aurium et contra tumores et contra dolorem nervorum recipe '
    'semen cannabis et contere et misce cum oleo et instilla in '
    'aures et sedabit dolorem item folia cannabis cocta et posita '
    'super tumorem resolvunt et maturant',

    # Centaurea (Centaury)
    'centaurea est calida et sicca in secundo gradu habet virtutem '
    'aperitivam et mundificativam et febrifugam valet contra febrem '
    'et contra vermes et contra vulnera et contra obstructiones '
    'recipe centauream et coque in aqua et da bibere contra febrem '
    'per tres dies et liberabitur item succus centaureae positus '
    'super vulnera mundificat et consolidat',

    # Chelidonium (Celandine)
    'chelidonium est calidum et siccum in tertio gradu habet virtutem '
    'mundificativam et incisivam et aperitivam valet contra '
    'ictericiam et contra obstructiones hepatis et contra maculas '
    'oculorum recipe chelidonium et exprime succum et instilla in '
    'oculos et mundificabit maculas nota quod hirundines utuntur '
    'hac herba ad curandum oculos pullorum suorum et est verum',

    # Cinnamomum (Cinnamon)
    'cinnamomum est calidum et siccum in secundo gradu habet virtutem '
    'confortativam et calefacientem et digestivam valet contra '
    'frigiditatem stomachi et contra nausiam et contra debilitatem '
    'recipe cinnamomum et tere in pulverem et misce cum melle et '
    'da post cibum et confortabit stomachum et cerebrum et est '
    'probatum',

    # Crocus (Saffron)
    'crocus est calidus et siccus in secundo gradu habet virtutem '
    'cordialem et laetificantem et aperitivam valet contra '
    'dolorem cordis et contra obstructiones hepatis et contra '
    'tristitiam recipe crocum et dissolve in aqua rosarum et da '
    'bibere et laetificabit cor sed cave ne des nimis quia in '
    'magna dosi nocet et est verum',

    # Euphrasia (Eyebright)
    'euphrasia est calida et sicca in secundo gradu habet virtutem '
    'mundificativam oculorum et confortativam visus valet contra '
    'caliginem oculorum et contra lacrimationem et contra debilitatem '
    'visus recipe euphrasiam et coque in vino albo et lava oculos '
    'quotidie et confortabit visum mirabiliter praeterea euphrasia '
    'in pulvere sumpta cum vino valet contra dolorem capitis',

    # Feniculum (Fennel)
    'feniculum est calidum et siccum in secundo gradu habet virtutem '
    'aperitivam et diureticam et carminativam valet contra '
    'ventositatem et contra dolorem renum et contra obstructiones '
    'lactis recipe feniculum et coque in aqua et da bibere mulieri '
    'et provocabit lac item succus feniculi instillatus in oculos '
    'mundat visum et est probatum',

    # Gentiana (Gentian)
    'gentiana est calida et sicca in tertio gradu habet virtutem '
    'aperitivam et digestivam et febrifugam valet contra febrem '
    'et contra vermes et contra morsus venenosos recipe gentianam '
    'et tere in pulverem et misce cum vino et da bibere et '
    'expellet febrem et vermes similiter gentiana posita super '
    'vulnus venenosum extrahit venenum',

    # Hyssopus (Hyssop)
    'hyssopus est calidus et siccus in tertio gradu habet virtutem '
    'expectorantem et mundificativam et incisivam valet contra '
    'tussim et contra asthma et contra obstructiones pectoris '
    'recipe hyssopum et coque in aqua cum melle et fac syrupum et '
    'da bibere mane et sero et mundificabit pectus et pulmones '
    'et est medicina probata',

    # Juniperus (Juniper)
    'juniperus est calidus et siccus in tertio gradu habet virtutem '
    'calefacientem et dissolutivam et diureticam valet contra '
    'dolorem nervorum et contra ventositatem et contra calculis '
    'renum recipe baccas juniperi et contere et coque in vino et '
    'da bibere et dissolvet calculum et est probatum insuper '
    'suffumigium de junipero purificat aerem contra pestem',

    # Lactuca (Lettuce)
    'lactuca est frigida et humida in secundo gradu habet virtutem '
    'refrigerandi et sedandi et soporificam valet contra ardorem '
    'stomachi et contra insomnia et contra nimiam libidinem '
    'recipe lactucam et comede ante somnum et dormies placide '
    'item semen lactucae tritum cum aqua valet contra febrem '
    'et est probatum',

    # Mandragora (Mandrake)
    'mandragora est frigida et humida in tertio gradu habet virtutem '
    'soporificam fortissimam et anodynam et resolutivam valet '
    'contra dolorem vehementem et contra insomnia et contra tumores '
    'duros recipe corticem radicis mandragorae et dissolve in vino '
    'et da in parva dosi cum magna cautela quia est periculosa '
    'sed in dosi iusta est probata contra omnem dolorem',

    # Melissa (Lemon Balm)
    'melissa est calida et sicca in primo gradu habet virtutem '
    'cordialem et laetificantem et digestivam valet contra '
    'melancholiam et contra dolorem cordis et contra ventositatem '
    'recipe melissam et coque in aqua et bibe cum melle et '
    'laetificabit cor et confortabit stomachum et est probatum '
    'praeterea melissa posita in apibus retinet eas in alveari',

    # Origanum (Oregano)
    'origanum est calidum et siccum in tertio gradu habet virtutem '
    'incisivam et expectorantem et calefacientem valet contra '
    'tussim et contra dolorem dentium et contra morsus venenosos '
    'recipe origanum et coque in aqua et fac gargarismum contra '
    'dolorem gutturis item origanum tritum cum melle positum super '
    'morsus venenosos extrahit venenum et est probatum',

    # Paeonia (Peony)
    'paeonia est calida et sicca in secundo gradu habet virtutem '
    'contra epilepsiam et contra melancholiam et aperitivam '
    'valet contra epilepsiam infantium et contra obstructiones '
    'recipe radicem paeoniae et tere in pulverem et liga in '
    'panno circa collum infantis et praecavebit ab epilepsia '
    'item paeonia in potu data valet contra melancholiam',

    # Sambucus (Elder)
    'sambucus est calidus et siccus in secundo gradu habet virtutem '
    'purgativam et resolutivam et mundificativam valet contra '
    'hydropisis et contra tumores et contra dolorem nervorum '
    'recipe corticem sambuci et coque in aqua et da bibere et '
    'purgabit aquam per urinam item flores sambuci cocti in '
    'aqua et applicati super tumores resolvunt eos',

    # Thymus (Thyme)
    'thymus est calidus et siccus in tertio gradu habet virtutem '
    'incisivam et expectorantem et carminativam valet contra '
    'tussim et contra asthma et contra vermes intestinorum '
    'recipe thymum et coque cum melle et fac electuarium et '
    'da cochlear unum mane et sero et mundificabit pectus '
    'et expellet vermes et est probatissimum',

    # Urtica (Nettle)
    'urtica est calida et sicca in secundo gradu habet virtutem '
    'provocativam et aperitivam et mundificativam valet contra '
    'fluxum sanguinis nasi et contra dolorem articulorum et contra '
    'retentionem menstruorum recipe urticam et exprime succum et '
    'da bibere et provocabit menstrua item urtica cocta et comesta '
    'valet contra dolorem pectoris',

    # Valeriana (Valerian)
    'valeriana est calida et sicca in secundo gradu habet virtutem '
    'contra epilepsiam et contra insomnia et sudorificam valet '
    'contra tremorem et contra nervorum debilitatem et contra '
    'pestilentiam recipe valerianam et tere in pulverem et da cum '
    'vino ante somnum et dormiet pacifice et confortabit nervos '
    'et est medicina probata',

    # Verbena (Vervain)
    'verbena est frigida et sicca in primo gradu habet virtutem '
    'stypticam et vulnerariam et aperitivam valet contra dolorem '
    'capitis et contra vulnera et contra obstructiones hepatis '
    'recipe verbenam et contere et pone super frontem contra '
    'dolorem capitis item verbena in vino cocta et potata aperit '
    'obstructiones hepatis et splenis et est probatum',

    # Zingiber (Ginger)
    'zingiber est calidum et siccum in tertio gradu habet virtutem '
    'calefacientem et digestivam et carminativam valet contra '
    'frigiditatem stomachi et contra nausiam et contra ventositatem '
    'recipe zingiber et tere in pulverem et misce cum melle et da '
    'post cibum et confortabit digestionem insuper zingiber cum '
    'cassia et cinnamomo confectum valet contra debilitatem corporis',
]


# ============================================================================
# TEMPLATE GENERATORS (8 distinct structures)
# ============================================================================

def _template_standard(rng, plant, quality, moisture, degree):
    """Standard Circa Instans entry: name-quality-properties-remedy-closing."""
    props = rng.sample(EXPANDED_PROPERTY_WORDS, k=min(2, len(EXPANDED_PROPERTY_WORDS)))
    bodies = rng.sample(EXPANDED_BODY_WORDS, k=min(2, len(EXPANDED_BODY_WORDS)))
    condition = rng.choice(CONDITION_WORDS)
    prep = rng.choice(PREPARATION_WORDS)
    delivery = rng.choice(DELIVERY_WORDS)
    closing = rng.choice(EXPANDED_CLOSING_PHRASES)

    parts = [
        f'{plant} est {quality} et {moisture} in {degree} gradu',
        f'habet virtutem {props[0]}',
    ]
    if len(props) > 1:
        parts.append(f'et {props[1]}')
    parts.append(f'valet contra {condition} {bodies[0]}')
    if len(bodies) > 1:
        parts.append(f'et contra {rng.choice(CONDITION_WORDS)} {bodies[1]}')
    parts.append(f'recipe {plant} et {prep}')
    parts.append(delivery)
    parts.append(closing)
    return ' '.join(parts)


def _template_multi_remedy(rng, plant, quality, moisture, degree):
    """Entry with multiple remedies separated by 'item' or 'praeterea'."""
    props = rng.sample(EXPANDED_PROPERTY_WORDS, k=min(3, len(EXPANDED_PROPERTY_WORDS)))
    condition1 = rng.choice(CONDITION_WORDS)
    condition2 = rng.choice(CONDITION_WORDS)
    body1 = rng.choice(EXPANDED_BODY_WORDS)
    body2 = rng.choice(EXPANDED_BODY_WORDS)

    parts = [
        f'{plant} est {quality} et {moisture} in {degree} gradu',
        f'habet virtutem {props[0]} et {props[1]}',
        f'valet contra {condition1} {body1} et contra {condition2} {body2}',
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)}',
        rng.choice(DELIVERY_WORDS),
        rng.choice(EXPANDED_CLOSING_PHRASES),
        rng.choice(TRANSITIONAL_PHRASES),
        f'{plant} {rng.choice(PREPARATION_WORDS)}',
        f'valet contra {rng.choice(CONDITION_WORDS)}',
        rng.choice(DELIVERY_WORDS),
        rng.choice(EXPANDED_CLOSING_PHRASES),
    ]
    return ' '.join(parts)


def _template_short_warning(rng, plant, quality, moisture, degree):
    """Short entry with danger warning (for toxic plants)."""
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'sed est periculosus recipe {plant} cum magna cautela '
        f'et in parva dosi {rng.choice(DELIVERY_WORDS)} '
        f'cave ne des {rng.choice(["gravidis", "infantibus", "senibus", "debilibus"])}'
    )


def _template_compound_preparation(rng, plant, quality, moisture, degree):
    """Entry with compound preparation involving multiple ingredients."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)

    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'et {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {plant2} ana partes aequales '
        f'et contere et misce cum {substance} '
        f'et da {dosage} {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(TIME_WORDS)} {rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )


def _template_external_application(rng, plant, quality, moisture, degree):
    """Entry focused on external application (poultice, ointment, bath)."""
    application = rng.choice([
        f'fac emplastrum cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} et pone super locum dolentem',
        f'fac unguentum cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} et unge locum',
        f'fac cataplasma et applica super {rng.choice(EXPANDED_BODY_WORDS)}',
        f'fac balneum et immitte membrum in aqua',
        f'fac fomentum et pone super locum dolentem',
    ])

    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'et {application} '
        f'{rng.choice(TIME_WORDS)} et {rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )


def _template_detailed_dosage(rng, plant, quality, moisture, degree):
    """Entry with detailed dosage instructions."""
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'et {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)} '
        f'et {rng.choice(PREPARATION_WORDS)} '
        f'et da {rng.choice(DOSAGE_WORDS)} '
        f'{rng.choice(TIME_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )


def _template_comparative(rng, plant, quality, moisture, degree):
    """Entry comparing properties to another plant."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)

    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'similiter ut {plant2} sed est fortior '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )


def _template_seasonal(rng, plant, quality, moisture, degree):
    """Entry with seasonal harvesting instructions."""
    season = rng.choice([
        'in vere collige', 'in aestate collige', 'in autumno collige',
        'ante florem collige', 'post florem collige',
        'in plenilunio collige',
    ])

    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'{season} {plant} et {rng.choice(["sicca in umbra", "sicca in sole", "conserva in loco sicco"])} '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )


TEMPLATE_GENERATORS = [
    _template_standard,
    _template_multi_remedy,
    _template_short_warning,
    _template_compound_preparation,
    _template_external_application,
    _template_detailed_dosage,
    _template_comparative,
    _template_seasonal,
]


# ============================================================================
# MAIN CLASS
# ============================================================================

class ExpandedLatinHerbalCorpus:
    """
    Provides an expanded medieval Latin herbal reference corpus for
    Phase 5's constrained SAA.

    Target: 20,000–50,000 tokens with rich template variety.
    """

    def __init__(self, target_tokens: int = 30000, seed: int = 42,
                 verbose: bool = True):
        self.target_tokens = target_tokens
        self.seed = seed
        self.verbose = verbose
        self._corpus_text = None
        self._tokens = None
        self._profile = None

    def build_corpus(self) -> str:
        """
        Build the expanded corpus from three sources:
        1. Circa Instans excerpts (hardcoded)
        2. Macer Floridus-style entries (hardcoded)
        3. Synthetic generation (to reach target_tokens)
        """
        if self._corpus_text is not None:
            return self._corpus_text

        parts = []

        # Source 1: Circa Instans (from Phase 4)
        parts.extend(CIRCA_INSTANS_ENTRIES)

        # Source 2: Macer Floridus-style entries
        parts.extend(MACER_FLORIDUS_ENTRIES)

        # Count tokens so far
        current_text = ' '.join(parts)
        current_tokens = len(current_text.split())

        # Source 3: Synthetic generation to reach target
        rng = random.Random(self.seed)
        remaining = self.target_tokens - current_tokens

        while remaining > 0:
            plant = rng.choice(EXPANDED_PLANT_NAMES)
            quality = rng.choice(['calida', 'frigida', 'calidus', 'frigidus',
                                  'calidum', 'frigidum'])
            moisture = rng.choice(['sicca', 'humida', 'siccus', 'humidus',
                                   'siccum', 'humidum'])
            degree = rng.choice(['primo', 'secundo', 'tertio', 'quarto'])

            template = rng.choice(TEMPLATE_GENERATORS)
            entry = template(rng, plant, quality, moisture, degree)
            parts.append(entry)
            remaining -= len(entry.split())

        self._corpus_text = ' '.join(parts)

        if self.verbose:
            tokens = self._corpus_text.split()
            print(f'  Expanded Latin corpus: {len(tokens)} tokens, '
                  f'{len(set(tokens))} types')

        return self._corpus_text

    def get_corpus(self) -> str:
        """Return the corpus text (alias for build_corpus)."""
        return self.build_corpus()

    def get_tokens(self) -> List[str]:
        """Get tokenized corpus."""
        if self._tokens is None:
            self._tokens = [t for t in self.build_corpus().split() if t]
        return self._tokens

    def get_top_n_words(self, n: int = 1000) -> List[Tuple[str, int]]:
        """Return the top N most frequent words with their counts."""
        return Counter(self.get_tokens()).most_common(n)

    def build_transition_matrix(self, top_n: int = 1000) -> Tuple[np.ndarray, List[str]]:
        """
        Build word-level transition matrix restricted to the top N
        most frequent words.

        Returns:
            (matrix, vocab) where matrix[i][j] = P(word_j | word_i)
        """
        tokens = self.get_tokens()
        freqs = Counter(tokens)
        vocab = [w for w, _ in freqs.most_common(top_n)]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        counts = np.zeros((n, n), dtype=float)
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 in word_to_idx and w2 in word_to_idx:
                counts[word_to_idx[w1]][word_to_idx[w2]] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = counts / row_sums

        return matrix, vocab

    def compute_word_bigram_h2(self) -> float:
        """Compute word-level bigram conditional entropy."""
        return word_conditional_entropy(self.get_tokens(), order=1)

    def compute_full_profile(self) -> Dict:
        """Compute comprehensive statistical profile."""
        if self._profile is None:
            text = self.get_corpus()
            tokens = self.get_tokens()

            self._profile = full_statistical_profile(text, 'latin_herbal_expanded')
            self._profile['word_entropy'] = {
                'H2_word': self.compute_word_bigram_h2(),
                'H3_word': word_conditional_entropy(tokens, order=2),
            }
            self._profile['word_level_zipf'] = zipf_analysis(tokens)

        return self._profile

    def run(self, verbose: bool = True) -> Dict:
        """Build corpus and compute all metrics."""
        tokens = self.get_tokens()
        h2_word = self.compute_word_bigram_h2()
        h3_word = word_conditional_entropy(tokens, order=2)
        profile = self.compute_full_profile()
        zipf = zipf_analysis(tokens)

        results = {
            'target_tokens': self.target_tokens,
            'actual_tokens': len(tokens),
            'vocabulary_size': len(set(tokens)),
            'type_token_ratio': len(set(tokens)) / max(1, len(tokens)),
            'word_bigram_h2': h2_word,
            'word_trigram_h3': h3_word,
            'char_entropy': profile['entropy'],
            'zipf': {
                'zipf_exponent': zipf['zipf_exponent'],
                'r_squared': zipf['r_squared'],
            },
            'top_30_words': Counter(tokens).most_common(30),
            'voynich_comparison': {
                'latin_word_bigram_h2': h2_word,
                'voynich_h2': 2.385,
                'delta': abs(h2_word - 2.385),
                'within_range': abs(h2_word - 2.385) < 0.3,
            },
        }

        if verbose:
            print(f'\n  Expanded Latin Herbal Corpus:')
            print(f'    Target:  {self.target_tokens} tokens')
            print(f'    Actual:  {len(tokens)} tokens, {len(set(tokens))} types')
            print(f'    TTR:     {len(set(tokens))/max(1,len(tokens)):.3f}')
            print(f'    Word H2: {h2_word:.3f}')
            print(f'    Word H3: {h3_word:.3f}')
            print(f'    Char H2: {profile["entropy"]["H2"]:.3f}')
            print(f'    Zipf:    {zipf["zipf_exponent"]:.3f}')
            print(f'    --- Voynich Comparison ---')
            print(f'    Latin word H2:   {h2_word:.3f}')
            print(f'    Voynich H2:      2.385')
            print(f'    Delta:           {abs(h2_word - 2.385):.3f}')
            print(f'    Within range:    {abs(h2_word - 2.385) < 0.3}')

        return results
