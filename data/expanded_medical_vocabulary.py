"""
Expanded medieval medical Latin vocabulary.

Sources: Dioscorides (De Materia Medica), Galen (Constantine the African),
Circa Instans (Platearius), Macer Floridus (De Viribus Herbarum),
Antidotarium Nicolai.

Each category maps lemma -> list of inflected surface forms.
"""

import json
import os

_DIR = os.path.dirname(__file__)
with open(os.path.join(_DIR, 'json', 'medical_vocabulary.json')) as _f:
    _data = json.load(_f)

MEDICAL_PLANT_NAMES = _data['MEDICAL_PLANT_NAMES']
MEDICAL_ANATOMICAL_TERMS = _data['MEDICAL_ANATOMICAL_TERMS']
MEDICAL_PHARMACEUTICAL_TERMS = _data['MEDICAL_PHARMACEUTICAL_TERMS']
MEDICAL_DISEASE_TERMS = _data['MEDICAL_DISEASE_TERMS']
MEDICAL_DOSAGE_TERMS = _data['MEDICAL_DOSAGE_TERMS']
MEDICAL_PROCESS_VERBS = _data['MEDICAL_PROCESS_VERBS']
CATEGORY_WEIGHTS = _data['CATEGORY_WEIGHTS']

ALL_MEDICAL_CATEGORIES = {
    'plant_names': MEDICAL_PLANT_NAMES,
    'anatomical_terms': MEDICAL_ANATOMICAL_TERMS,
    'pharmaceutical_terms': MEDICAL_PHARMACEUTICAL_TERMS,
    'disease_terms': MEDICAL_DISEASE_TERMS,
    'dosage_terms': MEDICAL_DOSAGE_TERMS,
    'process_verbs': MEDICAL_PROCESS_VERBS,
}

del _data, _f
