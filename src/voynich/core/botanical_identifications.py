"""
Botanical Identifications Database
====================================
Published plant identifications for Voynich Manuscript herbal illustrations.
Sources: Tucker & Talbert (2013), Bax (2014), Sherwood & Sherwood (2008),
         O'Neill (1944), Brumbaugh (1976), and consensus candidates.
"""

import json
import os

from voynich.core._paths import data_dir, json_dir
_DIR = str(data_dir())
with open(os.path.join(_DIR, 'json', 'botanical_identifications.json')) as _f:
    _data = json.load(_f)

HUMORAL_QUALITIES = _data['HUMORAL_QUALITIES']
PLANT_IDS = _data['PLANT_IDS']
PLANT_PART_TERMS = _data['PLANT_PART_TERMS']
HUMORAL_LABEL_TERMS = _data['HUMORAL_LABEL_TERMS']
DEGREE_TERMS = _data['DEGREE_TERMS']

del _data, _f

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
