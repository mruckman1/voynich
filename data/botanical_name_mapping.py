"""
Botanical Name Mapping: Scientific → Medieval Latin → Folio
============================================================
Bridges two data sources:
  - data/Voynich_Botanicals.csv: scientific_name → medieval Latin names
  - data/botanical_identifications.py: folio_id → scientific names (PLANT_IDS)

Provides:
  - load_botanical_csv() → {scientific_name: [latin_name, ...]}
  - build_folio_name_map() → {folio_id: {species, latin_names, ...}}
"""

import csv
import os
from typing import Dict, List, Set

_CSV_PATH = os.path.join(os.path.dirname(__file__), 'Voynich_Botanicals.csv')

MIN_STEM_LENGTH = 4

def load_botanical_csv(csv_path: str = _CSV_PATH) -> Dict[str, List[str]]:
    """Load Voynich_Botanicals.csv → {scientific_name: [medieval_latin_name, ...]}.

    Empty medieval Latin lists indicate unmappable (New World) species.
    """
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sci_name = row['Scientific Name'].strip()
            raw_names = row.get('Medieval Latin Names', '').strip()
            if not raw_names:
                mapping[sci_name] = []
                continue
            names = []
            for name in raw_names.split(','):
                name = name.strip().lower()
                if '(' in name:
                    name = name[:name.index('(')].strip()
                if name:
                    names.append(name)
            mapping[sci_name] = names
    return mapping

def build_folio_name_map() -> Dict[str, Dict]:
    """Build folio-indexed lookup: folio_id → botanical name data.

    Returns:
        {folio_id: {
            'species': [scientific names from PLANT_IDS],
            'common': common name string,
            'source': identification source,
            'confidence': HIGH/MODERATE/LOW,
            'latin_names': [all medieval Latin names, deduplicated],
            'single_word_names': [single-word names],
            'multi_word_names': [names with spaces],
            'stems': {stem_string: original_name},
            'testable': bool (False for New World / no mapping),
            'new_world': bool,
        }}
    """
    from data.botanical_identifications import PLANT_IDS

    csv_data = load_botanical_csv()
    result = {}

    for folio, plant_data in PLANT_IDS.items():
        entry = {
            'species': plant_data['candidates'],
            'common': plant_data.get('common', ''),
            'source': plant_data.get('source', ''),
            'confidence': plant_data.get('confidence', 'LOW'),
            'latin_names': [],
            'single_word_names': [],
            'multi_word_names': [],
            'stems': {},
            'testable': False,
            'new_world': False,
        }

        all_names: List[str] = []
        any_mapped = False

        for species in plant_data['candidates']:
            if species in csv_data:
                names = csv_data[species]
                if names:
                    all_names.extend(names)
                    any_mapped = True
                else:
                    entry['new_world'] = True

        seen: Set[str] = set()
        for name in all_names:
            name_lower = name.lower().strip()
            if name_lower in seen:
                continue
            seen.add(name_lower)
            entry['latin_names'].append(name_lower)

            if ' ' in name_lower:
                entry['multi_word_names'].append(name_lower)
            else:
                entry['single_word_names'].append(name_lower)
                _add_stems(entry['stems'], name_lower)

        entry['testable'] = any_mapped and len(entry['latin_names']) > 0
        result[folio] = entry

    return result

def _add_stems(stems_dict: Dict[str, str], name: str) -> None:
    """Add truncated stems for a single-word name.

    Strips 1-3 trailing characters to catch common Latin inflections:
    ruta → rut (catches rutae, rutam, rutarum)
    papaver → papave, papav (catches papaveris)
    """
    for strip_len in range(1, 4):
        stem = name[:-strip_len] if strip_len < len(name) else ''
        if len(stem) >= MIN_STEM_LENGTH:
            if stem not in stems_dict:
                stems_dict[stem] = name
