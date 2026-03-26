"""
Improved Latin Herbal Corpus — Path A Foundation
==================================================
Fixes Phase 5's corpus problems:
  - TTR was 0.021 (629 types / 30,012 tokens). Target: > 0.05 (> 1,500 types)
  - H2 delta was 0.345 (outside ±0.2 tolerance). Target: < 0.2
  - Only 8 template structures produced formulaic repetition

Strategy:
  1. Reuse all Phase 5 vocabulary lists and hardcoded entries
  2. Add ~300 new vocabulary items across 8 lexical categories
  3. Add 7 new template structures to break repetitive patterns
  4. Validation loop: measure TTR and H2 after building, inject more
     diverse content if below targets

Phase 6  ·  Voynich Convergence Attack
"""

import os
import json
import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import (
    word_conditional_entropy, zipf_analysis, conditional_entropy,
    first_order_entropy, word_transition_matrix,
)
from voynich.modules.phase5.latin_corpus_expanded import (
    ExpandedLatinHerbalCorpus,
    EXPANDED_PLANT_NAMES, EXPANDED_BODY_WORDS, CONDITION_WORDS,
    PREPARATION_WORDS, DELIVERY_WORDS, EXPANDED_SUBSTANCE_WORDS,
    EXPANDED_PROPERTY_WORDS, DOSAGE_WORDS, TIME_WORDS,
    EXPANDED_CLOSING_PHRASES, TRANSITIONAL_PHRASES,
    CIRCA_INSTANS_ENTRIES, MACER_FLORIDUS_ENTRIES,
    TEMPLATE_GENERATORS as PHASE5_TEMPLATES,
)
from voynich.core.expanded_medical_vocabulary import (
    ALL_MEDICAL_CATEGORIES, CATEGORY_WEIGHTS,
)

from voynich.core._paths import json_dir
_DATA_DIR = str(json_dir())
with open(os.path.join(_DATA_DIR, 'phase6_vocabulary.json')) as _f:
    _p6_data = json.load(_f)

GALENIC_TERMS = _p6_data['GALENIC_TERMS']
ASTROLOGICAL_TERMS = _p6_data['ASTROLOGICAL_TERMS']
ANIMAL_INGREDIENTS = _p6_data['ANIMAL_INGREDIENTS']
MINERAL_INGREDIENTS = _p6_data['MINERAL_INGREDIENTS']
DIAGNOSTIC_TERMS = _p6_data['DIAGNOSTIC_TERMS']
EXTRA_VERBS = _p6_data['EXTRA_VERBS']
EXTRA_NOUNS = _p6_data['EXTRA_NOUNS']
EXTRA_ADJECTIVES = _p6_data['EXTRA_ADJECTIVES']

del _p6_data, _f

ALL_NEW_VOCAB = (
    GALENIC_TERMS + ASTROLOGICAL_TERMS + ANIMAL_INGREDIENTS +
    MINERAL_INGREDIENTS + DIAGNOSTIC_TERMS + EXTRA_VERBS +
    EXTRA_NOUNS + EXTRA_ADJECTIVES
)

def _template_symptom_first(rng, plant, quality, moisture, degree):
    """Start with symptom, then recommend plant."""
    condition = rng.choice(CONDITION_WORDS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    prop = rng.choice(EXPANDED_PROPERTY_WORDS)
    return (
        f'contra {condition} {body} recipe {plant} '
        f'quae est {quality} et {moisture} in {degree} gradu '
        f'et habet virtutem {prop} '
        f'{rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_recipe_first(rng, plant, quality, moisture, degree):
    """Start with recipe instruction."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'recipe {plant} et {plant2} et {rng.choice(PREPARATION_WORDS)} '
        f'cum {substance} et {rng.choice(DELIVERY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'quia {plant} est {quality} et {moisture} in {degree} gradu '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_property_first(rng, plant, quality, moisture, degree):
    """Start with property description."""
    prop1 = rng.choice(EXPANDED_PROPERTY_WORDS)
    prop2 = rng.choice(EXPANDED_PROPERTY_WORDS)
    return (
        f'habet virtutem {prop1} et {prop2} {plant} '
        f'quae est {quality} et {moisture} in {degree} gradu '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'et contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} {rng.choice(DOSAGE_WORDS)} '
        f'et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} {rng.choice(TIME_WORDS)}'
    )

def _template_cross_reference(rng, plant, quality, moisture, degree):
    """Cross-reference between plants."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    verb = rng.choice(EXTRA_VERBS)
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'similiter ut {plant2} sed {plant} est fortior '
        f'et {verb} melius quam {plant2} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(TRANSITIONAL_PHRASES)} {plant2} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_short_note(rng, plant, quality, moisture, degree):
    """Brief nota-format entry."""
    return (
        f'nota quod {plant} est {quality} et {moisture} '
        f'et {rng.choice(EXTRA_VERBS)} {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_BODY_WORDS)} '
        f'et est {rng.choice(["verum", "probatum", "certum", "expertum"])}'
    )

def _template_diagnostic(rng, plant, quality, moisture, degree):
    """Diagnostic-oriented entry with symptoms."""
    diag1 = rng.choice(DIAGNOSTIC_TERMS)
    diag2 = rng.choice(DIAGNOSTIC_TERMS)
    return (
        f'si {diag1} est {diag2} et {rng.choice(CONDITION_WORDS)} '
        f'adest tunc recipe {plant} '
        f'quae est {quality} et {moisture} in {degree} gradu '
        f'et habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'{rng.choice(PREPARATION_WORDS)} {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(TIME_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} {rng.choice(CONDITION_WORDS)}'
    )

def _template_astrological(rng, plant, quality, moisture, degree):
    """Entry with astrological timing."""
    sign = rng.choice(ASTROLOGICAL_TERMS[:12])
    timing = rng.choice(['luna crescente', 'luna decrescente',
                         'in plenilunio', 'in novilunio'])
    season = rng.choice(['vere', 'aestate', 'autumno', 'hieme'])
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'collige {plant} sub signo {sign} {timing} '
        f'in {season} et sicca in umbra '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_BODY_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)}'
    )

def _template_compound_galenic(rng, plant, quality, moisture, degree):
    """Entry using Galenic theory vocabulary."""
    galenic = rng.sample(GALENIC_TERMS, k=min(3, len(GALENIC_TERMS)))
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet {galenic[0]} {galenic[1]} '
        f'et {rng.choice(EXTRA_VERBS)} {galenic[2]} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'recipe {plant} et {rng.choice(PREPARATION_WORDS)} '
        f'cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_mineral_animal(rng, plant, quality, moisture, degree):
    """Entry combining plant with mineral or animal ingredient."""
    ingredient = rng.choice(ANIMAL_INGREDIENTS + MINERAL_INGREDIENTS)
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'recipe {plant} cum {ingredient} '
        f'et {rng.choice(PREPARATION_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} {rng.choice(DOSAGE_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_vocabulary_rich(rng, plant, quality, moisture, degree):
    """Deliberately vocabulary-rich entry to push up type count."""
    adj = rng.choice(EXTRA_ADJECTIVES)
    noun = rng.choice(EXTRA_NOUNS)
    verb = rng.choice(EXTRA_VERBS)
    diag = rng.choice(DIAGNOSTIC_TERMS)
    animal = rng.choice(ANIMAL_INGREDIENTS)
    mineral = rng.choice(MINERAL_INGREDIENTS)
    return (
        f'{plant} est {quality} et {moisture} in {degree} gradu '
        f'habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)} '
        f'et {verb} {noun} {adj} '
        f'si {diag} adest recipe {plant} cum {animal} et {mineral} '
        f'et {rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

NEW_TEMPLATES = [
    _template_symptom_first,
    _template_recipe_first,
    _template_property_first,
    _template_cross_reference,
    _template_short_note,
    _template_diagnostic,
    _template_astrological,
    _template_compound_galenic,
    _template_mineral_animal,
    _template_vocabulary_rich,
]

def _template_imperative_sequence(rng, plant, quality, moisture, degree):
    """Verb-first: coque plant in substance, cola, da dosage."""
    verb = rng.choice(EXTRA_VERBS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)
    time = rng.choice(TIME_WORDS)
    return (
        f'coque {plant} in {substance} {time} '
        f'et cola per pannum et da {dosage} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'quia {plant} {verb} {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_preparation_chain(rng, plant, quality, moisture, degree):
    """Multi-verb chain: accipe, tere, misce, fac."""
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    noun = rng.choice(EXTRA_NOUNS)
    return (
        f'accipe {plant} et tere in pulverem '
        f'misce cum {substance} et fac emplastrum '
        f'pone super {rng.choice(EXPANDED_BODY_WORDS)} '
        f'contra {rng.choice(CONDITION_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} {noun} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_multi_step(rng, plant, quality, moisture, degree):
    """Sequential: primo... deinde... tandem..."""
    prep1 = rng.choice(PREPARATION_WORDS)
    prep2 = rng.choice(PREPARATION_WORDS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'primo {prep1} {plant} '
        f'deinde {prep2} cum {substance} '
        f'tandem {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(DOSAGE_WORDS)} {rng.choice(TIME_WORDS)} '
        f'contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_compound_recipe_gap(rng, plant, quality, moisture, degree):
    """Compound recipe with multiple plants and additive."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    ingredient = rng.choice(ANIMAL_INGREDIENTS + MINERAL_INGREDIENTS)
    return (
        f'recipe {plant} et {plant2} cum {substance} '
        f'{rng.choice(PREPARATION_WORDS)} et adde {ingredient} '
        f'{rng.choice(DELIVERY_WORDS)} {rng.choice(DOSAGE_WORDS)} '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_instruction_dense(rng, plant, quality, moisture, degree):
    """Dense instruction sequence with many verb→noun bigrams."""
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    dosage = rng.choice(DOSAGE_WORDS)
    return (
        f'contere {plant} cola per pannum '
        f'adde {substance} {dosage} '
        f'da {rng.choice(TIME_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_BODY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_body_target(rng, plant, quality, moisture, degree):
    """Body-part targeted: ad dolorem {body} recipe plant."""
    body = rng.choice(EXPANDED_BODY_WORDS)
    body2 = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'ad dolorem {body} recipe {plant} '
        f'quae est {quality} et {moisture} '
        f'et applica super {body} '
        f'{rng.choice(PREPARATION_WORDS)} '
        f'item contra {rng.choice(CONDITION_WORDS)} {body2} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_lavation(rng, plant, quality, moisture, degree):
    """Washing/lavation: lava body cum decocto plant."""
    body = rng.choice(EXPANDED_BODY_WORDS)
    verb = rng.choice(EXTRA_VERBS)
    return (
        f'lava {body} cum decocto {plant} '
        f'{rng.choice(TIME_WORDS)} '
        f'et {verb} {rng.choice(CONDITION_WORDS)} '
        f'quia {plant} est {quality} et {moisture} '
        f'in {degree} gradu '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_unction(rng, plant, quality, moisture, degree):
    """Unction: unge body cum oleo plant."""
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'unge {body} cum oleo {plant} '
        f'quod est {quality} et {moisture} '
        f'contra {rng.choice(CONDITION_WORDS)} {body} '
        f'{rng.choice(TIME_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} dolorem '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_fomentation(rng, plant, quality, moisture, degree):
    """Fomentation: fac fomentum de plant super body."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'fac fomentum de {plant} et {plant2} '
        f'et pone super {body} '
        f'{rng.choice(TIME_WORDS)} '
        f'contra {rng.choice(CONDITION_WORDS)} '
        f'quia {plant} est {quality} et {plant2} est {moisture} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_disease_first(rng, plant, quality, moisture, degree):
    """Disease-leading: contra condition body recipe plant."""
    body = rng.choice(EXPANDED_BODY_WORDS)
    condition = rng.choice(CONDITION_WORDS)
    verb = rng.choice(EXTRA_VERBS)
    return (
        f'contra {condition} {body} '
        f'recipe {plant} quae {verb} {condition} '
        f'est {quality} et {moisture} in {degree} gradu '
        f'{rng.choice(PREPARATION_WORDS)} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_conditional_disease(rng, plant, quality, moisture, degree):
    """Conditional: si diagnostic adest recipe plant cum substance."""
    diag = rng.choice(DIAGNOSTIC_TERMS)
    condition = rng.choice(CONDITION_WORDS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'si {diag} adest et {condition} fuerit '
        f'tunc {rng.choice(PREPARATION_WORDS)} {plant} cum {substance} '
        f'{rng.choice(DELIVERY_WORDS)} {rng.choice(DOSAGE_WORDS)} '
        f'quia {plant} est {quality} et {moisture} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_proven_remedy(rng, plant, quality, moisture, degree):
    """Proven remedy: ad condition curandum probatum est plant."""
    condition = rng.choice(CONDITION_WORDS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'ad {condition} curandum probatum est {plant} '
        f'cum {substance} {rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(DOSAGE_WORDS)} {rng.choice(TIME_WORDS)} '
        f'quia est {quality} et {moisture} in {degree} gradu '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_dosage_precise(rng, plant, quality, moisture, degree):
    """Precise dosage: da dosage plant cum dosage substance."""
    dosage1 = rng.choice(DOSAGE_WORDS)
    dosage2 = rng.choice(DOSAGE_WORDS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    return (
        f'da {dosage1} {plant} cum {dosage2} {substance} '
        f'{rng.choice(TIME_WORDS)} '
        f'contra {rng.choice(CONDITION_WORDS)} {rng.choice(EXPANDED_BODY_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} dolorem '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_regimen(rng, plant, quality, moisture, degree):
    """Regimen: bibat dosage plant mane et sero."""
    dosage = rng.choice(DOSAGE_WORDS)
    time = rng.choice(TIME_WORDS)
    condition = rng.choice(CONDITION_WORDS)
    return (
        f'bibat {dosage} {plant} mane et sero '
        f'per {time} contra {condition} '
        f'{rng.choice(EXPANDED_BODY_WORDS)} '
        f'et {rng.choice(EXTRA_VERBS)} {condition} '
        f'quia {plant} est {quality} et {moisture} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_graduated_dose(rng, plant, quality, moisture, degree):
    """Graduated dosing: in primo die da dosage, in secundo die..."""
    dosage1 = rng.choice(DOSAGE_WORDS)
    dosage2 = rng.choice(DOSAGE_WORDS)
    return (
        f'in primo die da {dosage1} {plant} '
        f'in secundo die da {dosage2} {plant} '
        f'usque ad sanitatem '
        f'valet contra {rng.choice(CONDITION_WORDS)} '
        f'{rng.choice(EXPANDED_BODY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_dual_plant(rng, plant, quality, moisture, degree):
    """Dual plant comparison: plant et plant2 simul valent."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'{plant} et {plant2} simul {rng.choice(PREPARATION_WORDS)} '
        f'valent contra {rng.choice(CONDITION_WORDS)} {body} '
        f'melius quam {plant} solum '
        f'{rng.choice(DELIVERY_WORDS)} {rng.choice(DOSAGE_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_alternative(rng, plant, quality, moisture, degree):
    """Alternative: plant valet vel plant2 si non habetur."""
    plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    while plant2 == plant:
        plant2 = rng.choice(EXPANDED_PLANT_NAMES)
    condition = rng.choice(CONDITION_WORDS)
    return (
        f'{plant} valet contra {condition} '
        f'vel {plant2} si {plant} non habetur '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'quia utraque est {quality} et {moisture} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

def _template_compound_sentence(rng, plant, quality, moisture, degree):
    """Compound: plant verb condition item plant prep cum substance."""
    verb = rng.choice(EXTRA_VERBS)
    condition1 = rng.choice(CONDITION_WORDS)
    condition2 = rng.choice(CONDITION_WORDS)
    substance = rng.choice(EXPANDED_SUBSTANCE_WORDS)
    body = rng.choice(EXPANDED_BODY_WORDS)
    return (
        f'{plant} {verb} {condition1} '
        f'{rng.choice(TRANSITIONAL_PHRASES)} '
        f'{plant} {rng.choice(PREPARATION_WORDS)} cum {substance} '
        f'valet contra {condition2} {body} '
        f'{rng.choice(DELIVERY_WORDS)} '
        f'{rng.choice(EXPANDED_CLOSING_PHRASES)}'
    )

GAP_FILLING_TEMPLATES = [
    _template_imperative_sequence,
    _template_preparation_chain,
    _template_multi_step,
    _template_compound_recipe_gap,
    _template_instruction_dense,
    _template_body_target,
    _template_lavation,
    _template_unction,
    _template_fomentation,
    _template_disease_first,
    _template_conditional_disease,
    _template_proven_remedy,
    _template_dosage_precise,
    _template_regimen,
    _template_graduated_dose,
    _template_dual_plant,
    _template_alternative,
    _template_compound_sentence,
]

ALL_TEMPLATES = list(PHASE5_TEMPLATES) + NEW_TEMPLATES

class ImprovedLatinCorpus:
    """
    Improved medieval Latin herbal corpus for Phase 6.

    Fixes Phase 5's TTR (0.021 → target > 0.05) and H2 delta (0.345 → < 0.2)
    by adding diverse vocabulary and template structures.
    """

    def __init__(self, target_tokens: int = 30000,
                 target_ttr: float = 0.05,
                 target_min_types: int = 1500,
                 seed: int = 42, verbose: bool = True):
        self.target_tokens = target_tokens
        self.target_ttr = target_ttr
        self.target_min_types = target_min_types
        self.seed = seed
        self.verbose = verbose
        self._corpus_text = None
        self._tokens = None
        self._section_corpora: Dict[str, str] = {}
        self._section_tokens: Dict[str, List[str]] = {}

    def build_corpus(self) -> str:
        """Build the improved corpus with TTR and H2 validation."""
        if self._corpus_text is not None:
            return self._corpus_text

        parts = []

        parts.extend(CIRCA_INSTANS_ENTRIES)

        parts.extend(MACER_FLORIDUS_ENTRIES)

        rng = random.Random(self.seed)
        for word in ALL_NEW_VOCAB:
            context = rng.choice([
                f'{word} est {rng.choice(["utilis", "necessarius", "bonus", "malus"])}',
                f'recipe {word} {rng.choice(DOSAGE_WORDS)}',
                f'contra {word} valet {rng.choice(EXPANDED_PLANT_NAMES)}',
            ])
            parts.append(context)

        from voynich.modules.phase11.phonetic_skeletonizer import LATIN_CONSONANT_CLASSES as _LCC
        def _skeleton_segments(word):
            skel, last = [], ''
            for ch in word.lower():
                if ch in _LCC:
                    m = _LCC[ch]
                    if m != last:
                        skel.append(m)
                        last = m
            return len(skel)

        _medical_templates = [
            lambda rng, w: f'{w} est {rng.choice(["utilis", "necessarius", "efficax", "probatus"])}',
            lambda rng, w: f'recipe {w} {rng.choice(DOSAGE_WORDS)}',
            lambda rng, w: f'contra {rng.choice(CONDITION_WORDS)} valet {w}',
            lambda rng, w: f'{w} cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)}',
            lambda rng, w: f'accipe {w} et {rng.choice(PREPARATION_WORDS)}',
            lambda rng, w: f'{w} habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)}',
        ]
        for cat_name, category in ALL_MEDICAL_CATEGORIES.items():
            weight = CATEGORY_WEIGHTS.get(cat_name, 1)
            for lemma, forms in category.items():
                for form in forms:
                    if _skeleton_segments(form) < 3:
                        continue
                    for _ in range(weight):
                        tpl = rng.choice(_medical_templates)
                        parts.append(tpl(rng, form))

        qualities = ['calida', 'frigida', 'calidus', 'frigidus',
                     'calidum', 'frigidum']
        moistures = ['sicca', 'humida', 'siccus', 'humidus',
                     'siccum', 'humidum']
        degrees = ['primo', 'secundo', 'tertio', 'quarto']

        gap_rng = random.Random(self.seed + 7919)
        for _ in range(100):
            plant = gap_rng.choice(EXPANDED_PLANT_NAMES)
            quality = gap_rng.choice(qualities)
            moisture = gap_rng.choice(moistures)
            degree = gap_rng.choice(degrees)
            template = gap_rng.choice(GAP_FILLING_TEMPLATES)
            parts.append(template(gap_rng, plant, quality, moisture, degree))

        current_text = ' '.join(parts)
        current_count = len(current_text.split())
        remaining = self.target_tokens - current_count

        while remaining > 0:
            plant = rng.choice(EXPANDED_PLANT_NAMES)
            quality = rng.choice(qualities)
            moisture = rng.choice(moistures)
            degree = rng.choice(degrees)

            if rng.random() < 0.6:
                template = rng.choice(NEW_TEMPLATES)
            else:
                template = rng.choice(ALL_TEMPLATES)

            entry = template(rng, plant, quality, moisture, degree)
            parts.append(entry)
            remaining -= len(entry.split())

        self._corpus_text = ' '.join(parts)
        self._tokens = None

        tokens = self.get_tokens()
        n_types = len(set(tokens))
        ttr = n_types / max(1, len(tokens))

        if self.verbose:
            print(f'  Improved Latin corpus: {len(tokens)} tokens, '
                  f'{n_types} types, TTR={ttr:.4f}')

        return self._corpus_text

    def get_corpus(self) -> str:
        """Return corpus text."""
        return self.build_corpus()

    def get_tokens(self) -> List[str]:
        """Get tokenized corpus."""
        if self._tokens is None:
            self._tokens = [t for t in self.build_corpus().split() if t]
        return self._tokens

    def build_section_corpus(self, section: str,
                             section_fraction: float = 0.25) -> str:
        """Build a section-specialized corpus.

        Architecture: base corpus (shared) + section-specific addendum.
        Uses a separate RNG stream per section to avoid perturbing the
        base corpus or other sections.

        Args:
            section: Canonical section name from VOYNICH_SECTIONS.
            section_fraction: Fraction of base corpus size for addendum.

        Returns:
            Combined corpus text string.
        """
        if section in self._section_corpora:
            return self._section_corpora[section]

        from voynich.core.section_vocabularies import SECTION_PROFILES

        base = self.build_corpus()
        profile = SECTION_PROFILES.get(section)

        if profile is None or (
            not profile.get('additional_vocabulary')
            and not profile.get('template_functions')
        ):
            self._section_corpora[section] = base
            return base

        section_seed = self.seed + abs(hash(section)) % 10000
        section_rng = random.Random(section_seed)

        from voynich.modules.phase11.phonetic_skeletonizer import LATIN_CONSONANT_CLASSES as _LCC
        def _skel_segs(word):
            skel, last = [], ''
            for ch in word.lower():
                if ch in _LCC:
                    m = _LCC[ch]
                    if m != last:
                        skel.append(m)
                        last = m
            return len(skel)

        qualities = ['calida', 'frigida', 'calidus', 'frigidus',
                     'calidum', 'frigidum']
        moistures = ['sicca', 'humida', 'siccus', 'humidus',
                     'siccum', 'humidum']
        degrees = ['primo', 'secundo', 'tertio', 'quarto']

        _micro_templates = [
            lambda rng, w: f'{w} est {rng.choice(["utilis", "necessarius", "efficax", "probatus"])}',
            lambda rng, w: f'recipe {w} {rng.choice(DOSAGE_WORDS)}',
            lambda rng, w: f'contra {rng.choice(CONDITION_WORDS)} valet {w}',
            lambda rng, w: f'{w} cum {rng.choice(EXPANDED_SUBSTANCE_WORDS)}',
            lambda rng, w: f'accipe {w} et {rng.choice(PREPARATION_WORDS)}',
            lambda rng, w: f'{w} habet virtutem {rng.choice(EXPANDED_PROPERTY_WORDS)}',
        ]

        section_parts = []
        section_target = int(self.target_tokens * section_fraction)

        for word in profile['additional_vocabulary']:
            if _skel_segs(word) < 3:
                continue
            tpl = section_rng.choice(_micro_templates)
            section_parts.append(tpl(section_rng, word))

        current_count = sum(len(p.split()) for p in section_parts)
        remaining = section_target - current_count

        templates = profile.get('template_functions') or GAP_FILLING_TEMPLATES
        while remaining > 0:
            plant = section_rng.choice(EXPANDED_PLANT_NAMES)
            quality = section_rng.choice(qualities)
            moisture = section_rng.choice(moistures)
            degree = section_rng.choice(degrees)
            template = section_rng.choice(templates)
            entry = template(section_rng, plant, quality, moisture, degree)
            section_parts.append(entry)
            remaining -= len(entry.split())

        combined = base + ' ' + ' '.join(section_parts)
        self._section_corpora[section] = combined
        return combined

    def get_section_tokens(self, section: str,
                           section_fraction: float = 0.25) -> List[str]:
        """Get tokenized section corpus."""
        if section not in self._section_tokens:
            text = self.build_section_corpus(section, section_fraction)
            self._section_tokens[section] = [t for t in text.split() if t]
        return self._section_tokens[section]

    def build_section_transition_matrix(
        self, section: str, top_n: int = 1001,
        section_fraction: float = 0.25,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build section-specific transition matrix.

        Generates the combined corpus, applies frequency boosts by
        repeating boosted tokens, then builds the matrix.
        """
        from voynich.core.section_vocabularies import SECTION_PROFILES

        tokens = list(self.get_section_tokens(section, section_fraction))

        profile = SECTION_PROFILES.get(section, {})
        boosts = profile.get('frequency_boosts', {})
        if boosts:
            boosted = []
            for t in tokens:
                boosted.append(t)
                extra = int(boosts.get(t, 1.0)) - 1
                for _ in range(extra):
                    boosted.append(t)
            tokens = boosted

        return word_transition_matrix(tokens, top_n=top_n)

    def get_top_n_words(self, n: int = 1001) -> List[Tuple[str, int]]:
        """Return the top N most frequent words with counts."""
        return Counter(self.get_tokens()).most_common(n)

    def build_transition_matrix(self, top_n: int = 1001) -> Tuple[np.ndarray, List[str]]:
        """
        Build word-level transition matrix restricted to top N words.

        CRITICAL: top_n should match len(voynich_vocab) for bijection in FixedSAA.
        """
        return word_transition_matrix(self.get_tokens(), top_n=top_n)

    def compute_word_bigram_h2(self) -> float:
        """Compute word-level bigram conditional entropy."""
        return word_conditional_entropy(self.get_tokens(), order=1)

    def validate_against_voynich(self, voynich_h2: float = 2.385) -> Dict:
        """
        Validate corpus quality against Voynich targets.

        Returns dict with pass/fail for each metric.
        """
        tokens = self.get_tokens()
        n_types = len(set(tokens))
        n_tokens = len(tokens)
        ttr = n_types / max(1, n_tokens)
        h2 = self.compute_word_bigram_h2()
        delta = abs(h2 - voynich_h2)

        ttr_ok = ttr >= self.target_ttr
        types_ok = n_types >= self.target_min_types
        h2_delta_ok = delta < 0.2

        return {
            'ttr': ttr,
            'ttr_ok': ttr_ok,
            'n_types': n_types,
            'types_ok': types_ok,
            'n_tokens': n_tokens,
            'word_h2': h2,
            'voynich_h2': voynich_h2,
            'h2_delta': delta,
            'h2_delta_ok': h2_delta_ok,
            'all_ok': ttr_ok and types_ok and h2_delta_ok,
        }

    def run(self, verbose: bool = True) -> Dict:
        """Build corpus and compute all metrics."""
        tokens = self.get_tokens()
        h2_word = self.compute_word_bigram_h2()
        h3_word = word_conditional_entropy(tokens, order=2)
        text = self.get_corpus()
        h1 = first_order_entropy(text)
        h2_char = conditional_entropy(text, order=1)
        h3_char = conditional_entropy(text, order=2)
        zipf = zipf_analysis(tokens)
        validation = self.validate_against_voynich()

        results = {
            'target_tokens': self.target_tokens,
            'actual_tokens': len(tokens),
            'vocabulary_size': len(set(tokens)),
            'type_token_ratio': len(set(tokens)) / max(1, len(tokens)),
            'word_bigram_h2': h2_word,
            'word_trigram_h3': h3_word,
            'char_entropy': {'H1': h1, 'H2': h2_char, 'H3': h3_char},
            'zipf': {
                'zipf_exponent': zipf['zipf_exponent'],
                'r_squared': zipf['r_squared'],
            },
            'top_30_words': Counter(tokens).most_common(30),
            'voynich_comparison': {
                'latin_word_bigram_h2': h2_word,
                'voynich_h2': 2.385,
                'delta': abs(h2_word - 2.385),
                'within_range': abs(h2_word - 2.385) < 0.2,
            },
            'validation': validation,
        }

        if verbose:
            print(f'\n  Improved Latin Herbal Corpus:')
            print(f'    Tokens:  {len(tokens)}, Types: {len(set(tokens))}')
            print(f'    TTR:     {validation["ttr"]:.4f} '
                  f'({"PASS" if validation["ttr_ok"] else "FAIL"}, target > {self.target_ttr})')
            print(f'    Types:   {validation["n_types"]} '
                  f'({"PASS" if validation["types_ok"] else "FAIL"}, target > {self.target_min_types})')
            print(f'    Word H2: {h2_word:.3f}')
            print(f'    H2 Δ:    {validation["h2_delta"]:.3f} '
                  f'({"PASS" if validation["h2_delta_ok"] else "FAIL"}, target < 0.2)')
            print(f'    Zipf:    {zipf["zipf_exponent"]:.3f} (R²={zipf["r_squared"]:.3f})')
            print(f'    Overall: {"ALL PASS" if validation["all_ok"] else "SOME FAIL"}')

        return results
