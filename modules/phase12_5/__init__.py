"""
Phase 12.5: Adversarial Defense Suite
======================================
Subjects the Phase 11/12 decoding pipeline to adversarial conditions
(randomized text, wrong domains, wrong languages, ablated dictionaries)
to mathematically prove that its success on real Voynich text is genuine
and not an artifact of algorithmic overfitting.

Five adversarial tests:
  12.5.1  Unicity Distance     — scrambled/random text baseline
  12.5.2  Domain Swap          — Bible/Legal transition matrix injection
  12.5.3  Polyglot Dictionary  — Italian/Occitan dictionary substitution
  12.5.4  EVA Collapse         — re-tokenized ligature collapse
  12.5.5  Ablation Study       — function word removal + grammar trace

Phase 12.5  ·  Voynich Convergence Attack
"""

from modules.phase12_5.adv_1_unicity_distance import UnicityDistanceTest
from modules.phase12_5.adv_2_domain_swap import DomainSwapTest
from modules.phase12_5.adv_3_polyglot_dict import PolyglotDictTest
from modules.phase12_5.adv_4_eva_collapse import EvaCollapseTest
from modules.phase12_5.adv_5_ablation_study import AblationStudyTest
from modules.phase12_5.dictionary_diagnostic import DictionaryDiagnostic

__all__ = [
    'UnicityDistanceTest',
    'DomainSwapTest',
    'PolyglotDictTest',
    'EvaCollapseTest',
    'AblationStudyTest',
    'DictionaryDiagnostic',
]
