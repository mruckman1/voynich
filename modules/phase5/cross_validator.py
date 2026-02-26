"""
Cross-Validation Suite (Attack C)
====================================
The ultimate test: does the combined decryption (Tier 1 codebook + Tier 2
cipher) produce coherent Latin herbal text on full pages?

Five validation checks:
  1. Humoral consistency: hot-dry plants → "calida" + "sicca", etc.
  2. Structural consistency: decoded pages match herbal entry formula.
  3. Language B alignment: zodiac pages thematically related.
  4. Entropy gradient preservation: decoded text preserves H2 gradient.
  5. The "fachys" test: first word of f1r decodes appropriately.

Phase 5  ·  Voynich Convergence Attack
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter
from modules.statistical_analysis import conditional_entropy, word_conditional_entropy

try:
    from data.botanical_identifications import PLANT_IDS, HUMORAL_QUALITIES
except ImportError:
    PLANT_IDS = {}
    HUMORAL_QUALITIES = {}


# Humoral quality keywords expected in decoded text
HUMORAL_KEYWORDS = {
    'hot_dry': {
        'expected': {'calida', 'calidus', 'calidum', 'sicca', 'siccus', 'siccum'},
        'forbidden': {'frigida', 'frigidus', 'humida', 'humidus'},
    },
    'hot_wet': {
        'expected': {'calida', 'calidus', 'calidum', 'humida', 'humidus', 'humidum'},
        'forbidden': {'frigida', 'frigidus', 'sicca', 'siccus'},
    },
    'cold_dry': {
        'expected': {'frigida', 'frigidus', 'frigidum', 'sicca', 'siccus', 'siccum'},
        'forbidden': {'calida', 'calidus', 'humida', 'humidus'},
    },
    'cold_wet': {
        'expected': {'frigida', 'frigidus', 'frigidum', 'humida', 'humidus', 'humidum'},
        'forbidden': {'calida', 'calidus', 'sicca', 'siccus'},
    },
}

# Herbal entry structural elements (in expected order)
STRUCTURAL_ELEMENTS = [
    'plant_name',    # Plant name at the start
    'quality',       # calida/frigida + sicca/humida
    'degree',        # primo/secundo/tertio/quarto gradu
    'virtue',        # habet virtutem + property
    'indication',    # valet contra + condition
    'preparation',   # recipe + method
    'closing',       # et est probatum, etc.
]

QUALITY_MARKERS = {'calida', 'calidu', 'frigida', 'frigidus', 'calidum', 'frigidum'}
DEGREE_MARKERS = {'primo', 'secundo', 'tertio', 'quarto', 'gradu'}
VIRTUE_MARKERS = {'habet', 'virtutem'}
INDICATION_MARKERS = {'valet', 'contra'}
PREPARATION_MARKERS = {'recipe', 'accipe', 'coque', 'contere', 'misce', 'fac'}
CLOSING_MARKERS = {'probatum', 'sanabitur', 'curabitur', 'verum', 'volente'}


class CrossValidator:
    """
    Cross-validate the combined Tier 1 + Tier 2 decryption against
    multiple independent criteria.
    """

    def __init__(self, tier1_mapping: Dict[str, str],
                 tier2_mapping: Optional[Dict[str, str]],
                 extractor: LanguageAExtractor,
                 splitter: TierSplitter):
        self.tier1_mapping = tier1_mapping
        self.tier2_mapping = tier2_mapping or {}
        self.extractor = extractor
        self.splitter = splitter

    def _decode_tokens(self, tokens: List[str]) -> List[str]:
        """Decode a token sequence using both tier mappings."""
        decoded = []
        for t in tokens:
            if t in self.tier1_mapping:
                decoded.append(self.tier1_mapping[t])
            elif t in self.tier2_mapping:
                decoded.append(self.tier2_mapping[t])
            else:
                decoded.append(f'[{t}]')
        return decoded

    def _decode_page(self, tokens: List[str]) -> str:
        """Decode a page's tokens to a string."""
        return ' '.join(self._decode_tokens(tokens))

    # =========================================================
    # Check 1: Humoral Consistency
    # =========================================================

    def check_humoral_consistency(self) -> Dict:
        """
        For each page with an identified plant, check if the decoded text
        contains the expected humoral quality words and lacks the forbidden ones.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()
        page_results = []
        correct = 0
        tested = 0

        for folio, plant_data in PLANT_IDS.items():
            if folio not in by_folio:
                continue

            humoral = plant_data.get('humoral', 'unknown')
            if humoral not in HUMORAL_KEYWORDS:
                continue

            decoded_text = self._decode_page(by_folio[folio])
            decoded_words = set(decoded_text.lower().split())

            expected = HUMORAL_KEYWORDS[humoral]['expected']
            forbidden = HUMORAL_KEYWORDS[humoral]['forbidden']

            has_expected = bool(decoded_words & expected)
            has_forbidden = bool(decoded_words & forbidden)

            is_consistent = has_expected and not has_forbidden
            tested += 1
            if is_consistent:
                correct += 1

            page_results.append({
                'folio': folio,
                'plant': plant_data.get('name', 'unknown'),
                'humoral': humoral,
                'has_expected': has_expected,
                'has_forbidden': has_forbidden,
                'consistent': is_consistent,
                'expected_found': list(decoded_words & expected),
                'forbidden_found': list(decoded_words & forbidden),
            })

        consistency_rate = correct / max(1, tested)

        return {
            'n_tested': tested,
            'n_correct': correct,
            'consistency_rate': consistency_rate,
            'passes': consistency_rate >= 0.70,
            'page_results': page_results[:20],
            'conclusion': (
                f'Humoral consistency: {correct}/{tested} pages correct '
                f'({consistency_rate:.1%}). '
                f'{"PASSES" if consistency_rate >= 0.70 else "FAILS"} threshold (≥70%).'
            ),
        }

    # =========================================================
    # Check 2: Structural Consistency
    # =========================================================

    def check_structural_consistency(self) -> Dict:
        """
        Check if decoded pages follow the herbal entry formula:
        plant name → quality → degree → virtue → indication → preparation → closing.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()
        page_results = []
        n_structured = 0
        n_total = 0

        for folio, tokens in by_folio.items():
            if len(tokens) < 5:
                continue

            decoded = self._decode_tokens(tokens)
            decoded_lower = [w.lower() for w in decoded]
            decoded_set = set(decoded_lower)

            # Check for structural elements
            has_quality = bool(decoded_set & QUALITY_MARKERS)
            has_degree = bool(decoded_set & DEGREE_MARKERS)
            has_virtue = bool(decoded_set & VIRTUE_MARKERS)
            has_indication = bool(decoded_set & INDICATION_MARKERS)
            has_preparation = bool(decoded_set & PREPARATION_MARKERS)
            has_closing = bool(decoded_set & CLOSING_MARKERS)

            elements_found = sum([
                has_quality, has_degree, has_virtue,
                has_indication, has_preparation, has_closing,
            ])

            is_structured = elements_found >= 3
            n_total += 1
            if is_structured:
                n_structured += 1

            page_results.append({
                'folio': folio,
                'n_tokens': len(tokens),
                'elements_found': elements_found,
                'has_quality': has_quality,
                'has_degree': has_degree,
                'has_virtue': has_virtue,
                'has_indication': has_indication,
                'has_preparation': has_preparation,
                'has_closing': has_closing,
                'is_structured': is_structured,
            })

        structure_rate = n_structured / max(1, n_total)

        return {
            'n_pages': n_total,
            'n_structured': n_structured,
            'structure_rate': structure_rate,
            'passes': structure_rate >= 0.50,
            'page_results': page_results[:20],
            'conclusion': (
                f'Structural consistency: {n_structured}/{n_total} pages match '
                f'herbal formula ({structure_rate:.1%}). '
                f'{"PASSES" if structure_rate >= 0.50 else "FAILS"} threshold (≥50%).'
            ),
        }

    # =========================================================
    # Check 3: Language B Alignment
    # =========================================================

    def check_language_b_alignment(self) -> Dict:
        """
        On zodiac pages, check if decoded Language A is thematically
        related to the zodiacal/medical content expected there.

        Zodiac pages should contain body part references and astrological
        medical terms when decoded.
        """
        from data.voynich_corpus import ZODIAC_LABELS

        zodiac_medical_terms = {
            'caput', 'collum', 'pectus', 'venter', 'bracchium',
            'manus', 'pes', 'genu', 'femur', 'cor', 'ren',
            'hepar', 'splen', 'pulmo', 'vesica', 'matrix',
        }

        by_folio = self.extractor.extract_lang_a_by_folio()
        zodiac_results = []
        n_aligned = 0
        n_tested = 0

        for folio, label in ZODIAC_LABELS.items():
            if folio not in by_folio:
                continue

            decoded = self._decode_page(by_folio[folio])
            decoded_words = set(decoded.lower().split())

            medical_found = decoded_words & zodiac_medical_terms
            n_tested += 1
            is_aligned = len(medical_found) >= 1
            if is_aligned:
                n_aligned += 1

            zodiac_results.append({
                'folio': folio,
                'zodiac_sign': label,
                'medical_terms_found': list(medical_found),
                'aligned': is_aligned,
            })

        alignment_rate = n_aligned / max(1, n_tested)

        return {
            'n_zodiac_pages': n_tested,
            'n_aligned': n_aligned,
            'alignment_rate': alignment_rate,
            'zodiac_results': zodiac_results,
            'conclusion': (
                f'Language B alignment: {n_aligned}/{n_tested} zodiac pages '
                f'contain medical terms ({alignment_rate:.1%}).'
            ),
        }

    # =========================================================
    # Check 4: Entropy Gradient Preservation
    # =========================================================

    def check_entropy_gradient_preservation(self) -> Dict:
        """
        Verify that decoded text shows higher lexical diversity at page
        starts and lower at page ends, matching the observed H2 gradient.
        """
        by_folio = self.extractor.extract_lang_a_by_folio()

        q1_decoded = []
        q4_decoded = []

        for folio, tokens in by_folio.items():
            n = len(tokens)
            if n < 4:
                continue

            decoded = self._decode_tokens(tokens)
            q_size = max(1, n // 4)

            q1_decoded.extend(decoded[:q_size])
            q4_decoded.extend(decoded[-q_size:])

        if not q1_decoded or not q4_decoded:
            return {
                'error': 'Insufficient data for gradient analysis',
                'passes': False,
            }

        # Compute word-level entropy for each quartile
        q1_h2 = word_conditional_entropy(q1_decoded, order=1) if len(q1_decoded) > 2 else 0
        q4_h2 = word_conditional_entropy(q4_decoded, order=1) if len(q4_decoded) > 2 else 0

        # Vocabulary diversity
        q1_ttr = len(set(q1_decoded)) / max(1, len(q1_decoded))
        q4_ttr = len(set(q4_decoded)) / max(1, len(q4_decoded))

        gradient = q1_h2 - q4_h2
        ttr_gradient = q1_ttr - q4_ttr

        # Gradient should be positive (Q1 more diverse than Q4)
        gradient_preserved = gradient > 0

        return {
            'q1_word_h2': q1_h2,
            'q4_word_h2': q4_h2,
            'gradient': gradient,
            'q1_ttr': q1_ttr,
            'q4_ttr': q4_ttr,
            'ttr_gradient': ttr_gradient,
            'gradient_preserved': gradient_preserved,
            'passes': gradient_preserved,
            'n_q1_tokens': len(q1_decoded),
            'n_q4_tokens': len(q4_decoded),
            'conclusion': (
                f'Entropy gradient: Q1 H2={q1_h2:.3f}, Q4 H2={q4_h2:.3f}, '
                f'gradient={gradient:.3f}. '
                f'{"PRESERVED" if gradient_preserved else "NOT PRESERVED"}.'
            ),
        }

    # =========================================================
    # Check 5: The "fachys" Test
    # =========================================================

    def check_fachys_test(self) -> Dict:
        """
        "fachys" appears only as the first word of folio f1r.
        If it decodes to a Latin plant name or opening formula
        ("incipit", "de" + plant name), the page-header structure
        is confirmed.
        """
        fachys_decoded = self.tier1_mapping.get('fachys',
                         self.tier2_mapping.get('fachys', '[fachys]'))

        # Check if it's a plant name
        from modules.phase5.latin_corpus_expanded import EXPANDED_PLANT_NAMES
        is_plant_name = fachys_decoded.lower() in {p.lower() for p in EXPANDED_PLANT_NAMES}

        # Check if it's an opening formula
        opening_words = {'incipit', 'de', 'liber', 'herba', 'capitulum'}
        is_opening = fachys_decoded.lower() in opening_words

        # Check if it's still unmapped
        is_unmapped = fachys_decoded.startswith('[')

        plausible = is_plant_name or is_opening

        return {
            'fachys_decoded_to': fachys_decoded,
            'is_plant_name': is_plant_name,
            'is_opening_formula': is_opening,
            'is_unmapped': is_unmapped,
            'plausible': plausible,
            'conclusion': (
                f'"fachys" → "{fachys_decoded}". '
                f'{"Plant name" if is_plant_name else ""}'
                f'{"Opening formula" if is_opening else ""}'
                f'{"Unmapped" if is_unmapped else ""}'
                f'{" — PLAUSIBLE" if plausible else " — inconclusive"}.'
            ),
        }

    # =========================================================
    # Overall Score
    # =========================================================

    def compute_overall_score(self) -> Dict:
        """
        Compute overall cross-validation score from all five checks.

        Pass thresholds:
        - Humoral consistency: ≥70%
        - Structural consistency: ≥50%
        - Entropy gradient: preserved
        - Overall: ≥3/5 checks pass
        """
        humoral = self.check_humoral_consistency()
        structural = self.check_structural_consistency()
        lang_b = self.check_language_b_alignment()
        gradient = self.check_entropy_gradient_preservation()
        fachys = self.check_fachys_test()

        checks = {
            'humoral': humoral.get('passes', False),
            'structural': structural.get('passes', False),
            'language_b': lang_b.get('alignment_rate', 0) > 0.3,
            'gradient': gradient.get('passes', False),
            'fachys': fachys.get('plausible', False),
        }

        n_passed = sum(checks.values())
        overall_pass = n_passed >= 3

        return {
            'checks': checks,
            'n_passed': n_passed,
            'n_total': 5,
            'overall_pass': overall_pass,
            'conclusion': (
                f'Cross-validation: {n_passed}/5 checks passed. '
                f'{"OVERALL PASS" if overall_pass else "OVERALL FAIL"}.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all cross-validation checks."""
        humoral = self.check_humoral_consistency()
        structural = self.check_structural_consistency()
        lang_b = self.check_language_b_alignment()
        gradient = self.check_entropy_gradient_preservation()
        fachys = self.check_fachys_test()
        overall = self.compute_overall_score()

        results = {
            'humoral_consistency': humoral,
            'structural_consistency': structural,
            'language_b_alignment': lang_b,
            'entropy_gradient': gradient,
            'fachys_test': fachys,
            'overall': overall,
        }

        if verbose:
            print(f'\n  Cross-Validation (Attack C):')
            print(f'    1. Humoral:    {humoral["conclusion"]}')
            print(f'    2. Structural: {structural["conclusion"]}')
            print(f'    3. Language B: {lang_b["conclusion"]}')
            print(f'    4. Gradient:   {gradient["conclusion"]}')
            print(f'    5. Fachys:     {fachys["conclusion"]}')
            print(f'    --- Overall ---')
            print(f'    {overall["conclusion"]}')

        return results
