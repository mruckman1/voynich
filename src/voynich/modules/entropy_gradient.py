"""
Track 9: Intra-Paragraph Entropy Gradient
============================================
Tests whether Voynich paragraphs show the formulaic opening / variable middle /
formulaic closing pattern characteristic of medieval scientific text.

Medieval herbals, recipe collections, and zodiac texts are highly formulaic.
They open with stock phrases, elaborate in the middle, and close with stock
phrases. If encrypted versions preserve this pattern as an entropy U-curve
(low-high-low across paragraph quartiles), that's independent evidence for
the content hypothesis — and the low-entropy regions become targets for
known-plaintext matching.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import conditional_entropy, first_order_entropy
from voynich.core.voynich_corpus import get_all_tokens, SAMPLE_CORPUS, SECTIONS

class EntropyGradient:
    """
    Computes and analyzes intra-paragraph entropy gradients to detect
    formulaic text structure.
    """

    def __init__(self, min_paragraph_tokens: int = 12, verbose: bool = True):
        self.min_tokens = min_paragraph_tokens
        self.verbose = verbose

    def _extract_paragraphs(self) -> Dict[str, List[List[str]]]:
        """Extract paragraphs from corpus, grouped by section."""
        section_paragraphs: Dict[str, List[List[str]]] = defaultdict(list)

        for folio, data in SAMPLE_CORPUS.items():
            section = data.get('section', 'unknown')
            text_lines = data.get('text', [])

            current_paragraph = []
            for line in text_lines:
                tokens = line.split()
                if not tokens:
                    if current_paragraph:
                        section_paragraphs[section].append(current_paragraph)
                        current_paragraph = []
                    continue
                current_paragraph.extend(tokens)
                if len(tokens) > 15:
                    section_paragraphs[section].append(current_paragraph)
                    current_paragraph = []

            if current_paragraph:
                section_paragraphs[section].append(current_paragraph)

        return dict(section_paragraphs)

    def paragraph_quartile_entropy(
        self, paragraph_tokens: List[str]
    ) -> Optional[np.ndarray]:
        """
        Split paragraph into 4 quartiles, compute H2 per quartile.
        Returns array [Q1_H2, Q2_H2, Q3_H2, Q4_H2] or None if too short.
        """
        n = len(paragraph_tokens)
        if n < self.min_tokens:
            return None

        quarter = n // 4
        if quarter < 3:
            return None

        quartiles = [
            paragraph_tokens[:quarter],
            paragraph_tokens[quarter:2 * quarter],
            paragraph_tokens[2 * quarter:3 * quarter],
            paragraph_tokens[3 * quarter:],
        ]

        entropies = []
        for q_tokens in quartiles:
            text = ' '.join(q_tokens)
            if len(text) < 5:
                entropies.append(0.0)
            else:
                h2 = conditional_entropy(text, order=2)
                entropies.append(h2)

        return np.array(entropies)

    def corpus_entropy_gradient(
        self, section: Optional[str] = None
    ) -> Dict:
        """
        Average entropy gradient across all paragraphs in a section.
        """
        all_paragraphs = self._extract_paragraphs()

        if section:
            paragraphs = all_paragraphs.get(section, [])
        else:
            paragraphs = []
            for paras in all_paragraphs.values():
                paragraphs.extend(paras)

        gradients = []
        for para in paragraphs:
            gradient = self.paragraph_quartile_entropy(para)
            if gradient is not None:
                gradients.append(gradient)

        if not gradients:
            return {
                'mean_gradient': [0, 0, 0, 0],
                'std_gradient': [0, 0, 0, 0],
                'n_paragraphs': 0,
            }

        gradients_arr = np.array(gradients)

        return {
            'mean_gradient': gradients_arr.mean(axis=0).tolist(),
            'std_gradient': gradients_arr.std(axis=0).tolist(),
            'n_paragraphs': len(gradients),
        }

    def u_curve_test(self, gradient: List[float]) -> Dict:
        """
        Test for U-shape: Q1 < Q2 and Q4 < Q3?
        U-score = (Q2+Q3)/2 - (Q1+Q4)/2
        Positive U-score = U-shape (formulaic edges, variable middle).
        """
        if len(gradient) < 4:
            return {'u_score': 0, 'is_u_curve': False}

        q1, q2, q3, q4 = gradient[:4]

        u_score = (q2 + q3) / 2 - (q1 + q4) / 2

        rising = q1 < q2 < q3 < q4
        falling = q1 > q2 > q3 > q4

        is_u = q1 < q2 and q4 < q3
        is_inverted_u = q1 > q2 and q4 > q3

        return {
            'u_score': float(u_score),
            'is_u_curve': is_u,
            'is_inverted_u': is_inverted_u,
            'is_rising': rising,
            'is_falling': falling,
            'q1': float(q1),
            'q2': float(q2),
            'q3': float(q3),
            'q4': float(q4),
            'pattern': (
                'U-curve (formulaic edges)' if is_u else
                'Inverted U (formulaic middle)' if is_inverted_u else
                'Rising (increasingly variable)' if rising else
                'Falling (decreasingly variable)' if falling else
                'Irregular'
            ),
        }

    def reference_gradients(self) -> Dict[str, Dict]:
        """
        Expected entropy gradients for medieval text types.
        Based on the formulaic structure of medieval writing conventions.
        """
        return {
            'herbal_entry': {
                'expected_pattern': 'U-curve',
                'reason': 'Herbals open with "Plant_name est quality" and close with '
                          '"et est probatum", with variable middle description.',
                'expected_gradient': [1.8, 2.5, 2.3, 1.9],
            },
            'recipe_entry': {
                'expected_pattern': 'Falling',
                'reason': 'Recipes open with variable "Recipe X et Y" and become '
                          'more formulaic toward "et sanabitur" closing.',
                'expected_gradient': [2.4, 2.2, 2.0, 1.8],
            },
            'zodiac_medical': {
                'expected_pattern': 'U-curve',
                'reason': 'Zodiac texts open with formulaic "Sign regit Body_part" '
                          'and close with stock advice.',
                'expected_gradient': [1.7, 2.3, 2.4, 1.8],
            },
        }

    def compare_sections(self) -> Dict[str, Dict]:
        """
        Compute and compare entropy gradients across sections.
        Returns gradient shape classification per section.
        """
        all_paragraphs = self._extract_paragraphs()
        results = {}

        for section in all_paragraphs:
            gradient_data = self.corpus_entropy_gradient(section=section)
            if gradient_data['n_paragraphs'] < 2:
                continue

            u_test = self.u_curve_test(gradient_data['mean_gradient'])

            results[section] = {
                'gradient': gradient_data['mean_gradient'],
                'n_paragraphs': gradient_data['n_paragraphs'],
                'u_test': u_test,
            }

        return results

    def identify_anchor_regions(
        self, section_gradients: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Identify low-entropy quartiles that become targets for
        known-plaintext matching against formulaic openings/closings.
        """
        anchors = []

        for section, data in section_gradients.items():
            gradient = data['gradient']
            u_test = data['u_test']

            if len(gradient) < 4:
                continue

            if gradient[0] < np.mean(gradient) - 0.1:
                anchors.append({
                    'section': section,
                    'region': 'opening (Q1)',
                    'entropy': gradient[0],
                    'mean_entropy': float(np.mean(gradient)),
                    'delta': float(gradient[0] - np.mean(gradient)),
                    'suggestion': 'Formulaic opening — match against stock opening phrases '
                                  'for this text type.',
                })

            if gradient[3] < np.mean(gradient) - 0.1:
                anchors.append({
                    'section': section,
                    'region': 'closing (Q4)',
                    'entropy': gradient[3],
                    'mean_entropy': float(np.mean(gradient)),
                    'delta': float(gradient[3] - np.mean(gradient)),
                    'suggestion': 'Formulaic closing — match against stock closing phrases '
                                  'for this text type.',
                })

        return anchors

def run(verbose: bool = True) -> Dict:
    """
    Run intra-paragraph entropy gradient analysis.

    Returns:
        Dict with per-section gradients, U-curve tests, reference comparisons,
        and anchor region identifications.
    """
    analyzer = EntropyGradient(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 9: INTRA-PARAGRAPH ENTROPY GRADIENT")
        print("=" * 70)

    if verbose:
        print("\n  Computing overall entropy gradient...")
    overall = analyzer.corpus_entropy_gradient()
    overall_u = analyzer.u_curve_test(overall['mean_gradient'])
    if verbose:
        print(f"    Mean gradient: {[f'{v:.3f}' for v in overall['mean_gradient']]}")
        print(f"    U-score: {overall_u['u_score']:.4f}")
        print(f"    Pattern: {overall_u['pattern']}")

    if verbose:
        print("\n  Per-section entropy gradients:")
    section_gradients = analyzer.compare_sections()
    if verbose:
        for section, data in section_gradients.items():
            g = data['gradient']
            pattern = data['u_test']['pattern']
            u_score = data['u_test']['u_score']
            print(f"    {section}: [{', '.join(f'{v:.3f}' for v in g)}]  "
                  f"U={u_score:.3f}  {pattern}")

    reference = analyzer.reference_gradients()

    if verbose:
        print("\n  Identifying anchor regions (low-entropy quartiles)...")
    anchors = analyzer.identify_anchor_regions(section_gradients)
    if verbose:
        if anchors:
            for a in anchors:
                print(f"    {a['section']} {a['region']}: "
                      f"H2={a['entropy']:.3f} (Δ={a['delta']:.3f})")
        else:
            print("    No clear anchor regions identified.")

    u_curve_sections = [s for s, d in section_gradients.items()
                        if d['u_test']['is_u_curve']]

    results = {
        'track': 'entropy_gradient',
        'track_number': 9,
        'overall_gradient': overall['mean_gradient'],
        'overall_u_test': overall_u,
        'section_gradients': {
            s: {
                'gradient': d['gradient'],
                'n_paragraphs': d['n_paragraphs'],
                'u_score': d['u_test']['u_score'],
                'pattern': d['u_test']['pattern'],
            }
            for s, d in section_gradients.items()
        },
        'reference_gradients': reference,
        'anchor_regions': anchors,
        'n_anchor_regions': len(anchors),
        'u_curve_sections': u_curve_sections,
        'n_u_curve_sections': len(u_curve_sections),
    }

    if verbose:
        print("\n" + "─" * 70)
        print("ENTROPY GRADIENT SUMMARY")
        print("─" * 70)
        print(f"  Overall pattern: {overall_u['pattern']}")
        print(f"  Sections with U-curve: {u_curve_sections if u_curve_sections else 'none'}")
        print(f"  Anchor regions identified: {len(anchors)}")
        if u_curve_sections:
            print("  → U-curve pattern supports formulaic medieval text hypothesis")
            print("  → Low-entropy openings/closings are targets for known-plaintext attack")

    return results
