"""
Boundary Analyzer — Path C Step 2
====================================
Analyzes character bigrams across word boundaries to test whether
Voynich "words" are truly independent tokens or whether strong
cross-boundary dependencies suggest the word boundaries are artificial.

If MI(boundary) > MI(within-word), the word boundaries may not
correspond to meaningful linguistic units.

Phase 6  ·  Voynich Convergence Attack
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter


class BoundaryAnalyzer:
    """
    Analyze character-level statistics at word boundaries to determine
    whether word boundaries in the Voynich manuscript are linguistically
    meaningful or potentially artificial.
    """

    def __init__(self, extractor: LanguageAExtractor,
                 splitter: TierSplitter):
        self.extractor = extractor
        self.splitter = splitter

    def extract_boundary_bigrams(self) -> List[Tuple[str, str]]:
        """
        Extract (last_char_of_word_N, first_char_of_word_N+1) pairs
        for all adjacent word pairs in the Tier 1 corpus.
        """
        tokens = self.splitter.get_tier1_tokens()
        bigrams = []
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 and w2:
                bigrams.append((w1[-1], w2[0]))
        return bigrams

    def extract_within_word_bigrams(self) -> List[Tuple[str, str]]:
        """
        Extract all (char_N, char_N+1) pairs within words.
        """
        tokens = self.splitter.get_tier1_tokens()
        bigrams = []
        for word in tokens:
            for i in range(len(word) - 1):
                bigrams.append((word[i], word[i + 1]))
        return bigrams

    def _compute_conditional_entropy(self, bigrams: List[Tuple[str, str]]) -> float:
        """
        Compute H(char2 | char1) from a list of (char1, char2) bigrams.
        """
        joint = Counter(bigrams)
        context = Counter(c1 for c1, _ in bigrams)

        total = sum(joint.values())
        if total == 0:
            return 0.0

        h = 0.0
        for (c1, c2), count in joint.items():
            p_joint = count / total
            p_context = context[c1] / total
            if p_joint > 0 and p_context > 0:
                p_cond = count / context[c1]
                h -= p_joint * math.log2(p_cond)
        return h

    def _compute_mutual_information(self, bigrams: List[Tuple[str, str]]) -> float:
        """
        Compute MI(char1, char2) from bigram pairs.
        MI = H(char1) + H(char2) - H(char1, char2)
        """
        if not bigrams:
            return 0.0

        joint = Counter(bigrams)
        c1_counts = Counter(c1 for c1, _ in bigrams)
        c2_counts = Counter(c2 for _, c2 in bigrams)
        total = sum(joint.values())

        if total == 0:
            return 0.0

        # H(c1)
        h_c1 = -sum((c / total) * math.log2(c / total)
                     for c in c1_counts.values() if c > 0)
        # H(c2)
        h_c2 = -sum((c / total) * math.log2(c / total)
                     for c in c2_counts.values() if c > 0)
        # H(c1, c2)
        h_joint = -sum((c / total) * math.log2(c / total)
                       for c in joint.values() if c > 0)

        return h_c1 + h_c2 - h_joint

    def compute_boundary_entropy(self) -> Dict:
        """
        Compare conditional entropy across vs within word boundaries.

        H(first_char | last_char) across boundaries
        H(char_N+1 | char_N) within words
        """
        boundary_bigrams = self.extract_boundary_bigrams()
        within_bigrams = self.extract_within_word_bigrams()

        boundary_h = self._compute_conditional_entropy(boundary_bigrams)
        within_h = self._compute_conditional_entropy(within_bigrams)

        return {
            'boundary_h': boundary_h,
            'within_h': within_h,
            'ratio': boundary_h / max(within_h, 0.001),
            'n_boundary_bigrams': len(boundary_bigrams),
            'n_within_bigrams': len(within_bigrams),
            'interpretation': (
                f'Boundary H={boundary_h:.3f}, Within H={within_h:.3f}. '
                f'{"Boundary entropy LOWER than within-word — word boundaries may be artificial" if boundary_h < within_h else "Boundary entropy HIGHER — word boundaries appear linguistically meaningful"}'
            ),
        }

    def compute_mutual_information(self) -> Dict:
        """
        Compare MI across vs within word boundaries.

        High boundary MI suggests characters on either side of word
        boundaries are statistically dependent — the boundaries may
        not be linguistically real.
        """
        boundary_bigrams = self.extract_boundary_bigrams()
        within_bigrams = self.extract_within_word_bigrams()

        boundary_mi = self._compute_mutual_information(boundary_bigrams)
        within_mi = self._compute_mutual_information(within_bigrams)

        return {
            'boundary_mi': boundary_mi,
            'within_mi': within_mi,
            'ratio': boundary_mi / max(within_mi, 0.001),
            'interpretation': (
                f'Boundary MI={boundary_mi:.3f}, Within MI={within_mi:.3f}. '
                f'Ratio={boundary_mi/max(within_mi, 0.001):.3f}. '
                f'{"HIGH boundary MI — word boundaries may be artificial" if boundary_mi > within_mi * 0.8 else "Low boundary MI — word boundaries appear real"}'
            ),
        }

    def find_strong_boundary_bigrams(self, threshold: float = 0.3) -> List[Dict]:
        """
        Find cross-boundary character bigrams with high conditional
        probability P(first_char | last_char) > threshold.

        These are specific character pairs that "want" to be together
        across word boundaries, suggesting the boundary is artificial.
        """
        bigrams = self.extract_boundary_bigrams()
        joint = Counter(bigrams)
        context = Counter(c1 for c1, _ in bigrams)

        strong = []
        for (c1, c2), count in joint.items():
            p_cond = count / max(context[c1], 1)
            if p_cond >= threshold:
                strong.append({
                    'last_char': c1,
                    'first_char': c2,
                    'p_conditional': round(p_cond, 4),
                    'count': count,
                    'context_count': context[c1],
                })

        strong.sort(key=lambda x: -x['p_conditional'])
        return strong

    def run(self, verbose: bool = True) -> Dict:
        """Run full boundary analysis."""
        entropy = self.compute_boundary_entropy()
        mi = self.compute_mutual_information()
        strong_bigrams = self.find_strong_boundary_bigrams()

        # Interpretation
        boundary_h_lower = entropy['boundary_h'] < entropy['within_h']
        boundary_mi_high = mi['boundary_mi'] > mi['within_mi'] * 0.8
        n_strong = len(strong_bigrams)

        if boundary_h_lower and boundary_mi_high:
            interpretation = (
                'Word boundaries show LOWER entropy and HIGH mutual information '
                'compared to within-word positions. This suggests word boundaries '
                'may be artificial — characters across boundaries are more predictable '
                'than characters within words.'
            )
            boundaries_artificial = True
        elif boundary_mi_high:
            interpretation = (
                'Word boundaries show HIGH mutual information, suggesting some '
                'cross-boundary character dependencies. Boundaries may be partially '
                'artificial or reflect strong phonotactic constraints.'
            )
            boundaries_artificial = False
        else:
            interpretation = (
                'Word boundaries show HIGHER entropy and LOWER mutual information '
                'than within-word positions. This is consistent with natural language '
                'word boundaries being linguistically meaningful.'
            )
            boundaries_artificial = False

        results = {
            'entropy_comparison': entropy,
            'mutual_information': mi,
            'strong_boundary_bigrams': strong_bigrams[:20],
            'n_strong_bigrams': n_strong,
            'boundaries_artificial': boundaries_artificial,
            'interpretation': interpretation,
            'synthesis': {
                'boundary_h': entropy['boundary_h'],
                'within_h': entropy['within_h'],
                'boundary_mi': mi['boundary_mi'],
                'within_mi': mi['within_mi'],
                'n_strong_bigrams': n_strong,
                'boundaries_artificial': boundaries_artificial,
                'conclusion': interpretation,
            },
        }

        if verbose:
            print(f'\n  Boundary Analyzer:')
            print(f'    --- Conditional Entropy ---')
            print(f'    Across boundaries: {entropy["boundary_h"]:.3f}')
            print(f'    Within words:      {entropy["within_h"]:.3f}')
            print(f'    Ratio:             {entropy["ratio"]:.3f}')
            print(f'    --- Mutual Information ---')
            print(f'    Across boundaries: {mi["boundary_mi"]:.3f}')
            print(f'    Within words:      {mi["within_mi"]:.3f}')
            print(f'    Ratio:             {mi["ratio"]:.3f}')
            print(f'    Strong boundary bigrams: {n_strong}')
            if strong_bigrams:
                print(f'    --- Top Boundary Bigrams ---')
                for b in strong_bigrams[:5]:
                    print(f'      {b["last_char"]}→{b["first_char"]}: '
                          f'P={b["p_conditional"]:.3f} (n={b["count"]})')
            print(f'    Interpretation: {interpretation}')

        return results
