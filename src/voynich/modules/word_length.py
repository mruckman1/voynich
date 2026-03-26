"""
Track 4: Word-Length Distribution Analysis
============================================
Determines whether word boundaries in the Voynich carry semantic information
or are cipher artifacts. If word boundaries are artificial chunking points,
we should strip them and work at the character-stream level. If they're
meaningful, word-level statistics are legitimate features.

Key methods: KS tests, word-boundary information content, cross-section
comparison of length distributions.
"""

import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
from scipy import stats

from voynich.core.stats import (
    compute_all_entropy, first_order_entropy, conditional_entropy,
    word_positional_entropy
)
from voynich.core.voynich_corpus import (
    get_all_tokens, get_section_text, SAMPLE_CORPUS, SECTIONS
)
from voynich.core.medieval_text_templates import (
    generate_italian_text, generate_german_text
)
from voynich.modules.strategy1_parameter_search import generate_medical_plaintext

class WordLengthAnalyzer:
    """Analyzes word-length distributions to validate word boundary semantics."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def voynich_word_lengths(
        self, section: Optional[str] = None, language: Optional[str] = None
    ) -> np.ndarray:
        """Extract word-length distribution from Voynich corpus."""
        if section:
            text = get_section_text(section)
            tokens = text.split() if text else []
        else:
            tokens = get_all_tokens()

        if language:
            lang_tokens = []
            for folio, data in SAMPLE_CORPUS.items():
                if data.get('lang') == language:
                    for line in data.get('text', []):
                        lang_tokens.extend(line.split())
            tokens = lang_tokens if lang_tokens else tokens

        lengths = np.array([len(t) for t in tokens if t])
        return lengths

    def natural_language_word_lengths(self, language: str, n_words: int = 2000) -> np.ndarray:
        """Get word-length distribution from synthetic plaintext."""
        if language == 'latin':
            text = generate_medical_plaintext(n_words=n_words)
        elif language == 'italian':
            text = generate_italian_text(n_words=n_words)
        elif language == 'german':
            text = generate_german_text(n_words=n_words)
        else:
            raise ValueError(f"Unknown language: {language}")

        tokens = text.split()
        return np.array([len(t) for t in tokens if t])

    def ks_test(self, dist_a: np.ndarray, dist_b: np.ndarray) -> Tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test.
        Returns (KS statistic, p-value).
        Low p-value → distributions differ significantly.
        """
        if len(dist_a) < 2 or len(dist_b) < 2:
            return (1.0, 0.0)
        stat, pval = stats.ks_2samp(dist_a, dist_b)
        return (float(stat), float(pval))

    def word_boundary_information_test(self, tokens: List[str]) -> Dict:
        """
        Test whether word boundaries carry information by comparing entropy
        with and without boundaries.

        Method:
        1. Compute H2 of tokens joined with spaces (word-aware)
        2. Compute H2 of tokens concatenated without spaces (character stream)
        3. If H2_with_spaces << H2_without, boundaries add predictability → meaningful
        4. If H2_with_spaces ≈ H2_without, boundaries add little → possibly artificial
        """
        text_with_spaces = ' '.join(tokens)
        entropy_with = compute_all_entropy(text_with_spaces)

        text_without = ''.join(tokens)
        entropy_without = compute_all_entropy(text_without)

        delta_h2 = entropy_with['H2'] - entropy_without['H2']
        delta_h3 = entropy_with['H3'] - entropy_without['H3']

        boundary_informative = delta_h2 < -0.2

        return {
            'H2_with_boundaries': entropy_with['H2'],
            'H2_without_boundaries': entropy_without['H2'],
            'H3_with_boundaries': entropy_with['H3'],
            'H3_without_boundaries': entropy_without['H3'],
            'delta_H2': delta_h2,
            'delta_H3': delta_h3,
            'boundary_informative': boundary_informative,
            'interpretation': (
                'Word boundaries CARRY semantic information (reduce entropy by '
                f'{abs(delta_h2):.4f} bits). Word-level analysis is valid.'
                if boundary_informative else
                'Word boundaries add MINIMAL information. Consider character-stream '
                'analysis as primary approach.'
            ),
        }

    def compare_against_languages(self) -> Dict:
        """
        Compare Voynich word-length distribution against natural languages.
        Uses KS test to find best-matching language.
        """
        voynich_lengths = self.voynich_word_lengths()

        results = {}
        for lang in ['latin', 'italian', 'german']:
            lang_lengths = self.natural_language_word_lengths(lang)

            ks_stat, ks_p = self.ks_test(voynich_lengths, lang_lengths)

            results[lang] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'voynich_mean': float(np.mean(voynich_lengths)),
                'voynich_std': float(np.std(voynich_lengths)),
                'language_mean': float(np.mean(lang_lengths)),
                'language_std': float(np.std(lang_lengths)),
                'mean_difference': float(np.mean(voynich_lengths) - np.mean(lang_lengths)),
                'compatible': ks_p > 0.05,
            }

        ranked = sorted(results.items(), key=lambda x: x[1]['ks_statistic'])
        results['ranking'] = [lang for lang, _ in ranked]
        results['best_match'] = ranked[0][0] if ranked else None

        return results

    def compare_sections(self) -> Dict:
        """
        Compare word-length distributions across Voynich sections.
        Tests whether different sections have significantly different
        word-length profiles (which would suggest different content types).
        """
        section_lengths = {}
        for section in SECTIONS:
            lengths = self.voynich_word_lengths(section=section)
            if len(lengths) > 10:
                section_lengths[section] = lengths

        pairwise = {}
        sections = list(section_lengths.keys())
        for i, s1 in enumerate(sections):
            for s2 in sections[i + 1:]:
                ks_stat, ks_p = self.ks_test(section_lengths[s1], section_lengths[s2])
                key = f"{s1}_vs_{s2}"
                pairwise[key] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'significantly_different': ks_p < 0.05,
                }

        section_stats = {}
        for section, lengths in section_lengths.items():
            hist = Counter(lengths.tolist())
            section_stats[section] = {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'median': float(np.median(lengths)),
                'mode': int(hist.most_common(1)[0][0]) if hist else 0,
                'n_tokens': len(lengths),
                'length_histogram': {int(k): int(v) for k, v in
                                     sorted(hist.items())},
            }

        n_different = sum(1 for v in pairwise.values()
                         if v['significantly_different'])

        return {
            'section_stats': section_stats,
            'pairwise_tests': pairwise,
            'n_significantly_different': n_different,
            'total_pairs': len(pairwise),
            'interpretation': (
                f'{n_different}/{len(pairwise)} section pairs have significantly '
                f'different word-length distributions. '
                + ('This supports different content types per section.'
                   if n_different > len(pairwise) * 0.5 else
                   'Word-length distribution is relatively uniform across sections.')
            ),
        }

    def length_distribution_histogram(self) -> Dict:
        """Compute overall word-length histogram with statistics."""
        lengths = self.voynich_word_lengths()
        hist = Counter(lengths.tolist())

        return {
            'histogram': {int(k): int(v) for k, v in sorted(hist.items())},
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'median': float(np.median(lengths)),
            'skewness': float(stats.skew(lengths)) if len(lengths) > 2 else 0.0,
            'kurtosis': float(stats.kurtosis(lengths)) if len(lengths) > 2 else 0.0,
            'total_tokens': len(lengths),
        }

def run(verbose: bool = True) -> Dict:
    """
    Run the word-length distribution analysis.

    Returns:
        Dict with word boundary test, language comparisons, section comparisons,
        and overall histogram.
    """
    analyzer = WordLengthAnalyzer(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 4: WORD-LENGTH DISTRIBUTION ANALYSIS")
        print("=" * 70)

    if verbose:
        print("\n  Test 1: Word boundary information content...")
    tokens = get_all_tokens()
    boundary_test = analyzer.word_boundary_information_test(tokens)

    if verbose:
        print(f"    H2 with boundaries:    {boundary_test['H2_with_boundaries']:.4f}")
        print(f"    H2 without boundaries: {boundary_test['H2_without_boundaries']:.4f}")
        print(f"    ΔH2 = {boundary_test['delta_H2']:.4f}")
        print(f"    → {boundary_test['interpretation']}")

    if verbose:
        print("\n  Test 2: Language comparison (KS tests)...")
    language_comparison = analyzer.compare_against_languages()

    if verbose:
        for lang in ['latin', 'italian', 'german']:
            data = language_comparison.get(lang, {})
            compat = "COMPATIBLE" if data.get('compatible') else "INCOMPATIBLE"
            print(f"    {lang}: KS={data.get('ks_statistic', 0):.4f}  "
                  f"p={data.get('ks_p_value', 0):.4f}  [{compat}]  "
                  f"(Voynich μ={data.get('voynich_mean', 0):.2f} vs "
                  f"{lang} μ={data.get('language_mean', 0):.2f})")
        print(f"    Best match: {language_comparison.get('best_match', '?')}")

    if verbose:
        print("\n  Test 3: Cross-section comparison...")
    section_comparison = analyzer.compare_sections()

    if verbose:
        print(f"    {section_comparison['interpretation']}")
        for section, stats_data in section_comparison.get('section_stats', {}).items():
            print(f"    {section}: μ={stats_data['mean']:.2f}  "
                  f"σ={stats_data['std']:.2f}  n={stats_data['n_tokens']}")

    histogram = analyzer.length_distribution_histogram()

    results = {
        'track': 'word_length_analysis',
        'track_number': 4,
        'word_boundary_test': boundary_test,
        'word_boundary_valid': boundary_test['boundary_informative'],
        'language_comparison': language_comparison,
        'best_matching_language': language_comparison.get('best_match'),
        'section_comparison': section_comparison,
        'histogram': histogram,
    }

    if verbose:
        print("\n" + "─" * 70)
        print("WORD-LENGTH SUMMARY")
        print("─" * 70)
        valid = "YES" if results['word_boundary_valid'] else "NO"
        print(f"  Word boundaries carry semantic information: {valid}")
        print(f"  Best-matching language word-length: {results['best_matching_language']}")
        print(f"  Mean word length: {histogram['mean']:.2f} ± {histogram['std']:.2f}")

    return results
