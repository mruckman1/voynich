"""
Track 6: Paragraph Structure Analysis
========================================
Estimates the plaintext-to-ciphertext length ratio by comparing Voynich
paragraph structures against known medieval text types.

If Voynich herbal paragraphs average 50 words and medieval Latin herbal
entries average 60 words, the cipher is roughly 1:1. If Voynich paragraphs
are 30 words where Latin entries are 60, the cipher is 1:2 compressing,
which constrains the mechanism.
"""

import os
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from data.voynich_corpus import get_all_tokens, SAMPLE_CORPUS, SECTIONS
from data.medieval_text_templates import (
    PARAGRAPH_STATS, SECTION_TEXT_TYPE_MAP, get_text_type_for_section
)

class ParagraphAnalyzer:
    """Analyzes paragraph structure and estimates plaintext-ciphertext ratios."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def extract_paragraphs(self) -> Dict[str, List[List[str]]]:
        """
        Group tokens into paragraphs per section.
        Uses line breaks as paragraph boundaries (each text line in SAMPLE_CORPUS
        represents a continuous text block).
        Groups consecutive short lines into paragraphs, and treats long lines
        as individual paragraphs.
        """
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

    def paragraph_statistics(
        self, paragraphs: Dict[str, List[List[str]]]
    ) -> Dict[str, Dict]:
        """Compute word count and related statistics per section."""
        stats = {}

        for section, paras in paragraphs.items():
            if not paras:
                continue

            word_counts = [len(p) for p in paras]
            unique_counts = [len(set(p)) for p in paras]

            stats[section] = {
                'n_paragraphs': len(paras),
                'mean_words': float(np.mean(word_counts)),
                'std_words': float(np.std(word_counts)),
                'median_words': float(np.median(word_counts)),
                'min_words': int(np.min(word_counts)),
                'max_words': int(np.max(word_counts)),
                'mean_unique_words': float(np.mean(unique_counts)),
                'type_token_ratio': float(np.mean(
                    [u / max(w, 1) for u, w in zip(unique_counts, word_counts)]
                )),
                'word_count_histogram': dict(Counter(
                    [(w // 10) * 10 for w in word_counts]
                )),
            }

        return stats

    def reference_paragraph_stats(self) -> Dict[str, Dict]:
        """Get reference paragraph statistics for medieval text types."""
        return PARAGRAPH_STATS

    def length_ratio_estimate(
        self, voynich_stats: Dict[str, Dict], reference_stats: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Estimate the plaintext-to-ciphertext length ratio per section.
        Ratio = Voynich_mean / Reference_mean
        """
        ratios = {}

        for section, v_stats in voynich_stats.items():
            text_type = SECTION_TEXT_TYPE_MAP.get(section)
            if not text_type or text_type not in reference_stats:
                continue

            ref = reference_stats[text_type]
            voynich_mean = v_stats['mean_words']
            ref_mean = ref['mean_words']

            ratio = voynich_mean / max(ref_mean, 1)

            ratios[section] = {
                'voynich_mean_words': voynich_mean,
                'reference_mean_words': ref_mean,
                'reference_text_type': text_type,
                'ratio': ratio,
                'interpretation': self._interpret_ratio(ratio),
            }

        return ratios

    def _interpret_ratio(self, ratio: float) -> str:
        """Interpret the plaintext-ciphertext length ratio."""
        if ratio < 0.5:
            return (f'Ratio {ratio:.2f}: STRONG COMPRESSION. '
                    'Cipher compresses text significantly, suggesting information-dense '
                    'encoding (e.g., abbreviations, code words, or multi-character mappings).')
        elif ratio < 0.8:
            return (f'Ratio {ratio:.2f}: MODERATE COMPRESSION. '
                    'Cipher somewhat compresses text. May use abbreviations or omit '
                    'common words.')
        elif ratio < 1.2:
            return (f'Ratio {ratio:.2f}: APPROXIMATELY 1:1. '
                    'Cipher roughly preserves text length. Consistent with simple '
                    'substitution or mild transformation.')
        elif ratio < 1.5:
            return (f'Ratio {ratio:.2f}: MILD EXPANSION. '
                    'Cipher slightly expands text, possibly due to prefix/suffix additions '
                    'or multi-glyph substitutions.')
        else:
            return (f'Ratio {ratio:.2f}: SIGNIFICANT EXPANSION. '
                    'Cipher expands text substantially, suggesting nulls, padding, '
                    'or verbose multi-glyph substitutions.')

    def ratio_implications(self, ratios: Dict[str, Dict]) -> Dict:
        """Derive cipher mechanism constraints from length ratios."""
        all_ratios = [r['ratio'] for r in ratios.values()]
        if not all_ratios:
            return {'insufficient_data': True}

        mean_ratio = np.mean(all_ratios)
        std_ratio = np.std(all_ratios)

        consistent = std_ratio < 0.3

        if mean_ratio < 0.7:
            mechanism = ('Compression cipher: the mechanism must encode more information '
                         'per output character than input. Candidate mechanisms: '
                         'abbreviation cipher, code book, syllabic encoding.')
        elif mean_ratio < 1.3:
            mechanism = ('Length-preserving cipher: approximately 1-to-1 mapping. '
                         'Candidate mechanisms: simple substitution, monoalphabetic, '
                         'polyalphabetic with same-alphabet output.')
        else:
            mechanism = ('Expansion cipher: the mechanism adds material to output. '
                         'Candidate mechanisms: homophonic (multi-symbol per letter), '
                         'null-insertion, prefix/suffix decoration (Naibbe-type).')

        return {
            'mean_ratio': float(mean_ratio),
            'std_ratio': float(std_ratio),
            'cross_section_consistent': consistent,
            'mechanism_constraint': mechanism,
            'per_section_ratios': {s: r['ratio'] for s, r in ratios.items()},
        }

def run(verbose: bool = True) -> Dict:
    """
    Run paragraph structure analysis.

    Returns:
        Dict with paragraph statistics, length ratios, and mechanism constraints.
    """
    analyzer = ParagraphAnalyzer(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 6: PARAGRAPH STRUCTURE ANALYSIS")
        print("=" * 70)

    if verbose:
        print("\n  Extracting paragraphs from corpus...")
    paragraphs = analyzer.extract_paragraphs()
    if verbose:
        for section, paras in paragraphs.items():
            print(f"    {section}: {len(paras)} paragraphs")

    if verbose:
        print("\n  Computing paragraph statistics...")
    voynich_stats = analyzer.paragraph_statistics(paragraphs)
    if verbose:
        for section, stats in voynich_stats.items():
            print(f"    {section}: μ={stats['mean_words']:.1f} ± "
                  f"{stats['std_words']:.1f} words  "
                  f"(n={stats['n_paragraphs']})")

    reference_stats = analyzer.reference_paragraph_stats()

    if verbose:
        print("\n  Estimating plaintext-ciphertext length ratios...")
    ratios = analyzer.length_ratio_estimate(voynich_stats, reference_stats)
    if verbose:
        for section, data in ratios.items():
            print(f"    {section}: {data['voynich_mean_words']:.1f} / "
                  f"{data['reference_mean_words']:.1f} = {data['ratio']:.2f}")
            print(f"      {data['interpretation'][:80]}")

    if verbose:
        print("\n  Deriving mechanism constraints...")
    implications = analyzer.ratio_implications(ratios)
    if verbose:
        print(f"    Mean ratio: {implications.get('mean_ratio', 0):.2f} ± "
              f"{implications.get('std_ratio', 0):.2f}")
        print(f"    Cross-section consistent: "
              f"{implications.get('cross_section_consistent', False)}")
        print(f"    {implications.get('mechanism_constraint', '')[:100]}")

    results = {
        'track': 'paragraph_analysis',
        'track_number': 6,
        'voynich_stats': voynich_stats,
        'reference_stats': {
            k: {sk: sv for sk, sv in v.items() if sk != 'examples'}
            for k, v in reference_stats.items()
        },
        'length_ratios': {
            s: {k: v for k, v in r.items()}
            for s, r in ratios.items()
        },
        'implications': implications,
    }

    if verbose:
        print("\n" + "─" * 70)
        print("PARAGRAPH ANALYSIS SUMMARY")
        print("─" * 70)
        print(f"  Mean length ratio: {implications.get('mean_ratio', 0):.2f}")
        print(f"  {implications.get('mechanism_constraint', '')[:100]}")

    return results
