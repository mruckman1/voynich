"""
Track 5: qo- Functional Analysis
====================================
Determines the linguistic function of "qo-" words — the single strongest
distributional signal in the Voynich Manuscript.

The character 'q' appears 98.3% word-initial, almost always followed by 'o'.
These words dominate Language B but are rare in Language A. If qo- words are
grammatical markers (articles, prepositions, demonstratives), their fixed
positions and restricted collocations make them the most crackable units.
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from modules.statistical_analysis import (
    first_order_entropy, conditional_entropy, compute_all_entropy
)
from data.voynich_corpus import get_all_tokens, get_section_text, SAMPLE_CORPUS, SECTIONS

class QoAnalyzer:
    """Analyzes the distributional and functional properties of qo- words."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def extract_qo_words(self, tokens: List[str]) -> List[str]:
        """Extract all words starting with 'qo' or consisting of 'q' alone."""
        return [t for t in tokens if t.startswith('qo') or t == 'q']

    def q_positional_stats(self, tokens: List[str]) -> Dict:
        """Analyze where 'q' appears within words."""
        q_initial = 0
        q_medial = 0
        q_final = 0
        q_total = 0

        for token in tokens:
            for i, ch in enumerate(token):
                if ch == 'q':
                    q_total += 1
                    if i == 0:
                        q_initial += 1
                    elif i == len(token) - 1:
                        q_final += 1
                    else:
                        q_medial += 1

        return {
            'q_initial': q_initial,
            'q_medial': q_medial,
            'q_final': q_final,
            'q_total': q_total,
            'initial_fraction': q_initial / max(q_total, 1),
            'q_followed_by_o': self._q_followed_by_o_rate(tokens),
        }

    def _q_followed_by_o_rate(self, tokens: List[str]) -> float:
        """Fraction of 'q' characters followed by 'o'."""
        q_total = 0
        qo_count = 0
        for token in tokens:
            for i, ch in enumerate(token):
                if ch == 'q':
                    q_total += 1
                    if i + 1 < len(token) and token[i + 1] == 'o':
                        qo_count += 1
        return qo_count / max(q_total, 1)

    def qo_frequency_by_section(self) -> Dict[str, Dict]:
        """Compute qo-word proportion per section."""
        results = {}
        for section in SECTIONS:
            text = get_section_text(section)
            tokens = text.split() if text else []
            if not tokens:
                continue
            qo_words = self.extract_qo_words(tokens)
            results[section] = {
                'total_tokens': len(tokens),
                'qo_tokens': len(qo_words),
                'qo_proportion': len(qo_words) / len(tokens),
                'qo_types': len(set(qo_words)),
                'top_qo_words': Counter(qo_words).most_common(5),
            }
        return results

    def qo_frequency_by_language(self) -> Dict[str, Dict]:
        """Compare qo-word frequency in Language A vs Language B."""
        lang_tokens = {'A': [], 'B': []}

        for folio, data in SAMPLE_CORPUS.items():
            lang = data.get('lang', '')
            if lang in lang_tokens:
                for line in data.get('text', []):
                    lang_tokens[lang].extend(line.split())

        results = {}
        for lang, tokens in lang_tokens.items():
            if tokens:
                qo_words = self.extract_qo_words(tokens)
                results[lang] = {
                    'total_tokens': len(tokens),
                    'qo_tokens': len(qo_words),
                    'qo_proportion': len(qo_words) / len(tokens),
                    'qo_types': len(set(qo_words)),
                    'top_qo_words': Counter(qo_words).most_common(10),
                }

        if 'A' in results and 'B' in results:
            ratio = results['B']['qo_proportion'] / max(results['A']['qo_proportion'], 0.001)
            results['b_to_a_ratio'] = ratio
            results['asymmetry'] = (
                f'qo- words are {ratio:.1f}x more frequent in Language B '
                f'({results["B"]["qo_proportion"]:.1%}) than Language A '
                f'({results["A"]["qo_proportion"]:.1%})'
            )

        return results

    def qo_positional_analysis(self) -> Dict:
        """
        Where do qo-words appear within paragraphs?
        Tests whether qo-words cluster at paragraph/sentence beginnings,
        which would suggest article/demonstrative function.
        """
        positions = []
        first_word_qo = 0
        total_blocks = 0

        for folio, data in SAMPLE_CORPUS.items():
            for line in data.get('text', []):
                tokens = line.split()
                if len(tokens) < 3:
                    continue
                total_blocks += 1
                if tokens[0].startswith('qo') or tokens[0] == 'q':
                    first_word_qo += 1

                for i, token in enumerate(tokens):
                    if token.startswith('qo') or token == 'q':
                        positions.append(i / max(len(tokens) - 1, 1))

        if not positions:
            return {'insufficient_data': True}

        pos_arr = np.array(positions)

        quartile_counts = {
            'first_quarter': float(np.sum(pos_arr < 0.25) / len(pos_arr)),
            'second_quarter': float(np.sum((pos_arr >= 0.25) & (pos_arr < 0.5)) / len(pos_arr)),
            'third_quarter': float(np.sum((pos_arr >= 0.5) & (pos_arr < 0.75)) / len(pos_arr)),
            'fourth_quarter': float(np.sum(pos_arr >= 0.75) / len(pos_arr)),
        }

        first_word_rate = first_word_qo / max(total_blocks, 1)

        return {
            'mean_position': float(np.mean(pos_arr)),
            'std_position': float(np.std(pos_arr)),
            'median_position': float(np.median(pos_arr)),
            'quartile_distribution': quartile_counts,
            'first_word_rate': first_word_rate,
            'total_qo_occurrences': len(positions),
            'total_blocks': total_blocks,
            'front_biased': quartile_counts['first_quarter'] > 0.35,
        }

    def collocation_analysis(self, window: int = 2) -> Dict:
        """
        Analyze what words co-occur with qo-words.
        Computes Pointwise Mutual Information (PMI) for collocates.
        """
        all_tokens = get_all_tokens()
        total = len(all_tokens)
        if total < 10:
            return {'insufficient_data': True}

        word_freq = Counter(all_tokens)
        qo_indices = [i for i, t in enumerate(all_tokens)
                      if t.startswith('qo') or t == 'q']

        left_collocates = Counter()
        right_collocates = Counter()

        for idx in qo_indices:
            for offset in range(1, window + 1):
                if idx - offset >= 0:
                    left_collocates[all_tokens[idx - offset]] += 1
                if idx + offset < total:
                    right_collocates[all_tokens[idx + offset]] += 1

        qo_freq = len(qo_indices) / total

        def compute_pmi(collocate_counts, position):
            pmi_scores = []
            for word, count in collocate_counts.most_common(20):
                word_prob = word_freq[word] / total
                joint_prob = count / total
                if word_prob > 0 and joint_prob > 0:
                    pmi = math.log2(joint_prob / (qo_freq * word_prob))
                    pmi_scores.append((word, float(pmi), count))
            return sorted(pmi_scores, key=lambda x: -x[1])

        left_pmi = compute_pmi(left_collocates, 'left')
        right_pmi = compute_pmi(right_collocates, 'right')

        return {
            'left_collocates': left_pmi[:10],
            'right_collocates': right_pmi[:10],
            'n_qo_occurrences': len(qo_indices),
            'strongest_left': left_pmi[0] if left_pmi else None,
            'strongest_right': right_pmi[0] if right_pmi else None,
        }

    def functional_classification(
        self, positional: Dict, collocation: Dict, language: Dict
    ) -> Dict:
        """
        Classify the functional role of qo-words based on evidence.
        """
        evidence = []
        scores = {
            'article_demonstrative': 0,
            'preposition': 0,
            'conjunction': 0,
            'grammatical_particle': 0,
        }

        if positional.get('front_biased', False):
            scores['article_demonstrative'] += 2
            evidence.append('Front-biased position supports article/demonstrative role')
        if positional.get('first_word_rate', 0) > 0.15:
            scores['article_demonstrative'] += 1
            evidence.append(f'Appears as first word {positional["first_word_rate"]:.0%} of the time')

        right_collocs = collocation.get('right_collocates', [])
        if right_collocs:
            scores['article_demonstrative'] += 1
            scores['preposition'] += 1
            evidence.append(f'Strong right collocate: {right_collocs[0][0]}')

        ratio = language.get('b_to_a_ratio', 1.0)
        if ratio > 3:
            scores['grammatical_particle'] += 1
            evidence.append(f'Language B/A ratio = {ratio:.1f}x (asymmetric)')

        best = max(scores, key=scores.get)
        confidence = 'HIGH' if scores[best] >= 3 else 'MODERATE' if scores[best] >= 2 else 'LOW'

        interpretations = {
            'article_demonstrative': (
                'qo- words likely function as ARTICLES or DEMONSTRATIVES. '
                'They appear at the beginnings of text blocks and before specific '
                'word classes, analogous to "the/this/that" or "le/la/il".'
            ),
            'preposition': (
                'qo- words likely function as PREPOSITIONS. '
                'They connect nominal phrases and show position-dependent collocations.'
            ),
            'conjunction': (
                'qo- words likely function as CONJUNCTIONS. '
                'They connect clauses and appear between content words.'
            ),
            'grammatical_particle': (
                'qo- words likely function as GRAMMATICAL PARTICLES. '
                'Their asymmetric distribution suggests a language-specific grammatical '
                'feature present in Language B but not Language A.'
            ),
        }

        return {
            'classification': best,
            'confidence': confidence,
            'scores': scores,
            'evidence': evidence,
            'interpretation': interpretations[best],
        }

    def substitution_test(self) -> Dict:
        """
        Replace all qo-words with candidate grammatical words and test
        if the resulting entropy profile better matches known languages.
        """
        all_tokens = get_all_tokens()
        text_original = ' '.join(all_tokens)
        entropy_original = compute_all_entropy(text_original)

        candidates = {
            'latin_article': ['hoc', 'hic', 'ille', 'iste'],
            'latin_preposition': ['in', 'de', 'ad', 'cum', 'per'],
            'italian_article': ['il', 'la', 'lo', 'le', 'un'],
            'german_article': ['der', 'die', 'das', 'ein', 'dem'],
        }

        results = {}
        for category, words in candidates.items():
            import random
            rng = random.Random(42)
            modified_tokens = []
            for t in all_tokens:
                if t.startswith('qo') or t == 'q':
                    modified_tokens.append(rng.choice(words))
                else:
                    modified_tokens.append(t)

            text_modified = ' '.join(modified_tokens)
            entropy_modified = compute_all_entropy(text_modified)

            delta_h2 = entropy_modified['H2'] - entropy_original['H2']

            results[category] = {
                'replacement_words': words,
                'H2_original': entropy_original['H2'],
                'H2_modified': entropy_modified['H2'],
                'delta_H2': delta_h2,
            }

        target_h2 = 3.5
        best = min(results.items(),
                   key=lambda x: abs(x[1]['H2_modified'] - target_h2))

        results['best_substitution'] = best[0]
        results['best_delta_H2'] = best[1]['delta_H2']

        return results

def run(verbose: bool = True) -> Dict:
    """
    Run qo- functional analysis.

    Returns:
        Dict with q positional stats, section/language frequencies,
        positional analysis, collocations, functional classification,
        and substitution test.
    """
    analyzer = QoAnalyzer(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 5: qo- FUNCTIONAL ANALYSIS")
        print("=" * 70)

    tokens = get_all_tokens()

    if verbose:
        print("\n  Analyzing 'q' positional behavior...")
    q_stats = analyzer.q_positional_stats(tokens)
    if verbose:
        print(f"    q initial: {q_stats['initial_fraction']:.1%}")
        print(f"    q followed by 'o': {q_stats['q_followed_by_o']:.1%}")

    qo_words = analyzer.extract_qo_words(tokens)
    qo_vocab = Counter(qo_words)
    if verbose:
        print(f"\n  qo-words: {len(qo_words)} tokens, {len(qo_vocab)} types")
        print(f"  Top qo-words: {qo_vocab.most_common(10)}")

    if verbose:
        print("\n  Section-level qo-word frequencies...")
    section_freq = analyzer.qo_frequency_by_section()
    if verbose:
        for section, data in section_freq.items():
            print(f"    {section}: {data['qo_proportion']:.1%} "
                  f"({data['qo_tokens']}/{data['total_tokens']})")

    if verbose:
        print("\n  Language A vs B qo-word frequencies...")
    language_freq = analyzer.qo_frequency_by_language()
    if verbose:
        for lang in ['A', 'B']:
            if lang in language_freq:
                data = language_freq[lang]
                print(f"    Language {lang}: {data['qo_proportion']:.1%}")
        if 'asymmetry' in language_freq:
            print(f"    {language_freq['asymmetry']}")

    if verbose:
        print("\n  Analyzing qo-word positions within text blocks...")
    positional = analyzer.qo_positional_analysis()
    if verbose and not positional.get('insufficient_data'):
        print(f"    Mean position: {positional['mean_position']:.2f}")
        print(f"    First-word rate: {positional['first_word_rate']:.1%}")
        print(f"    Front-biased: {positional.get('front_biased', False)}")

    if verbose:
        print("\n  Computing collocations (PMI)...")
    collocations = analyzer.collocation_analysis()
    if verbose and not collocations.get('insufficient_data'):
        if collocations.get('strongest_right'):
            w, pmi, n = collocations['strongest_right']
            print(f"    Strongest right collocate: {w} (PMI={pmi:.2f}, n={n})")
        if collocations.get('strongest_left'):
            w, pmi, n = collocations['strongest_left']
            print(f"    Strongest left collocate: {w} (PMI={pmi:.2f}, n={n})")

    if verbose:
        print("\n  Classifying functional role...")
    classification = analyzer.functional_classification(
        positional, collocations, language_freq
    )
    if verbose:
        print(f"    Classification: {classification['classification']} "
              f"[{classification['confidence']}]")
        print(f"    {classification['interpretation']}")

    if verbose:
        print("\n  Running substitution test...")
    substitution = analyzer.substitution_test()
    if verbose:
        print(f"    Best substitution class: {substitution['best_substitution']}")
        print(f"    ΔH2: {substitution['best_delta_H2']:+.4f}")

    results = {
        'track': 'qo_analysis',
        'track_number': 5,
        'q_positional_stats': q_stats,
        'qo_word_count': len(qo_words),
        'qo_type_count': len(qo_vocab),
        'top_qo_words': qo_vocab.most_common(20),
        'section_frequencies': {
            s: {k: v for k, v in d.items() if k != 'top_qo_words'}
            for s, d in section_freq.items()
        },
        'language_frequencies': {
            k: v for k, v in language_freq.items()
            if k not in ('A', 'B') or not isinstance(v, dict) or 'top_qo_words' not in v
        },
        'language_a_proportion': language_freq.get('A', {}).get('qo_proportion', 0),
        'language_b_proportion': language_freq.get('B', {}).get('qo_proportion', 0),
        'positional_analysis': positional,
        'collocations': {
            k: v for k, v in collocations.items()
            if k in ('n_qo_occurrences', 'strongest_left', 'strongest_right')
        },
        'functional_classification': classification,
        'substitution_test': {
            'best_substitution': substitution.get('best_substitution'),
            'best_delta_H2': substitution.get('best_delta_H2'),
        },
    }

    if verbose:
        print("\n" + "─" * 70)
        print("qo- ANALYSIS SUMMARY")
        print("─" * 70)
        print(f"  qo-words: {len(qo_words)} tokens ({len(qo_vocab)} unique)")
        print(f"  Functional role: {classification['classification']} "
              f"[{classification['confidence']}]")
        print(f"  {classification['interpretation']}")

    return results
