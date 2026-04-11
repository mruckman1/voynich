"""
Attack 1: Two-Pattern Hypothesis
==================================
Tests whether Language B's two word families (edy/aiin) correlate
with known content types from the zodiac/astronomical sections.

Three sub-hypotheses:
  A) Two semantic categories (zodiac signs vs body parts)
  B) Verb/noun distinction (aiin = verbs at line starts, edy = nouns)
  C) Plaintext letter encoding (edy = consonants, aiin = vowels)

Priority: HIGHEST
"""

import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.modules.phase3.lang_b_profiler import LanguageBProfiler, LANG_B_TARGETS, LANG_B_VOCABULARY
from voynich.core.voynich_corpus import ZODIAC_LABELS, SAMPLE_CORPUS

ZODIAC_ELEMENTS = {
    'Aries': 'fire', 'Leo': 'fire', 'Sagittarius': 'fire',
    'Taurus': 'earth', 'Virgo': 'earth', 'Capricorn': 'earth',
    'Gemini': 'air', 'Libra': 'air', 'Aquarius': 'air',
    'Cancer': 'water', 'Scorpio': 'water', 'Pisces': 'water',
}

ANATOMICAL_REGIONS = {
    'head': 'upper', 'neck/throat': 'upper', 'arms/shoulders': 'upper',
    'chest/breast': 'upper', 'heart/stomach': 'middle',
    'intestines/womb': 'middle', 'kidneys/loins': 'middle',
    'genitals': 'lower', 'thighs': 'lower', 'knees': 'lower',
    'legs/shins': 'lower', 'feet': 'lower',
}

class TwoPatternAttack:
    """
    Tests whether edy/aiin word families encode distinct content categories.
    """

    def __init__(self, profiler: Optional[LanguageBProfiler] = None):
        self.profiler = profiler or LanguageBProfiler()
        self.profiler.extract_corpus()
        self.profiler.classify_word_families()
        self.folio_data = self.profiler.extract_folio_level_data()

        families = self.profiler.word_families
        self.edy_words = set(w for w, _ in families.get('edy', []))
        self.aiin_words = set(w for w, _ in families.get('aiin', []))

    def hypothesis_a_semantic_categories(self) -> Dict:
        """
        Hypothesis A: edy and aiin encode two semantic categories.

        Groups zodiac folios by element (fire/earth/air/water) and
        tests if edy/aiin ratios correlate with element grouping via
        chi-squared test for independence.
        """
        zodiac_folios = [f for f in self.folio_data if 'zodiac_sign' in f]

        if len(zodiac_folios) < 2:
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'n_zodiac_folios': len(zodiac_folios),
                'interpretation': (
                    f'Only {len(zodiac_folios)} zodiac folios in sample — '
                    f'need at least 2 for chi-squared test. '
                    f'Run on full IVTFF corpus for meaningful results.'
                ),
            }

        element_counts = defaultdict(lambda: {'edy': 0, 'aiin': 0})
        folio_ratios = []

        for f in zodiac_folios:
            sign = f.get('zodiac_sign', '')
            element = ZODIAC_ELEMENTS.get(sign, 'unknown')
            element_counts[element]['edy'] += f['edy_count']
            element_counts[element]['aiin'] += f['aiin_count']
            folio_ratios.append({
                'folio': f['folio'],
                'sign': sign,
                'element': element,
                'edy_ratio': f['edy_ratio'],
                'aiin_ratio': f['aiin_ratio'],
            })

        elements = sorted(element_counts.keys())
        if len(elements) < 2:
            return {
                'verdict': 'INSUFFICIENT_ELEMENTS',
                'elements_found': elements,
                'folio_ratios': folio_ratios,
                'interpretation': (
                    f'Only {len(elements)} zodiac elements represented — '
                    f'need at least 2 for contingency test.'
                ),
            }

        observed = np.array([
            [element_counts[e]['edy'] for e in elements],
            [element_counts[e]['aiin'] for e in elements],
        ], dtype=float)

        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        grand_total = observed.sum()

        if grand_total == 0:
            return {
                'verdict': 'NO_DATA',
                'folio_ratios': folio_ratios,
                'interpretation': 'No edy/aiin tokens found in zodiac folios.',
            }

        expected = row_totals * col_totals / grand_total
        expected[expected == 0] = 1e-10

        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        df = (observed.shape[0] - 1) * (observed.shape[1] - 1)

        if df > 0 and chi2 > 0:
            z = ((chi2 / df) ** (1/3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
        else:
            p_value = 1.0

        min_dim = min(observed.shape[0] - 1, observed.shape[1] - 1)
        cramers_v = math.sqrt(chi2 / (grand_total * max(min_dim, 1))) if grand_total > 0 else 0

        element_edy_ratios = {}
        for e in elements:
            total = element_counts[e]['edy'] + element_counts[e]['aiin']
            element_edy_ratios[e] = element_counts[e]['edy'] / total if total > 0 else 0

        return {
            'verdict': 'CONSISTENT' if p_value < 0.05 else 'INCONSISTENT',
            'chi2': chi2,
            'df': df,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'contingency_table': {
                'elements': elements,
                'edy_counts': [element_counts[e]['edy'] for e in elements],
                'aiin_counts': [element_counts[e]['aiin'] for e in elements],
            },
            'element_edy_ratios': element_edy_ratios,
            'folio_ratios': folio_ratios,
            'interpretation': (
                f'Chi2={chi2:.3f}, df={df}, p={p_value:.4f}, V={cramers_v:.3f}. '
                f'{"Significant" if p_value < 0.05 else "Not significant"} — '
                f'edy/aiin ratio {"varies" if p_value < 0.05 else "does NOT vary"} '
                f'by zodiac element.'
            ),
        }

    def hypothesis_b_verb_noun(self) -> Dict:
        """
        Hypothesis B: edy = nouns, aiin = verbs (or vice versa).

        Tests if aiin-family words cluster at line-initial positions
        (where imperative verbs like 'recipe' would appear in medical text)
        vs edy words distributed more uniformly.
        """
        edy_positions = []
        aiin_positions = []

        for folio_id, data in SAMPLE_CORPUS.items():
            if data['lang'] != 'B':
                continue
            for line in data['text']:
                tokens = line.split()
                n = len(tokens)
                if n == 0:
                    continue
                for i, tok in enumerate(tokens):
                    pos = i / max(n - 1, 1)
                    if tok in self.edy_words:
                        edy_positions.append(pos)
                    elif tok in self.aiin_words:
                        aiin_positions.append(pos)

        if not edy_positions or not aiin_positions:
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'n_edy': len(edy_positions),
                'n_aiin': len(aiin_positions),
                'interpretation': (
                    f'Insufficient positional data: {len(edy_positions)} edy, '
                    f'{len(aiin_positions)} aiin positions.'
                ),
            }

        edy_arr = np.array(edy_positions)
        aiin_arr = np.array(aiin_positions)

        edy_initial = np.mean(edy_arr == 0.0)
        aiin_initial = np.mean(aiin_arr == 0.0)

        combined = np.sort(np.concatenate([edy_arr, aiin_arr]))
        edy_cdf = np.searchsorted(np.sort(edy_arr), combined, side='right') / len(edy_arr)
        aiin_cdf = np.searchsorted(np.sort(aiin_arr), combined, side='right') / len(aiin_arr)
        ks_stat = float(np.max(np.abs(edy_cdf - aiin_cdf)))

        n_eff = len(edy_arr) * len(aiin_arr) / (len(edy_arr) + len(aiin_arr))
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * ks_stat
        if lambda_val > 0:
            p_value = 2 * math.exp(-2 * lambda_val ** 2)
            p_value = max(0.0, min(1.0, p_value))
        else:
            p_value = 1.0

        edy_mean_pos = float(np.mean(edy_arr))
        aiin_mean_pos = float(np.mean(aiin_arr))

        return {
            'verdict': 'CONSISTENT' if p_value < 0.05 else 'INCONSISTENT',
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'edy_line_initial_rate': float(edy_initial),
            'aiin_line_initial_rate': float(aiin_initial),
            'edy_mean_position': edy_mean_pos,
            'aiin_mean_position': aiin_mean_pos,
            'edy_n': len(edy_positions),
            'aiin_n': len(aiin_positions),
            'interpretation': (
                f'KS={ks_stat:.3f}, p={p_value:.4f}. '
                f'edy mean_pos={edy_mean_pos:.3f}, aiin mean_pos={aiin_mean_pos:.3f}. '
                f'{"Significant positional difference" if p_value < 0.05 else "No significant positional difference"} — '
                f'{"supports" if p_value < 0.05 else "does NOT support"} verb/noun distinction.'
            ),
        }

    def hypothesis_c_letter_encoding(self) -> Dict:
        """
        Hypothesis C: edy/aiin encode consonants/vowels of plaintext.

        Compares the observed edy:aiin ratio against expected letter-class
        splits in Latin medical text.
        """
        family_stats = self.profiler.compute_family_statistics()
        edy_prop = family_stats.get('edy', {}).get('proportion', 0)
        aiin_prop = family_stats.get('aiin', {}).get('proportion', 0)

        binary_splits = {
            'consonant_vs_vowel': {
                'description': 'Consonants (BCDFGHLMNPQRSTVX) vs vowels (AEIOU)',
                'expected_major': 0.60, 'expected_minor': 0.40,
            },
            'common_vs_rare': {
                'description': 'Top 10 letters vs bottom 11',
                'expected_major': 0.80, 'expected_minor': 0.20,
            },
            'first_half_vs_second': {
                'description': 'Letters A-M vs N-Z',
                'expected_major': 0.55, 'expected_minor': 0.45,
            },
            'labial_dental_vs_other': {
                'description': 'Labial/dental (BDFLMNPT) vs other consonants + vowels',
                'expected_major': 0.45, 'expected_minor': 0.55,
            },
        }

        test_results = []
        for split_name, split in binary_splits.items():
            mismatch_edy_major = abs(edy_prop - split['expected_major'])
            mismatch_edy_minor = abs(edy_prop - split['expected_minor'])
            best_mismatch = min(mismatch_edy_major, mismatch_edy_minor)

            test_results.append({
                'split': split_name,
                'description': split['description'],
                'expected_ratio': f'{split["expected_major"]:.0%}:{split["expected_minor"]:.0%}',
                'observed_ratio': f'{edy_prop:.1%}:{aiin_prop:.1%}',
                'mismatch': best_mismatch,
                'plausible': best_mismatch < 0.10,
            })

        any_plausible = any(t['plausible'] for t in test_results)

        return {
            'verdict': 'POSSIBLE' if any_plausible else 'ELIMINATED',
            'edy_proportion': edy_prop,
            'aiin_proportion': aiin_prop,
            'binary_split_tests': test_results,
            'any_plausible_split': any_plausible,
            'interpretation': (
                f'Observed edy:aiin = {edy_prop:.1%}:{aiin_prop:.1%}. '
                f'{"At least one" if any_plausible else "No"} plausible binary letter-class '
                f'split matches this ratio (threshold: mismatch < 10%). '
                f'Hypothesis C is {"still viable" if any_plausible else "ELIMINATED"}.'
            ),
        }

    def cross_folio_distribution_test(self) -> Dict:
        """
        Test whether edy/aiin ratios vary across folios or stay constant.

        Constant ratio = structural/mechanical split (not semantic).
        Variable ratio = content-dependent (semantic information).
        """
        ratios = []
        for f in self.folio_data:
            if f['n_tokens'] >= 5:
                ratios.append({
                    'folio': f['folio'],
                    'section': f['section'],
                    'edy_ratio': f['edy_ratio'],
                    'aiin_ratio': f['aiin_ratio'],
                    'n_tokens': f['n_tokens'],
                })

        if len(ratios) < 3:
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'n_folios': len(ratios),
                'interpretation': (
                    f'Only {len(ratios)} folios with 5+ tokens — '
                    f'need at least 3 for distribution test.'
                ),
            }

        edy_ratios = np.array([r['edy_ratio'] for r in ratios])
        aiin_ratios = np.array([r['aiin_ratio'] for r in ratios])

        cv_edy = float(np.std(edy_ratios) / np.mean(edy_ratios)) if np.mean(edy_ratios) > 0 else 0
        cv_aiin = float(np.std(aiin_ratios) / np.mean(aiin_ratios)) if np.mean(aiin_ratios) > 0 else 0

        ratio_constant = cv_edy < 0.15

        return {
            'verdict': 'CONSTANT' if ratio_constant else 'VARIABLE',
            'cv_edy': cv_edy,
            'cv_aiin': cv_aiin,
            'edy_mean': float(np.mean(edy_ratios)),
            'edy_std': float(np.std(edy_ratios)),
            'aiin_mean': float(np.mean(aiin_ratios)),
            'aiin_std': float(np.std(aiin_ratios)),
            'per_folio_ratios': ratios,
            'interpretation': (
                f'CV(edy)={cv_edy:.3f}, CV(aiin)={cv_aiin:.3f}. '
                f'Edy/aiin ratio is {"approximately CONSTANT" if ratio_constant else "VARIABLE"} '
                f'across folios. '
                f'{"Suggests mechanical/structural split." if ratio_constant else "Suggests content-dependent (semantic) variation."}'
            ),
        }

    def zodiac_content_correlation(self) -> Dict:
        """
        Map word positions to zodiac page content and test if
        family ratios correlate with anatomical region changes.
        """
        zodiac_folios = [f for f in self.folio_data if 'zodiac_sign' in f]

        if len(zodiac_folios) < 3:
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'n_zodiac': len(zodiac_folios),
                'interpretation': (
                    f'Only {len(zodiac_folios)} zodiac folios — '
                    f'need at least 3 for anatomical correlation.'
                ),
            }

        region_counts = defaultdict(lambda: {'edy': 0, 'aiin': 0, 'total': 0})
        mapping = []

        for f in zodiac_folios:
            body_part = f.get('body_part', '')
            region = ANATOMICAL_REGIONS.get(body_part, 'unknown')
            region_counts[region]['edy'] += f['edy_count']
            region_counts[region]['aiin'] += f['aiin_count']
            region_counts[region]['total'] += f['n_tokens']
            mapping.append({
                'folio': f['folio'],
                'sign': f.get('zodiac_sign', ''),
                'body_part': body_part,
                'region': region,
                'edy_ratio': f['edy_ratio'],
            })

        region_edy_ratios = {}
        for region, counts in region_counts.items():
            if counts['total'] > 0:
                region_edy_ratios[region] = counts['edy'] / counts['total']

        ordered_regions = ['upper', 'middle', 'lower']
        ordered_ratios = [region_edy_ratios.get(r, 0) for r in ordered_regions
                          if r in region_edy_ratios]

        has_monotonic_trend = False
        if len(ordered_ratios) >= 3:
            diffs = [ordered_ratios[i+1] - ordered_ratios[i]
                     for i in range(len(ordered_ratios)-1)]
            has_monotonic_trend = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)

        return {
            'verdict': 'CORRELATED' if has_monotonic_trend else 'NO_CORRELATION',
            'region_edy_ratios': region_edy_ratios,
            'zodiac_mapping': mapping,
            'has_monotonic_trend': has_monotonic_trend,
            'interpretation': (
                f'Anatomical region edy ratios: {region_edy_ratios}. '
                f'{"Monotonic trend detected" if has_monotonic_trend else "No monotonic trend"} '
                f'across upper->middle->lower body regions.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all three hypotheses plus supporting tests."""
        if verbose:
            print('\n=== Attack 1: Two-Pattern Hypothesis ===')

        results = {}

        if verbose:
            print('\n  --- Hypothesis A: Semantic Categories ---')
        results['hypothesis_a'] = self.hypothesis_a_semantic_categories()
        if verbose:
            print(f'  {results["hypothesis_a"]["interpretation"]}')

        if verbose:
            print('\n  --- Hypothesis B: Verb/Noun Distinction ---')
        results['hypothesis_b'] = self.hypothesis_b_verb_noun()
        if verbose:
            print(f'  {results["hypothesis_b"]["interpretation"]}')

        if verbose:
            print('\n  --- Hypothesis C: Letter Encoding ---')
        results['hypothesis_c'] = self.hypothesis_c_letter_encoding()
        if verbose:
            print(f'  {results["hypothesis_c"]["interpretation"]}')

        if verbose:
            print('\n  --- Cross-Folio Distribution Test ---')
        results['cross_folio'] = self.cross_folio_distribution_test()
        if verbose:
            print(f'  {results["cross_folio"]["interpretation"]}')

        if verbose:
            print('\n  --- Zodiac Content Correlation ---')
        results['zodiac_correlation'] = self.zodiac_content_correlation()
        if verbose:
            print(f'  {results["zodiac_correlation"]["interpretation"]}')

        verdicts = {
            'hypothesis_a': results['hypothesis_a'].get('verdict', 'UNKNOWN'),
            'hypothesis_b': results['hypothesis_b'].get('verdict', 'UNKNOWN'),
            'hypothesis_c': results['hypothesis_c'].get('verdict', 'UNKNOWN'),
            'cross_folio': results['cross_folio'].get('verdict', 'UNKNOWN'),
            'zodiac': results['zodiac_correlation'].get('verdict', 'UNKNOWN'),
        }
        results['synthesis'] = {
            'verdicts': verdicts,
            'strongest_hypothesis': _pick_strongest(verdicts),
        }

        if verbose:
            print(f'\n  Synthesis: {results["synthesis"]}')

        return results

def _pick_strongest(verdicts: Dict[str, str]) -> str:
    """Pick the most supported hypothesis."""
    scores = {'A': 0, 'B': 0, 'C': 0}

    if verdicts.get('hypothesis_a') == 'CONSISTENT':
        scores['A'] += 2
    if verdicts.get('hypothesis_b') == 'CONSISTENT':
        scores['B'] += 2
    if verdicts.get('hypothesis_c') == 'POSSIBLE':
        scores['C'] += 1
    elif verdicts.get('hypothesis_c') == 'ELIMINATED':
        scores['C'] -= 2

    if verdicts.get('cross_folio') == 'VARIABLE':
        scores['A'] += 1
        scores['B'] += 1
    elif verdicts.get('cross_folio') == 'CONSTANT':
        scores['C'] += 1

    if verdicts.get('zodiac') == 'CORRELATED':
        scores['A'] += 2

    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return 'NONE — all hypotheses weak or eliminated'
    return f'Hypothesis {best} (score: {scores[best]})'
