"""
Module 13.5: Illustration-Text Correlation
============================================
Correlates Phase 12 decoded plant names against independent botanical
identifications from manuscript illustrations.

If the decoded text on a folio contains the same plant that was
independently identified from the illustration, that's external
validation that connects the decipherment to physical reality.

February 2026  ·  Voynich Convergence Attack  ·  Module 13.5
"""

import json
import os
import random
from math import comb
from typing import Dict, List, Optional, Set, Tuple

class IllustrationTextCorrelator:
    """Correlates botanical illustration IDs with Phase 12 decoded text.

    For each folio with a proposed plant identification (from PLANT_IDS),
    searches the decoded Latin text for medieval Latin names of that plant.
    A match validates the decoding: the text independently names the plant
    depicted in the illustration.
    """

    MIN_STEM = 4

    def __init__(
        self,
        folio_name_map: Dict[str, Dict],
        final_translations: Dict[str, str],
    ):
        """
        Args:
            folio_name_map: from build_folio_name_map() — folio→botanical data
            final_translations: from Phase 12 — {folio_id: decoded_text_string}
        """
        self.folio_map = folio_name_map
        self.translations = final_translations

        self._vocab: Optional[Set[str]] = None

    @property
    def corpus_vocab(self) -> Set[str]:
        """Unique resolved words across the entire decoded corpus."""
        if self._vocab is None:
            self._vocab = set()
            for text in self.translations.values():
                self._vocab.update(self._extract_resolved_words(text))
        return self._vocab

    @staticmethod
    def _extract_resolved_words(text: str) -> List[str]:
        """Extract only resolved (non-bracketed) words from folio text."""
        return [
            w.lower()
            for w in text.split()
            if not w.startswith('[') and not w.startswith('<')
        ]

    @staticmethod
    def _check_exact(word_set: Set[str], name: str) -> bool:
        """Check if a single-word name appears exactly in the word set."""
        return name in word_set

    @classmethod
    def _check_stem(cls, word_set: Set[str], name: str) -> Optional[str]:
        """Check if any resolved word shares a stem with the name.

        Returns the matched word, or None.
        """
        if len(name) < cls.MIN_STEM:
            return None
        for strip_len in range(1, min(4, len(name) - cls.MIN_STEM + 1)):
            stem = name[:-strip_len]
            for w in word_set:
                if len(w) >= cls.MIN_STEM and w.startswith(stem):
                    return w
        return None

    @staticmethod
    def _check_bigram(words: List[str], multi_name: str) -> bool:
        """Check if a multi-word name appears as adjacent words.

        Also checks within 2 positions for slight separation.
        """
        parts = multi_name.split()
        if len(parts) < 2:
            return False
        first, second = parts[0], parts[1]
        for i in range(len(words)):
            if words[i] == first:
                for j in range(i + 1, min(i + 3, len(words))):
                    if words[j] == second:
                        return True
        return False

    def correlate_folio(self, folio_id: str) -> Dict:
        """Run correlation analysis for a single folio.

        Returns a result dict with match details.
        """
        entry = self.folio_map[folio_id]
        result = {
            'folio': folio_id,
            'species': entry['species'],
            'common': entry['common'],
            'source': entry['source'],
            'confidence': entry['confidence'],
            'testable': entry['testable'],
            'new_world': entry['new_world'],
            'has_decoded_text': folio_id in self.translations,
            'resolved_word_count': 0,
            'searched_names': entry['latin_names'],
            'exact_matches': [],
            'stem_matches': {},
            'bigram_matches': [],
            'any_match': False,
            'match_type': 'none',
        }

        if not entry['testable']:
            return result
        if folio_id not in self.translations:
            return result

        text = self.translations[folio_id]
        words = self._extract_resolved_words(text)
        result['resolved_word_count'] = len(words)
        word_set = set(words)

        if not words:
            return result

        for name in entry['single_word_names']:
            if self._check_exact(word_set, name):
                result['exact_matches'].append(name)

        for name in entry['multi_word_names']:
            if self._check_bigram(words, name):
                result['bigram_matches'].append(name)

        exact_set = set(result['exact_matches'])
        for name in entry['single_word_names']:
            if name in exact_set:
                continue
            matched_word = self._check_stem(word_set, name)
            if matched_word and matched_word not in exact_set:
                result['stem_matches'][name] = matched_word

        if result['exact_matches']:
            result['any_match'] = True
            result['match_type'] = 'exact'
        elif result['bigram_matches']:
            result['any_match'] = True
            result['match_type'] = 'bigram'
        elif result['stem_matches']:
            result['any_match'] = True
            result['match_type'] = 'stem'

        return result

    def correlate_all(self) -> Dict:
        """Run correlation across all botanical folios.

        Returns:
            {
                'per_folio': {folio_id: result_dict, ...},
                'summary': aggregate counts,
                'rates': match rates,
            }
        """
        per_folio = {}
        for folio_id in sorted(self.folio_map.keys()):
            per_folio[folio_id] = self.correlate_folio(folio_id)

        total = len(per_folio)
        testable = sum(
            1 for r in per_folio.values()
            if r['testable'] and r['has_decoded_text']
        )
        matched = sum(1 for r in per_folio.values() if r['any_match'])
        new_world = sum(1 for r in per_folio.values() if r['new_world'])
        no_text = sum(
            1 for r in per_folio.values()
            if r['testable'] and not r['has_decoded_text']
        )
        no_match = sum(
            1 for r in per_folio.values()
            if r['testable'] and r['has_decoded_text'] and not r['any_match']
        )

        exact_count = sum(
            1 for r in per_folio.values() if r['match_type'] == 'exact'
        )
        stem_count = sum(
            1 for r in per_folio.values() if r['match_type'] == 'stem'
        )
        bigram_count = sum(
            1 for r in per_folio.values() if r['match_type'] == 'bigram'
        )

        rate_of_testable = matched / max(1, testable)
        rate_of_all = matched / max(1, total)

        return {
            'per_folio': per_folio,
            'summary': {
                'total_botanical_folios': total,
                'testable_folios': testable,
                'matched_folios': matched,
                'no_match_folios': no_match,
                'new_world_excluded': new_world,
                'no_decoded_text': no_text,
                'match_by_type': {
                    'exact': exact_count,
                    'stem': stem_count,
                    'bigram': bigram_count,
                },
            },
            'rates': {
                'of_testable': round(rate_of_testable, 4),
                'of_all_botanical': round(rate_of_all, 4),
            },
        }

class CorrelationStatistics:
    """Statistical significance tests for illustration-text correlation."""

    def __init__(
        self,
        correlator: IllustrationTextCorrelator,
        correlation_results: Dict,
    ):
        self.correlator = correlator
        self.results = correlation_results

    def _testable_folios(self) -> List[str]:
        """Get list of testable folio IDs."""
        return [
            fid for fid, r in self.results['per_folio'].items()
            if r['testable'] and r['has_decoded_text']
        ]

    def _p_folio(self, folio_id: str) -> float:
        """Probability of random match for a single folio.

        p = 1 - (1 - n_names/V)^W
        """
        entry = self.correlator.folio_map[folio_id]
        n_names = len(entry['latin_names'])
        V = len(self.correlator.corpus_vocab)
        if V == 0 or n_names == 0:
            return 0.0

        r = self.results['per_folio'][folio_id]
        W = r['resolved_word_count']
        if W == 0:
            return 0.0

        p_no_match = (1.0 - n_names / V) ** W
        return 1.0 - p_no_match

    def binomial_test(self) -> Dict:
        """Binomial test: are matches more frequent than random chance?

        Uses average per-folio match probability as the null rate.
        """
        testable = self._testable_folios()
        if not testable:
            return {
                'n_testable': 0, 'observed_matches': 0,
                'p_value': 1.0, 'significant_005': False,
            }

        observed = sum(
            1 for fid in testable
            if self.results['per_folio'][fid]['any_match']
        )

        p_values = [self._p_folio(fid) for fid in testable]
        avg_p = sum(p_values) / len(p_values)
        expected = avg_p * len(testable)
        n = len(testable)

        p_value = _binomial_sf(observed, n, avg_p)

        return {
            'n_testable': n,
            'observed_matches': observed,
            'expected_matches_random': round(expected, 2),
            'avg_p_random': round(avg_p, 4),
            'p_value': round(p_value, 6),
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01,
        }

    def permutation_test(
        self, n_permutations: int = 10000, seed: int = 42,
    ) -> Dict:
        """Permutation test: shuffle folio-text mapping, count matches.

        Null hypothesis: decoded text is unrelated to illustrations.
        Directly answers: if decoded text were randomly assigned to folios,
        how often would we see this many matches?
        """
        testable = self._testable_folios()
        if not testable:
            return {
                'observed_matches': 0, 'p_value': 1.0,
                'significant_005': False,
            }

        observed = sum(
            1 for fid in testable
            if self.results['per_folio'][fid]['any_match']
        )

        all_texts = list(self.correlator.translations.values())
        rng = random.Random(seed)
        null_counts = []

        for _ in range(n_permutations):
            shuffled = rng.sample(all_texts, len(testable))
            shuffled_trans = dict(zip(testable, shuffled))

            count = 0
            for fid, text in shuffled_trans.items():
                entry = self.correlator.folio_map[fid]
                words = IllustrationTextCorrelator._extract_resolved_words(text)
                word_set = set(words)

                matched = False
                for name in entry['single_word_names']:
                    if name in word_set:
                        matched = True
                        break
                if not matched:
                    for name in entry['multi_word_names']:
                        if IllustrationTextCorrelator._check_bigram(words, name):
                            matched = True
                            break
                if not matched:
                    for name in entry['single_word_names']:
                        if IllustrationTextCorrelator._check_stem(word_set, name):
                            matched = True
                            break
                if matched:
                    count += 1

            null_counts.append(count)

        p_value = sum(1 for c in null_counts if c >= observed) / n_permutations
        null_sorted = sorted(null_counts)

        return {
            'observed_matches': observed,
            'n_permutations': n_permutations,
            'null_mean': round(sum(null_counts) / len(null_counts), 2),
            'null_median': null_sorted[len(null_sorted) // 2],
            'null_max': max(null_counts),
            'null_95th': null_sorted[int(0.95 * len(null_sorted))],
            'p_value': round(p_value, 6),
            'significant_005': p_value < 0.05,
        }

def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Pure Python, no scipy needed."""
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1:
        return 1.0
    total = 0.0
    for i in range(k):
        pmf = comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        total += pmf
    return max(0.0, 1.0 - total)

def _print_report(
    results: Dict, binomial: Dict, permutation: Dict, verbose: bool = True,
) -> None:
    """Print formatted console report."""
    s = results['summary']
    r = results['rates']

    print(f'\n  Data:')
    print(f'    Botanical folios:     {s["total_botanical_folios"]}')
    print(f'    Testable:             {s["testable_folios"]}')
    print(f'    New World excluded:   {s["new_world_excluded"]}')
    print(f'    No decoded text:      {s["no_decoded_text"]}')

    print(f'\n  Results (of {s["testable_folios"]} testable folios):')
    print(f'    Exact matches:        {s["match_by_type"]["exact"]}')
    print(f'    Stem matches:         {s["match_by_type"]["stem"]}')
    print(f'    Bigram matches:       {s["match_by_type"]["bigram"]}')
    print(f'    No match:             {s["no_match_folios"]}')
    print(f'    Combined match rate:  {r["of_testable"]:.1%}')

    print(f'\n  Statistical significance:')
    print(f'    Chance baseline:      {binomial["avg_p_random"]:.1%}')
    print(f'    Expected (random):    {binomial["expected_matches_random"]}')
    print(f'    Observed:             {binomial["observed_matches"]}')
    print(f'    Binomial p-value:     {binomial["p_value"]:.6f}')
    print(f'    Permutation p-value:  {permutation["p_value"]:.6f} '
          f'({permutation["n_permutations"]} trials)')
    sig = binomial['significant_005'] or permutation['significant_005']
    verdict = 'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'
    print(f'    -> {verdict} at p < 0.05')

    matches = [
        (fid, r) for fid, r in results['per_folio'].items()
        if r['any_match']
    ]
    if matches:
        print(f'\n  Matches:')
        for fid, r in matches:
            names = r['exact_matches'] or list(r['stem_matches'].keys()) or r['bigram_matches']
            name_str = ', '.join(names)
            sci = ', '.join(r['species'])
            print(f'    + {fid}: {r["common"]} ({sci}) <-> {name_str} '
                  f'[{r["match_type"]}, {r["source"]}]')

    non_matches = [
        (fid, r) for fid, r in results['per_folio'].items()
        if r['testable'] and r['has_decoded_text'] and not r['any_match']
    ]
    if non_matches and verbose:
        print(f'\n  Non-matches ({len(non_matches)} testable folios):')
        for fid, r in non_matches:
            sci = ', '.join(r['species'])
            print(f'    - {fid}: {r["common"]} ({sci}) '
                  f'— searched: {", ".join(r["searched_names"][:5])}')

    nw = [
        (fid, r) for fid, r in results['per_folio'].items()
        if r['new_world']
    ]
    if nw:
        print(f'\n  New World (excluded):')
        for fid, r in nw:
            print(f'    ~ {fid}: {r["common"]} ({", ".join(r["species"])})')

def run_illustration_correlation(
    phase12_data: Dict,
    output_dir: str,
    n_permutations: int = 10000,
    verbose: bool = True,
) -> Dict:
    """Module 13.5: Illustration-Text Correlation.

    Args:
        phase12_data: Phase 12/13 data with 'final_translations' key
        output_dir: Directory for output JSON
        n_permutations: Number of permutation test iterations
        verbose: Print progress and results

    Returns:
        Metrics dict for inclusion in Phase 13 results.
    """
    from data.botanical_name_mapping import build_folio_name_map

    folio_map = build_folio_name_map()
    final_translations = phase12_data.get('final_translations', {})

    if verbose:
        print(f'  -> Loaded {len(folio_map)} botanical folios, '
              f'{len(final_translations)} decoded folios')

    correlator = IllustrationTextCorrelator(folio_map, final_translations)
    results = correlator.correlate_all()

    if verbose:
        testable = results['summary']['testable_folios']
        matched = results['summary']['matched_folios']
        print(f'  -> Correlation: {matched}/{testable} testable folios matched')

    stats = CorrelationStatistics(correlator, results)
    binomial = stats.binomial_test()

    if verbose:
        print(f'  -> Running permutation test ({n_permutations} trials)...')
    permutation = stats.permutation_test(n_permutations=n_permutations)

    output = {
        'correlation': results,
        'binomial_test': binomial,
        'permutation_test': permutation,
    }

    output_path = os.path.join(output_dir, 'illustration_correlation.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    if verbose:
        _print_report(results, binomial, permutation, verbose=verbose)
        print(f'\n  -> Saved: {output_path}')

    return {
        'matched_folios': results['summary']['matched_folios'],
        'testable_folios': results['summary']['testable_folios'],
        'match_rate_testable': results['rates']['of_testable'],
        'match_by_type': results['summary']['match_by_type'],
        'binomial_p': binomial['p_value'],
        'permutation_p': permutation['p_value'],
        'output_path': output_path,
    }
