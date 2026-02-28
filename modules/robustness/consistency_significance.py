"""
Test 4c: Consistency Significance
===================================
For each cross-folio consistent skeleton->word mapping, computes a p-value
to assess statistical significance. Also checks bidirectional consistency
(word -> skeleton).
"""
import json
import os
import re
from collections import Counter, defaultdict
from math import comb
from typing import Dict, List, Tuple


class ConsistencySignificance:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        fuzzy_skel = self.components['fuzzy_skel']
        latin_skel = self.components['latin_skel']
        by_folio = self.components['by_folio']
        folio_metadata = self.components['folio_metadata']

        # Load Phase 12 output
        phase12_path = os.path.join('./output/phase12', 'phase12_reconstruction.json')
        with open(phase12_path) as f:
            phase12_data = json.load(f)
        final_translations = phase12_data['final_translations']

        # Collect skeleton -> word mappings per folio
        skeleton_folio_words = defaultdict(lambda: defaultdict(list))
        word_skeletons = defaultdict(list)

        for folio, tokens in by_folio.items():
            if folio not in final_translations:
                continue
            decoded_words = final_translations[folio].split()
            if len(tokens) != len(decoded_words):
                continue

            for voynich_token, decoded_word in zip(tokens, decoded_words):
                is_bracket = (decoded_word.startswith('[') or
                              decoded_word.startswith('<'))
                if is_bracket:
                    continue

                stem = voynich_token.split('_')[0]
                skeleton_candidates = fuzzy_skel.get_skeleton_candidates(stem)
                if not skeleton_candidates:
                    continue

                primary_skeleton = skeleton_candidates[0][0]
                skeleton_folio_words[primary_skeleton][folio].append(decoded_word)
                word_skeletons[decoded_word].append(primary_skeleton)

        # Forward consistency: skeleton -> word p-values
        forward_results = []
        for skeleton, folio_words in skeleton_folio_words.items():
            # Count how many folios have this skeleton resolved
            m_folios = len(folio_words)
            if m_folios < 2:
                continue

            # Count word agreements across folios
            word_counter = Counter()
            for folio, words in folio_words.items():
                # Most common word for this skeleton on this folio
                folio_counter = Counter(words)
                most_common_word = folio_counter.most_common(1)[0][0]
                word_counter[most_common_word] += 1

            best_word, k_agreements = word_counter.most_common(1)[0]
            agreement_rate = k_agreements / m_folios

            # Count dictionary candidates for this skeleton
            n_candidates = 1
            if skeleton in latin_skel.skeleton_index:
                n_candidates = max(1, len(latin_skel.skeleton_index[skeleton]))

            # Compute p-value: P(X >= k) where X ~ Binomial(m, 1/n)
            p_value = _binomial_p_value(n_candidates, k_agreements, m_folios)

            forward_results.append({
                'skeleton': skeleton,
                'word': best_word,
                'folios': m_folios,
                'agreements': k_agreements,
                'agreement_rate': round(agreement_rate, 4),
                'candidates': n_candidates,
                'p_value': p_value,
            })

        forward_results.sort(key=lambda x: x['p_value'])

        # Significance counts
        sig_001 = sum(1 for r in forward_results if r['p_value'] < 0.01)
        sig_005 = sum(1 for r in forward_results if r['p_value'] < 0.05)
        total_mappings = len(forward_results)

        # Bidirectional consistency: word -> skeleton
        bidirectional_results = []
        for word, skeletons in word_skeletons.items():
            if len(skeletons) < 3:
                continue
            skel_counter = Counter(skeletons)
            unique_skeletons = len(skel_counter)
            most_common_skel, most_common_count = skel_counter.most_common(1)[0]
            consistency_rate = most_common_count / len(skeletons)

            bidirectional_results.append({
                'word': word,
                'total_occurrences': len(skeletons),
                'unique_skeletons': unique_skeletons,
                'primary_skeleton': most_common_skel,
                'primary_fraction': round(consistency_rate, 4),
            })

        bidirectional_results.sort(key=lambda x: -x['primary_fraction'])

        unique_source = sum(1 for r in bidirectional_results if r['unique_skeletons'] == 1)
        two_source = sum(1 for r in bidirectional_results if r['unique_skeletons'] == 2)
        three_plus = sum(1 for r in bidirectional_results if r['unique_skeletons'] >= 3)

        # Combined significance: forward significant AND bidirectional unique
        sig_forward_set = {r['skeleton'] for r in forward_results if r['p_value'] < 0.01}
        unique_bidir_words = {r['word'] for r in bidirectional_results if r['unique_skeletons'] == 1}
        # Find mappings that are significant in both directions
        both_significant = 0
        for r in forward_results:
            if r['p_value'] < 0.01 and r['word'] in unique_bidir_words:
                both_significant += 1

        result = {
            'test': 'consistency_significance',
            'forward_consistency': {
                'total_mappings': total_mappings,
                'significant_p001': sig_001,
                'significant_p005': sig_005,
                'top_mappings': forward_results[:20],
                'all_mappings': forward_results,
            },
            'bidirectional_consistency': {
                'total_words': len(bidirectional_results),
                'unique_source_skeleton': unique_source,
                'two_source_skeletons': two_source,
                'three_plus_source_skeletons': three_plus,
                'strong_examples': [r for r in bidirectional_results
                                    if r['unique_skeletons'] == 1][:10],
                'weak_examples': [r for r in bidirectional_results
                                  if r['unique_skeletons'] >= 3][:10],
            },
            'combined_significance': {
                'both_directions_significant': both_significant,
            },
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _print_report(self, result):
        fwd = result['forward_consistency']
        bidir = result['bidirectional_consistency']

        print('\nCross-Folio Consistency Significance')
        print('=' * 70)

        print(f'\nForward consistency (skeleton -> word):')
        print(f'  Significant mappings (p < 0.01): {fwd["significant_p001"]} / {fwd["total_mappings"]}')
        print(f'  Significant mappings (p < 0.05): {fwd["significant_p005"]} / {fwd["total_mappings"]}')
        print()
        print(f'  {"Skeleton":<15} {"Word":<15} {"Folios":>8} {"Agree":>7} '
              f'{"Cands":>7} {"p-value":>12}')
        print('  ' + '-' * 66)
        for r in fwd['top_mappings'][:15]:
            print(f'  {r["skeleton"]:<15} {r["word"]:<15} '
                  f'{r["folios"]:>8} {r["agreements"]:>7} '
                  f'{r["candidates"]:>7} {r["p_value"]:>12.6f}')

        print(f'\nBidirectional consistency (word -> skeleton):')
        print(f'  Words with unique source skeleton: '
              f'{bidir["unique_source_skeleton"]} / {bidir["total_words"]}')
        print(f'  Words with 2 source skeletons:     '
              f'{bidir["two_source_skeletons"]} / {bidir["total_words"]}')
        print(f'  Words with 3+ source skeletons:    '
              f'{bidir["three_plus_source_skeletons"]} / {bidir["total_words"]}')

        if bidir['strong_examples']:
            print(f'\n  Strong bidirectional consistency:')
            for r in bidir['strong_examples'][:5]:
                print(f'    {r["word"]} <- {r["primary_skeleton"]} '
                      f'(100% of {r["total_occurrences"]} occurrences)')

        if bidir['weak_examples']:
            print(f'\n  Weak bidirectional consistency:')
            for r in bidir['weak_examples'][:5]:
                print(f'    {r["word"]} <- {r["primary_skeleton"]} '
                      f'({100 * r["primary_fraction"]:.0f}% of '
                      f'{r["total_occurrences"]} occurrences, '
                      f'{r["unique_skeletons"]} different skeletons)')

        combined = result['combined_significance']
        print(f'\nCombined significance:')
        print(f'  Mappings significant in BOTH directions: '
              f'{combined["both_directions_significant"]}')


def _binomial_p_value(n_candidates, k_agreements, m_folios):
    """P(X >= k) where X ~ Binomial(m, 1/n)."""
    if n_candidates <= 1:
        return 1.0
    p = 1.0 / n_candidates
    q = 1.0 - p
    p_value = 0.0
    for k in range(k_agreements, m_folios + 1):
        p_value += comb(m_folios, k) * (p ** k) * (q ** (m_folios - k))
    return min(1.0, p_value)
