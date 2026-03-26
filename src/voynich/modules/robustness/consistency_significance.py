"""
Test 4c: Consistency Significance
Test 4b: Bidirectional Consistency (Strengthened)
===================================================
Test 4c: For each cross-folio consistent skeleton->word mapping, computes a
p-value to assess statistical significance. Also checks bidirectional
consistency (word -> skeleton).

Test 4b: Extends 4c with reverse p-values (word -> skeleton direction),
identifies strongest bidirectional mappings, and quantifies cipher character.
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
        phase12_path = os.path.join('./results/phase12', 'phase12_reconstruction.json')
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


class BidirectionalConsistency:
    """Test 4b: Strengthened bidirectional consistency with reverse p-values.

    For each resolved Latin word appearing on 3+ folios:
    1. Collect all Voynich tokens that decoded to this word
    2. Compute each token's skeleton
    3. Count how often the most common skeleton produced this word
    4. Compute reverse p-value: given the global pool of N distinct
       skeleton types observed in the decoded output, what's the probability
       that one skeleton dominates K out of M occurrences of word W?

    The null hypothesis: if the cipher were random, each occurrence of W
    could come from any of the N globally-observed skeleton types. Skeleton
    dominance under this null follows Binomial(M, 1/N).
    """

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self) -> Dict:
        fuzzy_skel = self.components['fuzzy_skel']
        latin_skel = self.components['latin_skel']
        by_folio = self.components['by_folio']

        # Load Phase 12 output
        phase12_path = os.path.join('./results/phase12', 'phase12_reconstruction.json')
        with open(phase12_path) as f:
            phase12_data = json.load(f)
        final_translations = phase12_data['final_translations']

        # Collect observed word -> skeleton mappings from decoded output
        word_observed_skeletons = defaultdict(list)
        skeleton_to_words = defaultdict(list)
        all_observed_skeletons = set()

        for folio, tokens in by_folio.items():
            if folio not in final_translations:
                continue
            decoded_words = final_translations[folio].split()
            if len(tokens) != len(decoded_words):
                continue

            for voynich_token, decoded_word in zip(tokens, decoded_words):
                if decoded_word.startswith('[') or decoded_word.startswith('<'):
                    continue

                stem = voynich_token.split('_')[0]
                skeleton_candidates = fuzzy_skel.get_skeleton_candidates(stem)
                if not skeleton_candidates:
                    continue

                primary_skeleton = skeleton_candidates[0][0]
                word_observed_skeletons[decoded_word].append(primary_skeleton)
                skeleton_to_words[primary_skeleton].append(decoded_word)
                all_observed_skeletons.add(primary_skeleton)

        # Global pool size: how many distinct skeleton types appear
        n_global_skeletons = len(all_observed_skeletons)

        # Compute reverse p-values for each word with 3+ occurrences
        reverse_results = []
        for word, observed_skels in word_observed_skeletons.items():
            m_total = len(observed_skels)
            if m_total < 3:
                continue

            skel_counter = Counter(observed_skels)
            dominant_skel, k_dominant = skel_counter.most_common(1)[0]
            unique_observed = len(skel_counter)

            # Use the global pool of observed skeletons as N
            p_value = _binomial_p_value(n_global_skeletons, k_dominant, m_total)

            reverse_results.append({
                'word': word,
                'dominant_skeleton': dominant_skel,
                'occurrences': m_total,
                'dominant_count': k_dominant,
                'dominance_pct': round(100.0 * k_dominant / m_total, 1),
                'unique_skeletons': unique_observed,
                'global_skeleton_pool': n_global_skeletons,
                'p_value': p_value,
            })

        reverse_results.sort(key=lambda x: x['p_value'])

        rev_sig_001 = sum(1 for r in reverse_results if r['p_value'] < 0.01)
        rev_sig_005 = sum(1 for r in reverse_results if r['p_value'] < 0.05)

        # Also run forward consistency to get combined results
        forward_sig_words = self._get_forward_significant_words(
            fuzzy_skel, latin_skel, by_folio, final_translations,
        )

        # Combined: significant in BOTH forward AND reverse
        rev_sig_words = {r['word'] for r in reverse_results if r['p_value'] < 0.01}
        both_sig = forward_sig_words & rev_sig_words

        # Cipher character analysis
        one_to_one = 0       # skeleton <-> word (unique in both directions)
        many_to_one_fwd = 0  # multiple skeletons -> same word
        one_to_many_fwd = 0  # one skeleton -> multiple words
        for skel, words in skeleton_to_words.items():
            unique_words = set(words)
            if len(unique_words) == 1:
                word = unique_words.pop()
                unique_skels = set(word_observed_skeletons.get(word, []))
                if len(unique_skels) == 1:
                    one_to_one += 1
                else:
                    many_to_one_fwd += 1
            else:
                one_to_many_fwd += 1

        result = {
            'test': 'bidirectional_consistency',
            'reverse_consistency': {
                'total_words': len(reverse_results),
                'global_skeleton_pool': n_global_skeletons,
                'significant_p001': rev_sig_001,
                'significant_p005': rev_sig_005,
                'top_mappings': reverse_results[:20],
                'all_mappings': reverse_results,
            },
            'combined_bidirectional': {
                'forward_significant_p001': len(forward_sig_words),
                'reverse_significant_p001': len(rev_sig_words),
                'both_significant_p001': len(both_sig),
                'both_significant_words': sorted(both_sig)[:30],
            },
            'cipher_character': {
                'one_to_one': one_to_one,
                'many_to_one_forward': many_to_one_fwd,
                'one_to_many_forward': one_to_many_fwd,
            },
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _get_forward_significant_words(
        self, fuzzy_skel, latin_skel, by_folio, final_translations,
    ) -> set:
        """Run forward consistency analysis and return words with p < 0.01."""
        skeleton_folio_words = defaultdict(lambda: defaultdict(list))

        for folio, tokens in by_folio.items():
            if folio not in final_translations:
                continue
            decoded_words = final_translations[folio].split()
            if len(tokens) != len(decoded_words):
                continue
            for vt, dw in zip(tokens, decoded_words):
                if dw.startswith('[') or dw.startswith('<'):
                    continue
                stem = vt.split('_')[0]
                sc = fuzzy_skel.get_skeleton_candidates(stem)
                if not sc:
                    continue
                skeleton_folio_words[sc[0][0]][folio].append(dw)

        sig_words = set()
        for skeleton, folio_words in skeleton_folio_words.items():
            m = len(folio_words)
            if m < 2:
                continue
            wc = Counter()
            for words in folio_words.values():
                wc[Counter(words).most_common(1)[0][0]] += 1
            best_word, k = wc.most_common(1)[0]
            n = max(1, len(latin_skel.skeleton_index.get(skeleton, [])))
            if _binomial_p_value(n, k, m) < 0.01:
                sig_words.add(best_word)
        return sig_words

    def _print_report(self, result):
        rev = result['reverse_consistency']
        comb = result['combined_bidirectional']
        cipher = result['cipher_character']

        print('\nBidirectional Consistency (Extended)')
        print('=' * 70)

        print(f'\nReverse consistency (word -> skeleton):')
        print(f'  Total words analyzed (3+ occurrences): {rev["total_words"]}')
        print(f'  Global skeleton pool (null N):         {rev["global_skeleton_pool"]}')
        print(f'  Significant at p < 0.01:               {rev["significant_p001"]}')
        print(f'  Significant at p < 0.05:               {rev["significant_p005"]}')
        print()
        print(f'  {"Word":<18} {"Skeleton":<12} {"Occur":>7} {"Dom":>5} '
              f'{"Dom%":>6} {"Uniq":>6} {"p-value":>12}')
        print('  ' + '-' * 68)
        for r in rev['top_mappings'][:15]:
            print(f'  {r["word"]:<18} {r["dominant_skeleton"]:<12} '
                  f'{r["occurrences"]:>7} {r["dominant_count"]:>5} '
                  f'{r["dominance_pct"]:>5.1f}% {r["unique_skeletons"]:>6} '
                  f'{r["p_value"]:>12.6f}')

        print(f'\nCombined bidirectional significance:')
        print(f'  Forward significant (p<0.01):  {comb["forward_significant_p001"]}')
        print(f'  Reverse significant (p<0.01):  {comb["reverse_significant_p001"]}')
        print(f'  BOTH significant (p<0.01):     {comb["both_significant_p001"]}')

        print(f'\nCipher character:')
        total_skels = cipher['one_to_one'] + cipher['many_to_one_forward'] + cipher['one_to_many_forward']
        print(f'  One-to-one (skel <-> word):     {cipher["one_to_one"]}')
        print(f'  Many-to-one (N skel -> 1 word): {cipher["many_to_one_forward"]}')
        print(f'  One-to-many (1 skel -> N word): {cipher["one_to_many_forward"]}')
        if total_skels > 0:
            dominant = max(
                ('one-to-one', cipher['one_to_one']),
                ('many-to-one', cipher['many_to_one_forward']),
                ('one-to-many', cipher['one_to_many_forward']),
                key=lambda x: x[1],
            )
            print(f'  Dominant pattern: {dominant[0]} '
                  f'({100.0 * dominant[1] / total_skels:.0f}%)')


def _binomial_p_value(n_candidates, k_agreements, m_folios):
    """P(X >= k) where X ~ Binomial(m, 1/n).

    Uses log-space arithmetic to avoid overflow for large m values.
    """
    if n_candidates <= 1:
        return 1.0
    from math import lgamma, log, exp
    log_p = -log(n_candidates)
    log_q = log(1.0 - 1.0 / n_candidates)

    # Sum P(X = k) for k = k_agreements..m_folios in log space
    log_terms = []
    for k in range(k_agreements, m_folios + 1):
        log_binom = lgamma(m_folios + 1) - lgamma(k + 1) - lgamma(m_folios - k + 1)
        log_term = log_binom + k * log_p + (m_folios - k) * log_q
        log_terms.append(log_term)

    if not log_terms:
        return 1.0

    # Log-sum-exp for numerical stability
    max_log = max(log_terms)
    p_value = exp(max_log) * sum(exp(lt - max_log) for lt in log_terms)
    return min(1.0, max(0.0, p_value))
