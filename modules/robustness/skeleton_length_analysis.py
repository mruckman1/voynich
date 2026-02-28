"""
Test 3a: Resolution by Skeleton Length
=======================================
Breaks down resolution rate by consonant skeleton segment count to answer:
"Is the pipeline resolving specific, informative words, or just common short words?"
"""
import json
import os
import re
from collections import defaultdict
from typing import Dict


class SkeletonLengthAnalysis:

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

        # Buckets: segment_count -> lists of stats
        buckets = defaultdict(lambda: {
            'tokens': 0, 'resolved': 0, 'unresolved': 0,
            'candidate_counts': [], 'resolved_words': [],
            'high_ambiguity_resolved': 0,
        })

        for folio, tokens in by_folio.items():
            if folio not in final_translations:
                continue
            decoded_words = final_translations[folio].split()
            if len(tokens) != len(decoded_words):
                continue

            for voynich_token, decoded_word in zip(tokens, decoded_words):
                is_bracket = (decoded_word.startswith('[') or
                              decoded_word.startswith('<'))

                # Get skeleton
                skeleton_candidates = fuzzy_skel.get_skeleton_candidates(
                    voynich_token.split('_')[0]
                )
                if not skeleton_candidates:
                    seg_count = 0
                else:
                    primary_skeleton = skeleton_candidates[0][0]
                    seg_count = len(primary_skeleton.split('-'))

                # Count dictionary candidates for this skeleton
                n_candidates = 0
                if skeleton_candidates:
                    primary_skel = skeleton_candidates[0][0]
                    if primary_skel in latin_skel.skeleton_index:
                        n_candidates = len(latin_skel.skeleton_index[primary_skel])

                # Bucket key: clamp at 5+
                bucket_key = min(seg_count, 5)

                buckets[bucket_key]['tokens'] += 1
                buckets[bucket_key]['candidate_counts'].append(n_candidates)

                if is_bracket:
                    buckets[bucket_key]['unresolved'] += 1
                else:
                    buckets[bucket_key]['resolved'] += 1
                    buckets[bucket_key]['resolved_words'].append(decoded_word)
                    if n_candidates > 10:
                        buckets[bucket_key]['high_ambiguity_resolved'] += 1

        # Compute per-bucket stats
        results_table = []
        total_resolved = 0
        total_tokens = 0
        for seg in sorted(buckets.keys()):
            b = buckets[seg]
            rate = b['resolved'] / max(1, b['tokens'])
            avg_cands = (sum(b['candidate_counts']) /
                         max(1, len(b['candidate_counts'])))
            false_discovery = (b['high_ambiguity_resolved'] /
                               max(1, b['resolved']))

            # Top resolved words (by frequency)
            from collections import Counter
            word_counts = Counter(b['resolved_words'])
            top_words = [w for w, _ in word_counts.most_common(10)]

            label = f'{seg}' if seg < 5 else '5+'
            results_table.append({
                'segments': label,
                'tokens': b['tokens'],
                'resolved': b['resolved'],
                'unresolved': b['unresolved'],
                'resolution_rate': round(rate, 4),
                'avg_candidates': round(avg_cands, 1),
                'false_discovery_risk': round(false_discovery, 4),
                'top_resolved_words': top_words,
            })
            total_resolved += b['resolved']
            total_tokens += b['tokens']

        overall_rate = total_resolved / max(1, total_tokens)

        # Compute content word fraction
        short_resolved = sum(
            buckets[k]['resolved'] for k in buckets if k <= 2
        )
        long_resolved = sum(
            buckets[k]['resolved'] for k in buckets if k >= 3
        )
        short_frac = short_resolved / max(1, total_resolved)
        long_frac = long_resolved / max(1, total_resolved)

        result = {
            'test': 'skeleton_length_analysis',
            'buckets': results_table,
            'overall_resolution_rate': round(overall_rate, 4),
            'short_skeleton_fraction': round(short_frac, 4),
            'long_skeleton_fraction': round(long_frac, 4),
            'total_tokens': total_tokens,
            'total_resolved': total_resolved,
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _print_report(self, result):
        print('\nSkeleton Length Analysis')
        print('=' * 70)
        print()
        print(f'{"Segments":<10} {"Tokens":>8} {"Resolved":>10} {"Rate":>8} '
              f'{"Avg Cands":>10} {"FD Risk":>8}')
        print('-' * 70)
        for row in result['buckets']:
            print(f'{row["segments"]:<10} {row["tokens"]:>8} '
                  f'{row["resolved"]:>10} '
                  f'{100 * row["resolution_rate"]:>7.1f}% '
                  f'{row["avg_candidates"]:>10.1f} '
                  f'{100 * row["false_discovery_risk"]:>7.1f}%')
        print('-' * 70)
        print(f'{"Total":<10} {result["total_tokens"]:>8} '
              f'{result["total_resolved"]:>10} '
              f'{100 * result["overall_resolution_rate"]:>7.1f}%')

        print(f'\nResolved words by length:')
        for row in result['buckets']:
            words_str = ', '.join(row['top_resolved_words'][:8])
            print(f'  {row["segments"]}-segment: {words_str}')

        print(f'\nKey finding: {100 * result["short_skeleton_fraction"]:.1f}% of '
              f'resolutions come from 1-2 segment skeletons (common/function words)')
        print(f'             {100 * result["long_skeleton_fraction"]:.1f}% from '
              f'3+ segments (content words)')
