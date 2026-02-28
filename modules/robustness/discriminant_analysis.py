"""
Discriminant Analysis: Which metrics separate real Voynich from null text?
=========================================================================
For each of three null text types (within-folio shuffle, cross-folio
shuffle, character-level random), runs the full Phase 12 pipeline and
then collects Phase 14 semantic coherence metrics, Phase 13 illustration
correlation counts, and cross-folio consistency significance tests.
Compares real-text metrics against null-text metrics to determine which
measurements actually discriminate.

Metrics collected per condition (real + 3 nulls):
  - Phase 14: medical_rate, entropy, template_coverage, collocation_plausible
  - Phase 13: match_count, match_rate, binomial_p
  - Consistency: forward significant (p<0.01), bidirectional unique source
  - Content: resolution_rate, narrow medical_vocab_rate, function_word_frac,
             unique_resolved_types, mean_skeleton_segments

Usage:
  uv run cli.py --robustness discriminant
"""

import random
import time
from collections import Counter, defaultdict
from typing import Dict, List

from orchestrators.robustness import run_full_pipeline, _compute_content_metrics


class DiscriminantAnalysis:
    """Determines which metrics discriminate real Voynich from null text."""

    EFFECT_THRESHOLD = 0.20

    NULL_TYPES = [
        'within_folio_shuffle',
        'cross_folio_shuffle',
        'char_random',
    ]

    HIGHER_IS_BETTER = {
        'phase14_medical_rate',
        'phase14_template_coverage',
        'phase14_collocation_plausible',
        'phase13_matches',
        'phase13_match_rate',
        'consistency_forward_sig_p001',
        'consistency_bidir_unique_source',
        'resolution_rate',
        'content_medical_vocab_rate',
        'content_function_word_frac',
        'content_unique_resolved_types',
        'content_mean_skeleton_segments',
    }

    LOWER_IS_BETTER = {
        'phase14_entropy',
    }

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, seed: int = 42) -> Dict:
        """Run discriminant analysis across 3 null types.

        Returns structured dict with real_metrics, null_metrics, and
        per-metric discrimination assessment.
        """
        t0 = time.time()
        by_folio = self.components['by_folio']

        # Step 1: Real Voynich metrics
        if self.verbose:
            print('  Running real Voynich pipeline...')
        real_result = run_full_pipeline(
            self.components, verbose=False, compute_content_metrics=True,
        )
        real_translations = real_result['final_translations']
        real_metrics = self._collect_all_metrics(real_translations, by_folio)
        real_metrics['resolution_rate'] = real_result['overall_resolution']
        for k, v in real_result.get('content_metrics', {}).items():
            real_metrics[f'content_{k}'] = v

        # Step 2: Null type metrics
        null_metrics = {}
        for null_type in self.NULL_TYPES:
            if self.verbose:
                print(f'  Running {null_type} pipeline...')
            rng = random.Random(seed)
            null_folios = self._generate_null(null_type, by_folio, rng)
            null_result = run_full_pipeline(
                self.components, by_folio_override=null_folios,
                verbose=False, compute_content_metrics=True,
            )
            null_trans = null_result['final_translations']
            nm = self._collect_all_metrics(null_trans, null_folios)
            nm['resolution_rate'] = null_result['overall_resolution']
            for k, v in null_result.get('content_metrics', {}).items():
                nm[f'content_{k}'] = v
            null_metrics[null_type] = nm

        # Step 3: Assess discrimination
        assessment = self._assess_discrimination(real_metrics, null_metrics)

        elapsed = time.time() - t0
        result = {
            'test': 'discriminant_analysis',
            'real_metrics': {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in real_metrics.items()
            },
            'null_metrics': {
                nt: {
                    k: round(v, 6) if isinstance(v, float) else v
                    for k, v in nm.items()
                }
                for nt, nm in null_metrics.items()
            },
            'assessment': assessment,
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_report(result)

        return result

    # ── Metric collection ──────────────────────────────────────────

    def _collect_all_metrics(
        self,
        final_translations: Dict[str, str],
        by_folio: Dict[str, List[str]],
    ) -> Dict:
        """Collect Phase 14, Phase 13, and consistency metrics."""
        metrics = {}

        p14 = self._collect_phase14_metrics(final_translations)
        metrics['phase14_medical_rate'] = p14['medical_rate']
        metrics['phase14_entropy'] = p14['entropy']
        metrics['phase14_template_coverage'] = p14['template_coverage']
        metrics['phase14_collocation_plausible'] = p14['collocation_plausible']

        p13 = self._collect_phase13_metrics(final_translations)
        metrics['phase13_matches'] = p13['matches']
        metrics['phase13_match_rate'] = p13['match_rate']
        metrics['phase13_binomial_p'] = p13['binomial_p']

        con = self._collect_consistency_metrics(final_translations, by_folio)
        metrics['consistency_forward_sig_p001'] = con['forward_sig_p001']
        metrics['consistency_forward_total'] = con['forward_total']
        metrics['consistency_bidir_unique_source'] = con['bidir_unique_source']
        metrics['consistency_bidir_total'] = con['bidir_total']

        return metrics

    def _collect_phase14_metrics(
        self, final_translations: Dict[str, str],
    ) -> Dict:
        """Run SemanticCoherenceAnalyzer.analyze_all() — fast, no significance."""
        from modules.phase14.semantic_coherence import SemanticCoherenceAnalyzer

        per_folio_stats = self.components['folio_metadata']
        analyzer = SemanticCoherenceAnalyzer(final_translations, per_folio_stats)
        analysis = analyzer.analyze_all()

        if 'error' in analysis:
            return {
                'medical_rate': 0.0,
                'entropy': 1.0,
                'template_coverage': 0.0,
                'collocation_plausible': 0.0,
            }

        s = analysis['summary']
        return {
            'medical_rate': s['overall_medical_rate'],
            'entropy': s['overall_entropy'],
            'template_coverage': s['overall_template_coverage'],
            'collocation_plausible': s['overall_collocation_plausible'],
        }

    def _collect_phase13_metrics(
        self, final_translations: Dict[str, str],
    ) -> Dict:
        """Run IllustrationTextCorrelator — fast binomial test only."""
        from data.botanical_name_mapping import build_folio_name_map
        from modules.phase13.illustration_correlation import (
            IllustrationTextCorrelator, CorrelationStatistics,
        )

        folio_map = build_folio_name_map()
        correlator = IllustrationTextCorrelator(folio_map, final_translations)
        results = correlator.correlate_all()

        stats = CorrelationStatistics(correlator, results)
        binomial = stats.binomial_test()

        return {
            'matches': results['summary']['matched_folios'],
            'testable': results['summary']['testable_folios'],
            'match_rate': results['rates']['of_testable'],
            'binomial_p': binomial['p_value'],
        }

    def _collect_consistency_metrics(
        self,
        final_translations: Dict[str, str],
        by_folio: Dict[str, List[str]],
    ) -> Dict:
        """Compute forward and bidirectional consistency inline (no disk I/O).

        Reimplements the core logic from ConsistencySignificance.run(),
        accepting translations directly as a parameter.
        """
        from modules.robustness.consistency_significance import _binomial_p_value

        fuzzy_skel = self.components['fuzzy_skel']
        latin_skel = self.components['latin_skel']

        skeleton_folio_words = defaultdict(lambda: defaultdict(list))
        word_skeletons = defaultdict(list)

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
                skeleton_folio_words[primary_skeleton][folio].append(
                    decoded_word,
                )
                word_skeletons[decoded_word].append(primary_skeleton)

        # Forward consistency: skeleton→word p-values
        sig_001 = 0
        total_mappings = 0
        for skeleton, folio_words in skeleton_folio_words.items():
            m_folios = len(folio_words)
            if m_folios < 2:
                continue
            total_mappings += 1

            word_counter = Counter()
            for words in folio_words.values():
                most_common_word = Counter(words).most_common(1)[0][0]
                word_counter[most_common_word] += 1

            _best_word, k_agreements = word_counter.most_common(1)[0]
            n_candidates = max(
                1, len(latin_skel.skeleton_index.get(skeleton, [])),
            )
            p_value = _binomial_p_value(n_candidates, k_agreements, m_folios)
            if p_value < 0.01:
                sig_001 += 1

        # Bidirectional: words with unique source skeleton
        unique_source = 0
        total_bidir = 0
        for word, skels in word_skeletons.items():
            if len(skels) < 3:
                continue
            total_bidir += 1
            if len(set(skels)) == 1:
                unique_source += 1

        return {
            'forward_sig_p001': sig_001,
            'forward_total': total_mappings,
            'bidir_unique_source': unique_source,
            'bidir_total': total_bidir,
        }

    # ── Null text generation ───────────────────────────────────────

    def _generate_null(
        self,
        null_type: str,
        by_folio: Dict[str, List[str]],
        rng: random.Random,
    ) -> Dict[str, List[str]]:
        """Generate null tokens by delegating to MultipleBaselines generators."""
        from modules.robustness.multiple_baselines import (
            MultipleBaselines, _build_eva_char_weights,
        )

        baselines = MultipleBaselines(self.components, verbose=False)

        all_tokens = []
        for tokens in by_folio.values():
            all_tokens.extend(tokens)

        if null_type == 'within_folio_shuffle':
            return baselines._gen_shuffled_tokens(by_folio, rng, None)
        elif null_type == 'cross_folio_shuffle':
            return baselines._gen_cross_folio_shuffle(
                by_folio, rng, all_tokens,
            )
        elif null_type == 'char_random':
            char_weights = _build_eva_char_weights(all_tokens)
            return baselines._gen_char_random(by_folio, rng, char_weights)
        else:
            raise ValueError(f'Unknown null type: {null_type}')

    # ── Discrimination assessment ──────────────────────────────────

    def _assess_discrimination(
        self, real_metrics: Dict, null_metrics: Dict[str, Dict],
    ) -> Dict:
        """Compare real vs each null type, compute effect sizes and verdicts."""
        per_metric = {}

        for metric, real_val in real_metrics.items():
            if not isinstance(real_val, (int, float)):
                continue

            per_null = {}
            disc_count = 0

            for null_type, nm in null_metrics.items():
                null_val = nm.get(metric)
                if not isinstance(null_val, (int, float)):
                    per_null[null_type] = {
                        'null_value': null_val,
                        'discriminates': False,
                        'reason': 'non_numeric',
                    }
                    continue

                diff = real_val - null_val
                denom = max(abs(real_val), abs(null_val), 1e-6)
                effect = abs(diff) / denom

                if metric in self.HIGHER_IS_BETTER:
                    correct_direction = diff > 0
                elif metric in self.LOWER_IS_BETTER:
                    correct_direction = diff < 0
                else:
                    correct_direction = True

                disc = effect >= self.EFFECT_THRESHOLD and correct_direction
                if disc:
                    disc_count += 1

                per_null[null_type] = {
                    'null_value': (
                        round(null_val, 6) if isinstance(null_val, float)
                        else null_val
                    ),
                    'difference': round(diff, 6),
                    'effect_size': round(effect, 4),
                    'correct_direction': correct_direction,
                    'discriminates': disc,
                }

            total_null_types = len(null_metrics)
            if disc_count == total_null_types:
                verdict = 'YES'
            elif disc_count > 0:
                verdict = 'WEAK'
            else:
                verdict = 'NO'

            per_metric[metric] = {
                'real_value': (
                    round(real_val, 6) if isinstance(real_val, float)
                    else real_val
                ),
                'per_null_type': per_null,
                'verdict': verdict,
            }

        yes_metrics = [m for m, r in per_metric.items() if r['verdict'] == 'YES']
        weak_metrics = [m for m, r in per_metric.items() if r['verdict'] == 'WEAK']
        no_metrics = [m for m, r in per_metric.items() if r['verdict'] == 'NO']

        per_metric['_summary'] = {
            'discriminating': yes_metrics,
            'weakly_discriminating': weak_metrics,
            'non_discriminating': no_metrics,
            'counts': {
                'yes': len(yes_metrics),
                'weak': len(weak_metrics),
                'no': len(no_metrics),
                'total': len(yes_metrics) + len(weak_metrics) + len(no_metrics),
            },
        }

        return per_metric

    # ── Console report ─────────────────────────────────────────────

    def _print_report(self, result: Dict) -> None:
        """Print formatted discrimination comparison table."""
        real = result['real_metrics']
        nulls = result['null_metrics']
        assessment = result['assessment']
        summary = assessment.get('_summary', {})

        print(f'\nDiscriminant Analysis')
        print('=' * 78)

        display_order = [
            ('phase14_medical_rate',          'Phase14: Medical Rate',       True),
            ('phase14_entropy',               'Phase14: Entropy',            False),
            ('phase14_template_coverage',     'Phase14: Template Coverage',  True),
            ('phase14_collocation_plausible', 'Phase14: Collocation',        True),
            ('phase13_matches',               'Phase13: Match Count',        False),
            ('phase13_match_rate',            'Phase13: Match Rate',         True),
            ('consistency_forward_sig_p001',  'Consistency: Sig p<0.01',     False),
            ('consistency_bidir_unique_source', 'Consistency: UniqueSource', False),
            ('resolution_rate',               'Resolution Rate',             True),
            ('content_medical_vocab_rate',    'Content: Med Vocab (narrow)', True),
            ('content_function_word_frac',    'Content: Function Words',     True),
            ('content_unique_resolved_types', 'Content: Unique Types',       False),
        ]

        null_keys = [
            'within_folio_shuffle', 'cross_folio_shuffle', 'char_random',
        ]

        print(f'\n  {"Metric":<32s} {"Real":>8s} '
              f'{"Shuffle":>8s} {"XFolio":>8s} '
              f'{"CharRnd":>8s} {"Disc?":>6s}')
        print('  ' + '-' * 74)

        for metric_key, label, is_pct in display_order:
            if metric_key not in real:
                continue

            real_val = real[metric_key]
            info = assessment.get(metric_key, {})
            verdict = info.get('verdict', '?')

            def _fmt(val, pct=is_pct):
                if isinstance(val, int):
                    return f'{val:>8d}'
                if pct and isinstance(val, float):
                    return f'{100 * val:>7.1f}%'
                if isinstance(val, float):
                    return f'{val:>8.4f}'
                return f'{str(val):>8s}'

            null_strs = []
            for nk in null_keys:
                nv = nulls.get(nk, {}).get(metric_key, '-')
                null_strs.append(_fmt(nv))

            print(f'  {label:<32s} {_fmt(real_val)} '
                  f'{null_strs[0]} {null_strs[1]} '
                  f'{null_strs[2]} {verdict:>6s}')

        print('  ' + '-' * 74)

        counts = summary.get('counts', {})
        print(f'\n  Discriminating:       '
              f'{counts.get("yes", 0)}/{counts.get("total", 0)} metrics')
        if counts.get('weak', 0) > 0:
            print(f'  Weakly discrim:       '
                  f'{counts["weak"]}/{counts["total"]} metrics')
            for m in summary.get('weakly_discriminating', []):
                print(f'    - {m}')
        if summary.get('non_discriminating'):
            print(f'  Non-discriminating:   '
                  f'{counts["no"]}/{counts["total"]} metrics')
            for m in summary['non_discriminating']:
                print(f'    - {m}')

        print(f'\n  Elapsed: {result["elapsed_seconds"]:.1f}s')
