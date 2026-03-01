"""
Selectivity Analysis: Characterize skeleton matching selectivity + gated resolution.
====================================================================================
Phase 1 characterizes why skeleton matching is non-discriminative:
  1a  Skeleton Ambiguity Profile — candidate counts by segment length
  1b  Token Information Content  — bits discarded by skeletonization
  1c  Per-Length Null Selectivity — real vs null resolution per segment count

Phase 2 tests whether gating resolution to 3+ segments (content words only)
produces discriminative output by comparing real vs 3 null types.

Usage:
  uv run cli.py --robustness selectivity
"""

import math
import random
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Callable

from orchestrators.robustness import run_full_pipeline, _compute_content_metrics
import orchestrators._config as _cfg


# EVA alphabet size (for information content calculation)
EVA_ALPHABET_SIZE = 26  # approximate: a-z mapping of EVA characters
# Skeleton alphabet: consonant segments use a reduced set
SKELETON_ALPHABET_SIZE = 20  # approximate consonant inventory


class SelectivityAnalysis:
    """Characterize skeleton matching selectivity and test gated resolution."""

    NULL_TYPES = ['within_folio_shuffle', 'cross_folio_shuffle', 'char_random']

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, seed: int = 42) -> Dict:
        """Run full selectivity analysis (Phase 1 + Phase 2)."""
        t0 = time.time()
        results = {}

        # Phase 1a: Ambiguity profile
        if self.verbose:
            print(f'\n{"=" * 70}')
            print('PHASE 1a: Skeleton Ambiguity Profile')
            print('=' * 70)
        profile = self.skeleton_ambiguity_profile()
        if self.verbose:
            self.print_ambiguity_profile(profile)
        results['ambiguity_profile'] = profile

        # Phase 1b: Information content
        if self.verbose:
            print(f'\n{"=" * 70}')
            print('PHASE 1b: Token Information Content')
            print('=' * 70)
        info = self.token_information_content()
        if self.verbose:
            self.print_info_content(info)
        results['information_content'] = info

        # Phase 1c: Per-length null selectivity
        if self.verbose:
            print(f'\n{"=" * 70}')
            print('PHASE 1c: Per-Length Null Selectivity')
            print('=' * 70)
        selectivity = self.per_length_null_selectivity(n_trials=3, seed=seed)
        if self.verbose:
            self.print_selectivity(selectivity)
        results['per_length_selectivity'] = selectivity

        # Phase 2: Gated discriminant test at min_segments=3 and 4
        for min_seg in [3, 4]:
            if self.verbose:
                print(f'\n{"=" * 70}')
                print(f'PHASE 2: Gated Discriminant Test (min_segments={min_seg})')
                print('=' * 70)
            gated = self.gated_discriminant_test(
                min_segments=min_seg, n_null_trials=1, seed=seed,
            )
            if self.verbose:
                self.print_gated_discriminant(gated)
            results[f'gated_discriminant_min{min_seg}'] = gated

        elapsed = time.time() - t0
        results['elapsed_seconds'] = round(elapsed, 1)

        if self.verbose:
            print(f'\n  Total elapsed: {elapsed:.1f}s')

        return results

    # ── Phase 1a: Skeleton Ambiguity Profile ─────────────────────────

    def skeleton_ambiguity_profile(self) -> Dict:
        """For each segment count, compute ambiguity statistics."""
        latin_skel = self.components['latin_skel']
        skeleton_index = latin_skel.skeleton_index

        # Bucket skeletons by segment count
        buckets = defaultdict(list)  # seg_count -> [(skeleton, n_candidates)]
        for skeleton, words in skeleton_index.items():
            seg_count = len(skeleton.split('-')) if skeleton else 0
            seg_count = min(seg_count, 6)  # clamp 6+
            buckets[seg_count].append((skeleton, len(words)))

        rows = []
        for seg in sorted(buckets.keys()):
            entries = buckets[seg]
            n_skeletons = len(entries)
            cand_counts = [c for _, c in entries]
            avg_cands = sum(cand_counts) / n_skeletons if n_skeletons else 0
            match_any = sum(1 for c in cand_counts if c >= 1) / n_skeletons if n_skeletons else 0
            match_5plus = sum(1 for c in cand_counts if c >= 5) / n_skeletons if n_skeletons else 0
            max_cands = max(cand_counts) if cand_counts else 0

            rows.append({
                'segments': seg,
                'n_skeletons': n_skeletons,
                'avg_candidates': round(avg_cands, 1),
                'match_any_frac': round(match_any, 3),
                'match_5plus_frac': round(match_5plus, 3),
                'max_candidates': max_cands,
            })

        return {'rows': rows}

    def print_ambiguity_profile(self, profile: Dict) -> None:
        print(f'\n  {"Segments":<10s} {"Skeletons":>10s} {"Avg Cands":>10s} '
              f'{"Match≥1":>10s} {"Match≥5":>10s} {"Max Cands":>10s}')
        print('  ' + '-' * 60)
        for row in profile['rows']:
            seg_label = f'{row["segments"]}+' if row['segments'] == 6 else str(row['segments'])
            print(f'  {seg_label:<10s} {row["n_skeletons"]:>10d} '
                  f'{row["avg_candidates"]:>10.1f} '
                  f'{100*row["match_any_frac"]:>9.1f}% '
                  f'{100*row["match_5plus_frac"]:>9.1f}% '
                  f'{row["max_candidates"]:>10d}')

        # Summary
        for row in profile['rows']:
            if row['match_any_frac'] < 0.50:
                print(f'\n  -> Selectivity threshold: {row["segments"]}+ segments '
                      f'(match rate drops below 50%)')
                break

    # ── Phase 1b: Token Information Content ──────────────────────────

    def token_information_content(self) -> Dict:
        """Compute raw vs skeleton information content per token."""
        by_folio = self.components['by_folio']
        fuzzy_skel = self.components['fuzzy_skel']

        # Per-segment-count buckets
        buckets = defaultdict(lambda: {'tokens': 0, 'raw_bits': 0.0, 'skel_bits': 0.0})
        total_tokens = 0
        total_raw = 0.0
        total_skel = 0.0

        for tokens in by_folio.values():
            for token in tokens:
                raw_len = len(token)
                raw_bits = raw_len * math.log2(EVA_ALPHABET_SIZE) if raw_len > 0 else 0

                skeleton_candidates = fuzzy_skel.get_skeleton_candidates(token)
                if skeleton_candidates:
                    primary = skeleton_candidates[0][0]
                    seg_count = len(primary.split('-')) if primary else 0
                    skel_bits = seg_count * math.log2(SKELETON_ALPHABET_SIZE) if seg_count > 0 else 0
                else:
                    seg_count = 0
                    skel_bits = 0

                seg_key = min(seg_count, 5)  # clamp 5+
                buckets[seg_key]['tokens'] += 1
                buckets[seg_key]['raw_bits'] += raw_bits
                buckets[seg_key]['skel_bits'] += skel_bits
                total_tokens += 1
                total_raw += raw_bits
                total_skel += skel_bits

        preservation = total_skel / total_raw if total_raw > 0 else 0

        rows = []
        for seg in sorted(buckets.keys()):
            b = buckets[seg]
            n = b['tokens']
            avg_raw = b['raw_bits'] / n if n else 0
            avg_skel = b['skel_bits'] / n if n else 0
            pres = avg_skel / avg_raw if avg_raw > 0 else 0
            rows.append({
                'segments': seg,
                'tokens': n,
                'avg_raw_bits': round(avg_raw, 1),
                'avg_skel_bits': round(avg_skel, 1),
                'preserved_frac': round(pres, 3),
            })

        return {
            'total_tokens': total_tokens,
            'total_raw_bits': round(total_raw, 0),
            'total_skel_bits': round(total_skel, 0),
            'preservation_ratio': round(preservation, 3),
            'rows': rows,
        }

    def print_info_content(self, info: Dict) -> None:
        print(f'\n  Total tokens:        {info["total_tokens"]:,d}')
        print(f'  Total raw bits:      {info["total_raw_bits"]:,.0f}')
        print(f'  Total skeleton bits: {info["total_skel_bits"]:,.0f}')
        print(f'  Preservation ratio:  {100*info["preservation_ratio"]:.1f}%')
        print(f'  Discarded:           {100*(1-info["preservation_ratio"]):.1f}%')
        print()
        print(f'  {"Segments":<10s} {"Tokens":>10s} {"Avg Raw":>10s} '
              f'{"Avg Skel":>10s} {"Preserved":>10s}')
        print('  ' + '-' * 50)
        for row in info['rows']:
            seg_label = f'{row["segments"]}+' if row['segments'] == 5 else str(row['segments'])
            print(f'  {seg_label:<10s} {row["tokens"]:>10,d} '
                  f'{row["avg_raw_bits"]:>10.1f} '
                  f'{row["avg_skel_bits"]:>10.1f} '
                  f'{100*row["preserved_frac"]:>9.1f}%')

    # ── Phase 1c: Per-Length Null Selectivity ─────────────────────────

    def per_length_null_selectivity(
        self, n_trials: int = 3, seed: int = 42,
    ) -> Dict:
        """Compare real vs shuffled-token null resolution rate per segment count."""
        by_folio = self.components['by_folio']
        fuzzy_skel = self.components['fuzzy_skel']

        # Run real pipeline
        if self.verbose:
            print('  Running real pipeline...')
        real_result = run_full_pipeline(self.components, verbose=False)
        real_buckets = self._bucket_by_segments(
            real_result['final_translations'], by_folio, fuzzy_skel,
        )

        # Run null trials (shuffled tokens)
        null_buckets_list = []
        for trial in range(n_trials):
            if self.verbose:
                print(f'  Running null trial {trial+1}/{n_trials}...')
            rng = random.Random(seed + trial)
            null_folios = self._generate_null('within_folio_shuffle', by_folio, rng)
            null_result = run_full_pipeline(
                self.components, by_folio_override=null_folios, verbose=False,
            )
            nb = self._bucket_by_segments(
                null_result['final_translations'], null_folios, fuzzy_skel,
            )
            null_buckets_list.append(nb)

        # Average null buckets
        rows = []
        for seg in sorted(real_buckets.keys()):
            rb = real_buckets[seg]
            real_rate = rb['resolved'] / rb['total'] if rb['total'] > 0 else 0

            null_rates = []
            for nb_dict in null_buckets_list:
                if seg in nb_dict:
                    nb = nb_dict[seg]
                    null_rates.append(nb['resolved'] / nb['total'] if nb['total'] > 0 else 0)

            avg_null = sum(null_rates) / len(null_rates) if null_rates else 0
            diff = real_rate - avg_null
            selective = diff > 0.02  # >2pp positive differential

            rows.append({
                'segments': seg,
                'tokens': rb['total'],
                'real_rate': round(real_rate, 3),
                'null_rate': round(avg_null, 3),
                'diff': round(diff, 3),
                'selective': selective,
            })

        return {'rows': rows, 'n_trials': n_trials}

    def _bucket_by_segments(
        self,
        translations: Dict[str, str],
        by_folio: Dict[str, List[str]],
        fuzzy_skel,
    ) -> Dict[int, Dict]:
        """Bucket tokens by skeleton segment count, tracking resolved/total."""
        buckets = defaultdict(lambda: {'resolved': 0, 'total': 0})

        for folio, tokens in by_folio.items():
            if folio not in translations:
                continue
            decoded_words = translations[folio].split()

            # Handle length mismatch (pipeline may merge/split tokens)
            n = min(len(tokens), len(decoded_words))
            for idx in range(n):
                voynich_token = tokens[idx]
                decoded = decoded_words[idx]

                # Get segment count
                skel_cands = fuzzy_skel.get_skeleton_candidates(voynich_token)
                if skel_cands:
                    seg_count = len(skel_cands[0][0].split('-')) if skel_cands[0][0] else 0
                else:
                    seg_count = 0
                seg_key = min(seg_count, 5)  # clamp 5+

                is_resolved = not (decoded.startswith('[') or decoded.startswith('<'))
                buckets[seg_key]['total'] += 1
                if is_resolved:
                    buckets[seg_key]['resolved'] += 1

        return dict(buckets)

    def print_selectivity(self, selectivity: Dict) -> None:
        print(f'\n  {"Segments":<10s} {"Real":>8s} {"Null":>8s} '
              f'{"Diff":>8s} {"Selective?":>12s} {"Tokens":>8s}')
        print('  ' + '-' * 58)
        for row in selectivity['rows']:
            seg_label = f'{row["segments"]}+' if row['segments'] == 5 else str(row['segments'])
            sel_str = 'YES' if row['selective'] else 'no'
            print(f'  {seg_label:<10s} {100*row["real_rate"]:>7.1f}% '
                  f'{100*row["null_rate"]:>7.1f}% '
                  f'{100*row["diff"]:>+7.1f}% '
                  f'{sel_str:>12s} {row["tokens"]:>8,d}')

        # Key finding
        selective_segs = [r for r in selectivity['rows'] if r['selective']]
        if selective_segs:
            print(f'\n  -> Positive differential at segments: '
                  f'{", ".join(str(r["segments"]) for r in selective_segs)}')
        else:
            print(f'\n  -> No segment count shows positive differential '
                  f'(skeleton matching non-selective at all lengths)')

    # ── Phase 2: Gated Discriminant Test ─────────────────────────────

    def gated_discriminant_test(
        self,
        min_segments: int = 3,
        n_null_trials: int = 1,
        seed: int = 42,
    ) -> Dict:
        """Run pipeline with skeleton gate, compare real vs null types."""
        by_folio = self.components['by_folio']
        latin_skel = self.components['latin_skel']

        # Run real gated
        if self.verbose:
            print(f'  Running real pipeline (gated min_segments={min_segments})...')
        real_result = self._run_gated_pipeline(
            min_segments=min_segments, by_folio_override=None,
        )
        real_metrics = self._extract_gated_metrics(real_result, latin_skel)

        # Also record ungated baseline for reference
        if self.verbose:
            print('  Running real pipeline (ungated, for reference)...')
        ungated_result = run_full_pipeline(
            self.components, verbose=False, compute_content_metrics=True,
        )
        ungated_metrics = self._extract_gated_metrics(ungated_result, latin_skel)

        # Run null types
        null_metrics = {}
        for null_type in self.NULL_TYPES:
            if self.verbose:
                print(f'  Running {null_type} (gated min_segments={min_segments})...')
            rng = random.Random(seed)
            null_folios = self._generate_null(null_type, by_folio, rng)
            null_result = self._run_gated_pipeline(
                min_segments=min_segments, by_folio_override=null_folios,
            )
            null_metrics[null_type] = self._extract_gated_metrics(null_result, latin_skel)

        # Assess discrimination
        assessment = self._assess_discrimination(real_metrics, null_metrics)

        return {
            'min_segments': min_segments,
            'real_gated': real_metrics,
            'real_ungated': ungated_metrics,
            'null_metrics': null_metrics,
            'assessment': assessment,
        }

    def _run_gated_pipeline(
        self, min_segments: int, by_folio_override=None,
    ) -> Dict:
        """Run pipeline with temporary config override for skeleton gate."""
        old_val = _cfg.MIN_SKELETON_SEGMENTS_FOR_RESOLUTION
        _cfg.MIN_SKELETON_SEGMENTS_FOR_RESOLUTION = min_segments
        try:
            result = run_full_pipeline(
                self.components,
                by_folio_override=by_folio_override,
                verbose=False,
                compute_content_metrics=True,
            )
        finally:
            _cfg.MIN_SKELETON_SEGMENTS_FOR_RESOLUTION = old_val
        return result

    def _extract_gated_metrics(self, result: Dict, latin_skel) -> Dict:
        """Extract key metrics from a pipeline result."""
        metrics = {
            'resolution_rate': result['overall_resolution'],
        }
        for k, v in result.get('content_metrics', {}).items():
            metrics[f'content_{k}'] = v
        return metrics

    def _assess_discrimination(
        self, real_metrics: Dict, null_metrics: Dict[str, Dict],
    ) -> Dict:
        """Compare real vs each null type with effect size thresholds."""
        EFFECT_THRESHOLD = 0.20

        HIGHER_IS_BETTER = {
            'resolution_rate', 'content_medical_vocab_rate',
            'content_function_word_frac', 'content_unique_resolved_types',
            'content_mean_skeleton_segments',
        }

        per_metric = {}
        for metric, real_val in real_metrics.items():
            if not isinstance(real_val, (int, float)):
                continue

            disc_count = 0
            per_null = {}
            for null_type, nm in null_metrics.items():
                null_val = nm.get(metric)
                if not isinstance(null_val, (int, float)):
                    per_null[null_type] = {'discriminates': False, 'reason': 'non_numeric'}
                    continue

                diff = real_val - null_val
                denom = max(abs(real_val), abs(null_val), 1e-6)
                effect = abs(diff) / denom
                correct_dir = diff > 0 if metric in HIGHER_IS_BETTER else True
                disc = effect >= EFFECT_THRESHOLD and correct_dir
                if disc:
                    disc_count += 1

                per_null[null_type] = {
                    'null_value': round(null_val, 4) if isinstance(null_val, float) else null_val,
                    'diff': round(diff, 4),
                    'effect': round(effect, 4),
                    'discriminates': disc,
                }

            total = len(null_metrics)
            if disc_count == total:
                verdict = 'YES'
            elif disc_count > 0:
                verdict = 'WEAK'
            else:
                verdict = 'NO'

            per_metric[metric] = {
                'real_value': round(real_val, 4) if isinstance(real_val, float) else real_val,
                'per_null': per_null,
                'verdict': verdict,
            }

        # Summary
        yes_count = sum(1 for m in per_metric.values() if m.get('verdict') == 'YES')
        weak_count = sum(1 for m in per_metric.values() if m.get('verdict') == 'WEAK')
        no_count = sum(1 for m in per_metric.values() if m.get('verdict') == 'NO')

        if yes_count > 0:
            overall = 'DISCRIMINATIVE'
        elif weak_count > 0:
            overall = 'WEAKLY_DISCRIMINATIVE'
        else:
            overall = 'NON_DISCRIMINATIVE'

        per_metric['_summary'] = {
            'yes': yes_count,
            'weak': weak_count,
            'no': no_count,
            'overall_verdict': overall,
        }

        return per_metric

    def print_gated_discriminant(self, gated: Dict) -> None:
        real_g = gated['real_gated']
        real_u = gated['real_ungated']
        nulls = gated['null_metrics']
        assessment = gated['assessment']

        null_keys = self.NULL_TYPES

        display_metrics = [
            ('resolution_rate',              'Resolution Rate',     True),
            ('content_medical_vocab_rate',   'Medical Vocab Rate',  True),
            ('content_function_word_frac',   'Function Word Frac',  True),
            ('content_unique_resolved_types', 'Unique Types',       False),
            ('content_mean_skeleton_segments', 'Mean Skel Segments', False),
        ]

        print(f'\n  {"Metric":<24s} {"Real(gated)":>12s} {"Real(full)":>12s} '
              f'{"Shuffle":>10s} {"XFolio":>10s} {"CharRnd":>10s} {"Disc?":>6s}')
        print('  ' + '-' * 80)

        for metric_key, label, is_pct in display_metrics:
            real_val = real_g.get(metric_key, '-')
            ungated_val = real_u.get(metric_key, '-')
            info = assessment.get(metric_key, {})
            verdict = info.get('verdict', '?')

            def _fmt(val, pct=is_pct):
                if isinstance(val, int):
                    return f'{val:>10d}'
                if pct and isinstance(val, float):
                    return f'{100*val:>9.1f}%'
                if isinstance(val, float):
                    return f'{val:>10.2f}'
                return f'{str(val):>10s}'

            null_strs = []
            for nk in null_keys:
                nv = nulls.get(nk, {}).get(metric_key, '-')
                null_strs.append(_fmt(nv))

            print(f'  {label:<24s} {_fmt(real_val):>12s} {_fmt(ungated_val):>12s} '
                  f'{null_strs[0]:>10s} {null_strs[1]:>10s} '
                  f'{null_strs[2]:>10s} {verdict:>6s}')

        summary = assessment.get('_summary', {})
        print(f'\n  Verdict: {summary.get("overall_verdict", "?")}')
        print(f'  ({summary.get("yes", 0)} discriminative, '
              f'{summary.get("weak", 0)} weak, '
              f'{summary.get("no", 0)} non-discriminative)')

    # ── Null text generation ─────────────────────────────────────────

    def _generate_null(
        self,
        null_type: str,
        by_folio: Dict[str, List[str]],
        rng: random.Random,
    ) -> Dict[str, List[str]]:
        """Generate null tokens (delegates to MultipleBaselines)."""
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
            return baselines._gen_cross_folio_shuffle(by_folio, rng, all_tokens)
        elif null_type == 'char_random':
            char_weights = _build_eva_char_weights(all_tokens)
            return baselines._gen_char_random(by_folio, rng, char_weights)
        else:
            raise ValueError(f'Unknown null type: {null_type}')
