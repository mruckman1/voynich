"""
Test 1a-ext: Multiple Random Baselines
========================================
Generates 6 types of null text and runs each through the pipeline.
For each type, generates 10 independent samples and reports mean +/- std
resolution plus content-quality metrics. Total: 6 types x 10 samples = 60 runs.

Null types:
  1. RANDOM_TOKENS       — uniform random draw from Voynich vocabulary
  2. SHUFFLED_TOKENS     — real tokens with positions shuffled within folio
  3. RANDOM_SKELETONS    — tokens randomly reassigned across all positions
  4. CROSS_FOLIO_SHUFFLE — tokens redistributed randomly across folios
  5. FREQUENCY_MATCHED   — tokens sampled from real frequency distribution
  6. CHAR_RANDOM         — random EVA character strings (matches unicity test)

Also runs:
  - Single-pass comparison (no consistency/refinement) for unicity-comparable results
  - Controlled ablation on random text to quantify amplification per stage
"""
import random
import time
from collections import Counter
from typing import Dict, List

from orchestrators.robustness import (
    run_full_pipeline, run_single_pass_pipeline, _compute_content_metrics,
)

EVA_CHARS = list('oainedylrsktchpfqmgx')


def _build_eva_char_weights(all_tokens):
    """Build character frequency distribution from real Voynich tokens."""
    char_counts = Counter()
    eva_set = set(EVA_CHARS)
    for token in all_tokens:
        for ch in token:
            if ch in eva_set:
                char_counts[ch] += 1
    total = sum(char_counts.values()) or 1
    return [char_counts.get(ch, 0) / total for ch in EVA_CHARS]


class MultipleBaselines:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, n_trials: int = 10) -> Dict:
        by_folio = self.components['by_folio']

        # Build vocabulary pool and frequency distribution
        all_tokens = []
        for tokens in by_folio.values():
            all_tokens.extend(tokens)
        vocab = sorted(set(all_tokens))
        freq_pool = list(all_tokens)
        char_weights = _build_eva_char_weights(all_tokens)

        null_types = [
            ('random_tokens', self._gen_random_tokens, vocab),
            ('shuffled_tokens', self._gen_shuffled_tokens, None),
            ('random_skeletons', self._gen_random_skeletons, all_tokens),
            ('cross_folio_shuffle', self._gen_cross_folio_shuffle, all_tokens),
            ('frequency_matched', self._gen_frequency_matched, freq_pool),
            ('char_random', self._gen_char_random, char_weights),
        ]

        # ---- Real Voynich baseline (full + single-pass + metrics) ----
        if self.verbose:
            print('  Running real Voynich baseline (full pipeline)...')
        real_result = run_full_pipeline(
            self.components, verbose=False, compute_content_metrics=True,
        )
        real_resolution = real_result['overall_resolution']
        real_content = real_result.get('content_metrics', {})

        if self.verbose:
            print('  Running real Voynich baseline (single-pass)...')
        real_sp_result = run_single_pass_pipeline(self.components, verbose=False)
        real_sp_resolution = real_sp_result['overall_resolution']

        # ---- Null baselines (full pipeline + content metrics) ----
        results_by_type = {}
        all_null_resolutions = []
        all_null_content = {
            'medical_vocab_rate': [],
            'function_word_frac': [],
            'unique_resolved_types': [],
            'mean_skeleton_segments': [],
        }
        t0 = time.time()

        for type_name, gen_func, gen_data in null_types:
            if self.verbose:
                print(f'  Running {type_name} ({n_trials} trials)...')

            type_resolutions = []
            type_content = {k: [] for k in all_null_content}
            for trial in range(n_trials):
                rng = random.Random(42 + trial)
                null_folios = gen_func(by_folio, rng, gen_data)
                result = run_full_pipeline(
                    self.components, by_folio_override=null_folios,
                    verbose=False, compute_content_metrics=True,
                )
                rate = result['overall_resolution']
                type_resolutions.append(rate)
                all_null_resolutions.append(rate)

                cm = result.get('content_metrics', {})
                for k in type_content:
                    type_content[k].append(cm.get(k, 0))
                    all_null_content[k].append(cm.get(k, 0))

            mean_res = sum(type_resolutions) / len(type_resolutions)
            std_res = _std(type_resolutions)
            sorted_res = sorted(type_resolutions)
            p95 = sorted_res[int(0.95 * len(sorted_res))] if len(sorted_res) > 1 else sorted_res[0]

            results_by_type[type_name] = {
                'mean': round(mean_res, 4),
                'std': round(std_res, 4),
                'p95': round(p95, 4),
                'min': round(min(type_resolutions), 4),
                'max': round(max(type_resolutions), 4),
                'delta_from_real': round(real_resolution - mean_res, 4),
                'trials': [round(r, 4) for r in type_resolutions],
                'content_metrics': {
                    k: round(sum(v) / len(v), 4) if v else 0.0
                    for k, v in type_content.items()
                },
            }

        # ---- Single-pass comparison (unicity-comparable) ----
        sp_results_by_type = {}
        if self.verbose:
            print('  Running single-pass baselines...')
        for type_name, gen_func, gen_data in null_types:
            rng = random.Random(42)
            null_folios = gen_func(by_folio, rng, gen_data)
            result = run_single_pass_pipeline(
                self.components, by_folio_override=null_folios, verbose=False,
            )
            sp_results_by_type[type_name] = round(result['overall_resolution'], 4)

        # ---- Ablation on random text ----
        if self.verbose:
            print('  Running amplification ablation diagnostics...')
        ablation = self._run_ablation_on_random(by_folio, vocab)

        # ---- Pooled statistics ----
        pooled_mean = sum(all_null_resolutions) / len(all_null_resolutions)
        pooled_std = _std(all_null_resolutions)
        res_z = (real_resolution - pooled_mean) / max(pooled_std, 1e-6)

        # Content-quality z-scores
        content_z = {}
        for k in all_null_content:
            null_vals = all_null_content[k]
            null_mean = sum(null_vals) / len(null_vals) if null_vals else 0
            null_std = _std(null_vals) if null_vals else 1e-6
            real_val = real_content.get(k, 0)
            content_z[k] = {
                'real': real_val if isinstance(real_val, int) else round(real_val, 4),
                'null_mean': round(null_mean, 4),
                'null_std': round(null_std, 4),
                'z_score': round(
                    (real_val - null_mean) / max(null_std, 1e-6), 2
                ),
            }

        elapsed = time.time() - t0

        result = {
            'test': 'multiple_baselines',
            'resolution_comparison': {
                'real': round(real_resolution, 4),
                'null_types': {
                    k: {
                        'mean': v['mean'], 'std': v['std'],
                        'delta_from_real': v['delta_from_real'],
                    }
                    for k, v in results_by_type.items()
                },
                'pooled_z_score': round(res_z, 2),
                'note': (
                    'Resolution rate alone does not discriminate real from null. '
                    'See content_quality for discriminative metrics.'
                ),
            },
            'content_quality': {
                'real': {
                    k: v if isinstance(v, int) else round(v, 4)
                    for k, v in real_content.items()
                },
                'null_types': {
                    k: v['content_metrics']
                    for k, v in results_by_type.items()
                },
                'z_scores': content_z,
            },
            'single_pass_comparison': {
                'real': round(real_sp_resolution, 4),
                'null_types': sp_results_by_type,
                'note': (
                    'Single-pass = CSP + scaffold + solve only (no consistency, '
                    'no iterative refinement). Comparable to Phase 12.5 unicity test.'
                ),
            },
            'diagnostics': {
                'amplification_ablation': ablation,
            },
            # Keep full per-type data for backward compatibility
            'full_results_by_type': results_by_type,
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _run_ablation_on_random(self, by_folio, vocab):
        """Run random_tokens through pipeline with each stage disabled.

        Quantifies each stage's contribution to random text inflation.
        """
        rng = random.Random(42)
        null_folios = self._gen_random_tokens(by_folio, rng, vocab)

        stages = [
            None,  # full pipeline (no disabled features)
            {'iterative_refinement'},
            {'cross_folio_consistency', 'relaxed_consistency'},
            {'pos_backoff'},
            {'char_ngram'},
            {'relaxed_consistency'},
            {'graduated_csp'},
            # All amplification disabled (single-pass)
            {'cross_folio_consistency', 'relaxed_consistency', 'iterative_refinement'},
        ]
        labels = [
            'full_pipeline',
            'no_iterative_refinement',
            'no_cross_folio_consistency',
            'no_pos_backoff',
            'no_char_ngram',
            'no_relaxed_consistency',
            'no_graduated_csp',
            'single_pass_only',
        ]

        results = {}
        for label, disabled in zip(labels, stages):
            result = run_full_pipeline(
                self.components, by_folio_override=null_folios,
                disabled_features=disabled, verbose=False,
            )
            results[label] = round(result['overall_resolution'], 4)

        return results

    # ---- Null text generators ----

    def _gen_random_tokens(self, by_folio, rng, vocab):
        """Type 1: Uniform random draw from Voynich vocabulary."""
        null_folios = {}
        for folio, tokens in by_folio.items():
            null_folios[folio] = [rng.choice(vocab) for _ in range(len(tokens))]
        return null_folios

    def _gen_shuffled_tokens(self, by_folio, rng, _data):
        """Type 2: Real tokens with positions shuffled within each folio."""
        null_folios = {}
        for folio, tokens in by_folio.items():
            shuffled = list(tokens)
            rng.shuffle(shuffled)
            null_folios[folio] = shuffled
        return null_folios

    def _gen_random_skeletons(self, by_folio, rng, all_tokens):
        """Type 3: For each position, draw a random token from the global pool."""
        null_folios = {}
        for folio, tokens in by_folio.items():
            null_folios[folio] = [rng.choice(all_tokens) for _ in range(len(tokens))]
        return null_folios

    def _gen_cross_folio_shuffle(self, by_folio, rng, all_tokens):
        """Type 4: Pool all tokens, redistribute randomly across folios."""
        pool = list(all_tokens)
        rng.shuffle(pool)
        null_folios = {}
        idx = 0
        for folio, tokens in by_folio.items():
            n = len(tokens)
            null_folios[folio] = pool[idx:idx + n]
            idx += n
        return null_folios

    def _gen_frequency_matched(self, by_folio, rng, freq_pool):
        """Type 5: Sample tokens from the real frequency distribution."""
        null_folios = {}
        for folio, tokens in by_folio.items():
            null_folios[folio] = [rng.choice(freq_pool) for _ in range(len(tokens))]
        return null_folios

    def _gen_char_random(self, by_folio, rng, char_weights):
        """Type 6: Random EVA character strings matching word lengths.

        Matches the Phase 12.5 unicity test's frequency-matched random method.
        """
        null_folios = {}
        for folio, tokens in by_folio.items():
            random_tokens = []
            for token in tokens:
                length = len(token)
                chars = rng.choices(EVA_CHARS, weights=char_weights, k=length)
                random_tokens.append(''.join(chars))
            null_folios[folio] = random_tokens
        return null_folios

    # ---- Reporting ----

    def _print_report(self, result):
        print(f'\nMultiple Random Baselines')
        print('=' * 70)

        # Resolution comparison
        res = result['resolution_comparison']
        print(f'\n  Resolution Rate Comparison')
        print('  ' + '-' * 66)
        print(f'  {"Null Type":<28} {"Mean":>8} {"Std":>8} {"Delta":>8}')
        print('  ' + '-' * 66)

        for type_name, data in res['null_types'].items():
            delta = data['delta_from_real']
            sign = '+' if delta >= 0 else ''
            print(f'  {type_name:<28} {100*data["mean"]:>7.1f}% '
                  f'{100*data["std"]:>7.1f}% '
                  f'{sign}{100*delta:>6.1f}pp')

        print()
        print(f'  Real Voynich: {100*res["real"]:.1f}%')
        print(f'  Resolution z-score: {res["pooled_z_score"]:.1f} '
              f'(NOTE: resolution alone is not discriminative)')

        # Content quality comparison
        cq = result['content_quality']
        print(f'\n  Content Quality Comparison')
        print('  ' + '-' * 66)
        zs = cq['z_scores']
        pct_metrics = {'medical_vocab_rate', 'function_word_frac'}
        for metric, data in zs.items():
            label = metric.replace('_', ' ').title()
            z = data['z_score']
            real_v = data['real']
            null_m = data['null_mean']
            if isinstance(real_v, int):
                print(f'  {label:<30} real={real_v:>6d}  '
                      f'null={null_m:>8.1f}  z={z:>6.1f}')
            elif metric in pct_metrics:
                print(f'  {label:<30} real={100*real_v:>6.1f}%  '
                      f'null={100*null_m:>5.1f}%  z={z:>6.1f}')
            else:
                print(f'  {label:<30} real={real_v:>6.2f}  '
                      f'null={null_m:>8.2f}  z={z:>6.1f}')

        # Single-pass comparison
        sp = result['single_pass_comparison']
        print(f'\n  Single-Pass Comparison (unicity-comparable)')
        print('  ' + '-' * 66)
        print(f'  Real Voynich (single-pass): {100*sp["real"]:.1f}%')
        for type_name, rate in sp['null_types'].items():
            print(f'  {type_name:<28} {100*rate:>7.1f}%')

        # Ablation diagnostics
        diag = result['diagnostics']['amplification_ablation']
        print(f'\n  Amplification Ablation (random_tokens, seed=42)')
        print('  ' + '-' * 66)
        full = diag.get('full_pipeline', 0)
        for label, rate in diag.items():
            delta = rate - full
            sign = '+' if delta >= 0 else ''
            note = '' if label == 'full_pipeline' else f' ({sign}{100*delta:.1f}pp)'
            print(f'  {label:<36} {100*rate:>7.1f}%{note}')

        print(f'\n  Elapsed: {result["elapsed_seconds"]:.1f}s')


def _std(vals):
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
