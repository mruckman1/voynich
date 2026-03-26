"""
Test 2a: Leave-One-Out Validation
====================================
For a sample of folios, removes each folio's resolved words from the
transition matrix and re-decodes to test for circularity.

If resolution is similar with depleted corpus -> not circular.
If resolution drops sharply -> potential circularity.

Uses matrix depletion (zeroing rows/columns) rather than full corpus
regeneration for speed.
"""
import re
import time
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from voynich.phases.robustness import (
    build_default_solver_kwargs, run_full_pipeline, NgramMaskSolver,
)


def _count_brackets(text: str) -> int:
    return len(re.findall(r'\[[^\]]+\]|<[^>]+>', text))


class LeaveOneOutValidation:

    def __init__(self, components: Dict, verbose: bool = True):
        self.components = components
        self.verbose = verbose

    def run(self, n_sample: int = 30) -> Dict:
        t0 = time.time()
        by_folio = self.components['by_folio']
        folio_metadata = self.components['folio_metadata']
        trans_vocab = self.components['trans_vocab']

        # Run baseline: each folio individually without cross-folio consistency
        # (for fair comparison, since LOO single-folio runs can't use consistency)
        if self.verbose:
            print('  Computing per-folio baselines (no consistency)...')

        disabled_no_consistency = {
            'cross_folio_consistency', 'relaxed_consistency',
        }

        baseline_result = run_full_pipeline(
            self.components,
            disabled_features=disabled_no_consistency,
            verbose=False,
        )
        baseline_translations = baseline_result['final_translations']

        # Compute per-folio resolution rates from baseline
        folio_baselines = {}
        for folio, text in baseline_translations.items():
            wc = len(text.split())
            bc = _count_brackets(text)
            if wc > 0:
                folio_baselines[folio] = {
                    'resolution': (wc - bc) / wc,
                    'words': wc,
                    'brackets': bc,
                }

        # Select sample folios
        sample_folios = self._select_sample(folio_baselines, folio_metadata, n_sample)

        if self.verbose:
            print(f'  Selected {len(sample_folios)} folios for LOO testing')

        # Build vocab index for matrix depletion
        vocab_index = {w: i for i, w in enumerate(trans_vocab)}

        # Run LOO for each sample folio
        loo_results = []
        for i, folio in enumerate(sample_folios):
            if self.verbose:
                print(f'  LOO {i+1}/{len(sample_folios)}: {folio}...')

            if folio not in baseline_translations:
                continue

            # Get resolved words on this folio
            resolved_words = self._get_resolved_words(baseline_translations[folio])
            if not resolved_words:
                continue

            # Deplete the matrix: zero out rows/columns for these words
            depleted_solver = self._build_depleted_solver(resolved_words, vocab_index)

            # Run pipeline on just this folio with depleted solver
            single_folio = {folio: by_folio[folio]}
            loo_result = run_full_pipeline(
                self.components,
                by_folio_override=single_folio,
                solver_override=depleted_solver,
                disabled_features=disabled_no_consistency,
                verbose=False,
            )

            loo_text = loo_result['final_translations'].get(folio, '')
            loo_wc = len(loo_text.split()) if loo_text else 0
            loo_bc = _count_brackets(loo_text) if loo_text else 0
            loo_rate = (loo_wc - loo_bc) / max(1, loo_wc)

            base_rate = folio_baselines.get(folio, {}).get('resolution', 0)
            delta = loo_rate - base_rate

            # Find which words were lost/gained
            base_resolved = self._get_resolved_words(baseline_translations.get(folio, ''))
            loo_resolved = self._get_resolved_words(loo_text)
            lost = base_resolved - loo_resolved
            gained = loo_resolved - base_resolved

            meta = folio_metadata.get(folio, {})
            loo_results.append({
                'folio': folio,
                'section': meta.get('section', 'unknown'),
                'baseline_resolution': round(base_rate, 4),
                'loo_resolution': round(loo_rate, 4),
                'delta': round(delta, 4),
                'words_depleted': len(resolved_words),
                'words_lost': len(lost),
                'words_gained': len(gained),
            })

        # Summary statistics
        deltas = [r['delta'] for r in loo_results]
        mean_delta = sum(deltas) / max(1, len(deltas))
        std_delta = (sum((d - mean_delta) ** 2 for d in deltas) / max(1, len(deltas))) ** 0.5
        max_drop = min(deltas) if deltas else 0
        min_drop = max(deltas) if deltas else 0
        folios_under_2pp = sum(1 for d in deltas if abs(d) < 0.02)
        folios_over_5pp = sum(1 for d in deltas if d < -0.05)

        elapsed = time.time() - t0

        # Interpretation
        abs_mean = abs(mean_delta)
        if abs_mean < 0.03:
            interpretation = 'MINIMAL circularity'
        elif abs_mean < 0.08:
            interpretation = 'MODERATE overlap'
        else:
            interpretation = 'HIGH circularity risk'

        result = {
            'test': 'leave_one_out',
            'n_folios_tested': len(loo_results),
            'per_folio': loo_results,
            'summary': {
                'mean_delta': round(mean_delta, 4),
                'std_delta': round(std_delta, 4),
                'max_drop': round(max_drop, 4),
                'min_drop': round(min_drop, 4),
                'folios_under_2pp': folios_under_2pp,
                'folios_over_5pp': folios_over_5pp,
                'interpretation': interpretation,
            },
            'elapsed_seconds': round(elapsed, 1),
        }

        if self.verbose:
            self._print_report(result)

        return result

    def _select_sample(
        self, folio_baselines: Dict, folio_metadata: Dict, n_sample: int,
    ) -> List[str]:
        """Select a stratified sample of folios for LOO testing."""
        # Sort by resolution
        by_res = sorted(folio_baselines.items(), key=lambda x: x[1]['resolution'], reverse=True)
        folios_sorted = [f for f, _ in by_res]

        sample = set()

        # Top 5 highest resolution
        for f in folios_sorted[:5]:
            sample.add(f)

        # Bottom 5 lowest resolution
        for f in folios_sorted[-5:]:
            sample.add(f)

        # Middle folios from each section
        sections = ['herbal_a', 'pharmaceutical', 'biological', 'astronomical', 'recipes']
        for section in sections:
            section_folios = [
                f for f in folios_sorted
                if folio_metadata.get(f, {}).get('section') == section
            ]
            if section_folios:
                # Take from the middle
                mid = len(section_folios) // 2
                start = max(0, mid - 2)
                for f in section_folios[start:start + 4]:
                    sample.add(f)

        # Cap at n_sample
        sample_list = sorted(sample)
        return sample_list[:n_sample]

    def _get_resolved_words(self, text: str) -> Set[str]:
        """Extract the set of resolved (non-bracket) words from decoded text."""
        if not text:
            return set()
        words = set()
        for w in text.split():
            if not (w.startswith('[') or w.startswith('<')):
                words.add(w)
        return words

    def _build_depleted_solver(
        self, words_to_remove: Set[str], vocab_index: Dict[str, int],
    ) -> NgramMaskSolver:
        """Build a solver with zeroed-out matrix entries for removed words."""
        c = self.components
        trans_matrix = c['trans_matrix'].copy()

        for word in words_to_remove:
            idx = vocab_index.get(word)
            if idx is not None:
                trans_matrix[idx, :] = 0
                trans_matrix[:, idx] = 0

        kwargs = build_default_solver_kwargs(
            c['pos_tagger'], c['pos_matrix'], c['pos_vocab'],
            c['char_ngram_model'], c['illustration_prior'],
        )
        solver = NgramMaskSolver(
            trans_matrix, c['trans_vocab'],
            c['latin_skel'], c['fuzzy_skel'],
            **kwargs,
        )

        # Set depleted corpus frequencies (remove words from frequency dict)
        # Use original frequencies but zero out removed words
        solver.set_corpus_frequencies(c['l_tokens'])
        if hasattr(solver, '_unigram_freq'):
            for word in words_to_remove:
                solver._unigram_freq.pop(word, None)

        return solver

    def _print_report(self, result):
        per_folio = result['per_folio']
        summary = result['summary']

        print(f'\nLeave-One-Out Validation ({result["n_folios_tested"]} folios)')
        print('=' * 70)
        print()
        print(f'  {"Folio":<10} {"Section":<18} {"Base%":>8} {"LOO%":>8} '
              f'{"Delta":>8} {"Lost":>6} {"Gained":>8}')
        print('  ' + '-' * 68)

        for r in sorted(per_folio, key=lambda x: x['delta']):
            print(f'  {r["folio"]:<10} {r["section"]:<18} '
                  f'{100*r["baseline_resolution"]:>7.1f}% '
                  f'{100*r["loo_resolution"]:>7.1f}% '
                  f'{100*r["delta"]:>+7.1f}pp '
                  f'{r["words_lost"]:>6} {r["words_gained"]:>8}')

        print()
        print(f'  Summary:')
        print(f'    Mean delta:       {100*summary["mean_delta"]:+.1f}pp')
        print(f'    Std delta:        {100*summary["std_delta"]:.1f}pp')
        print(f'    Max drop:         {100*summary["max_drop"]:+.1f}pp')
        print(f'    Folios < 2pp:     {summary["folios_under_2pp"]} / {result["n_folios_tested"]}')
        print(f'    Folios > 5pp:     {summary["folios_over_5pp"]} / {result["n_folios_tested"]}')
        print(f'    Interpretation:   {summary["interpretation"]}')
        print(f'    Elapsed:          {result["elapsed_seconds"]:.1f}s')
