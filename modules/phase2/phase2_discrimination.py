"""
Phase 2a: Quick Discrimination Sweep
=======================================
Tests all 6 generative models for the H2/TTR/Zipf triple match before
investing in deep implementation.

For each model:
1. Generate text with default + coarse-grid parameters across 3 source languages
2. Check H2/TTR/Zipf triple match
3. Rank models by composite distance
4. Classify as: triple_match, partial_match, or eliminated

Decision gate: Drop any model that cannot get within 50% of the gap.
Proceed with top 2-3 models for deep implementation.
"""

import time
import json
from typing import Dict, List, Optional

from modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS, TRIPLE_THRESHOLDS
from modules.phase2.verbose_cipher import VerboseCipher
from modules.phase2.syllabary_code import SyllabaryCode
from modules.phase2.slot_machine import SlotMachine
from modules.phase2.steganographic_carrier import SteganographicCarrier
from modules.phase2.grammar_induction import GrammarInduction
from modules.phase2.glyph_decomposition import GlyphDecomposition
from modules.strategy1_parameter_search import generate_medical_plaintext
from data.medieval_text_templates import generate_italian_text, generate_german_text

MODEL_REGISTRY = {
    'verbose_cipher': VerboseCipher,
    'syllabary_code': SyllabaryCode,
    'slot_machine': SlotMachine,
    'steganographic_carrier': SteganographicCarrier,
    'grammar_induction': GrammarInduction,
    'glyph_decomposition': GlyphDecomposition,
}

def _get_plaintext(language: str, n_words: int = 600) -> str:
    """Generate plaintext in the specified language."""
    if language == 'latin':
        return generate_medical_plaintext(n_words)
    elif language == 'italian':
        return generate_italian_text(n_words)
    elif language == 'german':
        return generate_german_text(n_words)
    else:
        return generate_medical_plaintext(n_words)

def run_quick_discrimination(
    models: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    resolution: str = 'coarse',
    verbose: bool = True,
) -> Dict:
    """
    Phase 2a: Quick screening of all 6 models.

    For each model × language:
    1. Generate text with coarse parameter sweep
    2. Check H2/TTR/Zipf triple
    3. Classify and rank

    Parameters:
        models: List of model names to test (default: all 6)
        languages: List of source languages (default: ['latin', 'italian', 'german'])
        resolution: Parameter grid resolution ('coarse' for speed)
        verbose: Print progress

    Returns:
        {model_rankings: [...], triple_matches: [...],
         partial_matches: [...], eliminated: [...],
         per_model: {...}, elapsed_seconds: float}
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    if languages is None:
        languages = ['latin', 'italian', 'german']

    t0 = time.time()
    per_model = {}

    if verbose:
        print('=' * 70)
        print('PHASE 2a: QUICK DISCRIMINATION SWEEP')
        print(f'Testing {len(models)} models × {len(languages)} languages')
        print(f'Resolution: {resolution}')
        print('=' * 70)

    for model_name in models:
        if model_name not in MODEL_REGISTRY:
            if verbose:
                print(f'  Skipping unknown model: {model_name}')
            continue

        model_class = MODEL_REGISTRY[model_name]

        if verbose:
            print(f'\n--- {model_name} (priority: {model_class.MODEL_PRIORITY}) ---')

        best_result = None
        best_distance = float('inf')
        all_results = []

        if model_name == 'glyph_decomposition':
            result = _test_glyph_decomposition(model_class, verbose)
            per_model[model_name] = result
            continue

        if model_name == 'steganographic_carrier':
            result = _test_steganographic(model_class, verbose)
            per_model[model_name] = result
            continue

        if model_name == 'grammar_induction':
            result = _test_grammar_induction(model_class, verbose)
            per_model[model_name] = result
            continue

        for language in languages:
            plaintext = _get_plaintext(language)

            if verbose:
                print(f'  Language: {language}')

            default_model = model_class(seed=42)
            sweep = default_model.run_sweep(
                plaintext=plaintext,
                resolution=resolution,
                n_best=5,
                verbose=verbose,
            )

            for entry in sweep.get('best_results', []):
                score = entry.get('score', {}).get('quick', {})
                dist = score.get('distance', float('inf'))

                result_entry = {
                    'language': language,
                    'params': entry.get('params', {}),
                    'H2': score.get('H2', 0),
                    'TTR': score.get('TTR', 0),
                    'zipf_exponent': score.get('zipf_exponent', 0),
                    'triple_match': score.get('triple_match', False),
                    'distance': dist,
                    'critical_test': entry.get('critical_test', {}),
                }
                all_results.append(result_entry)

                if dist < best_distance:
                    best_distance = dist
                    best_result = result_entry

        per_model[model_name] = _summarize_model_results(
            model_name, all_results, best_result, best_distance
        )

    triple_matches = []
    partial_matches = []
    eliminated = []

    for name, result in per_model.items():
        if result.get('has_triple_match', False):
            triple_matches.append(name)
        elif result.get('partial_match_count', 0) >= 2:
            partial_matches.append(name)
        else:
            eliminated.append(name)

    ranked = sorted(per_model.items(), key=lambda x: x[1].get('best_distance', float('inf')))
    rankings = [{'model': name, 'distance': result.get('best_distance', float('inf')),
                 'category': 'triple' if name in triple_matches
                 else 'partial' if name in partial_matches else 'eliminated'}
                for name, result in ranked]

    elapsed = time.time() - t0

    if verbose:
        print('\n' + '=' * 70)
        print('DISCRIMINATION RESULTS')
        print('=' * 70)
        print(f'Triple matches: {triple_matches or "none"}')
        print(f'Partial matches: {partial_matches or "none"}')
        print(f'Eliminated: {eliminated or "none"}')
        print(f'\nRanking by distance:')
        for r in rankings:
            print(f'  {r["model"]:25s} dist={r["distance"]:.4f} [{r["category"]}]')
        print(f'\nTotal time: {elapsed:.1f}s')

    return {
        'model_rankings': rankings,
        'triple_matches': triple_matches,
        'partial_matches': partial_matches,
        'eliminated': eliminated,
        'per_model': per_model,
        'elapsed_seconds': elapsed,
    }

def _test_glyph_decomposition(model_class, verbose: bool) -> Dict:
    """Test Model 6 (glyph decomposition) — no parameter sweep needed."""
    model = model_class()
    results = model.run_all_alphabets(verbose=verbose)

    best_alpha = results.get('best_alphabet', '')
    best_h2 = results.get('best_H2', 0)

    return {
        'model': 'glyph_decomposition',
        'has_triple_match': results.get('any_in_cipher_range', False),
        'partial_match_count': 0,
        'best_distance': abs(best_h2 - VOYNICH_TARGETS['H2']),
        'best_params': {'alphabet': best_alpha},
        'best_H2': best_h2,
        'best_TTR': 0,
        'best_zipf': 0,
        'n_tested': len(results.get('alphabets', [])),
        'conclusion': results.get('conclusion', ''),
        'all_alphabets': results.get('alphabets', []),
    }

def _test_steganographic(model_class, verbose: bool) -> Dict:
    """Test Model 4 (steganographic carrier) — tests against actual Voynich."""
    if verbose:
        print('  Testing deviation stream against actual Voynich corpus...')

    best_result = None
    best_distance = float('inf')

    for order in [1, 2, 3]:
        model = model_class(carrier_order=order, choices_per_position=3, seed=42)
        text = model.generate(n_words=500)
        if not text:
            continue

        profile = model.get_profile(text)
        score = model.quick_score(profile)

        crit = model.critical_test(profile)

        dist = score.get('distance', float('inf'))
        if dist < best_distance:
            best_distance = dist
            best_result = {
                'order': order,
                'H2': score.get('H2', 0),
                'TTR': score.get('TTR', 0),
                'zipf_exponent': score.get('zipf_exponent', 0),
                'triple_match': score.get('triple_match', False),
                'deviation_test': crit,
            }

        if verbose:
            stego_pass = 'PASS' if crit.get('passes', False) else 'FAIL'
            print(f'  Order {order}: distance={dist:.4f}, deviation test={stego_pass}')

    partial_count = 0
    if best_result:
        checks = [
            (best_result.get('H2', 0), VOYNICH_TARGETS['H2'], TRIPLE_THRESHOLDS['H2']),
            (best_result.get('TTR', 0), VOYNICH_TARGETS['type_token_ratio'], TRIPLE_THRESHOLDS['TTR']),
            (best_result.get('zipf_exponent', 0), VOYNICH_TARGETS['zipf_exponent'], TRIPLE_THRESHOLDS['zipf_exponent']),
        ]
        partial_count = sum(1 for m, t, th in checks if abs(m - t) < th)

    return {
        'model': 'steganographic_carrier',
        'has_triple_match': best_result.get('triple_match', False) if best_result else False,
        'partial_match_count': partial_count,
        'best_distance': best_distance,
        'best_params': {'order': best_result.get('order', 0)} if best_result else {},
        'best_H2': best_result.get('H2', 0) if best_result else 0,
        'best_TTR': best_result.get('TTR', 0) if best_result else 0,
        'best_zipf': best_result.get('zipf_exponent', 0) if best_result else 0,
        'deviation_test': best_result.get('deviation_test', {}) if best_result else {},
        'n_tested': 3,
    }

def _test_grammar_induction(model_class, verbose: bool) -> Dict:
    """Test Model 5 (grammar induction) — runs evolutionary search."""
    if verbose:
        print('  Running grammar induction (evolutionary search)...')

    model = model_class(
        max_rules=20, max_symbols=50,
        evolution_generations=100, population_size=20, seed=42
    )
    induction_result = model.induce_grammar(verbose=verbose)

    text = model.generate(n_words=500)
    if not text:
        return {
            'model': 'grammar_induction',
            'has_triple_match': False,
            'partial_match_count': 0,
            'best_distance': float('inf'),
            'best_params': {},
            'n_tested': 1,
        }

    profile = model.get_profile(text)
    score = model.quick_score(profile)
    crit = model.critical_test(profile)

    return {
        'model': 'grammar_induction',
        'has_triple_match': score.get('triple_match', False),
        'partial_match_count': sum(1 for delta, thresh in [
            (score.get('H2_delta', float('inf')), TRIPLE_THRESHOLDS['H2']),
            (score.get('TTR_delta', float('inf')), TRIPLE_THRESHOLDS['TTR']),
            (score.get('zipf_delta', float('inf')), TRIPLE_THRESHOLDS['zipf_exponent']),
        ] if delta < thresh),
        'best_distance': score.get('distance', float('inf')),
        'best_params': {'max_rules': 20, 'max_symbols': 50, 'generations': 100},
        'best_H2': score.get('H2', 0),
        'best_TTR': score.get('TTR', 0),
        'best_zipf': score.get('zipf_exponent', 0),
        'grammar_complexity': crit.get('details', {}),
        'induction_result': {
            'best_distance': induction_result.get('best_distance', float('inf')),
            'generations': induction_result.get('generations', 0),
        },
        'n_tested': 1,
    }

def _summarize_model_results(model_name: str, all_results: List[Dict],
                             best_result: Optional[Dict],
                             best_distance: float) -> Dict:
    """Summarize results across all parameter×language combinations."""
    has_triple = any(r.get('triple_match', False) for r in all_results)
    n_triple = sum(1 for r in all_results if r.get('triple_match', False))

    partial_count = 0
    if best_result:
        for metric, target, threshold in [
            ('H2', VOYNICH_TARGETS['H2'], TRIPLE_THRESHOLDS['H2']),
            ('TTR', VOYNICH_TARGETS['type_token_ratio'], TRIPLE_THRESHOLDS['TTR']),
            ('zipf_exponent', VOYNICH_TARGETS['zipf_exponent'], TRIPLE_THRESHOLDS['zipf_exponent']),
        ]:
            val = best_result.get(metric, 0)
            if abs(val - target) < threshold:
                partial_count += 1

    return {
        'model': model_name,
        'has_triple_match': has_triple,
        'n_triple_matches': n_triple,
        'partial_match_count': partial_count,
        'best_distance': best_distance,
        'best_params': best_result.get('params', {}) if best_result else {},
        'best_H2': best_result.get('H2', 0) if best_result else 0,
        'best_TTR': best_result.get('TTR', 0) if best_result else 0,
        'best_zipf': best_result.get('zipf_exponent', 0) if best_result else 0,
        'n_tested': len(all_results),
        'results_by_language': _group_by_language(all_results),
    }

def _group_by_language(results: List[Dict]) -> Dict:
    """Group results by language and find best per language."""
    by_lang = {}
    for r in results:
        lang = r.get('language', 'unknown')
        if lang not in by_lang:
            by_lang[lang] = {'best_distance': float('inf'), 'n_tested': 0}
        by_lang[lang]['n_tested'] += 1
        if r.get('distance', float('inf')) < by_lang[lang]['best_distance']:
            by_lang[lang]['best_distance'] = r['distance']
            by_lang[lang]['best_H2'] = r.get('H2', 0)
            by_lang[lang]['best_TTR'] = r.get('TTR', 0)
            by_lang[lang]['best_zipf'] = r.get('zipf_exponent', 0)
    return by_lang
