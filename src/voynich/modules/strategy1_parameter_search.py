"""
Strategy 1: Naibbe Parameter Space Search
===========================================
Reverse-engineer the Naibbe cipher parameters by encrypting probable
Hartlieb-era medical Latin through thousands of parameter variations
and comparing bigram transition matrices against the real Voynich text.

The key insight: we're not decrypting. We're finding which cipher parameters
produce output statistically indistinguishable from the actual manuscript.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional


from voynich.modules.naibbe_cipher import (
    NaibbeCipher, generate_parameter_grid
)
from voynich.core.stats import (
    full_statistical_profile, profile_distance,
    bigram_transition_matrix, compare_bigram_matrices,
    compute_all_entropy
)
from voynich.core.voynich_corpus import (
    HARTLIEB_MEDICAL_VOCAB, LATIN_RECIPE_FORMULAS,
    get_all_tokens, get_section_text
)

def generate_medical_plaintext(n_words: int = 500) -> str:
    """
    Generate a synthetic but realistic medieval Latin medical text
    using Hartlieb-era vocabulary and formulaic recipe structures.
    
    This serves as the probable plaintext for the parameter search.
    We generate enough text to produce stable statistical profiles.
    """
    import random
    rng = random.Random(12345)

    vocab = list(HARTLIEB_MEDICAL_VOCAB.keys())
    formulas = LATIN_RECIPE_FORMULAS.copy()

    weighted_vocab = []
    high_freq = ['recipe', 'accipe', 'cum', 'et', 'in', 'de', 'ad',
                 'aqua', 'herba', 'matrix', 'balneum', 'contra']
    for word in vocab:
        weight = 5 if word in high_freq else 2 if len(word) <= 4 else 1
        weighted_vocab.extend([word] * weight)

    lines = []
    words_generated = 0

    while words_generated < n_words:
        if rng.random() < 0.35:
            formula = rng.choice(formulas)
            herb = rng.choice([v for v in vocab if len(v) > 5])
            line = formula.replace('{herb}', herb)
        else:
            sent_len = rng.randint(4, 10)
            words = [rng.choice(weighted_vocab) for _ in range(sent_len)]
            line = ' '.join(words)

        lines.append(line)
        words_generated += len(line.split())

    return ' '.join(lines)

def generate_section_specific_plaintext(section: str, n_words: int = 300) -> str:
    """
    Generate plaintext biased toward the vocabulary expected in a specific
    manuscript section.
    """
    import random
    rng = random.Random(section.__hash__())

    section_vocab = {
        'herbal_a': [
            'herba', 'radix', 'folia', 'flos', 'semen', 'cortex',
            'calida', 'frigida', 'humida', 'sicca', 'virtus',
            'contra', 'ad', 'cum', 'in', 'de',
            'artemisia', 'malva', 'ruta', 'sabina', 'petroselinum',
        ],
        'herbal_b': [
            'herba', 'radix', 'folia', 'flos', 'semen',
            'pulvis', 'unguentum', 'potio', 'dosis',
            'tanacetum', 'pulegium', 'dictamnus', 'myrrha',
        ],
        'pharmaceutical': [
            'recipe', 'accipe', 'misce', 'contere', 'coque',
            'destilla', 'pulvis', 'unguentum', 'emplastrum',
            'potio', 'dosis', 'aqua', 'vinum', 'mel', 'oleum',
        ],
        'recipes': [
            'recipe', 'accipe', 'misce', 'contere', 'coque',
            'cum', 'et', 'in', 'ad', 'de', 'contra', 'pro',
            'matrix', 'menstrua', 'partus', 'conceptio',
            'fomenta', 'balneum', 'calida', 'super', 'ventrem',
        ],
        'biological': [
            'matrix', 'uterus', 'vulva', 'menstrua', 'conceptio',
            'partus', 'fetus', 'balneum', 'aqua', 'calida',
            'frigida', 'suffocatio', 'sterilitas', 'mola',
        ],
        'astronomical': [
            'caput', 'collum', 'pectus', 'cor', 'ventriculus',
            'renes', 'genitalia', 'femur', 'genu', 'tibia', 'pedes',
            'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
            'sanguis', 'cholera', 'phlegma', 'melancholia',
        ],
    }

    vocab = section_vocab.get(section, list(HARTLIEB_MEDICAL_VOCAB.keys()))
    filler = ['et', 'cum', 'in', 'de', 'ad', 'contra', 'pro', 'super']

    words = []
    for _ in range(n_words):
        if rng.random() < 0.3:
            words.append(rng.choice(filler))
        else:
            words.append(rng.choice(vocab))
    return ' '.join(words)

def compute_voynich_target_profiles() -> Dict[str, Dict]:
    """
    Compute the statistical profiles of the actual Voynich manuscript
    that cipher outputs must match.
    """
    profiles = {}

    all_tokens = get_all_tokens()
    all_text = ' '.join(all_tokens)
    profiles['overall'] = full_statistical_profile(all_text, 'Voynich-All')

    for section in ['herbal_a', 'herbal_b', 'astronomical',
                    'biological', 'pharmaceutical', 'recipes']:
        text = get_section_text(section)
        if text.strip():
            profiles[section] = full_statistical_profile(text, f'Voynich-{section}')

    for lang in ['A', 'B']:
        tokens = get_all_tokens(lang=lang)
        if tokens:
            text = ' '.join(tokens)
            profiles[f'lang_{lang}'] = full_statistical_profile(text, f'Voynich-Lang{lang}')

    return profiles

def search_parameter_space(
        resolution: str = 'coarse',
        target_section: str = 'overall',
        source_language: str = 'latin',
        max_results: int = 20,
        verbose: bool = True
) -> List[Dict]:
    """
    Main parameter space search.
    
    For each parameter combination:
    1. Instantiate a Naibbe cipher with those parameters
    2. Encrypt the appropriate medical Latin plaintext
    3. Compute the statistical profile of the ciphertext
    4. Compare against the real Voynich target profile
    5. Rank by similarity
    
    Returns the top-N most Voynich-like parameter sets.
    """
    if verbose:
        print("=" * 70)
        print("STRATEGY 1: NAIBBE PARAMETER SPACE SEARCH")
        print("=" * 70)
        print(f"  Resolution:      {resolution}")
        print(f"  Target section:  {target_section}")
        print(f"  Source language:  {source_language}")

    if verbose:
        print("\n[1/4] Computing Voynich target profiles...")
    targets = compute_voynich_target_profiles()
    target_prof = targets.get(target_section, targets['overall'])

    if verbose:
        e = target_prof['entropy']
        print(f"  Target entropy: H1={e['H1']:.3f} H2={e['H2']:.3f} H3={e['H3']:.3f}")

    if verbose:
        print("\n[2/4] Generating medical Latin plaintext corpus...")
    if target_section == 'overall':
        plaintext = generate_medical_plaintext(n_words=500)
    else:
        plaintext = generate_section_specific_plaintext(target_section, n_words=400)

    if verbose:
        print(f"  Plaintext length: {len(plaintext.split())} words, {len(plaintext)} chars")
        plain_prof = full_statistical_profile(plaintext, 'Plaintext')
        pe = plain_prof['entropy']
        print(f"  Plaintext entropy: H1={pe['H1']:.3f} H2={pe['H2']:.3f} H3={pe['H3']:.3f}")

    grid = generate_parameter_grid(resolution)
    if verbose:
        print(f"\n[3/4] Searching {len(grid)} parameter combinations...")

    results = []
    start_time = time.time()
    progress_interval = max(1, len(grid) // 10)

    for idx, params in enumerate(grid):
        if verbose and idx % progress_interval == 0:
            elapsed = time.time() - start_time
            pct = (idx / len(grid)) * 100
            print(f"  Progress: {pct:.0f}% ({idx}/{len(grid)}) [{elapsed:.1f}s]")

        try:
            cipher = NaibbeCipher(**params)
            ciphertext = cipher.encrypt(plaintext)

            if not ciphertext.strip():
                continue

            cipher_prof = full_statistical_profile(ciphertext, f'params_{idx}')
            distance = profile_distance(cipher_prof, target_prof)

            ct_mat, ct_alph = bigram_transition_matrix(ciphertext)
            vt_text = get_section_text(target_section) if target_section != 'overall' \
                else ' '.join(get_all_tokens())
            if vt_text.strip():
                vt_mat, vt_alph = bigram_transition_matrix(vt_text)
                bigram_dist = compare_bigram_matrices(ct_mat, vt_mat, ct_alph, vt_alph)
            else:
                bigram_dist = float('inf')

            composite = 0.6 * distance + 0.4 * bigram_dist

            results.append({
                'params': params,
                'profile_distance': round(distance, 6),
                'bigram_distance': round(bigram_dist, 6),
                'composite_score': round(composite, 6),
                'cipher_entropy': cipher_prof['entropy'],
                'cipher_zipf_exp': cipher_prof['zipf'].get('zipf_exponent', 0),
                'cipher_ttr': cipher_prof['zipf'].get('type_token_ratio', 0),
                'sample_output': ciphertext[:200],
            })

        except Exception as e:
            continue

    elapsed = time.time() - start_time
    results.sort(key=lambda r: r['composite_score'])

    if verbose:
        print(f"\n[4/4] Search complete in {elapsed:.1f}s")
        print(f"  Valid results: {len(results)}")
        print(f"\n{'='*70}")
        print(f"TOP {min(max_results, len(results))} PARAMETER SETS (lowest distance = best match)")
        print(f"{'='*70}")

        for i, r in enumerate(results[:max_results]):
            e = r['cipher_entropy']
            print(f"\n  Rank #{i+1}  [composite={r['composite_score']:.4f}  "
                  f"profile={r['profile_distance']:.4f}  "
                  f"bigram={r['bigram_distance']:.4f}]")
            print(f"    H1={e['H1']:.3f} H2={e['H2']:.3f} H3={e['H3']:.3f}  "
                  f"Zipf={r['cipher_zipf_exp']:.2f}  TTR={r['cipher_ttr']:.3f}")
            p = r['params']
            print(f"    tables={p['n_tables']}  bigram_p={p['bigram_probability']:.2f}  "
                  f"word_len={p['word_length_range']}  "
                  f"prefix_p={p['prefix_probability']:.2f}  "
                  f"suffix_p={p['suffix_probability']:.2f}")
            print(f"    Sample: {r['sample_output'][:100]}...")

    return results[:max_results]

def compare_source_languages(
        languages: Optional[List[str]] = None,
        resolution: str = 'coarse'
) -> Dict[str, List[Dict]]:
    """
    Run the parameter search for multiple possible source languages.
    
    For each language, generate appropriate plaintext and find the
    best-matching cipher parameters. The language whose best parameters
    produce the smallest distance to the Voynich profile is the most
    likely source language.
    """
    if languages is None:
        languages = ['latin_medical', 'latin_general', 'german_medical', 'italian']

    language_plaintexts = {
        'latin_medical': generate_medical_plaintext(n_words=400),
        'latin_general': _generate_latin_general(400),
        'german_medical': _generate_german_medical(400),
        'italian': _generate_italian_medical(400),
    }

    results = {}
    for lang in languages:
        print(f"\n{'='*70}")
        print(f"Testing source language: {lang}")
        print(f"{'='*70}")
        plaintext = language_plaintexts.get(lang, generate_medical_plaintext(400))
        lang_results = search_parameter_space(
            resolution=resolution,
            target_section='overall',
            max_results=5,
            verbose=True
        )
        results[lang] = lang_results

    print(f"\n{'='*70}")
    print("LANGUAGE COMPARISON SUMMARY")
    print(f"{'='*70}")
    for lang, res in results.items():
        best = res[0] if res else {'composite_score': float('inf')}
        print(f"  {lang:20s}: best composite = {best['composite_score']:.4f}")

    return results

def _generate_latin_general(n: int) -> str:
    """Generate generic Latin text (non-medical)."""
    import random
    rng = random.Random(999)
    words = ['et', 'in', 'de', 'ad', 'cum', 'per', 'ex', 'ab',
             'est', 'sunt', 'fuit', 'habet', 'dicit', 'facit',
             'homo', 'deus', 'terra', 'aqua', 'ignis', 'aer',
             'magnus', 'bonus', 'malus', 'sanctus', 'verus',
             'rex', 'dominus', 'ecclesia', 'anima', 'corpus',
             'liber', 'scientia', 'natura', 'virtus', 'gratia']
    return ' '.join(rng.choice(words) for _ in range(n))

def _generate_german_medical(n: int) -> str:
    """Generate Middle High German medical vocabulary."""
    import random
    rng = random.Random(888)
    words = ['und', 'in', 'mit', 'von', 'der', 'das', 'ein',
             'kraut', 'wurzel', 'blatt', 'blume', 'wasser',
             'nimm', 'mische', 'koche', 'trinke', 'salbe',
             'weib', 'kind', 'leib', 'blut', 'fieber',
             'geburt', 'mutter', 'bauch', 'haupt', 'herz',
             'arznei', 'heilung', 'bad', 'warm', 'kalt']
    return ' '.join(rng.choice(words) for _ in range(n))

def _generate_italian_medical(n: int) -> str:
    """Generate medieval Italian medical vocabulary."""
    import random
    rng = random.Random(777)
    words = ['e', 'in', 'con', 'di', 'per', 'la', 'il', 'una',
             'erba', 'radice', 'foglia', 'fiore', 'acqua', 'vino',
             'prendi', 'mescola', 'cuoci', 'bevi', 'unguento',
             'donna', 'madre', 'ventre', 'sangue', 'febbre',
             'parto', 'matrice', 'bagno', 'caldo', 'freddo',
             'rimedio', 'medicina', 'olio', 'miele', 'polvere']
    return ' '.join(rng.choice(words) for _ in range(n))

def run(verbose: bool = True) -> Dict:
    """Run Strategy 1 and return results."""
    results = {
        'strategy': 'naibbe_parameter_search',
        'sections_analyzed': {},
    }

    for section in ['overall', 'recipes', 'pharmaceutical', 'herbal_a']:
        section_results = search_parameter_space(
            resolution='coarse',
            target_section=section,
            max_results=10,
            verbose=verbose,
        )
        results['sections_analyzed'][section] = section_results

    return results

if __name__ == '__main__':
    run()
