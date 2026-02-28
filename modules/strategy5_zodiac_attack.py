"""
Strategy 5: Zodiac Month Labels as Known-Plaintext Attack
============================================================
The zodiac section contains Romance-language month names written
adjacent to Voynich text about each astrological sign.

Medieval zodiac-medical texts are FORMULAIC: each sign governs
a body part, with predictable phrases about humoral balance and
recommended treatments. This gives us constrained probable plaintext.

This module:
1. Generates probable plaintext for each zodiac section
2. Encrypts through Naibbe cipher parameter variants
3. Compares output against actual adjacent Voynich text
4. Identifies parameter sets that produce consistent mappings
5. If found, uses those mappings to bootstrap partial decryption
"""

import sys
import os
import math
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.naibbe_cipher import NaibbeCipher, SubstitutionTable
from modules.statistical_analysis import (
    full_statistical_profile, profile_distance,
    bigram_transition_matrix, compare_bigram_matrices,
    compute_all_entropy
)
from data.voynich_corpus import (
    ZODIAC_LABELS, SAMPLE_CORPUS, HARTLIEB_MEDICAL_VOCAB,
    get_folio_text, tokenize
)

ZODIAC_MEDICAL_FORMULAS = {
    'Aries': {
        'body': 'caput',
        'conditions': ['cephalea', 'dolor capitis', 'emigranea', 'vertigo'],
        'treatments': [
            'contra dolorem capitis recipe artemisia',
            'accipe ruta cum aqua et unge caput',
            'balneum capitis cum herba calida',
            'pulvis contra cephalea cum melle',
        ],
        'humoral': 'calidus et siccus sicut ignis',
    },
    'Taurus': {
        'body': 'collum guttur',
        'conditions': ['squinancia', 'dolor gutturis', 'angina'],
        'treatments': [
            'contra dolorem gutturis recipe malva',
            'gargarisma cum aqua calida et melle',
            'emplastrum super collum cum herba',
        ],
        'humoral': 'frigidus et siccus sicut terra',
    },
    'Gemini': {
        'body': 'bracchia humeri',
        'conditions': ['dolor brachii', 'paralysis'],
        'treatments': [
            'unguentum ad bracchia cum oleo',
            'fomenta calida super humeros',
        ],
        'humoral': 'calidus et humidus sicut aer',
    },
    'Cancer': {
        'body': 'pectus mamma pulmo',
        'conditions': ['tussis', 'asma', 'dolor pectoris'],
        'treatments': [
            'contra tussim recipe pulegium cum melle',
            'potio ad pectus cum aqua calida',
            'syrupus contra asma',
        ],
        'humoral': 'frigidus et humidus sicut aqua',
    },
    'Leo': {
        'body': 'cor stomachus',
        'conditions': ['palpitatio', 'dolor stomachi', 'sincopis'],
        'treatments': [
            'contra dolorem cordis recipe myrrha',
            'electuarium ad stomachus cum melle',
        ],
        'humoral': 'calidus et siccus sicut ignis',
    },
    'Virgo': {
        'body': 'viscera intestina matrix',
        'conditions': ['dolor ventris', 'suffocatio matricis', 'sterilitas'],
        'treatments': [
            'contra suffocationem matricis recipe artemisia',
            'balneum matricis cum herba calida',
            'fomenta super ventrem cum aqua',
            'ad provocandum menstrua recipe ruta',
        ],
        'humoral': 'frigidus et siccus sicut terra',
    },
    'Libra': {
        'body': 'renes lumbi',
        'conditions': ['dolor renum', 'calculus', 'dolor lumbi'],
        'treatments': [
            'contra dolorem renum recipe petroselinum',
            'potio ad calculum cum vino',
        ],
        'humoral': 'calidus et humidus sicut aer',
    },
    'Scorpio': {
        'body': 'genitalia pudenda vulva',
        'conditions': ['dolor genitalium', 'menstrua', 'conceptio'],
        'treatments': [
            'contra dolorem pudendorum recipe sabina',
            'ad conceptum recipe dictamnus cum melle',
            'emmenagogum cum ruta et artemisia',
            'balneum ad vulva cum aqua calida',
        ],
        'humoral': 'frigidus et humidus sicut aqua',
    },
    'Sagittarius': {
        'body': 'femur coxa',
        'conditions': ['dolor coxae', 'sciatica'],
        'treatments': [
            'unguentum ad coxam cum oleo',
            'fomenta calida super femur',
        ],
        'humoral': 'calidus et siccus sicut ignis',
    },
    'Capricorn': {
        'body': 'genu genua',
        'conditions': ['dolor genuum', 'articulatio'],
        'treatments': [
            'emplastrum super genu cum herba',
            'unguentum ad articulationem cum oleo',
        ],
        'humoral': 'frigidus et siccus sicut terra',
    },
    'Aquarius': {
        'body': 'tibia crus sura',
        'conditions': ['dolor tibiae', 'tumor cruris'],
        'treatments': [
            'fomenta super crus cum aqua calida',
            'unguentum ad tibiam cum oleo',
        ],
        'humoral': 'calidus et humidus sicut aer',
    },
    'Pisces': {
        'body': 'pedes',
        'conditions': ['podagra', 'dolor pedum', 'gutta'],
        'treatments': [
            'contra podagra recipe malva cum aqua',
            'balneum pedum cum herba calida',
            'emplastrum super pedes cum melle',
        ],
        'humoral': 'frigidus et humidus sicut aqua',
    },
}

def generate_zodiac_plaintext(sign: str) -> List[str]:
    """
    Generate all probable plaintext variants for a zodiac section.
    
    Returns multiple candidate plaintexts, as the actual content
    might be any combination of body-part description, conditions,
    treatments, and humoral theory.
    """
    data = ZODIAC_MEDICAL_FORMULAS.get(sign, {})
    if not data:
        return []

    variants = []

    body = data.get('body', '')
    conditions = ' et '.join(data.get('conditions', [])[:2])
    v1 = f"{sign.lower()} regit {body} {conditions}"
    variants.append(v1)

    for treatment in data.get('treatments', []):
        variants.append(treatment)

    humoral = data.get('humoral', '')
    v3 = f"{sign.lower()} est {humoral} et regit {body}"
    variants.append(v3)

    v4 = f"{body} {conditions} {data.get('treatments', [''])[0]}"
    variants.append(v4)

    return variants

def zodiac_known_plaintext_attack(
        n_params: int = 50,
        verbose: bool = True
) -> Dict:
    """
    The core known-plaintext attack.
    
    For each zodiac section with a known month label:
    1. Generate probable plaintext(s) for that sign
    2. Encrypt each through N Naibbe parameter variants
    3. Compare each encrypted output against the actual Voynich text
       adjacent to that month label
    4. Rank parameter sets by match quality
    5. Check for cross-section consistency (same params work for multiple signs)
    """
    if verbose:
        print("  Running zodiac known-plaintext attack...")

    param_variants = []
    import random
    for seed in range(n_params):
        rng = random.Random(seed)
        param_variants.append({
            'n_tables': rng.choice([2, 3, 4, 5, 6]),
            'bigram_probability': rng.uniform(0.15, 0.6),
            'word_length_range': rng.choice([(2, 6), (3, 7), (3, 8), (4, 9)]),
            'prefix_probability': rng.uniform(0.15, 0.55),
            'suffix_probability': rng.uniform(0.2, 0.6),
            'dice_sides': 6,
            'seed': seed * 7 + 13,
        })

    section_results = {}
    for folio, zodiac_data in ZODIAC_LABELS.items():
        voynich_text = get_folio_text(folio)
        if not voynich_text.strip():
            continue

        sign = zodiac_data['zodiac_sign']
        month = zodiac_data['month_label']

        probable_plains = generate_zodiac_plaintext(sign)
        if not probable_plains:
            continue

        voynich_profile = full_statistical_profile(voynich_text, f'Voynich-{folio}')

        best_matches = []
        for param_idx, params in enumerate(param_variants):
            for plain_idx, plaintext in enumerate(probable_plains):
                try:
                    cipher = NaibbeCipher(**params)
                    ciphertext = cipher.encrypt(plaintext)

                    if not ciphertext.strip():
                        continue

                    cipher_profile = full_statistical_profile(
                        ciphertext, f'cipher_{param_idx}_{plain_idx}'
                    )
                    distance = profile_distance(cipher_profile, voynich_profile)

                    voynich_tokens = tokenize(voynich_text)
                    cipher_tokens = tokenize(ciphertext)

                    v_lengths = Counter(len(t) for t in voynich_tokens)
                    c_lengths = Counter(len(t) for t in cipher_tokens)
                    length_overlap = _distribution_overlap(v_lengths, c_lengths)

                    composite = 0.7 * distance + 0.3 * (1 - length_overlap)

                    best_matches.append({
                        'param_idx': param_idx,
                        'params': params,
                        'plaintext_variant': plain_idx,
                        'plaintext': plaintext[:60],
                        'ciphertext': ciphertext[:60],
                        'profile_distance': round(distance, 4),
                        'length_overlap': round(length_overlap, 4),
                        'composite_score': round(composite, 4),
                    })

                except Exception:
                    continue

        best_matches.sort(key=lambda x: x['composite_score'])

        section_results[folio] = {
            'zodiac_sign': sign,
            'month_label': month,
            'voynich_text_preview': voynich_text[:80],
            'n_tested': len(best_matches),
            'top_5_matches': best_matches[:5],
        }

        if verbose:
            print(f"\n  {folio} ({sign}/{month}):")
            if best_matches:
                top = best_matches[0]
                print(f"    Best match: composite={top['composite_score']:.4f} "
                      f"profile_dist={top['profile_distance']:.4f}")
                print(f"    Plaintext: {top['plaintext']}")
                print(f"    Cipher:    {top['ciphertext']}")

    consistency = _check_cross_section_consistency(section_results)

    return {
        'section_results': section_results,
        'cross_section_consistency': consistency,
    }

def _distribution_overlap(d1: Counter, d2: Counter) -> float:
    """Compute overlap between two distributions (0-1)."""
    all_keys = set(d1.keys()) | set(d2.keys())
    if not all_keys:
        return 0.0
    total1 = sum(d1.values())
    total2 = sum(d2.values())
    if total1 == 0 or total2 == 0:
        return 0.0
    overlap = sum(min(d1.get(k, 0) / total1, d2.get(k, 0) / total2)
                  for k in all_keys)
    return overlap

def _check_cross_section_consistency(section_results: Dict) -> Dict:
    """
    Check if the same cipher parameters produce good matches across
    multiple zodiac sections. Consistency = strong evidence.
    """
    param_scores = defaultdict(list)

    for folio, data in section_results.items():
        for match in data.get('top_5_matches', []):
            param_key = match['param_idx']
            param_scores[param_key].append({
                'folio': folio,
                'score': match['composite_score'],
            })

    consistent = []
    for param_idx, scores in param_scores.items():
        if len(scores) >= 2:
            avg_score = sum(s['score'] for s in scores) / len(scores)
            consistent.append({
                'param_idx': param_idx,
                'sections_matched': len(scores),
                'avg_score': round(avg_score, 4),
                'sections': [s['folio'] for s in scores],
            })

    consistent.sort(key=lambda x: (-x['sections_matched'], x['avg_score']))

    best_consistent = consistent[0] if consistent else None

    return {
        'n_consistent_params': len(consistent),
        'top_consistent': consistent[:5],
        'best_param': best_consistent,
        'interpretation': (
            f"Found {len(consistent)} parameter sets that match across multiple "
            f"zodiac sections. {'STRONG' if consistent and consistent[0]['sections_matched'] >= 3 else 'MODERATE' if consistent else 'WEAK'} "
            f"evidence for a consistent cipher configuration."
        ),
    }

def bootstrap_glyph_mappings(
        best_params: Dict,
        top_n_sections: int = 3,
        verbose: bool = True
) -> Dict:
    """
    Given the best-matching cipher parameters, attempt to bootstrap
    preliminary glyph-to-plaintext mappings.
    
    This works by:
    1. Using the best parameters to encrypt known zodiac plaintexts
    2. Aligning the cipher output with actual Voynich text
    3. Identifying consistent glyph correspondences across multiple alignments
    """
    if verbose:
        print("\n  Bootstrapping glyph mappings from best parameters...")

    if not best_params:
        return {'error': 'No consistent parameters found for bootstrap'}

    cipher = NaibbeCipher(**best_params)

    alignments = []
    for folio, zodiac_data in ZODIAC_LABELS.items():
        voynich_text = get_folio_text(folio)
        if not voynich_text.strip():
            continue

        sign = zodiac_data['zodiac_sign']
        probable_plains = generate_zodiac_plaintext(sign)
        if not probable_plains:
            continue

        for plaintext in probable_plains[:2]:
            ciphertext = cipher.encrypt(plaintext)
            voynich_tokens = tokenize(voynich_text)
            cipher_tokens = tokenize(ciphertext)

            min_len = min(len(voynich_tokens), len(cipher_tokens))
            for i in range(min_len):
                vt = voynich_tokens[i]
                ct = cipher_tokens[i]
                min_chars = min(len(vt), len(ct))
                for j in range(min_chars):
                    alignments.append((ct[j], vt[j]))

    mapping_counts = defaultdict(Counter)
    for cipher_char, voynich_char in alignments:
        mapping_counts[cipher_char][voynich_char] += 1

    tentative_mapping = {}
    for cipher_char, voynich_counts in mapping_counts.items():
        best_match = voynich_counts.most_common(1)[0]
        total = sum(voynich_counts.values())
        confidence = best_match[1] / total if total > 0 else 0

        tentative_mapping[cipher_char] = {
            'maps_to': best_match[0],
            'confidence': round(confidence, 3),
            'total_observations': total,
            'alternatives': dict(voynich_counts.most_common(3)),
        }

    if verbose:
        print(f"  Tentative mappings ({len(tentative_mapping)} characters):")
        for char, data in sorted(tentative_mapping.items(),
                                  key=lambda x: x[1]['confidence'], reverse=True):
            print(f"    '{char}' → '{data['maps_to']}' "
                  f"(confidence={data['confidence']:.2f}, "
                  f"n={data['total_observations']})")

    return {
        'tentative_mapping': tentative_mapping,
        'total_alignments': len(alignments),
        'mapping_quality': _assess_mapping_quality(tentative_mapping),
    }

def _assess_mapping_quality(mapping: Dict) -> str:
    """Assess the quality of bootstrapped mappings."""
    if not mapping:
        return "No mappings generated."

    avg_confidence = sum(d['confidence'] for d in mapping.values()) / len(mapping)
    high_conf = sum(1 for d in mapping.values() if d['confidence'] > 0.5)

    if avg_confidence > 0.6 and high_conf > len(mapping) * 0.5:
        return ("HIGH QUALITY: Majority of mappings are high-confidence. "
                "These can be used as constraints for full decryption attempts.")
    elif avg_confidence > 0.4:
        return ("MODERATE QUALITY: Some consistent mappings found. "
                "These provide partial constraints but require validation.")
    else:
        return ("LOW QUALITY: Mappings are noisy and inconsistent. "
                "The cipher parameters may need refinement, or the "
                "plaintext hypotheses need revision.")

def run(verbose: bool = True) -> Dict:
    """Run the zodiac known-plaintext attack."""
    if verbose:
        print("=" * 70)
        print("STRATEGY 5: ZODIAC KNOWN-PLAINTEXT ATTACK")
        print("=" * 70)

    results = {}

    if verbose:
        print("\n[1/2] Running known-plaintext matching across zodiac sections...")
    attack_results = zodiac_known_plaintext_attack(n_params=50, verbose=verbose)
    results['attack'] = attack_results

    if verbose:
        print("\n[2/2] Attempting glyph mapping bootstrap...")
    consistency = attack_results.get('cross_section_consistency', {})
    best = consistency.get('best_param')

    if best:
        for folio, data in attack_results['section_results'].items():
            for match in data.get('top_5_matches', []):
                if match['param_idx'] == best['param_idx']:
                    bootstrap = bootstrap_glyph_mappings(
                        match['params'],
                        verbose=verbose,
                    )
                    results['bootstrap'] = bootstrap
                    break
            if 'bootstrap' in results:
                break

    if 'bootstrap' not in results:
        results['bootstrap'] = {'error': 'No consistent parameters for bootstrap'}
        if verbose:
            print("  No consistent parameters found for bootstrap.")

    if verbose:
        print(f"\n  Cross-section consistency: "
              f"{consistency.get('interpretation', 'N/A')}")

    return results

if __name__ == '__main__':
    run()
