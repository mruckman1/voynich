"""
Strategy 3: Binding Reconstruction & Sequential State Analysis
===============================================================
If the Naibbe cipher uses any running key, progressive table rotation,
or sequential state (dice/card progression), then the misbinding has been
feeding every analysis the wrong input sequence.

This module:
1. Models the original vs current quire ordering
2. Tests whether different orderings produce higher internal consistency
3. Detects sequential state patterns by measuring cross-folio entropy correlation
4. Identifies the statistical signature of missing quires 16 and 18
"""

import sys
import os
import math
import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistical_analysis import (
    compute_all_entropy, conditional_entropy, first_order_entropy,
    bigram_transition_matrix, compare_bigram_matrices,
    full_statistical_profile, profile_distance
)
from data.voynich_corpus import (
    SAMPLE_CORPUS, SECTIONS, SCRIBES, get_all_tokens,
    get_section_text, tokenize
)


# ============================================================================
# QUIRE ORDER MODELS
# ============================================================================

# Current binding order (as of Beinecke MS 408 today)
CURRENT_QUIRE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                       14, 15, 17, 19, 20, 21, 22, 23]
# Note: Quires 16 and 18 are MISSING

# Hypothetical original orders based on codicological evidence
# (waterstain patterns, catchwords, thematic continuity)
RECONSTRUCTED_ORDERS = {
    'hypothesis_A': {
        'description': 'Herbal-first, astronomical-middle, recipe-last (standard)',
        'order': [1, 2, 3, 4, 5, 6, 7, 17, 8, 9, 10, 11, 12,
                  13, 14, 15, 19, 20, 21, 22, 23],
        'rationale': 'Moves Herbal B (Q17) adjacent to Herbal A for thematic continuity',
    },
    'hypothesis_B': {
        'description': 'Biological before astronomical (anatomical sequence)',
        'order': [1, 2, 3, 4, 5, 6, 7, 17, 13, 14, 15,
                  8, 9, 10, 11, 12, 19, 20, 21, 22, 23],
        'rationale': 'Places biological/balneological before zodiac (body → stars logic)',
    },
    'hypothesis_C': {
        'description': 'Rosettes as central pivot (symmetric structure)',
        'order': [1, 2, 3, 4, 5, 6, 7, 17, 19, 20,
                  11, 12, 8, 9, 10, 13, 14, 15, 21, 22, 23],
        'rationale': 'Places cosmological rosettes at manuscript center',
    },
    'hypothesis_D': {
        'description': 'Pharmaceutical integrated with herbal (recipe book logic)',
        'order': [1, 2, 3, 4, 5, 6, 7, 17, 19, 20, 21, 22, 23,
                  8, 9, 10, 11, 12, 13, 14, 15],
        'rationale': 'Groups all plant/remedy sections together before theory sections',
    },
}

# Quire-to-folio mapping (simplified)
QUIRE_FOLIOS = {
    1: ['f1r', 'f2r', 'f3r'],
    2: ['f4r', 'f5r'],
    3: [],  # not in sample
    17: ['f87r', 'f88r'],
    8: ['f67r1'],
    9: ['f70v2'],
    13: ['f78r', 'f80r'],
    19: ['f99r'],
    20: ['f103r'],
    21: ['f108r'],
}


# ============================================================================
# SEQUENTIAL CONSISTENCY METRIC
# ============================================================================

def compute_sequential_consistency(folio_order: List[str]) -> Dict:
    """
    Measure how statistically consistent adjacent folios are
    when read in a given order.
    
    A correct ordering should show smoother entropy transitions
    between adjacent folios (assuming cipher state progresses sequentially).
    
    Returns:
        consistency_score: lower = smoother transitions = more likely correct
        entropy_deltas: list of entropy jumps between adjacent folios
    """
    entropies = []
    valid_folios = []

    for folio in folio_order:
        if folio in SAMPLE_CORPUS:
            text = ' '.join(SAMPLE_CORPUS[folio]['text'])
            if len(text) > 20:
                h = compute_all_entropy(text)
                entropies.append(h)
                valid_folios.append(folio)

    if len(entropies) < 2:
        return {
            'consistency_score': float('inf'),
            'entropy_deltas': [],
            'valid_folios': 0,
        }

    # Compute entropy deltas between adjacent folios
    deltas_h1 = []
    deltas_h2 = []
    for i in range(len(entropies) - 1):
        d1 = abs(entropies[i + 1]['H1'] - entropies[i]['H1'])
        d2 = abs(entropies[i + 1]['H2'] - entropies[i]['H2'])
        deltas_h1.append(d1)
        deltas_h2.append(d2)

    # Consistency = average entropy delta (lower = smoother)
    avg_delta_h1 = sum(deltas_h1) / len(deltas_h1) if deltas_h1 else 0
    avg_delta_h2 = sum(deltas_h2) / len(deltas_h2) if deltas_h2 else 0

    # Weighted composite
    consistency = 0.4 * avg_delta_h1 + 0.6 * avg_delta_h2

    return {
        'consistency_score': round(consistency, 6),
        'avg_delta_H1': round(avg_delta_h1, 6),
        'avg_delta_H2': round(avg_delta_h2, 6),
        'entropy_deltas_H1': [round(d, 4) for d in deltas_h1],
        'entropy_deltas_H2': [round(d, 4) for d in deltas_h2],
        'folio_entropies': [
            {'folio': f, 'H1': round(e['H1'], 4), 'H2': round(e['H2'], 4)}
            for f, e in zip(valid_folios, entropies)
        ],
        'valid_folios': len(valid_folios),
    }


def quire_order_to_folio_order(quire_order: List[int]) -> List[str]:
    """Convert a quire ordering to a folio ordering."""
    folios = []
    for q in quire_order:
        folios.extend(QUIRE_FOLIOS.get(q, []))
    return folios


# ============================================================================
# BINDING ORDER COMPARISON
# ============================================================================

def compare_binding_orders(verbose: bool = True) -> Dict:
    """
    Compare the sequential consistency of the current binding
    vs reconstructed orderings.
    """
    if verbose:
        print("\n  Comparing binding orders...")

    results = {}

    # Current order
    current_folios = quire_order_to_folio_order(CURRENT_QUIRE_ORDER)
    results['current'] = {
        'description': 'Current Beinecke binding',
        'metrics': compute_sequential_consistency(current_folios),
    }

    # Reconstructed orders
    for name, hyp in RECONSTRUCTED_ORDERS.items():
        folios = quire_order_to_folio_order(hyp['order'])
        results[name] = {
            'description': hyp['description'],
            'rationale': hyp['rationale'],
            'metrics': compute_sequential_consistency(folios),
        }

    # Rank by consistency
    ranked = sorted(results.items(),
                    key=lambda x: x[1]['metrics']['consistency_score'])

    if verbose:
        print(f"\n  {'Order':<20s} {'Score':<12s} {'Avg ΔH1':<10s} {'Avg ΔH2':<10s} Description")
        print(f"  {'-'*80}")
        for name, data in ranked:
            m = data['metrics']
            print(f"  {name:<20s} {m['consistency_score']:<12.6f} "
                  f"{m['avg_delta_H1']:<10.4f} {m['avg_delta_H2']:<10.4f} "
                  f"{data['description'][:40]}")

    results['ranking'] = [name for name, _ in ranked]
    return results


# ============================================================================
# MISSING QUIRE BOUNDARY ANALYSIS
# ============================================================================

def missing_quire_analysis(verbose: bool = True) -> Dict:
    """
    Analyze the statistical boundaries where Quires 16 and 18 are missing.
    
    If the cipher has sequential state, the gap should create a detectable
    discontinuity in the statistical progression.
    """
    if verbose:
        print("\n  Analyzing missing quire boundaries...")

    # Find folios adjacent to missing quire positions
    # Q16 sits between Q15 and Q17, Q18 between Q17 and Q19
    boundaries = {
        'Q15_Q16_gap': {
            'before_quire': 15,
            'missing_quire': 16,
            'after_quire': 17,
            'before_folios': [],  # Q15 not in our sample
            'after_folios': QUIRE_FOLIOS.get(17, []),
        },
        'Q17_Q18_gap': {
            'before_quire': 17,
            'missing_quire': 18,
            'after_quire': 19,
            'before_folios': QUIRE_FOLIOS.get(17, []),
            'after_folios': QUIRE_FOLIOS.get(19, []),
        },
    }

    results = {}
    for gap_name, gap in boundaries.items():
        before_text = ''
        after_text = ''

        for f in gap['before_folios']:
            if f in SAMPLE_CORPUS:
                before_text += ' '.join(SAMPLE_CORPUS[f]['text']) + ' '
        for f in gap['after_folios']:
            if f in SAMPLE_CORPUS:
                after_text += ' '.join(SAMPLE_CORPUS[f]['text']) + ' '

        before_entropy = compute_all_entropy(before_text) if len(before_text.strip()) > 20 else {}
        after_entropy = compute_all_entropy(after_text) if len(after_text.strip()) > 20 else {}

        # Compute entropy jump across the gap
        h2_jump = 0
        if before_entropy and after_entropy:
            h2_jump = abs(after_entropy.get('H2', 0) - before_entropy.get('H2', 0))

        # Word frequency discontinuity
        before_tokens = before_text.split() if before_text.strip() else []
        after_tokens = after_text.split() if after_text.strip() else []
        before_vocab = set(before_tokens)
        after_vocab = set(after_tokens)
        vocab_overlap = len(before_vocab & after_vocab) / max(1, len(before_vocab | after_vocab))

        results[gap_name] = {
            'missing_quire': gap['missing_quire'],
            'before_entropy': before_entropy,
            'after_entropy': after_entropy,
            'H2_jump': round(h2_jump, 4),
            'vocab_overlap': round(vocab_overlap, 4),
            'before_token_count': len(before_tokens),
            'after_token_count': len(after_tokens),
        }

        if verbose:
            print(f"  {gap_name}: H2_jump={h2_jump:.4f} "
                  f"vocab_overlap={vocab_overlap:.3f}")

    return results


# ============================================================================
# CROSS-FOLIO STATE PROGRESSION TEST
# ============================================================================

def state_progression_test(verbose: bool = True) -> Dict:
    """
    Test whether there's a detectable progressive state in the cipher.
    
    If the cipher uses running keys or progressive table rotation:
    - Adjacent folios should share more bigram patterns than distant ones
    - There should be a measurable correlation between folio distance and
      bigram matrix divergence
    """
    if verbose:
        print("\n  Testing for sequential state progression...")

    # Compute bigram matrix for each folio
    folio_matrices = {}
    sorted_folios = sorted(
        [f for f in SAMPLE_CORPUS.keys()],
        key=lambda f: (int(''.join(c for c in f if c.isdigit()) or '0'), f)
    )

    for folio in sorted_folios:
        text = ' '.join(SAMPLE_CORPUS[folio]['text'])
        if len(text.replace(' ', '')) > 30:
            mat, alph = bigram_transition_matrix(text)
            folio_matrices[folio] = (mat, alph)

    if len(folio_matrices) < 3:
        return {'error': 'Insufficient folios for state progression test'}

    # Compute pairwise bigram distances
    folios = list(folio_matrices.keys())
    distances = []
    for i in range(len(folios)):
        for j in range(i + 1, len(folios)):
            f1, f2 = folios[i], folios[j]
            m1, a1 = folio_matrices[f1]
            m2, a2 = folio_matrices[f2]
            dist = compare_bigram_matrices(m1, m2, a1, a2)
            seq_dist = abs(j - i)  # positional distance in current order
            distances.append({
                'folio_1': f1,
                'folio_2': f2,
                'bigram_distance': round(dist, 4) if dist != float('inf') else None,
                'sequential_distance': seq_dist,
            })

    # Compute correlation between sequential distance and bigram distance
    valid = [(d['sequential_distance'], d['bigram_distance'])
             for d in distances if d['bigram_distance'] is not None]

    correlation = 0.0
    if len(valid) >= 3:
        seq_dists, bg_dists = zip(*valid)
        mean_s = sum(seq_dists) / len(seq_dists)
        mean_b = sum(bg_dists) / len(bg_dists)
        cov = sum((s - mean_s) * (b - mean_b) for s, b in valid) / len(valid)
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in seq_dists) / len(seq_dists))
        std_b = math.sqrt(sum((b - mean_b) ** 2 for b in bg_dists) / len(bg_dists))
        if std_s > 0 and std_b > 0:
            correlation = cov / (std_s * std_b)

    state_detected = abs(correlation) > 0.3

    if verbose:
        print(f"  Correlation(sequential_distance, bigram_distance) = {correlation:.4f}")
        print(f"  Sequential state {'DETECTED' if state_detected else 'not detected'}")
        print(f"  (positive correlation = adjacent folios more similar = sequential state)")

    return {
        'correlation': round(correlation, 4),
        'state_detected': state_detected,
        'pairwise_distances': distances[:20],  # top 20 pairs
        'n_pairs': len(valid),
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

def run(verbose: bool = True) -> Dict:
    """Run all binding reconstruction analyses."""
    if verbose:
        print("=" * 70)
        print("STRATEGY 3: BINDING RECONSTRUCTION & SEQUENTIAL STATE ANALYSIS")
        print("=" * 70)

    results = {}

    # 1. Compare binding orders
    if verbose:
        print("\n[1/3] Comparing binding order hypotheses...")
    results['binding_comparison'] = compare_binding_orders(verbose)

    # 2. Missing quire boundary analysis
    if verbose:
        print("\n[2/3] Missing quire boundary analysis...")
    results['missing_quires'] = missing_quire_analysis(verbose)

    # 3. State progression test
    if verbose:
        print("\n[3/3] Sequential state progression test...")
    results['state_progression'] = state_progression_test(verbose)

    return results


if __name__ == '__main__':
    run()