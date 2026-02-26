"""
Strategy 2: Scribe Seam Cryptanalytic Analysis
================================================
Exploit the transition zones between scribes as potential information leaks.

In any collaborative cipher system, operator handoffs are the weakest link.
This module analyzes:
1. Token repetition at scribe boundaries (shared header words, "amen")
2. Entropy anomalies at transition points (cipher state resets)
3. First-word vs mid-section statistical deviations (cold start patterns)
4. Cross-scribe vocabulary overlap (shared vs unique cipher vocabulary)
"""

import sys
import os
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistical_analysis import (
    compute_all_entropy, word_frequencies, first_order_entropy,
    conditional_entropy, word_positional_entropy, classify_glyphs_by_position,
    full_statistical_profile, positional_glyph_distribution
)
from data.voynich_corpus import (
    SAMPLE_CORPUS, SCRIBES, get_all_tokens, get_scribe_text,
    get_scribe_transition_pairs, tokenize
)


# ============================================================================
# BOUNDARY TOKEN ANALYSIS
# ============================================================================

def extract_boundary_tokens(window_size: int = 10) -> List[Dict]:
    """
    Extract tokens at scribe transition boundaries.
    
    For each transition point, captures:
    - Last N tokens of outgoing scribe
    - First N tokens of incoming scribe
    - Any repeated tokens in both windows
    
    Parameters:
        window_size: Number of tokens to capture on each side of transition
    """
    transitions = get_scribe_transition_pairs()
    boundary_data = []

    for f_before, f_after, s_before, s_after in transitions:
        # Get tokens from each folio
        text_before = SAMPLE_CORPUS[f_before]['text']
        text_after = SAMPLE_CORPUS[f_after]['text']

        tokens_before = []
        for line in text_before:
            tokens_before.extend(tokenize(line))
        tokens_after = []
        for line in text_after:
            tokens_after.extend(tokenize(line))

        # Extract windows
        tail = tokens_before[-window_size:] if len(tokens_before) >= window_size \
            else tokens_before
        head = tokens_after[:window_size] if len(tokens_after) >= window_size \
            else tokens_after

        # Find repeated tokens
        tail_set = set(tail)
        head_set = set(head)
        shared = tail_set & head_set

        # Find tokens unique to boundary (not common in either scribe's full text)
        s1_all = set(get_all_tokens(scribe=s_before))
        s2_all = set(get_all_tokens(scribe=s_after))

        # Boundary-specific words: appear at boundary but rare overall
        s1_freq = Counter(get_all_tokens(scribe=s_before))
        s2_freq = Counter(get_all_tokens(scribe=s_after))
        boundary_specific = []
        for word in shared:
            s1_pct = s1_freq.get(word, 0) / max(1, sum(s1_freq.values()))
            s2_pct = s2_freq.get(word, 0) / max(1, sum(s2_freq.values()))
            avg_pct = (s1_pct + s2_pct) / 2
            boundary_specific.append((word, avg_pct))

        boundary_data.append({
            'folio_before': f_before,
            'folio_after': f_after,
            'scribe_before': s_before,
            'scribe_after': s_after,
            'tail_tokens': tail,
            'head_tokens': head,
            'shared_tokens': shared,
            'boundary_specific': sorted(boundary_specific, key=lambda x: x[1]),
            'tail_unique': tail_set - head_set,
            'head_unique': head_set - tail_set,
        })

    return boundary_data


# ============================================================================
# ENTROPY ANOMALY DETECTION AT TRANSITIONS
# ============================================================================

def transition_entropy_analysis(window_size: int = 15) -> List[Dict]:
    """
    Detect entropy anomalies at scribe transition zones.
    
    Hypothesis: If the cipher uses a running state (dice/card progression),
    a new scribe "resetting" the state would show detectable entropy spikes
    or dips in their first few words.
    
    For each transition:
    - Compute rolling character entropy in small windows across the boundary
    - Compare first-window entropy of new scribe vs their mid-section baseline
    - Flag statistically significant deviations
    """
    transitions = get_scribe_transition_pairs()
    results = []

    for f_before, f_after, s_before, s_after in transitions:
        # Get text
        text_before = ' '.join(SAMPLE_CORPUS[f_before]['text'])
        text_after = ' '.join(SAMPLE_CORPUS[f_after]['text'])

        # Compute baseline entropy for each scribe (from full section)
        baseline_before = compute_all_entropy(get_scribe_text(s_before))
        baseline_after = compute_all_entropy(get_scribe_text(s_after))

        # Rolling entropy across boundary
        combined = text_before + ' ' + text_after
        tokens = tokenize(combined)
        mid_point = len(tokenize(text_before))

        rolling_h1 = []
        rolling_h2 = []
        step = max(1, window_size // 3)

        for i in range(0, len(tokens) - window_size, step):
            window_text = ' '.join(tokens[i:i + window_size])
            h = compute_all_entropy(window_text)
            is_boundary = abs(i + window_size // 2 - mid_point) < window_size
            rolling_h1.append({
                'position': i,
                'H1': h['H1'],
                'H2': h['H2'],
                'is_boundary_zone': is_boundary,
                'scribe': s_before if i < mid_point else s_after,
            })

        # First-word entropy of incoming scribe
        first_words_after = ' '.join(tokenize(text_after)[:5])
        mid_words_after = ' '.join(tokenize(text_after)[10:20])

        first_entropy = compute_all_entropy(first_words_after) if len(first_words_after) > 10 else {}
        mid_entropy = compute_all_entropy(mid_words_after) if len(mid_words_after) > 10 else {}

        # Detect cold start: is first-window entropy significantly different from baseline?
        cold_start_detected = False
        h2_deviation = 0.0
        if first_entropy and baseline_after:
            h2_first = first_entropy.get('H2', 0)
            h2_base = baseline_after.get('H2', 0)
            h2_deviation = abs(h2_first - h2_base)
            cold_start_detected = h2_deviation > 0.3  # threshold

        results.append({
            'transition': f'{f_before} → {f_after}',
            'scribes': f'{s_before} → {s_after}',
            'baseline_before_H2': baseline_before.get('H2', 0),
            'baseline_after_H2': baseline_after.get('H2', 0),
            'first_words_H2': first_entropy.get('H2', 0),
            'mid_words_H2': mid_entropy.get('H2', 0),
            'H2_deviation': round(h2_deviation, 4),
            'cold_start_detected': cold_start_detected,
            'rolling_entropy': rolling_h1,
        })

    return results


# ============================================================================
# CROSS-SCRIBE VOCABULARY ANALYSIS
# ============================================================================

def cross_scribe_vocabulary() -> Dict:
    """
    Analyze vocabulary overlap and uniqueness across scribes.
    
    If the cipher uses shared tables, scribes should share most of their
    vocabulary. Unique words might represent:
    - Section-specific medical content
    - Different table selections
    - Cipher errors or corrections
    """
    scribe_vocabs = {}
    scribe_freqs = {}

    for scribe_id in range(1, 6):
        tokens = get_all_tokens(scribe=scribe_id)
        if tokens:
            scribe_vocabs[scribe_id] = set(tokens)
            scribe_freqs[scribe_id] = Counter(tokens)

    if len(scribe_vocabs) < 2:
        return {'error': 'Insufficient scribe data for comparison'}

    # Pairwise overlap analysis
    pairwise = {}
    for s1 in scribe_vocabs:
        for s2 in scribe_vocabs:
            if s1 >= s2:
                continue
            shared = scribe_vocabs[s1] & scribe_vocabs[s2]
            only_s1 = scribe_vocabs[s1] - scribe_vocabs[s2]
            only_s2 = scribe_vocabs[s2] - scribe_vocabs[s1]
            jaccard = len(shared) / len(scribe_vocabs[s1] | scribe_vocabs[s2]) \
                if scribe_vocabs[s1] | scribe_vocabs[s2] else 0

            pairwise[f'{s1}_vs_{s2}'] = {
                'shared_count': len(shared),
                'shared_words': sorted(shared)[:20],
                'unique_to_s1': sorted(only_s1)[:15],
                'unique_to_s2': sorted(only_s2)[:15],
                'jaccard_similarity': round(jaccard, 4),
                's1_vocab_size': len(scribe_vocabs[s1]),
                's2_vocab_size': len(scribe_vocabs[s2]),
            }

    # Universal vocabulary (words used by ALL scribes)
    if scribe_vocabs:
        universal = set.intersection(*scribe_vocabs.values())
    else:
        universal = set()

    # Hapax legomena per scribe (words occurring exactly once)
    hapax = {}
    for sid, freq in scribe_freqs.items():
        hapax[sid] = [w for w, c in freq.items() if c == 1]

    return {
        'pairwise_comparison': pairwise,
        'universal_vocabulary': sorted(universal),
        'universal_count': len(universal),
        'hapax_per_scribe': {s: len(h) for s, h in hapax.items()},
        'vocab_sizes': {s: len(v) for s, v in scribe_vocabs.items()},
    }


# ============================================================================
# FIRST-WORD PATTERN ANALYSIS
# ============================================================================

def first_word_patterns() -> Dict:
    """
    Analyze the distribution of first words on each folio/section.
    
    In cipher systems with state, the first token of a new section
    often reveals the initial state. If scribes reset to a common
    starting state, first words should cluster.
    """
    first_words_by_scribe = defaultdict(list)
    first_bigrams_by_scribe = defaultdict(list)

    for folio, data in SAMPLE_CORPUS.items():
        if data['text']:
            tokens = tokenize(data['text'][0])
            if tokens:
                first_words_by_scribe[data['scribe']].append(tokens[0])
                if len(tokens) >= 2:
                    first_bigrams_by_scribe[data['scribe']].append(
                        (tokens[0], tokens[1])
                    )

    results = {}
    for scribe_id in sorted(first_words_by_scribe.keys()):
        words = first_words_by_scribe[scribe_id]
        word_counts = Counter(words)

        # Entropy of first-word distribution
        total = len(words)
        if total > 0:
            probs = [c / total for c in word_counts.values()]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
        else:
            h = 0

        results[scribe_id] = {
            'first_word_counts': dict(word_counts.most_common(10)),
            'first_word_entropy': round(h, 4),
            'total_sections': total,
            'first_bigrams': [f"{a} {b}" for a, b in
                              Counter(first_bigrams_by_scribe[scribe_id]).most_common(5)],
        }

    return results


# ============================================================================
# POSITIONAL CLASS DIVERGENCE BETWEEN SCRIBES
# ============================================================================

def scribe_positional_divergence() -> Dict:
    """
    Compare positional glyph classifications across scribes.
    
    If scribes used different cipher tables or conventions,
    their positional glyph distributions should differ in measurable ways.
    This would indicate different encryption "dialects".
    """
    results = {}

    for scribe_id in range(1, 6):
        tokens = get_all_tokens(scribe=scribe_id)
        if not tokens:
            continue

        classifications = classify_glyphs_by_position(tokens, threshold=0.6)
        distributions = positional_glyph_distribution(tokens)

        results[scribe_id] = {
            'glyph_classes': classifications,
            'distributions': {
                char: pos_counts
                for char, pos_counts in sorted(distributions.items())
            },
        }

    # Compare classifications across scribes
    if len(results) >= 2:
        all_chars = set()
        for data in results.values():
            all_chars.update(data['glyph_classes'].keys())

        divergences = []
        for char in sorted(all_chars):
            classes = []
            for sid in sorted(results.keys()):
                cls = results[sid]['glyph_classes'].get(char, '?')
                classes.append((sid, cls))

            # Check if classification differs across scribes
            unique_classes = set(c for _, c in classes if c != '?')
            if len(unique_classes) > 1:
                divergences.append({
                    'glyph': char,
                    'classifications': dict(classes),
                    'divergent': True,
                })

        results['divergences'] = divergences
        results['divergence_count'] = len(divergences)

    return results


# ============================================================================
# COMPREHENSIVE SEAM REPORT
# ============================================================================

def run(verbose: bool = True) -> Dict:
    """Run all scribe seam analyses and compile results."""
    if verbose:
        print("=" * 70)
        print("STRATEGY 2: SCRIBE SEAM CRYPTANALYTIC ANALYSIS")
        print("=" * 70)

    results = {}

    # 1. Boundary token analysis
    if verbose:
        print("\n[1/5] Analyzing boundary tokens...")
    boundary = extract_boundary_tokens(window_size=10)
    results['boundary_tokens'] = boundary
    if verbose:
        for b in boundary:
            print(f"  Transition {b['folio_before']} → {b['folio_after']} "
                  f"(Scribe {b['scribe_before']} → {b['scribe_after']})")
            print(f"    Shared tokens: {b['shared_tokens']}")
            print(f"    Boundary-specific: {[w for w, _ in b['boundary_specific'][:5]]}")

    # 2. Entropy anomaly detection
    if verbose:
        print("\n[2/5] Detecting entropy anomalies at transitions...")
    entropy_anomalies = transition_entropy_analysis(window_size=15)
    results['entropy_anomalies'] = entropy_anomalies
    if verbose:
        for ea in entropy_anomalies:
            status = "⚡ COLD START" if ea['cold_start_detected'] else "  normal"
            print(f"  {ea['transition']} ({ea['scribes']}): "
                  f"H2 deviation={ea['H2_deviation']:.4f} {status}")

    # 3. Cross-scribe vocabulary
    if verbose:
        print("\n[3/5] Cross-scribe vocabulary analysis...")
    vocab_analysis = cross_scribe_vocabulary()
    results['vocabulary'] = vocab_analysis
    if verbose:
        print(f"  Universal vocabulary size: {vocab_analysis.get('universal_count', 0)}")
        print(f"  Universal words: {vocab_analysis.get('universal_vocabulary', [])[:15]}")
        for pair, data in vocab_analysis.get('pairwise_comparison', {}).items():
            print(f"  {pair}: Jaccard={data['jaccard_similarity']:.3f} "
                  f"shared={data['shared_count']}")

    # 4. First-word patterns
    if verbose:
        print("\n[4/5] First-word pattern analysis...")
    first_words = first_word_patterns()
    results['first_words'] = first_words
    if verbose:
        for sid, data in first_words.items():
            print(f"  Scribe {sid}: first_word_entropy={data['first_word_entropy']:.3f} "
                  f"top={list(data['first_word_counts'].keys())[:5]}")

    # 5. Positional class divergence
    if verbose:
        print("\n[5/5] Positional glyph class divergence...")
    pos_div = scribe_positional_divergence()
    results['positional_divergence'] = pos_div
    if verbose:
        div_count = pos_div.get('divergence_count', 0)
        print(f"  Divergent glyphs: {div_count}")
        for d in pos_div.get('divergences', [])[:10]:
            print(f"    '{d['glyph']}': {d['classifications']}")

    return results


if __name__ == '__main__':
    run()