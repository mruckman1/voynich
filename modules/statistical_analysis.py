"""
Statistical Analysis Module
============================
Implements character/word entropy, Zipf's law analysis, bigram transition
matrices, and positional glyph class extraction for Voynich text analysis.
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Any, List, Dict, Tuple, Optional

def char_frequencies(text: str) -> Dict[str, float]:
    """Compute character frequency distribution (ignoring spaces)."""
    chars = [c for c in text if c != ' ']
    total = len(chars)
    if total == 0:
        return {}
    counts = Counter(chars)
    return {c: n / total for c, n in counts.items()}

def first_order_entropy(text: str) -> float:
    """H1: Shannon entropy of individual characters."""
    freqs = char_frequencies(text)
    if not freqs:
        return 0.0
    return -sum(p * math.log2(p) for p in freqs.values() if p > 0)

def conditional_entropy(text: str, order: int = 2) -> float:
    """
    Compute conditional character entropy of given order.
    H2 = conditional entropy given 1 preceding character
    H3 = conditional entropy given 2 preceding characters
    """
    chars = [c for c in text if c != ' ']
    n = len(chars)
    if n <= order:
        return 0.0

    ngram_counts = Counter()
    context_counts = Counter()

    for i in range(n - order):
        context = tuple(chars[i:i + order])
        ngram = tuple(chars[i:i + order + 1])
        context_counts[context] += 1
        ngram_counts[ngram] += 1

    total = sum(ngram_counts.values())
    total_ctx = sum(context_counts.values())

    if total == 0 or total_ctx == 0:
        return 0.0

    h_joint = -sum((c / total) * math.log2(c / total)
                   for c in ngram_counts.values() if c > 0)
    h_ctx = -sum((c / total_ctx) * math.log2(c / total_ctx)
                 for c in context_counts.values() if c > 0)

    return h_joint - h_ctx

def compute_all_entropy(text: str) -> Dict[str, float]:
    """Compute H1, H2, H3 for a text."""
    return {
        'H1': first_order_entropy(text),
        'H2': conditional_entropy(text, order=1),
        'H3': conditional_entropy(text, order=2),
    }

def word_conditional_entropy(tokens: List[str], order: int = 1) -> float:
    """
    Compute word-level conditional entropy.

    H2_word = H(W_n | W_{n-1}) — entropy of next word given previous word(s).

    Analogous to character-level conditional_entropy() but operating on
    word tokens instead of characters. This is the critical metric for
    Phase 4's codebook model: if Language A is a whole-word codebook,
    then character-level H2 of the ciphertext equals word-bigram H2
    of the plaintext.

    Parameters:
        tokens: List of word tokens.
        order:  Context length (1 = bigram, 2 = trigram). Default 1.

    Returns:
        Conditional entropy in bits.
    """
    n = len(tokens)
    if n <= order:
        return 0.0

    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()

    for i in range(n - order):
        context = tuple(tokens[i:i + order])
        ngram = tuple(tokens[i:i + order + 1])
        context_counts[context] += 1
        ngram_counts[ngram] += 1

    total = sum(ngram_counts.values())
    total_ctx = sum(context_counts.values())

    if total == 0 or total_ctx == 0:
        return 0.0

    h_joint = -sum((c / total) * math.log2(c / total)
                   for c in ngram_counts.values() if c > 0)
    h_ctx = -sum((c / total_ctx) * math.log2(c / total_ctx)
                 for c in context_counts.values() if c > 0)

    return h_joint - h_ctx

def word_zipf_analysis(tokens: List[str]) -> Dict[str, Any]:
    """
    Compute Zipf's law analysis at the word level.
    Wrapper around zipf_analysis for clarity in Phase 4 code.
    """
    return zipf_analysis(tokens)

def word_frequencies(tokens: List[str]) -> Dict[str, int]:
    """Count word frequencies."""
    return dict(Counter(tokens))

def zipf_analysis(tokens: List[str]) -> Dict[str, Any]:
    """
    Analyze Zipf's law conformance.
    Returns rank-frequency data and R² fit.
    """
    counts = Counter(tokens)
    ranked = sorted(counts.values(), reverse=True)
    n = len(ranked)
    if n < 2:
        return {'r_squared': 0.0, 'ranks': [], 'frequencies': []}

    ranks = np.arange(1, n + 1, dtype=float)
    freqs = np.array(ranked, dtype=float)

    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs + 1e-10)

    A = np.vstack([log_ranks, np.ones(n)]).T
    try:
        result = np.linalg.lstsq(A, log_freqs, rcond=None)
        slope, intercept = result[0]
    except np.linalg.LinAlgError:
        slope, intercept = -1.0, 0.0

    predicted = slope * log_ranks + intercept
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'zipf_exponent': -slope,
        'r_squared': r_squared,
        'ranks': ranks.tolist(),
        'frequencies': freqs.tolist(),
        'top_20': counts.most_common(20),
        'vocabulary_size': n,
        'total_tokens': len(tokens),
        'type_token_ratio': n / len(tokens) if tokens else 0,
    }

def bigram_transition_matrix(text: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build a character bigram transition probability matrix.
    Returns: (matrix, alphabet) where matrix[i][j] = P(char_j | char_i)
    """
    chars = [c for c in text if c != ' ']
    alphabet = sorted(set(chars))
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    n = len(alphabet)

    counts = np.zeros((n, n), dtype=float)
    for i in range(len(chars) - 1):
        c1, c2 = chars[i], chars[i + 1]
        counts[char_to_idx[c1]][char_to_idx[c2]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, alphabet

def compare_bigram_matrices(mat_a: np.ndarray, mat_b: np.ndarray,
                            alph_a: List[str], alph_b: List[str]) -> float:
    """
    Compute Jensen-Shannon divergence between two bigram matrices.
    Only compares characters present in both alphabets.
    Lower value = more similar.
    """
    common = sorted(set(alph_a) & set(alph_b))
    if len(common) < 2:
        return float('inf')

    idx_a = [alph_a.index(c) for c in common]
    idx_b = [alph_b.index(c) for c in common]

    sub_a = mat_a[np.ix_(idx_a, idx_a)]
    sub_b = mat_b[np.ix_(idx_b, idx_b)]

    p = sub_a.flatten() + 1e-10
    q = sub_b.flatten() + 1e-10
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))

    return 0.5 * (kl_pm + kl_qm)

def positional_glyph_distribution(tokens: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Analyze character positions within words.
    Returns: {char: {'initial': n, 'medial': n, 'final': n, 'singleton': n}}
    """
    positions = defaultdict(lambda: {'initial': 0, 'medial': 0, 'final': 0, 'singleton': 0})

    for word in tokens:
        if len(word) == 1:
            positions[word[0]]['singleton'] += 1
        elif len(word) >= 2:
            positions[word[0]]['initial'] += 1
            positions[word[-1]]['final'] += 1
            for c in word[1:-1]:
                positions[c]['medial'] += 1

    return dict(positions)

def classify_glyphs_by_position(tokens: List[str],
                                threshold: float = 0.7) -> Dict[str, str]:
    """
    Classify each glyph as PREFIX (P), SUFFIX (S), MEDIAL (M), or ANY (A)
    based on positional dominance.
    
    A glyph is classified as P if >threshold of its occurrences are initial,
    S if >threshold are final, M if >threshold are medial, else A.
    """
    dist = positional_glyph_distribution(tokens)
    classifications = {}

    for char, pos_counts in dist.items():
        total = sum(pos_counts.values())
        if total == 0:
            classifications[char] = 'A'
            continue

        ratios = {k: v / total for k, v in pos_counts.items()}

        if ratios['initial'] >= threshold:
            classifications[char] = 'P'
        elif ratios['final'] >= threshold:
            classifications[char] = 'S'
        elif ratios['medial'] >= threshold:
            classifications[char] = 'M'
        else:
            classifications[char] = 'A'

    return classifications

def word_positional_entropy(tokens: List[str]) -> Dict[str, float]:
    """
    Compute entropy at each character position within words.
    Groups words by length, then computes entropy at each position.
    Returns: {'pos_0': H, 'pos_1': H, ...}
    """
    by_length = defaultdict(list)
    for t in tokens:
        by_length[len(t)].append(t)

    position_entropies = {}
    for pos in range(10):
        chars_at_pos = []
        for length, words in by_length.items():
            if length > pos:
                chars_at_pos.extend(w[pos] for w in words)
        if chars_at_pos:
            counts = Counter(chars_at_pos)
            total = len(chars_at_pos)
            h = -sum((c / total) * math.log2(c / total)
                     for c in counts.values() if c > 0)
            position_entropies[f'pos_{pos}'] = round(h, 4)

    return position_entropies

def full_statistical_profile(text: str, label: str = "text") -> Dict:
    """
    Compute a comprehensive statistical profile of a text.
    This is the core comparison fingerprint used across all attack strategies.
    """
    tokens = text.split()

    profile = {
        'label': label,
        'char_count': len(text.replace(' ', '')),
        'token_count': len(tokens),
        'entropy': compute_all_entropy(text),
        'zipf': zipf_analysis(tokens),
        'positional_entropy': word_positional_entropy(tokens),
        'glyph_classes': classify_glyphs_by_position(tokens),
        'positional_distribution': positional_glyph_distribution(tokens),
    }

    lengths = [len(t) for t in tokens]
    if lengths:
        profile['mean_word_length'] = np.mean(lengths)
        profile['std_word_length'] = np.std(lengths)
        profile['word_length_dist'] = dict(Counter(lengths))

    return profile

def profile_distance(prof_a: Dict, prof_b: Dict) -> float:
    """
    Compute a composite distance metric between two statistical profiles.
    Used to compare cipher outputs against the real Voynich profile.
    Lower = more similar.
    """
    distance = 0.0
    weights = {'H1': 1.0, 'H2': 3.0, 'H3': 3.0}

    for key, w in weights.items():
        ea = prof_a.get('entropy', {}).get(key, 0)
        eb = prof_b.get('entropy', {}).get(key, 0)
        distance += w * (ea - eb) ** 2

    za = prof_a.get('zipf', {}).get('zipf_exponent', 1.0)
    zb = prof_b.get('zipf', {}).get('zipf_exponent', 1.0)
    distance += 2.0 * (za - zb) ** 2

    mla = prof_a.get('mean_word_length', 4.0)
    mlb = prof_b.get('mean_word_length', 4.0)
    distance += 1.0 * (mla - mlb) ** 2

    ttr_a = prof_a.get('zipf', {}).get('type_token_ratio', 0.5)
    ttr_b = prof_b.get('zipf', {}).get('type_token_ratio', 0.5)
    distance += 1.5 * (ttr_a - ttr_b) ** 2

    pe_a = prof_a.get('positional_entropy', {})
    pe_b = prof_b.get('positional_entropy', {})
    common_pos = set(pe_a.keys()) & set(pe_b.keys())
    if common_pos:
        pos_dist = sum((pe_a[k] - pe_b[k]) ** 2 for k in common_pos)
        distance += 2.0 * pos_dist / len(common_pos)

    return math.sqrt(distance)

def word_length_distribution(tokens: List[str]) -> Dict:
    """
    Compute word-length distribution statistics.
    Returns histogram, mean, std, and per-length counts.
    """
    lengths = [len(t) for t in tokens if t]
    if not lengths:
        return {'lengths': [], 'mean': 0, 'std': 0, 'histogram': {}}
    arr = np.array(lengths)
    hist = Counter(lengths)
    return {
        'lengths': arr,
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr)),
        'histogram': {int(k): int(v) for k, v in sorted(hist.items())},
        'total': len(lengths),
    }
