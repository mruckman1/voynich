"""
Track 7: Bigram Matrix Factorization (NMF)
=============================================
Determines the effective dimensionality of the Voynich character transition
system via Non-negative Matrix Factorization. NMF decomposes the bigram matrix
into interpretable components corresponding to functional character classes.

If NMF classes align with positional classes from Strategy 4, that's convergent
evidence for the grammar-layer hypothesis.

Effective rank: 2-3 = strong structure preservation, 8+ = near-random transitions.
"""

import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import bigram_transition_matrix
from voynich.core.voynich_corpus import get_all_tokens, get_section_text, SECTIONS

def nmf_multiplicative_update(
    V: np.ndarray, rank: int, max_iter: int = 200, tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Non-negative Matrix Factorization via multiplicative update rules.
    V ≈ W @ H where V, W, H >= 0.

    Lee & Seung (2001) update rules:
        H <- H * (W^T V) / (W^T W H + eps)
        W <- W * (V H^T) / (W H H^T + eps)

    Returns: (W, H, reconstruction_error)
    """
    n, m = V.shape
    eps = 1e-10

    rng = np.random.RandomState(42)
    W = rng.rand(n, rank) + eps
    H = rng.rand(rank, m) + eps

    prev_error = None
    for iteration in range(max_iter):
        numerator_h = W.T @ V
        denominator_h = W.T @ W @ H + eps
        H *= numerator_h / denominator_h

        numerator_w = V @ H.T
        denominator_w = W @ H @ H.T + eps
        W *= numerator_w / denominator_w

        if iteration % 10 == 0:
            error = float(np.linalg.norm(V - W @ H, 'fro'))
            if prev_error is not None and prev_error > 0:
                if abs(prev_error - error) / prev_error < tol:
                    break
            prev_error = error

    final_error = float(np.linalg.norm(V - W @ H, 'fro'))
    return W, H, final_error

def simple_nmf(V: np.ndarray, k: int,
               max_iter: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple multiplicative-update NMF without early termination.

    Used by Phase 4 semantic compression and Phase 5 NMF scaffold.
    Differs from nmf_multiplicative_update() in initialization (0.1 vs eps)
    and convergence (always runs full max_iter iterations).
    """
    rng = np.random.RandomState(seed)
    n, m = V.shape

    W = rng.rand(n, k) + 0.1
    H = rng.rand(k, m) + 0.1

    for _ in range(max_iter):
        numerator = W.T @ V
        denominator = W.T @ W @ H + 1e-10
        H *= numerator / denominator

        numerator = V @ H.T
        denominator = W @ H @ H.T + 1e-10
        W *= numerator / denominator

    error = float(np.linalg.norm(V - W @ H, 'fro'))
    return W, H, error

class BigramNMF:
    """Applies NMF to bigram transition matrices to find character classes."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def build_bigram_matrix(self, tokens: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Build dense bigram transition matrix from tokens.
        Returns (matrix, alphabet_list).
        """
        text = ' '.join(tokens)
        matrix, alphabet = bigram_transition_matrix(text)

        if isinstance(matrix, list):
            matrix = np.array(matrix)

        return matrix, list(alphabet)

    def optimal_rank(
        self, matrix: np.ndarray, max_rank: int = 15
    ) -> Tuple[int, List[float]]:
        """
        Find optimal NMF rank using reconstruction error elbow method.
        Tests ranks 2 through max_rank and selects the elbow point.

        Returns: (optimal_rank, errors_by_rank)
        """
        errors = []
        ranks = list(range(2, min(max_rank + 1, min(matrix.shape))))

        for rank in ranks:
            _, _, error = nmf_multiplicative_update(matrix, rank, max_iter=100)
            norm = np.linalg.norm(matrix, 'fro')
            rel_error = error / max(norm, 1e-10)
            errors.append(rel_error)

        if len(errors) >= 3:
            diffs = np.diff(errors)
            second_diffs = np.diff(diffs)
            elbow_idx = int(np.argmax(np.abs(second_diffs))) + 2
            optimal = ranks[min(elbow_idx, len(ranks) - 1)]
        else:
            optimal = ranks[0] if ranks else 2

        return optimal, list(zip(ranks, errors))

    def factorize(
        self, matrix: np.ndarray, rank: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Factorize bigram matrix: V ≈ W @ H."""
        return nmf_multiplicative_update(matrix, rank, max_iter=300)

    def interpret_components(
        self, W: np.ndarray, H: np.ndarray, alphabet: List[str]
    ) -> List[Dict]:
        """
        Interpret NMF components as functional character classes.

        W columns show which characters are "produced by" each component.
        H rows show which characters "precede" each component.
        """
        n_components = W.shape[1]
        components = []

        for k in range(n_components):
            w_col = W[:, k]
            top_targets_idx = np.argsort(w_col)[::-1][:5]
            top_targets = [(alphabet[i], float(w_col[i])) for i in top_targets_idx
                          if i < len(alphabet)]

            h_row = H[k, :]
            top_sources_idx = np.argsort(h_row)[::-1][:5]
            top_sources = [(alphabet[i], float(h_row[i])) for i in top_sources_idx
                          if i < len(alphabet)]

            target_chars = set(c for c, _ in top_targets[:3])
            label = self._classify_component(target_chars)

            components.append({
                'component': k,
                'label': label,
                'top_targets': top_targets,
                'top_sources': top_sources,
                'target_variance': float(np.var(w_col)),
                'source_variance': float(np.var(h_row)),
            })

        return components

    def _classify_component(self, chars: set) -> str:
        """Classify a component based on its top characters."""
        vowel_like = {'o', 'a', 'e', 'i'}
        gallows = {'d', 'k', 't', 'p', 'f', 's'}
        finals = {'n', 'y', 'm'}
        bench = {'c', 'h', 'l', 'r'}

        n_vowel = len(chars & vowel_like)
        n_gallows = len(chars & gallows)
        n_final = len(chars & finals)
        n_bench = len(chars & bench)

        scores = {
            'vowel-like': n_vowel,
            'gallows-like': n_gallows,
            'final-like': n_final,
            'bench-like': n_bench,
        }
        return max(scores, key=scores.get)

    def compare_with_positional_classes(
        self, nmf_components: List[Dict], positional_classes: Dict
    ) -> Dict:
        """
        Compare NMF-derived character classes with Strategy 4's positional classes.
        Returns overlap score measuring convergent evidence.
        """
        nmf_classes = defaultdict(set)
        for comp in nmf_components:
            label = comp['label']
            for char, weight in comp['top_targets'][:3]:
                nmf_classes[label].add(char)

        pos_classes = defaultdict(set)
        for char, cls in positional_classes.items():
            pos_classes[cls].add(char)

        label_map = {
            'gallows-like': 'PREFIX',
            'final-like': 'SUFFIX',
            'vowel-like': 'MEDIAL',
            'bench-like': 'ANY',
        }

        total_overlap = 0
        total_chars = 0
        agreements = []

        for nmf_label, mapped_pos_label in label_map.items():
            nmf_set = nmf_classes.get(nmf_label, set())
            pos_set = pos_classes.get(mapped_pos_label, set())
            if nmf_set and pos_set:
                overlap = len(nmf_set & pos_set)
                union = len(nmf_set | pos_set)
                jaccard = overlap / max(union, 1)
                total_overlap += overlap
                total_chars += union
                agreements.append({
                    'nmf_class': nmf_label,
                    'positional_class': mapped_pos_label,
                    'nmf_chars': sorted(nmf_set),
                    'pos_chars': sorted(pos_set),
                    'jaccard': jaccard,
                })

        overall_overlap = total_overlap / max(total_chars, 1)

        return {
            'agreements': agreements,
            'overall_overlap': overall_overlap,
            'convergent_evidence': overall_overlap > 0.3,
            'interpretation': (
                f'NMF and positional classes show {overall_overlap:.0%} overlap. '
                + ('This is CONVERGENT EVIDENCE supporting the grammar-layer hypothesis.'
                   if overall_overlap > 0.3 else
                   'Limited overlap — NMF classes may capture different structure.')
            ),
        }

def run(verbose: bool = True) -> Dict:
    """
    Run bigram matrix factorization analysis.

    Returns:
        Dict with effective rank, NMF components, section comparison,
        and positional class convergence.
    """
    analyzer = BigramNMF(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 7: BIGRAM MATRIX FACTORIZATION (NMF)")
        print("=" * 70)

    tokens = get_all_tokens()
    if verbose:
        print(f"\n  Building bigram matrix ({len(tokens)} tokens)...")

    matrix, alphabet = analyzer.build_bigram_matrix(tokens)

    if verbose:
        print(f"    Matrix shape: {matrix.shape}")
        print(f"    Alphabet: {''.join(alphabet)}")

    if verbose:
        print("\n  Finding optimal NMF rank...")
    optimal, rank_errors = analyzer.optimal_rank(matrix)

    if verbose:
        print(f"    Optimal rank: {optimal}")
        for rank, err in rank_errors:
            marker = " ← optimal" if rank == optimal else ""
            print(f"      rank {rank}: error={err:.4f}{marker}")

    if verbose:
        print(f"\n  Factorizing at rank {optimal}...")
    W, H, error = analyzer.factorize(matrix, optimal)

    components = analyzer.interpret_components(W, H, alphabet)

    if verbose:
        print(f"    Reconstruction error: {error:.4f}")
        for comp in components:
            targets = ', '.join(f'{c}({w:.2f})' for c, w in comp['top_targets'][:3])
            print(f"    Component {comp['component']} [{comp['label']}]: {targets}")

    if verbose:
        print("\n  Per-section NMF rank analysis...")
    section_ranks = {}
    for section in SECTIONS:
        sec_text = get_section_text(section)
        sec_tokens = sec_text.split() if sec_text else []
        if len(sec_tokens) > 50:
            sec_matrix, sec_alpha = analyzer.build_bigram_matrix(sec_tokens)
            if sec_matrix.shape[0] >= 4:
                sec_optimal, _ = analyzer.optimal_rank(sec_matrix, max_rank=10)
                section_ranks[section] = sec_optimal
                if verbose:
                    print(f"    {section}: rank={sec_optimal}")

    try:
        from voynich.modules.strategy4_positional_grammar import extract_glyph_classes
        pos_classes = extract_glyph_classes()
        simple_pos = {}
        for char, data in pos_classes.items():
            if isinstance(data, dict):
                simple_pos[char] = data.get('class', 'ANY')
            else:
                simple_pos[char] = str(data)
        convergence = analyzer.compare_with_positional_classes(components, simple_pos)
    except Exception:
        convergence = {
            'overall_overlap': 0.0,
            'convergent_evidence': False,
            'interpretation': 'Could not compare with positional classes (Strategy 4 not available).',
            'agreements': [],
        }

    if optimal <= 3:
        rank_interpretation = ('VERY LOW rank: strong structure preservation. '
                               'The bigram system has only a few underlying patterns, '
                               'suggesting a structure-preserving cipher mechanism.')
    elif optimal <= 5:
        rank_interpretation = ('LOW-MODERATE rank: significant structure preserved. '
                               'The cipher retains meaningful character-class distinctions.')
    elif optimal <= 8:
        rank_interpretation = ('MODERATE rank: partial structure preservation. '
                               'Some character classes are distinguishable, others blurred.')
    else:
        rank_interpretation = ('HIGH rank: near-random transitions. '
                               'The cipher significantly destroys character-class structure.')

    results = {
        'track': 'nmf_analysis',
        'track_number': 7,
        'effective_rank': optimal,
        'rank_interpretation': rank_interpretation,
        'rank_errors': [{'rank': r, 'error': e} for r, e in rank_errors],
        'components': [
            {k: v for k, v in comp.items()
             if k not in ('target_variance', 'source_variance')}
            for comp in components
        ],
        'reconstruction_error': error,
        'section_ranks': section_ranks,
        'positional_class_convergence': convergence,
        'matrix_shape': list(matrix.shape),
        'alphabet': alphabet,
    }

    if verbose:
        print("\n" + "─" * 70)
        print("NMF ANALYSIS SUMMARY")
        print("─" * 70)
        print(f"  Effective rank: {optimal}")
        print(f"  {rank_interpretation}")
        print(f"  Positional class convergence: {convergence['interpretation']}")

    return results
