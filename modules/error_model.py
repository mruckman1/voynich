"""
Track 3: Scribe Error Modeling
================================
Maps cipher confidence across the manuscript by detecting local statistical
anomalies that may represent encryption errors. A scribe encrypting with
physical tools (dice, tables) makes mistakes that appear as tokens
statistically surprising in their local context.

The spatial distribution of errors tells us about the encryption process:
- Errors near illustrations → real-time encryption
- Errors near page ends → fatigue
- Random distribution → deliberate features
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistical_analysis import conditional_entropy
from data.voynich_corpus import get_all_tokens, SAMPLE_CORPUS, SECTIONS


# ============================================================================
# ERROR MODEL
# ============================================================================

class ErrorModel:
    """
    Detects and maps cipher-level errors across the Voynich Manuscript.
    Uses local surprisal (context-dependent token probability) to identify
    statistically anomalous tokens.
    """

    def __init__(self, window: int = 20, verbose: bool = True):
        self.window = window
        self.verbose = verbose

    def _build_local_bigram_model(
        self, tokens: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a bigram (word-level) transition probability model.
        P(token_i | token_{i-1})
        """
        bigram_counts = defaultdict(Counter)
        context_counts = Counter()

        for i in range(1, len(tokens)):
            prev = tokens[i - 1]
            curr = tokens[i]
            bigram_counts[prev][curr] += 1
            context_counts[prev] += 1

        model = {}
        for context, nexts in bigram_counts.items():
            total = context_counts[context]
            model[context] = {w: c / total for w, c in nexts.items()}

        return model

    def token_surprise(
        self, token: str, prev_token: Optional[str],
        bigram_model: Dict[str, Dict[str, float]],
        unigram_probs: Dict[str, float]
    ) -> float:
        """
        Compute surprisal: -log2(P(token|context)).
        Uses bigram model with unigram backoff.
        """
        if prev_token and prev_token in bigram_model:
            prob = bigram_model[prev_token].get(token, 0.0)
            if prob > 0:
                return -math.log2(prob)

        # Backoff to unigram
        prob = unigram_probs.get(token, 1e-6)
        return -math.log2(prob)

    def scan_manuscript(self) -> List[Dict]:
        """
        Compute surprise score for every token in the manuscript.
        Returns list of {folio, position, token, surprise, is_anomalous}.
        """
        # Build global models
        all_tokens = get_all_tokens()
        bigram_model = self._build_local_bigram_model(all_tokens)

        total_tokens = len(all_tokens)
        token_counts = Counter(all_tokens)
        unigram_probs = {t: c / total_tokens for t, c in token_counts.items()}

        # Scan each folio
        results = []
        for folio, data in SAMPLE_CORPUS.items():
            tokens = []
            for line in data.get('text', []):
                tokens.extend(line.split())

            for i, token in enumerate(tokens):
                prev = tokens[i - 1] if i > 0 else None
                surprise = self.token_surprise(token, prev, bigram_model, unigram_probs)

                results.append({
                    'folio': folio,
                    'position': i,
                    'total_tokens': len(tokens),
                    'token': token,
                    'surprise': surprise,
                })

        return results

    def anomaly_threshold(
        self, scores: List[float], method: str = 'iqr'
    ) -> float:
        """
        Determine the threshold for "anomalous" surprise scores.
        IQR method: Q3 + 1.5 * IQR
        """
        arr = np.array(scores)
        if method == 'iqr':
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            return float(q3 + 1.5 * iqr)
        elif method == 'percentile':
            return float(np.percentile(arr, 95))
        else:
            return float(np.mean(arr) + 2 * np.std(arr))

    def spatial_error_map(
        self, scan_results: List[Dict], threshold: float
    ) -> Dict[str, List[Dict]]:
        """
        Map anomalous tokens to their folios.
        Returns {folio: [{position, token, surprise}]}.
        """
        error_map = defaultdict(list)
        for r in scan_results:
            if r['surprise'] > threshold:
                error_map[r['folio']].append({
                    'position': r['position'],
                    'token': r['token'],
                    'surprise': r['surprise'],
                })
        return dict(error_map)

    def error_pattern_analysis(self, error_map: Dict, scan_results: List[Dict]) -> Dict:
        """
        Classify the spatial distribution of errors.
        Tests for: page-end clustering, random distribution, illustration proximity.
        """
        # Collect position fractions (relative position within page)
        position_fractions = []
        for folio, errors in error_map.items():
            for err in errors:
                total = next(
                    (r['total_tokens'] for r in scan_results
                     if r['folio'] == folio and r['position'] == 0),
                    max(err['position'] + 1, 1)
                )
                # Find total tokens for this folio
                for r in scan_results:
                    if r['folio'] == folio:
                        total = r['total_tokens']
                        break
                fraction = err['position'] / max(total, 1)
                position_fractions.append(fraction)

        if not position_fractions:
            return {
                'pattern': 'insufficient_data',
                'n_errors': 0,
            }

        fractions = np.array(position_fractions)

        # Test: are errors clustered near page ends (last 20%)?
        near_end = np.sum(fractions > 0.8) / len(fractions)
        near_start = np.sum(fractions < 0.2) / len(fractions)
        middle = np.sum((fractions >= 0.2) & (fractions <= 0.8)) / len(fractions)

        # Runs test for randomness
        median_frac = np.median(fractions)
        above = fractions > median_frac
        n_runs = 1 + np.sum(np.diff(above.astype(int)) != 0)
        n_above = np.sum(above)
        n_below = len(above) - n_above

        # Expected runs for random sequence
        if n_above > 0 and n_below > 0:
            expected_runs = (2 * n_above * n_below) / (n_above + n_below) + 1
            runs_ratio = n_runs / max(expected_runs, 1)
        else:
            runs_ratio = 1.0

        # Classify pattern
        if near_end > 0.35:
            pattern = 'page_end_clustering'
            interpretation = ('Errors cluster near page endings, suggesting scribe fatigue '
                              'during encryption. The encryption process was cognitively '
                              'demanding and accuracy degraded over time.')
        elif near_start > 0.35:
            pattern = 'page_start_clustering'
            interpretation = ('Errors cluster near page beginnings, suggesting difficulty '
                              'initializing the cipher for each new page (consistent with '
                              'cipher state reset at page boundaries).')
        elif abs(runs_ratio - 1.0) < 0.2:
            pattern = 'random'
            interpretation = ('Errors are randomly distributed, suggesting they may be '
                              'deliberate features rather than mistakes, or that the '
                              'encryption difficulty was uniform.')
        else:
            pattern = 'clustered'
            interpretation = ('Errors show non-random clustering, suggesting the cipher '
                              'mechanism has variable difficulty across different text types '
                              'or content sections.')

        return {
            'pattern': pattern,
            'interpretation': interpretation,
            'n_errors': len(position_fractions),
            'near_end_fraction': float(near_end),
            'near_start_fraction': float(near_start),
            'middle_fraction': float(middle),
            'runs_ratio': float(runs_ratio),
        }

    def reliable_pages(
        self, error_map: Dict, scan_results: List[Dict], n: int = 20
    ) -> List[Dict]:
        """
        Identify the top-N pages with fewest anomalies (highest cipher confidence).
        These are the best candidates for decryption attempts.
        """
        # Count errors per folio
        folio_error_counts = defaultdict(int)
        folio_token_counts = defaultdict(int)

        for r in scan_results:
            folio_token_counts[r['folio']] = max(
                folio_token_counts[r['folio']], r['total_tokens']
            )

        for folio, errors in error_map.items():
            folio_error_counts[folio] = len(errors)

        # All folios (including those with 0 errors)
        all_folios = set(r['folio'] for r in scan_results)

        folio_scores = []
        for folio in all_folios:
            n_tokens = folio_token_counts.get(folio, 1)
            n_errors = folio_error_counts.get(folio, 0)
            error_rate = n_errors / max(n_tokens, 1)
            folio_scores.append({
                'folio': folio,
                'n_tokens': n_tokens,
                'n_errors': n_errors,
                'error_rate': error_rate,
            })

        # Sort by error rate (ascending = most reliable)
        folio_scores.sort(key=lambda x: x['error_rate'])

        return folio_scores[:n]


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run scribe error modeling.

    Returns:
        Dict with confidence map, error patterns, and reliable page list.
    """
    model = ErrorModel(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 3: SCRIBE ERROR MODELING")
        print("=" * 70)

    # Scan manuscript
    if verbose:
        print("\n  Scanning manuscript for anomalous tokens...")
    scan_results = model.scan_manuscript()

    if verbose:
        print(f"    Scanned {len(scan_results)} tokens across "
              f"{len(set(r['folio'] for r in scan_results))} folios")

    # Compute threshold
    scores = [r['surprise'] for r in scan_results]
    threshold = model.anomaly_threshold(scores)

    n_anomalous = sum(1 for s in scores if s > threshold)
    if verbose:
        print(f"    Anomaly threshold: {threshold:.2f} bits")
        print(f"    Anomalous tokens: {n_anomalous}/{len(scores)} "
              f"({n_anomalous/max(len(scores),1)*100:.1f}%)")

    # Build error map
    error_map = model.spatial_error_map(scan_results, threshold)

    if verbose:
        print(f"    Folios with errors: {len(error_map)}")

    # Analyze error patterns
    if verbose:
        print("\n  Analyzing error distribution patterns...")
    patterns = model.error_pattern_analysis(error_map, scan_results)

    if verbose:
        print(f"    Pattern: {patterns['pattern']}")
        print(f"    {patterns['interpretation']}")
        print(f"    Near page end: {patterns.get('near_end_fraction', 0):.0%}")
        print(f"    Near page start: {patterns.get('near_start_fraction', 0):.0%}")

    # Reliable pages
    if verbose:
        print("\n  Identifying most reliable pages...")
    reliable = model.reliable_pages(error_map, scan_results, n=20)

    if verbose:
        print(f"    Top 10 most reliable pages:")
        for page in reliable[:10]:
            print(f"      {page['folio']}: {page['n_errors']} errors / "
                  f"{page['n_tokens']} tokens "
                  f"(rate={page['error_rate']:.4f})")

    # Summary statistics
    surprise_arr = np.array(scores)
    summary_stats = {
        'mean_surprise': float(np.mean(surprise_arr)),
        'std_surprise': float(np.std(surprise_arr)),
        'median_surprise': float(np.median(surprise_arr)),
        'threshold': threshold,
        'n_anomalous': n_anomalous,
        'anomaly_rate': n_anomalous / max(len(scores), 1),
    }

    results = {
        'track': 'error_model',
        'track_number': 3,
        'summary_stats': summary_stats,
        'error_pattern': patterns,
        'reliable_pages': reliable,
        'folios_with_errors': len(error_map),
        'error_map_summary': {
            folio: len(errors) for folio, errors in error_map.items()
        },
    }

    if verbose:
        print("\n" + "─" * 70)
        print("ERROR MODEL SUMMARY")
        print("─" * 70)
        print(f"  Mean surprisal: {summary_stats['mean_surprise']:.2f} bits")
        print(f"  Anomaly rate: {summary_stats['anomaly_rate']:.1%}")
        print(f"  Error pattern: {patterns['pattern']}")
        print(f"  Most reliable page: {reliable[0]['folio'] if reliable else 'N/A'}")

    return results
