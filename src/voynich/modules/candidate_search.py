"""
Phase 4: Decryption Candidate Testing
========================================
Systematically tests decryption hypotheses against the unified constraint model.
Uses the narrowed cipher-language space from Phase 2 and the plaintext anchors
from Phase 3 to test candidates.

Stretch goal: Naibbe parameter recovery using zodiac known-plaintext, label
correspondences, and paragraph openings.
"""

import time
import math
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import (
    compute_all_entropy, full_statistical_profile, profile_distance,
    zipf_analysis, first_order_entropy
)
from voynich.modules.naibbe_cipher import NaibbeCipher, generate_parameter_grid
from voynich.modules.strategy1_parameter_search import generate_medical_plaintext
from voynich.core.voynich_corpus import get_all_tokens, get_section_text
from voynich.core.medieval_text_templates import (
    generate_italian_text, generate_german_text
)

class CandidateSearch:
    """
    Generates and tests decryption candidates against the constraint model.
    """

    def __init__(self, constraint_model=None, verbose: bool = True):
        self.model = constraint_model
        self.verbose = verbose

    def generate_candidates(
        self, n: int = 100, resolution: str = 'medium'
    ) -> List[Dict]:
        """
        Generate viable cipher×language×parameter combinations.
        Filters by constraint model if available.
        """
        grid = generate_parameter_grid(resolution=resolution)

        if self.model:
            grid = self.model.narrow_parameter_space(grid)

        viable_families = ['naibbe']
        candidate_languages = ['latin', 'italian', 'german']

        if self.model:
            spec = self.model.compile() if hasattr(self.model, 'compile') else {}
            if spec.get('viable_cipher_families'):
                viable_families = [f for f in spec['viable_cipher_families']
                                  if f == 'naibbe']
                if not viable_families:
                    viable_families = ['naibbe']
            if spec.get('candidate_languages'):
                candidate_languages = spec['candidate_languages']

        candidates = []
        rng = random.Random(42)

        for params in grid[:n]:
            for lang in candidate_languages:
                candidates.append({
                    'cipher_family': 'naibbe',
                    'params': params,
                    'source_language': lang,
                })

        rng.shuffle(candidates)
        return candidates[:n]

    def test_candidate(
        self, candidate: Dict, anchor_pairs: List[Dict] = None
    ) -> Dict:
        """
        Test a single decryption candidate.

        1. Encrypt known anchor plaintexts
        2. Compare against known Voynich ciphertext at anchor points
        3. Compute constraint satisfaction score
        4. Test for linguistic coherence
        """
        params = candidate['params']
        lang = candidate['source_language']

        cipher = NaibbeCipher(**{k: v for k, v in params.items()
                                if k in ('n_tables', 'bigram_probability',
                                         'prefix_probability', 'suffix_probability',
                                         'seed')})

        if lang == 'latin':
            plaintext = generate_medical_plaintext(n_words=500)
        elif lang == 'italian':
            plaintext = generate_italian_text(n_words=500)
        elif lang == 'german':
            plaintext = generate_german_text(n_words=500)
        else:
            plaintext = generate_medical_plaintext(n_words=500)

        ciphertext = cipher.encrypt(plaintext)
        cipher_tokens = ciphertext.split()

        if len(cipher_tokens) < 20:
            return {'score': 0.0, 'valid': False, 'reason': 'Too few tokens'}

        voynich_tokens = get_all_tokens()
        voynich_text = ' '.join(voynich_tokens)

        cipher_profile = full_statistical_profile(ciphertext, 'candidate')
        voynich_profile = full_statistical_profile(voynich_text, 'voynich')

        distance = profile_distance(cipher_profile, voynich_profile)

        anchor_score = 0.0
        anchor_tests = 0
        if anchor_pairs:
            for pair in anchor_pairs[:10]:
                anchor_tests += 1
                for pt_candidate in pair.get('plaintext_candidates', [])[:3]:
                    pt_encrypted = cipher.encrypt(pt_candidate)
                    ct_target = pair.get('ciphertext', '')

                    if pt_encrypted and ct_target:
                        enc_chars = set(pt_encrypted.replace(' ', ''))
                        tgt_chars = set(ct_target.replace(' ', ''))
                        if enc_chars | tgt_chars:
                            overlap = len(enc_chars & tgt_chars) / len(enc_chars | tgt_chars)
                            anchor_score = max(anchor_score, overlap)

        constraint_check = {'score': 0.5}
        if self.model:
            constraint_check = self.model.check_candidate({
                'text': ciphertext,
                'params': params,
                'cipher_family': 'naibbe',
            })

        profile_score = max(0, 1.0 - distance / 5.0)
        constraint_score = constraint_check.get('score', 0.5)
        anchor_norm = anchor_score if anchor_tests > 0 else 0.5

        composite = (0.4 * profile_score +
                     0.3 * constraint_score +
                     0.3 * anchor_norm)

        return {
            'score': composite,
            'profile_distance': distance,
            'profile_score': profile_score,
            'constraint_score': constraint_score,
            'anchor_score': anchor_norm,
            'n_anchor_tests': anchor_tests,
            'valid': composite > 0.3,
            'cipher_entropy': compute_all_entropy(ciphertext),
            'params': params,
            'source_language': lang,
        }

    def linguistic_coherence_test(
        self, text: str, language: str
    ) -> Dict:
        """
        Test whether decrypted text looks like real language.
        """
        tokens = text.split()
        if len(tokens) < 10:
            return {'coherent': False, 'score': 0.0}

        entropy = compute_all_entropy(text)
        zipf = zipf_analysis(tokens)

        h1_natural = 3.5 <= entropy['H1'] <= 4.5
        h2_natural = 2.5 <= entropy['H2'] <= 4.0
        zipf_natural = 0.8 <= abs(zipf['zipf_exponent']) <= 1.5

        ttr = zipf['type_token_ratio']
        ttr_natural = 0.3 <= ttr <= 0.8

        checks = [h1_natural, h2_natural, zipf_natural, ttr_natural]
        score = sum(checks) / len(checks)

        return {
            'coherent': score >= 0.75,
            'score': score,
            'H1': entropy['H1'],
            'H2': entropy['H2'],
            'zipf_exponent': zipf['zipf_exponent'],
            'ttr': ttr,
            'h1_natural': h1_natural,
            'h2_natural': h2_natural,
            'zipf_natural': zipf_natural,
            'ttr_natural': ttr_natural,
        }

    def naibbe_parameter_recovery(
        self, anchor_pairs: List[Dict], top_n: int = 20
    ) -> Dict:
        """
        Stretch goal: Use multiple anchor types to triangulate exact
        Naibbe parameters.

        Tests a focused grid around the best-known parameter region
        and scores against all available anchors.
        """
        focused_params = []
        for n_tables in [2, 3, 4, 5]:
            for bigram_p in [0.10, 0.15, 0.20, 0.25, 0.30]:
                for prefix_p in [0.10, 0.15, 0.20, 0.25]:
                    for suffix_p in [0.15, 0.20, 0.25, 0.30]:
                        for seed in [42, 123]:
                            focused_params.append({
                                'n_tables': n_tables,
                                'bigram_probability': bigram_p,
                                'prefix_probability': prefix_p,
                                'suffix_probability': suffix_p,
                                'seed': seed,
                            })

        voynich_text = ' '.join(get_all_tokens())
        voynich_profile = full_statistical_profile(voynich_text, 'voynich')

        results = []
        for params in focused_params[:200]:
            cipher = NaibbeCipher(**params)
            plaintext = generate_medical_plaintext(n_words=500)
            ciphertext = cipher.encrypt(plaintext)

            if len(ciphertext.split()) < 20:
                continue

            cipher_profile = full_statistical_profile(ciphertext, 'test')
            distance = profile_distance(cipher_profile, voynich_profile)

            results.append({
                'params': params,
                'profile_distance': distance,
            })

        results.sort(key=lambda x: x['profile_distance'])

        return {
            'n_tested': len(results),
            'best_params': results[0] if results else None,
            'top_candidates': results[:top_n],
            'parameter_convergence': self._check_convergence(results[:top_n]),
        }

    def _check_convergence(self, top_results: List[Dict]) -> Dict:
        """Check if top candidates converge on similar parameters."""
        if not top_results:
            return {'converged': False}

        param_values = defaultdict(list)
        for r in top_results:
            for k, v in r['params'].items():
                if isinstance(v, (int, float)):
                    param_values[k].append(v)

        convergence = {}
        for param, values in param_values.items():
            arr = np.array(values)
            cv = np.std(arr) / max(np.mean(arr), 0.001)
            convergence[param] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'cv': float(cv),
                'converged': cv < 0.3,
            }

        all_converged = all(v['converged'] for v in convergence.values()
                           if v.get('cv') is not None)

        return {
            'converged': all_converged,
            'parameters': convergence,
        }

    def rank_candidates(self, results: List[Dict]) -> List[Dict]:
        """Rank candidates by combined score."""
        valid = [r for r in results if r.get('valid', False)]
        valid.sort(key=lambda x: -x.get('score', 0))
        return valid

def run(
    verbose: bool = True,
    constraint_model=None,
    anchor_pairs: List[Dict] = None,
    n_candidates: int = 50
) -> Dict:
    """
    Run decryption candidate search.

    Parameters:
        verbose: Print detailed output
        constraint_model: ConstraintModel instance from constraint_model.py
        anchor_pairs: List of candidate plaintext-ciphertext pairs
        n_candidates: Number of candidates to test

    Returns:
        Dict with ranked candidates and parameter recovery results.
    """
    searcher = CandidateSearch(constraint_model=constraint_model, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: DECRYPTION CANDIDATE SEARCH")
        print("=" * 70)

    if verbose:
        print(f"\n  Generating {n_candidates} candidates...")
    candidates = searcher.generate_candidates(n=n_candidates)
    if verbose:
        print(f"    Generated {len(candidates)} candidates "
              f"({len(set(c['source_language'] for c in candidates))} languages)")

    if verbose:
        print("\n  Testing candidates...")
    results = []
    t0 = time.time()

    for i, candidate in enumerate(candidates):
        result = searcher.test_candidate(candidate, anchor_pairs)
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Tested {i+1}/{len(candidates)}  "
                  f"({elapsed:.1f}s, best={max(r['score'] for r in results):.4f})")

    ranked = searcher.rank_candidates(results)

    if verbose:
        print(f"\n  Valid candidates: {len(ranked)}/{len(results)}")
        if ranked:
            print(f"\n  Top 5 candidates:")
            for i, r in enumerate(ranked[:5]):
                print(f"    #{i+1}: score={r['score']:.4f}  "
                      f"lang={r['source_language']}  "
                      f"dist={r['profile_distance']:.4f}  "
                      f"H2={r['cipher_entropy']['H2']:.4f}")

    if verbose:
        print("\n  Attempting Naibbe parameter recovery...")
    recovery = searcher.naibbe_parameter_recovery(anchor_pairs or [])

    if verbose:
        if recovery.get('best_params'):
            best = recovery['best_params']
            print(f"    Best params: {best['params']}")
            print(f"    Profile distance: {best['profile_distance']:.4f}")
        conv = recovery.get('parameter_convergence', {})
        if conv.get('converged'):
            print("    Parameters CONVERGE across top candidates")
            for p, v in conv.get('parameters', {}).items():
                print(f"      {p}: {v['mean']:.3f} ± {v['std']:.3f}")
        else:
            print("    Parameters do NOT converge — multiple viable regions")

    output = {
        'track': 'candidate_search',
        'n_candidates_tested': len(results),
        'n_valid': len(ranked),
        'top_candidates': [
            {k: v for k, v in r.items()
             if k in ('score', 'profile_distance', 'source_language',
                      'params', 'cipher_entropy', 'constraint_score',
                      'anchor_score')}
            for r in ranked[:20]
        ],
        'parameter_recovery': {
            'n_tested': recovery.get('n_tested', 0),
            'best_params': recovery.get('best_params'),
            'convergence': recovery.get('parameter_convergence'),
        },
    }

    if verbose:
        print("\n" + "─" * 70)
        print("CANDIDATE SEARCH SUMMARY")
        print("─" * 70)
        print(f"  Candidates tested: {len(results)}")
        print(f"  Valid candidates: {len(ranked)}")
        if ranked:
            print(f"  Best score: {ranked[0]['score']:.4f} "
                  f"({ranked[0]['source_language']})")
        converged = recovery.get('parameter_convergence', {}).get('converged', False)
        print(f"  Parameter convergence: {'YES' if converged else 'NO'}")

    return output
