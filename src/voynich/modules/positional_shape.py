"""
Track 1: Positional Entropy Shape Matching
=============================================
Uses the Voynich's positional entropy curve as a 10-dimensional fingerprint
to constrain the cipher-language space. Different cipher-language combinations
produce characteristic shapes; some shapes are geometrically impossible for
certain cipher families, enabling definitive exclusions.

The Voynich curve: low at positions 0-1, peak at 3-4, declining at 6+.
"""

import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine as cosine_distance

from voynich.core.stats import word_positional_entropy
from voynich.core.voynich_corpus import get_all_tokens, get_section_text, SECTIONS
from voynich.modules.null_framework import (
    CIPHER_FAMILIES, SOURCE_LANGUAGES, _generate_plaintext
)

class EntropyShapeAtlas:
    """
    Builds an atlas of positional entropy curves for known cipher-language
    combinations and matches the Voynich curve against it.
    """

    def __init__(self, n_positions: int = 10, n_samples: int = 100,
                 verbose: bool = True):
        self.n_positions = n_positions
        self.n_samples = n_samples
        self.verbose = verbose
        self.atlas: Dict[str, Dict] = {}

    def compute_voynich_shape(self, section: Optional[str] = None) -> np.ndarray:
        """Compute the Voynich positional entropy curve."""
        if section:
            text = get_section_text(section)
            tokens = text.split() if text else []
        else:
            tokens = get_all_tokens()

        pos_entropy = word_positional_entropy(tokens)
        curve = np.array([pos_entropy.get(i, 0.0) for i in range(self.n_positions)])
        return curve

    def compute_cipher_shape(
        self, cipher_family: str, source_language: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean ± std of positional entropy curves from synthetic ciphertext.
        Returns (mean_curve, std_curve).
        """
        factory = CIPHER_FAMILIES[cipher_family]['factory']
        curves = []

        for i in range(self.n_samples):
            seed = i * 13 + hash(cipher_family) % 997
            plaintext = _generate_plaintext(source_language, n_words=500, seed=seed)
            cipher = factory(seed)
            ciphertext = cipher.encrypt(plaintext)
            tokens = ciphertext.split()

            if len(tokens) < 20:
                continue

            pos_entropy = word_positional_entropy(tokens)
            curve = np.array([pos_entropy.get(j, 0.0) for j in range(self.n_positions)])
            curves.append(curve)

        if not curves:
            return np.zeros(self.n_positions), np.ones(self.n_positions)

        curves_arr = np.array(curves)
        return np.mean(curves_arr, axis=0), np.std(curves_arr, axis=0)

    def shape_distance(self, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
        """Cosine distance between two entropy curves."""
        norm_a = np.linalg.norm(curve_a)
        norm_b = np.linalg.norm(curve_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0
        return float(cosine_distance(curve_a, curve_b))

    def euclidean_distance(self, curve_a: np.ndarray, curve_b: np.ndarray) -> float:
        """Euclidean distance between two curves."""
        return float(np.linalg.norm(curve_a - curve_b))

    def build_atlas(self) -> Dict[str, Dict]:
        """
        Build the full atlas of cipher-language shapes.
        """
        if self.verbose:
            print("\n  Building entropy shape atlas...")

        total = len(CIPHER_FAMILIES) * len(SOURCE_LANGUAGES)
        done = 0

        for cipher_name in CIPHER_FAMILIES:
            self.atlas[cipher_name] = {}
            for lang in SOURCE_LANGUAGES:
                done += 1
                if self.verbose:
                    print(f"    [{done}/{total}] {cipher_name} × {lang}...", end='')

                t0 = time.time()
                mean_curve, std_curve = self.compute_cipher_shape(cipher_name, lang)
                elapsed = time.time() - t0

                self.atlas[cipher_name][lang] = {
                    'mean': mean_curve,
                    'std': std_curve,
                }

                if self.verbose:
                    print(f" {elapsed:.1f}s  peak_pos={np.argmax(mean_curve)}")

        return self.atlas

    def rank_cipher_families(
        self, voynich_shape: np.ndarray
    ) -> List[Tuple[str, str, float, float]]:
        """
        Rank each cipher×language combination by how well its typical shape
        matches the Voynich curve.

        Returns: List of (cipher, language, cosine_distance, z_score)
                 sorted by distance (ascending = better match).
        """
        rankings = []

        for cipher_name, lang_data in self.atlas.items():
            for lang, shape_data in lang_data.items():
                mean = shape_data['mean']
                std = shape_data['std']

                cos_dist = self.shape_distance(voynich_shape, mean)

                with np.errstate(divide='ignore', invalid='ignore'):
                    z_scores = np.where(std > 0.001,
                                        np.abs(voynich_shape - mean) / std,
                                        0.0)
                mean_z = float(np.mean(z_scores))

                rankings.append((cipher_name, lang, cos_dist, mean_z))

        rankings.sort(key=lambda x: x[2])
        return rankings

    def exclusion_test(
        self, voynich_shape: np.ndarray, alpha: float = 0.01
    ) -> List[Dict]:
        """
        Test which cipher families produce shapes incompatible with the Voynich.
        A family is excluded if the Voynich curve falls outside the 99% CI
        at the majority of positions.
        """
        excluded = []
        z_threshold = 2.576

        for cipher_name, lang_data in self.atlas.items():
            for lang, shape_data in lang_data.items():
                mean = shape_data['mean']
                std = shape_data['std']

                n_outside = 0
                for pos in range(self.n_positions):
                    if std[pos] > 0.001:
                        z = abs(voynich_shape[pos] - mean[pos]) / std[pos]
                        if z > z_threshold:
                            n_outside += 1

                fraction_outside = n_outside / self.n_positions
                if fraction_outside > 0.5:
                    excluded.append({
                        'cipher': cipher_name,
                        'language': lang,
                        'positions_outside_ci': n_outside,
                        'fraction_outside': fraction_outside,
                        'conclusion': f'{cipher_name}×{lang} EXCLUDED '
                                      f'({n_outside}/{self.n_positions} positions '
                                      f'outside 99% CI)',
                    })

        return excluded

def run(verbose: bool = True, n_samples: int = 100) -> Dict:
    """
    Run positional entropy shape matching.

    Returns:
        Dict with voynich_shape, atlas summary, rankings, exclusions,
        and per-section shapes.
    """
    atlas = EntropyShapeAtlas(n_samples=n_samples, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 1: POSITIONAL ENTROPY SHAPE MATCHING")
        print("=" * 70)

    if verbose:
        print("\n  Computing Voynich positional entropy shape...")
    voynich_shape = atlas.compute_voynich_shape()

    if verbose:
        print(f"    Curve: {[f'{v:.3f}' for v in voynich_shape]}")
        print(f"    Peak position: {np.argmax(voynich_shape)}")

    section_shapes = {}
    for section in SECTIONS:
        shape = atlas.compute_voynich_shape(section=section)
        if np.any(shape > 0):
            section_shapes[section] = {
                'curve': shape.tolist(),
                'peak_position': int(np.argmax(shape)),
            }

    atlas.build_atlas()

    if verbose:
        print("\n  Ranking cipher families by shape match...")
    rankings = atlas.rank_cipher_families(voynich_shape)

    if verbose:
        print(f"\n  {'Rank':<5} {'Cipher':<25} {'Language':<10} {'CosD':<8} {'MeanZ':<8}")
        print("  " + "─" * 56)
        for i, (cipher, lang, cos_d, mean_z) in enumerate(rankings[:10]):
            print(f"  {i+1:<5} {cipher:<25} {lang:<10} {cos_d:<8.4f} {mean_z:<8.2f}")

    if verbose:
        print("\n  Running exclusion test (99% CI)...")
    exclusions = atlas.exclusion_test(voynich_shape)

    if verbose:
        if exclusions:
            print(f"  EXCLUDED {len(exclusions)} cipher×language combinations:")
            for ex in exclusions:
                print(f"    ✗ {ex['conclusion']}")
        else:
            print("  No cipher families definitively excluded at 99% CI.")

    excluded_families = set()
    for cipher_name in CIPHER_FAMILIES:
        all_excluded = all(
            any(ex['cipher'] == cipher_name and ex['language'] == lang
                for ex in exclusions)
            for lang in SOURCE_LANGUAGES
        )
        if all_excluded:
            excluded_families.add(cipher_name)

    viable_families = [f for f in CIPHER_FAMILIES if f not in excluded_families]

    results = {
        'track': 'positional_entropy_shape',
        'track_number': 1,
        'voynich_shape': voynich_shape.tolist(),
        'voynich_peak_position': int(np.argmax(voynich_shape)),
        'section_shapes': section_shapes,
        'rankings': [
            {'cipher': c, 'language': l, 'cosine_distance': d, 'mean_z_score': z}
            for c, l, d, z in rankings
        ],
        'exclusions': exclusions,
        'excluded_families': list(excluded_families),
        'viable_families': viable_families,
        'best_match': {
            'cipher': rankings[0][0] if rankings else None,
            'language': rankings[0][1] if rankings else None,
            'distance': rankings[0][2] if rankings else None,
        },
    }

    if verbose:
        print("\n" + "─" * 70)
        print("SHAPE MATCHING SUMMARY")
        print("─" * 70)
        if excluded_families:
            print(f"  Definitively excluded families: {excluded_families}")
        print(f"  Viable cipher families: {viable_families}")
        print(f"  Best match: {results['best_match']}")

    return results
