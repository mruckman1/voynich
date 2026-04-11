"""
Track 10: Null Hypothesis Framework
=====================================
Builds reference distributions for every metric so Voynich observations
can be assigned p-values. Implements 4 additional cipher families alongside
the existing Naibbe cipher to create a comprehensive null distribution engine.

The central problem: a Voynich H2 of 2.36 means nothing until we know
what H2 looks like for encrypted text across a range of languages and ciphers.

Output: empirical distributions and percentile ranks for all Voynich metrics.
"""

import os
import json
import math
import random
import string
import time
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import (
    full_statistical_profile, compute_all_entropy, first_order_entropy,
    conditional_entropy, bigram_transition_matrix, compare_bigram_matrices,
    zipf_analysis, word_positional_entropy
)
from voynich.modules.naibbe_cipher import NaibbeCipher, generate_parameter_grid
from voynich.modules.strategy1_parameter_search import generate_medical_plaintext
from voynich.core.voynich_corpus import get_all_tokens, get_section_text
from voynich.core.medieval_text_templates import generate_italian_text, generate_german_text

class SimpleSubstitutionCipher:
    """
    Monoalphabetic substitution cipher.
    Each plaintext letter maps to exactly one ciphertext letter.
    Preserves word boundaries, word lengths, and letter frequencies.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        plain_alpha = list('abcdefghijklmnopqrstuvwxyz')
        cipher_alpha = plain_alpha.copy()
        self.rng.shuffle(cipher_alpha)
        self.table = dict(zip(plain_alpha, cipher_alpha))
        self.name = 'simple_substitution'

    def encrypt(self, plaintext: str) -> str:
        result = []
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch in self.table:
                result.append(self.table[ch])
            elif ch.isalpha():
                result.append(ch)
        return ''.join(result)

class VigenereCipher:
    """
    Polyalphabetic substitution cipher (Vigenere family).
    Uses a repeating key to shift each plaintext letter.
    Destroys single-letter frequency patterns but preserves word boundaries.
    """

    def __init__(self, key_length: int = 5, seed: int = 42):
        self.rng = random.Random(seed)
        self.key = [self.rng.randint(1, 25) for _ in range(key_length)]
        self.name = 'vigenere'

    def encrypt(self, plaintext: str) -> str:
        result = []
        key_idx = 0
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch.isalpha():
                shifted = (ord(ch) - ord('a') + self.key[key_idx % len(self.key)]) % 26
                result.append(chr(shifted + ord('a')))
                key_idx += 1
        return ''.join(result)

class HomophonicCipher:
    """
    Homophonic substitution cipher.
    High-frequency letters get multiple ciphertext symbols, flattening
    the frequency distribution. Uses a 40-symbol output alphabet.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.name = 'homophonic'

        freq_tiers = {
            'high': list('aeioust'),
            'medium': list('cnrldhm'),
            'low': list('bfgkpqvwxyz'),
        }

        symbols = [f'{chr(a)}{chr(b)}' for a in range(ord('a'), ord('z') + 1)
                   for b in range(ord('a'), ord('z') + 1)]
        self.rng.shuffle(symbols)
        sym_idx = 0

        self.table: Dict[str, List[str]] = {}
        for ch in freq_tiers['high']:
            n = self.rng.randint(3, 5)
            self.table[ch] = symbols[sym_idx:sym_idx + n]
            sym_idx += n
        for ch in freq_tiers['medium']:
            n = self.rng.randint(2, 3)
            self.table[ch] = symbols[sym_idx:sym_idx + n]
            sym_idx += n
        for ch in freq_tiers['low']:
            self.table[ch] = [symbols[sym_idx]]
            sym_idx += 1
        for ch in string.ascii_lowercase:
            if ch not in self.table:
                self.table[ch] = [symbols[sym_idx]]
                sym_idx += 1

    def encrypt(self, plaintext: str) -> str:
        result = []
        for ch in plaintext.lower():
            if ch == ' ':
                result.append(' ')
            elif ch in self.table:
                result.append(self.rng.choice(self.table[ch]))
        return ''.join(result)

class NomenclatorCipher:
    """
    Nomenclator: code + cipher hybrid.
    Common words get unique code symbols; remaining text uses simple substitution.
    This was the most common diplomatic cipher of the 15th century.
    """

    def __init__(self, n_code_words: int = 50, seed: int = 42):
        self.rng = random.Random(seed)
        self.name = 'nomenclator'

        code_symbols = [f'#{i:03d}' for i in range(n_code_words)]
        self.rng.shuffle(code_symbols)

        common_words = [
            'et', 'in', 'de', 'ad', 'cum', 'per', 'est', 'non', 'sed',
            'qui', 'que', 'ut', 'sic', 'hoc', 'aut', 'vel', 'pro', 'quod',
            'recipe', 'accipe', 'herba', 'aqua', 'contra', 'super', 'ante',
            'post', 'inter', 'sub', 'oleum', 'pulvis', 'misce', 'contere',
            'balneum', 'matrix', 'dolorem', 'febrem', 'sanguinem', 'ventrem',
            'caput', 'corpus', 'manum', 'oculum', 'aurem', 'pedem',
            'calida', 'frigida', 'sicca', 'humida', 'bona', 'mala',
        ]
        self.code_table = dict(zip(common_words[:n_code_words],
                                   code_symbols[:n_code_words]))

        plain_alpha = list('abcdefghijklmnopqrstuvwxyz')
        cipher_alpha = plain_alpha.copy()
        self.rng.shuffle(cipher_alpha)
        self.sub_table = dict(zip(plain_alpha, cipher_alpha))

    def encrypt(self, plaintext: str) -> str:
        words = plaintext.lower().split()
        result = []
        for word in words:
            clean = ''.join(c for c in word if c.isalpha())
            if clean in self.code_table:
                result.append(self.code_table[clean])
            else:
                encrypted = ''.join(self.sub_table.get(c, c) for c in clean)
                result.append(encrypted)
        return ' '.join(result)

CIPHER_FAMILIES = {
    'simple_substitution': {
        'factory': lambda seed: SimpleSubstitutionCipher(seed=seed),
        'description': 'Monoalphabetic 1-to-1 letter substitution',
        'preserves': 'word boundaries, word lengths, letter frequencies (permuted)',
    },
    'vigenere': {
        'factory': lambda seed: VigenereCipher(key_length=random.Random(seed).randint(3, 12), seed=seed),
        'description': 'Polyalphabetic cipher with random key length 3-12',
        'preserves': 'word boundaries, word lengths',
    },
    'homophonic': {
        'factory': lambda seed: HomophonicCipher(seed=seed),
        'description': 'Homophonic substitution flattening frequency distribution',
        'preserves': 'word boundaries (approximately)',
    },
    'nomenclator': {
        'factory': lambda seed: NomenclatorCipher(seed=seed),
        'description': 'Code+cipher hybrid, top-50 words get code symbols',
        'preserves': 'word count, partial word structure',
    },
    'naibbe': {
        'factory': lambda seed: NaibbeCipher(
            n_tables=random.Random(seed).randint(2, 6),
            bigram_probability=random.Random(seed).uniform(0.1, 0.4),
            prefix_probability=random.Random(seed).uniform(0.1, 0.4),
            suffix_probability=random.Random(seed).uniform(0.1, 0.4),
            seed=seed,
        ),
        'description': 'Multi-table combinatorial cipher (Voynich-targeted)',
        'preserves': 'Voynich-like positional structure',
    },
}

def _generate_plaintext(language: str, n_words: int = 500, seed: int = 42) -> str:
    """Generate synthetic plaintext for a given source language."""
    if language == 'latin':
        return generate_medical_plaintext(n_words=n_words)
    elif language == 'italian':
        return generate_italian_text(n_words=n_words, seed=seed)
    elif language == 'german':
        return generate_german_text(n_words=n_words, seed=seed)
    else:
        raise ValueError(f"Unknown language: {language}")

SOURCE_LANGUAGES = ['latin', 'italian', 'german']

class NullDistributionEngine:
    """
    Generates reference distributions for all metrics across cipher×language
    combinations, enabling p-value computation for Voynich observations.
    """

    def __init__(self, n_samples: int = 200, n_words: int = 500, verbose: bool = True):
        self.n_samples = n_samples
        self.n_words = n_words
        self.verbose = verbose
        self.distributions: Dict[str, Dict[str, np.ndarray]] = {}
        self.voynich_metrics: Dict[str, float] = {}

    def compute_voynich_baselines(self) -> Dict[str, float]:
        """Compute all metric values for the actual Voynich corpus."""
        tokens = get_all_tokens()
        text = ' '.join(tokens)

        entropy = compute_all_entropy(text)
        zipf = zipf_analysis(tokens)
        pos_entropy = word_positional_entropy(tokens)

        self.voynich_metrics = {
            'H1': entropy['H1'],
            'H2': entropy['H2'],
            'H3': entropy['H3'],
            'zipf_exponent': abs(zipf['zipf_exponent']),
            'type_token_ratio': zipf['type_token_ratio'],
            'mean_word_length': np.mean([len(t) for t in tokens]),
        }

        pos_values = [pos_entropy.get(i, 0.0) for i in range(10)]
        self.voynich_metrics['positional_entropy_curve'] = pos_values

        return self.voynich_metrics

    def generate_reference_corpus(
        self, cipher_family: str, source_language: str
    ) -> List[str]:
        """Generate encrypted samples for a given cipher×language pair."""
        factory = CIPHER_FAMILIES[cipher_family]['factory']
        samples = []

        for i in range(self.n_samples):
            seed = i * 7 + hash(cipher_family) % 1000 + hash(source_language) % 1000
            plaintext = _generate_plaintext(source_language, n_words=self.n_words,
                                            seed=seed)
            cipher = factory(seed)
            ciphertext = cipher.encrypt(plaintext)
            if ciphertext and len(ciphertext.split()) > 10:
                samples.append(ciphertext)

        return samples

    def compute_metric_distributions(
        self, samples: List[str]
    ) -> Dict[str, np.ndarray]:
        """Compute all metrics for a list of encrypted text samples."""
        metrics = {
            'H1': [], 'H2': [], 'H3': [],
            'zipf_exponent': [], 'type_token_ratio': [],
            'mean_word_length': [],
        }
        pos_entropy_curves = []

        for text in samples:
            tokens = text.split()
            if len(tokens) < 10:
                continue

            entropy = compute_all_entropy(text)
            metrics['H1'].append(entropy['H1'])
            metrics['H2'].append(entropy['H2'])
            metrics['H3'].append(entropy['H3'])

            zipf = zipf_analysis(tokens)
            metrics['zipf_exponent'].append(abs(zipf['zipf_exponent']))
            metrics['type_token_ratio'].append(zipf['type_token_ratio'])

            metrics['mean_word_length'].append(np.mean([len(t) for t in tokens]))

            pos_ent = word_positional_entropy(tokens)
            curve = [pos_ent.get(i, 0.0) for i in range(10)]
            pos_entropy_curves.append(curve)

        result = {k: np.array(v) for k, v in metrics.items()}
        result['positional_entropy_curves'] = np.array(pos_entropy_curves) if pos_entropy_curves else np.array([])

        return result

    @staticmethod
    def percentile_rank(value: float, distribution: np.ndarray) -> float:
        """Compute the percentile rank of a value within a distribution."""
        if len(distribution) == 0:
            return 50.0
        return float(np.sum(distribution <= value) / len(distribution) * 100)

    @staticmethod
    def compute_p_value(value: float, distribution: np.ndarray) -> float:
        """Compute two-tailed p-value: how extreme is this value?"""
        if len(distribution) == 0:
            return 1.0
        mean = np.mean(distribution)
        deviation = abs(value - mean)
        n_more_extreme = np.sum(np.abs(distribution - mean) >= deviation)
        return float(n_more_extreme / len(distribution))

    def run_all(self) -> Dict:
        """
        Generate null distributions for all cipher×language combinations
        and compute percentile ranks for Voynich metrics.
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("TRACK 10: NULL HYPOTHESIS FRAMEWORK")
            print("=" * 70)
            print(f"  Cipher families: {list(CIPHER_FAMILIES.keys())}")
            print(f"  Source languages: {SOURCE_LANGUAGES}")
            print(f"  Samples per combination: {self.n_samples}")

        if self.verbose:
            print("\n  Computing Voynich baselines...")
        self.compute_voynich_baselines()

        if self.verbose:
            print(f"    H1={self.voynich_metrics['H1']:.4f}  "
                  f"H2={self.voynich_metrics['H2']:.4f}  "
                  f"H3={self.voynich_metrics['H3']:.4f}")
            print(f"    Zipf={self.voynich_metrics['zipf_exponent']:.4f}  "
                  f"TTR={self.voynich_metrics['type_token_ratio']:.4f}  "
                  f"MWL={self.voynich_metrics['mean_word_length']:.2f}")

        results = {
            'voynich_metrics': self.voynich_metrics,
            'distributions': {},
            'percentile_ranks': {},
            'p_values': {},
            'anomaly_summary': {},
        }

        total = len(CIPHER_FAMILIES) * len(SOURCE_LANGUAGES)
        done = 0

        for cipher_name in CIPHER_FAMILIES:
            results['distributions'][cipher_name] = {}
            results['percentile_ranks'][cipher_name] = {}
            results['p_values'][cipher_name] = {}

            for lang in SOURCE_LANGUAGES:
                combo_key = f"{cipher_name}_{lang}"
                done += 1

                if self.verbose:
                    print(f"\n  [{done}/{total}] {cipher_name} × {lang}...", end='')

                t0 = time.time()
                samples = self.generate_reference_corpus(cipher_name, lang)

                if len(samples) < 10:
                    if self.verbose:
                        print(f" SKIP (only {len(samples)} valid samples)")
                    continue

                dist = self.compute_metric_distributions(samples)
                elapsed = time.time() - t0

                ranks = {}
                pvals = {}
                scalar_metrics = ['H1', 'H2', 'H3', 'zipf_exponent',
                                  'type_token_ratio', 'mean_word_length']

                for metric in scalar_metrics:
                    voynich_val = self.voynich_metrics.get(metric, 0.0)
                    if metric in dist and len(dist[metric]) > 0:
                        ranks[metric] = self.percentile_rank(voynich_val, dist[metric])
                        pvals[metric] = self.compute_p_value(voynich_val, dist[metric])

                results['distributions'][cipher_name][lang] = {
                    k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                         'min': float(np.min(v)), 'max': float(np.max(v)),
                         'n_samples': len(v)}
                    for k, v in dist.items()
                    if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) > 0
                }
                results['percentile_ranks'][cipher_name][lang] = ranks
                results['p_values'][cipher_name][lang] = pvals

                if self.verbose:
                    print(f" {len(samples)} samples, {elapsed:.1f}s")
                    h2_rank = ranks.get('H2', 50.0)
                    print(f"    H2 percentile: {h2_rank:.1f}%  "
                          f"(p={pvals.get('H2', 1.0):.4f})")

        results['anomaly_summary'] = self._anomaly_summary(results)

        return results

    def _anomaly_summary(self, results: Dict) -> Dict:
        """Identify which Voynich metrics are anomalous across cipher families."""
        summary = {}
        scalar_metrics = ['H1', 'H2', 'H3', 'zipf_exponent',
                          'type_token_ratio', 'mean_word_length']

        for metric in scalar_metrics:
            anomalous_for = []
            normal_for = []

            for cipher_name in CIPHER_FAMILIES:
                for lang in SOURCE_LANGUAGES:
                    pval = (results['p_values']
                            .get(cipher_name, {})
                            .get(lang, {})
                            .get(metric, 1.0))
                    combo = f"{cipher_name}_{lang}"
                    if pval < 0.05:
                        anomalous_for.append(combo)
                    else:
                        normal_for.append(combo)

            total = len(anomalous_for) + len(normal_for)
            summary[metric] = {
                'voynich_value': self.voynich_metrics.get(metric, 0.0),
                'anomalous_count': len(anomalous_for),
                'normal_count': len(normal_for),
                'total_tested': total,
                'anomalous_for': anomalous_for[:5],
                'normal_for': normal_for[:5],
                'globally_anomalous': len(anomalous_for) > total * 0.8,
            }

        return summary

def run(verbose: bool = True, n_samples: int = 200) -> Dict:
    """
    Run the null hypothesis framework.

    Parameters:
        verbose: Print detailed output
        n_samples: Number of samples per cipher×language combination

    Returns:
        Dict with distributions, percentile ranks, p-values, and anomaly summary
    """
    engine = NullDistributionEngine(n_samples=n_samples, verbose=verbose)
    results = engine.run_all()
    results['track'] = 'null_framework'
    results['track_number'] = 10

    if verbose:
        print("\n" + "─" * 70)
        print("NULL FRAMEWORK SUMMARY")
        print("─" * 70)

        summary = results.get('anomaly_summary', {})
        for metric, data in summary.items():
            status = "ANOMALOUS" if data['globally_anomalous'] else "NORMAL"
            print(f"  {metric}: {status} "
                  f"(anomalous for {data['anomalous_count']}/{data['total_tested']} "
                  f"combinations)")

        h2_data = summary.get('H2', {})
        if h2_data.get('globally_anomalous'):
            print(f"\n  KEY FINDING: Voynich H2={h2_data['voynich_value']:.4f} is "
                  f"statistically anomalous across {h2_data['anomalous_count']} "
                  f"of {h2_data['total_tested']} cipher×language combinations.")
            print("  → This suggests the cipher mechanism is NOT well-modeled "
                  "by standard historical ciphers.")
        else:
            normal = h2_data.get('normal_for', [])
            print(f"\n  KEY FINDING: Voynich H2 is within normal range for: "
                  f"{', '.join(normal[:3])}")
            print("  → These cipher families remain viable candidates.")

    try:
        os.makedirs('./output', exist_ok=True)
        save_data = {
            'voynich_metrics': {k: v for k, v in results['voynich_metrics'].items()
                               if not isinstance(v, list)},
            'percentile_ranks': results['percentile_ranks'],
            'p_values': results['p_values'],
            'anomaly_summary': results['anomaly_summary'],
        }
        with open('./results/null_distributions.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        if verbose:
            print("\n  Saved to output/null_distributions.json")
    except Exception as e:
        if verbose:
            print(f"\n  [WARN] Could not save: {e}")

    return results
