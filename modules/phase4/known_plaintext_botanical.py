"""
Approach 1: Known-Plaintext Attack via Botanical Illustrations
================================================================
Uses plant identifications from botanical_identifications.py to
constrain codebook assignments. Cross-references which Voynich words
appear on which pages to deduce word meanings.

Key insight: If a word appears only on cold/wet plant pages but not
on hot/dry pages, it probably encodes a cold/wet quality term.
"""

import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from modules.phase4.lang_a_extractor import LanguageAExtractor
from data.botanical_identifications import (
    PLANT_IDS, HUMORAL_QUALITIES, PLANT_PART_TERMS, HUMORAL_LABEL_TERMS,
    get_plants_by_humoral, get_high_confidence_ids,
)
from data.voynich_corpus import HARTLIEB_MEDICAL_VOCAB

HUMORAL_EXPECTED_VOCAB = {
    'hot_dry': [
        'calidus', 'siccus', 'ignis', 'cholera', 'calida', 'sicca',
        'stimulant', 'purgativam', 'emmenagogam', 'digestivam',
        'contra', 'venenum', 'febrem', 'vermifugam',
    ],
    'hot_wet': [
        'calidus', 'humidus', 'aer', 'sanguis', 'calida', 'humida',
        'sudorific', 'cordialem', 'laetificantem', 'emollientem',
        'diureticam',
    ],
    'cold_dry': [
        'frigidus', 'siccus', 'terra', 'melancholia', 'frigida', 'sicca',
        'stypticam', 'vulnerariam', 'astringentem', 'anti-inflammatory',
    ],
    'cold_wet': [
        'frigidus', 'humidus', 'aqua', 'phlegma', 'frigida', 'humida',
        'sedativam', 'refrigerandi', 'emollientem', 'laxativam',
    ],
}

class BotanicalKnownPlaintext:
    """
    Known-plaintext attack using botanical identifications.

    With 20 herbal_a plants identified (including humoral qualities),
    cross-referencing which words appear on which pages can constrain
    which Voynich words encode which Latin terms.
    """

    def __init__(self, extractor: LanguageAExtractor):
        self.extractor = extractor
        self._folio_words = None
        self._herbal_a_plants = None

    def _get_herbal_a_plants(self) -> Dict:
        """Get only herbal_a plant identifications."""
        if self._herbal_a_plants is not None:
            return self._herbal_a_plants

        self._herbal_a_plants = {}
        for folio, data in PLANT_IDS.items():
            folio_num = int(''.join(c for c in folio[1:] if c.isdigit()) or '0')
            if folio_num < 57 and not folio.startswith(('f87', 'f88', 'f89', 'f9')):
                self._herbal_a_plants[folio] = data

        return self._herbal_a_plants

    def build_folio_word_matrix(self) -> Dict:
        """
        For each identified herbal folio, list which Language A words
        appear on it with their frequencies.

        Returns: {folio: {word: count, ...}, ...}
        """
        if self._folio_words is not None:
            return self._folio_words

        by_folio = self.extractor.extract_lang_a_by_folio()
        plants = self._get_herbal_a_plants()

        self._folio_words = {}
        for folio in plants:
            if folio in by_folio:
                self._folio_words[folio] = Counter(by_folio[folio])

        return self._folio_words

    def build_humoral_groups(self) -> Dict[str, List[str]]:
        """
        Group identified folios by humoral quality.
        Returns: {quality: [folio1, folio2, ...]}
        """
        plants = self._get_herbal_a_plants()
        groups = defaultdict(list)
        for folio, data in plants.items():
            humoral = data.get('humoral', 'unknown')
            groups[humoral].append(folio)
        return dict(groups)

    def cross_reference_pages(self) -> Dict:
        """
        THE critical analysis: find Voynich words that are specific
        to certain humoral quality groups.

        A word appearing ONLY on cold/wet pages probably encodes a
        cold/wet quality term. A word appearing on ALL pages is
        likely a function word (et, in, ad, etc.).
        """
        folio_words = self.build_folio_word_matrix()
        humoral_groups = self.build_humoral_groups()

        if not folio_words:
            return {
                'error': 'No folio word data available',
                'n_folios_with_data': 0,
            }

        all_words = set()
        for fw in folio_words.values():
            all_words.update(fw.keys())

        word_stats = {}
        for word in sorted(all_words):
            present_in = {}
            for quality, folios in humoral_groups.items():
                count = 0
                pages = 0
                for f in folios:
                    if f in folio_words and word in folio_words[f]:
                        count += folio_words[f][word]
                        pages += 1
                if count > 0:
                    present_in[quality] = {'count': count, 'pages': pages}

            total_count = sum(d['count'] for d in present_in.values())
            total_pages = sum(d['pages'] for d in present_in.values())
            n_qualities = len(present_in)

            word_stats[word] = {
                'total_count': total_count,
                'total_pages': total_pages,
                'n_humoral_groups': n_qualities,
                'humoral_distribution': present_in,
                'specificity': 1.0 / max(n_qualities, 1),
            }

        universal_words = []
        specific_words = []
        moderate_words = []

        for word, stats in word_stats.items():
            if stats['n_humoral_groups'] >= 3:
                universal_words.append(word)
            elif stats['n_humoral_groups'] == 1:
                quality = list(stats['humoral_distribution'].keys())[0]
                specific_words.append({
                    'word': word,
                    'quality': quality,
                    'count': stats['total_count'],
                })
            elif stats['n_humoral_groups'] == 2:
                moderate_words.append(word)

        return {
            'n_folios_analyzed': len(folio_words),
            'n_words_analyzed': len(word_stats),
            'n_humoral_groups': len(humoral_groups),
            'humoral_groups': {k: len(v) for k, v in humoral_groups.items()},
            'universal_words': universal_words,
            'n_universal': len(universal_words),
            'specific_words': specific_words,
            'n_specific': len(specific_words),
            'moderate_words': moderate_words,
            'n_moderate': len(moderate_words),
            'word_stats': word_stats,
        }

    def compute_page_specificity_scores(self) -> Dict:
        """
        For each Language A word, compute how specific it is to certain
        folio groups using mutual information.

        High MI = word is strongly associated with certain pages.
        Low MI = word appears uniformly across all pages.
        """
        folio_words = self.build_folio_word_matrix()
        if not folio_words:
            return {'error': 'No folio word data'}

        all_words = set()
        for fw in folio_words.values():
            all_words.update(fw.keys())

        total_tokens = sum(sum(fw.values()) for fw in folio_words.values())
        n_folios = len(folio_words)

        specificity_scores = {}
        for word in sorted(all_words):
            word_total = sum(
                folio_words[f].get(word, 0) for f in folio_words
            )
            if word_total == 0:
                continue

            dist = []
            for f in folio_words:
                count = folio_words[f].get(word, 0)
                dist.append(count)

            dist = np.array(dist, dtype=float)
            dist_norm = dist / max(dist.sum(), 1)

            entropy = -sum(p * math.log2(p) for p in dist_norm if p > 0)
            max_entropy = math.log2(n_folios) if n_folios > 1 else 1

            specificity = 1.0 - (entropy / max(max_entropy, 1e-10))

            specificity_scores[word] = {
                'total_count': int(word_total),
                'n_pages': int(np.sum(dist > 0)),
                'entropy': entropy,
                'specificity': specificity,
            }

        ranked = sorted(specificity_scores.items(),
                        key=lambda x: -x[1]['specificity'])

        return {
            'scores': specificity_scores,
            'most_specific': [(w, s['specificity']) for w, s in ranked[:10]],
            'most_universal': [(w, s['specificity']) for w, s in ranked[-10:]],
        }

    def generate_crib_constraints(self) -> List[Dict]:
        """
        Produce a ranked list of (voynich_word, candidate_plaintext_set)
        pairs for use in the SAA attack.

        Higher-confidence cribs come from:
        - Words specific to a single humoral group
        - Universal words (likely function words like 'et', 'in')
        - Frequency-matched words
        """
        cross_ref = self.cross_reference_pages()
        specificity = self.compute_page_specificity_scores()
        voynich_freqs = self.extractor.compute_word_frequencies()

        cribs = []

        for word in cross_ref.get('universal_words', []):
            freq = voynich_freqs.get(word, 0)
            cribs.append({
                'voynich_word': word,
                'candidate_type': 'function_word',
                'candidates': ['et', 'in', 'est', 'ad', 'cum', 'de', 'habet'],
                'confidence': 'MODERATE' if freq > 5 else 'LOW',
                'rationale': f'Appears across 3+ humoral groups (freq={freq})',
            })

        for entry in cross_ref.get('specific_words', []):
            word = entry['word']
            quality = entry['quality']
            candidates = HUMORAL_EXPECTED_VOCAB.get(quality, [])

            cribs.append({
                'voynich_word': word,
                'candidate_type': f'humoral_{quality}',
                'candidates': candidates[:5],
                'confidence': 'LOW',
                'rationale': (
                    f'Appears only in {quality} pages '
                    f'(count={entry["count"]})'
                ),
            })

        top_voynich = voynich_freqs.most_common(3)
        common_latin = ['et', 'in', 'est', 'ad', 'cum', 'contra', 'habet']
        for i, (word, count) in enumerate(top_voynich):
            if not any(c['voynich_word'] == word for c in cribs):
                cribs.append({
                    'voynich_word': word,
                    'candidate_type': 'frequency_matched',
                    'candidates': common_latin[i:i+3],
                    'confidence': 'LOW',
                    'rationale': f'Top-{i+1} most frequent word (count={count})',
                })

        conf_order = {'MODERATE': 0, 'LOW': 1}
        cribs.sort(key=lambda x: conf_order.get(x['confidence'], 2))

        return cribs

    def run(self, verbose: bool = True) -> Dict:
        """Run the botanical known-plaintext attack."""
        folio_words = self.build_folio_word_matrix()
        humoral_groups = self.build_humoral_groups()
        cross_ref = self.cross_reference_pages()
        specificity = self.compute_page_specificity_scores()
        cribs = self.generate_crib_constraints()

        results = {
            'n_identified_plants': len(self._get_herbal_a_plants()),
            'n_folios_with_data': len(folio_words),
            'humoral_groups': {k: len(v) for k, v in humoral_groups.items()},
            'cross_reference': {
                'n_universal': cross_ref.get('n_universal', 0),
                'universal_words': cross_ref.get('universal_words', []),
                'n_specific': cross_ref.get('n_specific', 0),
                'specific_words': cross_ref.get('specific_words', []),
                'n_moderate': cross_ref.get('n_moderate', 0),
            },
            'specificity': {
                'most_specific': specificity.get('most_specific', []),
                'most_universal': specificity.get('most_universal', []),
            },
            'crib_constraints': cribs,
            'n_cribs': len(cribs),
            'synthesis': {
                'useful_cribs': len([c for c in cribs if c['confidence'] == 'MODERATE']),
                'total_cribs': len(cribs),
                'conclusion': (
                    f'{len(cribs)} crib constraints generated from '
                    f'{len(folio_words)} folios. '
                    f'{len([c for c in cribs if c["confidence"] == "MODERATE"])} '
                    f'moderate-confidence cribs available for SAA.'
                ),
            },
        }

        if verbose:
            print(f'\n  Approach 1: Botanical Known-Plaintext Attack')
            print(f'    Identified plants: {len(self._get_herbal_a_plants())}')
            print(f'    Folios with word data: {len(folio_words)}')
            print(f'    Humoral groups: {list(humoral_groups.keys())}')
            print(f'    Universal words: {cross_ref.get("n_universal", 0)}')
            print(f'    Quality-specific words: {cross_ref.get("n_specific", 0)}')
            print(f'    Total cribs generated: {len(cribs)}')
            for crib in cribs[:5]:
                print(f'      {crib["voynich_word"]} → '
                      f'{crib["candidates"][:3]} ({crib["confidence"]})')

        return results
