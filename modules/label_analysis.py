"""
Track 8: Label Extraction and Clustering
==========================================
Extracts every label from the herbal and pharmaceutical pages, clusters them
by co-occurrence and character overlap, and identifies candidate
plaintext-ciphertext pairs.

Labels are the closest thing to known plaintext outside the zodiac month names.
They're short (1-3 words), semantically constrained (plant names, body parts,
humoral qualities), and their position next to illustrations provides context.
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.voynich_corpus import get_all_tokens, SAMPLE_CORPUS, SECTIONS
from data.botanical_identifications import (
    PLANT_IDS, PLANT_PART_TERMS, HUMORAL_QUALITIES, HUMORAL_LABEL_TERMS,
    get_plants_by_humoral, get_high_confidence_ids
)


# ============================================================================
# LABEL ANALYZER
# ============================================================================

class LabelAnalyzer:
    """
    Extracts and analyzes labels from the Voynich Manuscript,
    clustering them to identify candidate plaintext-ciphertext pairs.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def extract_labels(self) -> List[Dict]:
        """
        Extract label-like tokens from the corpus.

        Uses heuristics to identify labels:
        1. Short isolated text segments (1-3 tokens)
        2. Tokens from herbal and pharmaceutical sections
        3. Tokens appearing near illustration markers
        """
        labels = []

        for folio, data in SAMPLE_CORPUS.items():
            section = data.get('section', '')
            scribe = data.get('scribe', 0)

            # Only look for labels in sections with illustrations
            if section not in ('herbal_a', 'herbal_b', 'pharmaceutical',
                               'astronomical', 'biological'):
                continue

            text_lines = data.get('text', [])

            for line_idx, line in enumerate(text_lines):
                tokens = line.split()

                # Heuristic 1: Very short lines (1-3 tokens) are likely labels
                if 1 <= len(tokens) <= 3:
                    labels.append({
                        'folio': folio,
                        'tokens': tokens,
                        'text': ' '.join(tokens),
                        'n_tokens': len(tokens),
                        'section': section,
                        'scribe': scribe,
                        'line_index': line_idx,
                        'source': 'short_line',
                    })

                # Heuristic 2: First token of lines in herbal sections
                # (often a plant label followed by description)
                if section in ('herbal_a', 'herbal_b') and len(tokens) > 3:
                    labels.append({
                        'folio': folio,
                        'tokens': [tokens[0]],
                        'text': tokens[0],
                        'n_tokens': 1,
                        'section': section,
                        'scribe': scribe,
                        'line_index': line_idx,
                        'source': 'line_initial',
                    })

        return labels

    def label_vocabulary(self, labels: List[Dict]) -> Dict:
        """Compute label vocabulary statistics."""
        all_label_tokens = []
        for label in labels:
            all_label_tokens.extend(label['tokens'])

        token_counts = Counter(all_label_tokens)
        section_dist = Counter(label['section'] for label in labels)

        return {
            'total_labels': len(labels),
            'total_tokens': len(all_label_tokens),
            'unique_tokens': len(token_counts),
            'top_tokens': token_counts.most_common(20),
            'section_distribution': dict(section_dist),
            'mean_label_length': np.mean([label['n_tokens'] for label in labels]) if labels else 0,
        }

    def cluster_labels(self, labels: List[Dict]) -> List[Dict]:
        """
        Cluster labels by character overlap and co-occurrence on same folio.
        Uses Jaccard similarity on character sets.
        """
        if not labels:
            return []

        # Build unique label texts
        unique_labels = list(set(label['text'] for label in labels))

        # Compute pairwise Jaccard similarity on character sets
        clusters = []
        assigned = set()

        for i, label_a in enumerate(unique_labels):
            if label_a in assigned:
                continue

            cluster = [label_a]
            assigned.add(label_a)
            chars_a = set(label_a.replace(' ', ''))

            for j in range(i + 1, len(unique_labels)):
                label_b = unique_labels[j]
                if label_b in assigned:
                    continue

                chars_b = set(label_b.replace(' ', ''))
                if chars_a and chars_b:
                    jaccard = len(chars_a & chars_b) / len(chars_a | chars_b)
                    if jaccard > 0.5:
                        cluster.append(label_b)
                        assigned.add(label_b)

            if len(cluster) > 1:
                # Find which folios this cluster appears on
                cluster_folios = set()
                for label in labels:
                    if label['text'] in cluster:
                        cluster_folios.add(label['folio'])

                clusters.append({
                    'labels': cluster,
                    'size': len(cluster),
                    'folios': sorted(cluster_folios),
                    'n_folios': len(cluster_folios),
                    'shared_chars': sorted(chars_a & set(''.join(cluster).replace(' ', ''))),
                })

        # Sort by cluster size
        clusters.sort(key=lambda c: -c['size'])
        return clusters

    def cross_reference_botanical(self, labels: List[Dict]) -> List[Dict]:
        """
        Match label clusters against known botanical identifications.
        Uses folio co-occurrence to link labels to plant IDs.
        """
        matches = []

        for folio, plant_data in PLANT_IDS.items():
            # Find labels on this folio
            folio_labels = [l for l in labels if l['folio'] == folio]
            if not folio_labels:
                continue

            label_texts = [l['text'] for l in folio_labels]

            matches.append({
                'folio': folio,
                'plant_candidates': plant_data['candidates'],
                'common_name': plant_data.get('common', ''),
                'humoral': plant_data.get('humoral', ''),
                'properties': plant_data.get('properties', []),
                'confidence': plant_data.get('confidence', 'LOW'),
                'labels_on_folio': label_texts,
                'n_labels': len(label_texts),
            })

        return matches

    def candidate_pairs(self, labels: List[Dict], botanical_matches: List[Dict]) -> List[Dict]:
        """
        Generate candidate plaintext-ciphertext pairs from labels.
        Uses botanical identifications and humoral theory to propose
        what each label might mean.
        """
        pairs = []

        # Strategy 1: Labels on folios with identified plants
        for match in botanical_matches:
            humoral = match.get('humoral', '')
            if humoral and humoral in HUMORAL_QUALITIES:
                qual = HUMORAL_QUALITIES[humoral]
                for label_text in match['labels_on_folio'][:3]:
                    pairs.append({
                        'ciphertext': label_text,
                        'plaintext_candidates': [
                            qual['latin'],
                            match['common_name'],
                        ] + match.get('properties', [])[:2],
                        'folio': match['folio'],
                        'reasoning': f'Plant identified as {match["plant_candidates"][0]} '
                                     f'({humoral})',
                        'confidence': match['confidence'],
                    })

        # Strategy 2: Recurring labels across herbal pages → generic plant-part terms
        label_freq = Counter(l['text'] for l in labels
                            if l['section'] in ('herbal_a', 'herbal_b'))
        for label_text, count in label_freq.most_common(10):
            if count >= 3:
                pairs.append({
                    'ciphertext': label_text,
                    'plaintext_candidates': list(PLANT_PART_TERMS.keys()),
                    'folio': 'multiple',
                    'reasoning': f'Appears on {count} herbal folios — likely a generic '
                                 f'plant-part term (radix, folia, flos, etc.)',
                    'confidence': 'MODERATE' if count >= 5 else 'LOW',
                })

        # Strategy 3: Labels in pharmaceutical section → preparation terms
        pharm_labels = Counter(l['text'] for l in labels
                              if l['section'] == 'pharmaceutical')
        for label_text, count in pharm_labels.most_common(5):
            if count >= 2:
                pairs.append({
                    'ciphertext': label_text,
                    'plaintext_candidates': [
                        'recipe', 'accipe', 'pulvis', 'unguentum',
                        'aqua', 'oleum', 'decoctio',
                    ],
                    'folio': 'pharmaceutical',
                    'reasoning': f'Recurring pharmaceutical label ({count} occurrences)',
                    'confidence': 'LOW',
                })

        return pairs


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run label extraction and clustering.

    Returns:
        Dict with labels, vocabulary, clusters, botanical matches,
        and candidate plaintext-ciphertext pairs.
    """
    analyzer = LabelAnalyzer(verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 8: LABEL EXTRACTION AND CLUSTERING")
        print("=" * 70)

    # Extract labels
    if verbose:
        print("\n  Extracting labels from corpus...")
    labels = analyzer.extract_labels()
    if verbose:
        print(f"    Found {len(labels)} label candidates")

    # Vocabulary
    vocab = analyzer.label_vocabulary(labels)
    if verbose:
        print(f"    Unique label tokens: {vocab['unique_tokens']}")
        print(f"    Section distribution: {vocab['section_distribution']}")
        if vocab['top_tokens']:
            print(f"    Top tokens: {vocab['top_tokens'][:10]}")

    # Clustering
    if verbose:
        print("\n  Clustering labels by character overlap...")
    clusters = analyzer.cluster_labels(labels)
    if verbose:
        print(f"    Found {len(clusters)} label clusters")
        for i, c in enumerate(clusters[:5]):
            print(f"      Cluster {i+1}: {c['labels'][:5]} "
                  f"({c['n_folios']} folios)")

    # Botanical cross-reference
    if verbose:
        print("\n  Cross-referencing with botanical identifications...")
    botanical = analyzer.cross_reference_botanical(labels)
    if verbose:
        print(f"    Matched {len(botanical)} folios with plant IDs")
        for m in botanical[:5]:
            print(f"      {m['folio']}: {m['plant_candidates'][0]} → "
                  f"{m['labels_on_folio'][:3]}")

    # Candidate pairs
    if verbose:
        print("\n  Generating candidate plaintext-ciphertext pairs...")
    pairs = analyzer.candidate_pairs(labels, botanical)
    if verbose:
        print(f"    Generated {len(pairs)} candidate pairs")
        for p in pairs[:5]:
            print(f"      '{p['ciphertext']}' → {p['plaintext_candidates'][:3]} "
                  f"[{p['confidence']}]")

    # Count high-confidence pairs
    high_conf = [p for p in pairs if p['confidence'] in ('HIGH', 'MODERATE')]

    results = {
        'track': 'label_analysis',
        'track_number': 8,
        'n_labels': len(labels),
        'vocabulary': {k: v for k, v in vocab.items() if k != 'top_tokens'},
        'top_label_tokens': vocab.get('top_tokens', [])[:20],
        'n_clusters': len(clusters),
        'clusters': clusters[:10],
        'n_botanical_matches': len(botanical),
        'botanical_matches': botanical[:10],
        'candidate_pairs': pairs,
        'n_candidate_pairs': len(pairs),
        'n_high_confidence_pairs': len(high_conf),
    }

    if verbose:
        print("\n" + "─" * 70)
        print("LABEL ANALYSIS SUMMARY")
        print("─" * 70)
        print(f"  Labels extracted: {len(labels)}")
        print(f"  Clusters found: {len(clusters)}")
        print(f"  Botanical matches: {len(botanical)}")
        print(f"  Candidate pairs: {len(pairs)} "
              f"({len(high_conf)} high/moderate confidence)")

    return results
