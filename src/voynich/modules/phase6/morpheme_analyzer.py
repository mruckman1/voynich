"""
Morpheme Analyzer — Path C Step 1
===================================
Tests whether Voynich words have productive morphological structure
(stems + affixes), which would indicate the cipher operates below
the word level and whole-word substitution is fundamentally wrong.

Analysis:
  1. Prefix/suffix extraction (lengths 1-4)
  2. Productive affix detection (≥ 10 words, above random baseline)
  3. Paradigm detection (words sharing stems, varying suffixes)
  4. Word decomposition into prefix + stem + suffix
  5. Entropy of stems vs affixes

Phase 6  ·  Voynich Convergence Attack
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

from voynich.modules.phase5.tier_splitter import TierSplitter

class MorphemeAnalyzer:
    """
    Analyze the internal morphological structure of Voynich Tier 1 words
    to test whether they are compositional (stem + affix) rather than
    opaque codebook entries.
    """

    def __init__(self, splitter: TierSplitter, min_affix_words: int = 10):
        self.splitter = splitter
        self.min_affix_words = min_affix_words
        self._words = None
        self._word_freqs = None

    def _ensure_words(self):
        if self._words is not None:
            return
        self._words = self.splitter.get_tier1_types()
        self._word_freqs = Counter(self.splitter.get_tier1_tokens())

    def extract_prefixes(self, max_len: int = 4) -> Dict[str, List[str]]:
        """
        Extract prefix → word list for prefix lengths 1 to max_len.
        Returns {prefix: [words containing that prefix]}.
        """
        self._ensure_words()
        prefixes = defaultdict(list)
        for word in self._words:
            for plen in range(1, min(max_len + 1, len(word))):
                prefix = word[:plen]
                prefixes[prefix].append(word)
        return dict(prefixes)

    def extract_suffixes(self, max_len: int = 4) -> Dict[str, List[str]]:
        """
        Extract suffix → word list for suffix lengths 1 to max_len.
        Returns {suffix: [words containing that suffix]}.
        """
        self._ensure_words()
        suffixes = defaultdict(list)
        for word in self._words:
            for slen in range(1, min(max_len + 1, len(word))):
                suffix = word[-slen:]
                suffixes[suffix].append(word)
        return dict(suffixes)

    def find_productive_affixes(self) -> Dict:
        """
        Find affixes that appear in more words than expected by chance.

        An affix is "productive" if:
        - It appears in >= min_affix_words words
        - Its frequency is > 2× what random character sequences would produce

        Returns dict with productive prefixes and suffixes.
        """
        self._ensure_words()
        n_words = len(self._words)

        all_chars = set(c for w in self._words for c in w)
        alpha_size = len(all_chars)

        prefixes = self.extract_prefixes()
        suffixes = self.extract_suffixes()

        productive_prefixes = []
        for prefix, words in sorted(prefixes.items(), key=lambda x: -len(x[1])):
            count = len(words)
            if count < self.min_affix_words:
                continue
            k = len(prefix)
            expected = n_words / max(1, alpha_size ** k)
            ratio = count / max(expected, 0.01)
            if ratio > 2.0:
                productive_prefixes.append({
                    'prefix': prefix,
                    'n_words': count,
                    'expected_random': round(expected, 2),
                    'enrichment': round(ratio, 2),
                    'example_words': words[:8],
                })

        productive_suffixes = []
        for suffix, words in sorted(suffixes.items(), key=lambda x: -len(x[1])):
            count = len(words)
            if count < self.min_affix_words:
                continue
            k = len(suffix)
            expected = n_words / max(1, alpha_size ** k)
            ratio = count / max(expected, 0.01)
            if ratio > 2.0:
                productive_suffixes.append({
                    'suffix': suffix,
                    'n_words': count,
                    'expected_random': round(expected, 2),
                    'enrichment': round(ratio, 2),
                    'example_words': words[:8],
                })

        return {
            'n_productive_prefixes': len(productive_prefixes),
            'n_productive_suffixes': len(productive_suffixes),
            'productive_prefixes': productive_prefixes[:30],
            'productive_suffixes': productive_suffixes[:30],
            'alphabet_size': alpha_size,
            'n_words': n_words,
        }

    def detect_paradigms(self) -> List[Dict]:
        """
        Find morphological paradigms: groups of words that share a stem
        and differ only in suffix.

        A paradigm is a set of words where:
        - All words share a common prefix of length >= 2
        - The suffix (remaining characters) varies
        - The group has >= 3 members

        Returns list of {stem, members: [{word, suffix}], n_members}.
        """
        self._ensure_words()

        paradigms = []
        seen_groups: Set[str] = set()

        for stem_len in range(2, 6):
            stem_groups = defaultdict(list)
            for word in self._words:
                if len(word) > stem_len:
                    stem = word[:stem_len]
                    suffix = word[stem_len:]
                    stem_groups[stem].append((word, suffix))

            for stem, members in stem_groups.items():
                if len(members) < 3:
                    continue

                member_key = ','.join(sorted(w for w, _ in members))
                if member_key in seen_groups:
                    continue
                seen_groups.add(member_key)

                unique_suffixes = set(s for _, s in members)
                if len(unique_suffixes) < 3:
                    continue

                members_sorted = sorted(
                    members,
                    key=lambda x: -self._word_freqs.get(x[0], 0)
                )

                paradigms.append({
                    'stem': stem,
                    'n_members': len(members),
                    'n_unique_suffixes': len(unique_suffixes),
                    'members': [{'word': w, 'suffix': s} for w, s in members_sorted[:10]],
                    'all_suffixes': sorted(unique_suffixes),
                    'combined_frequency': sum(
                        self._word_freqs.get(w, 0) for w, _ in members
                    ),
                })

        paradigms.sort(key=lambda x: -x['n_members'])
        return paradigms

    def decompose_words(self) -> Dict:
        """
        For each Tier 1 word, find the best decomposition into
        known productive prefixes and suffixes.

        Returns statistics about decomposition coverage.
        """
        self._ensure_words()
        affixes = self.find_productive_affixes()

        prod_prefixes = {a['prefix'] for a in affixes['productive_prefixes']}
        prod_suffixes = {a['suffix'] for a in affixes['productive_suffixes']}

        decomposed = 0
        has_prefix = 0
        has_suffix = 0
        has_both = 0
        examples = []

        for word in self._words:
            best_prefix = ''
            best_suffix = ''

            for plen in range(min(4, len(word)), 0, -1):
                if word[:plen] in prod_prefixes:
                    best_prefix = word[:plen]
                    break

            for slen in range(min(4, len(word)), 0, -1):
                if word[-slen:] in prod_suffixes:
                    best_suffix = word[-slen:]
                    break

            p = bool(best_prefix)
            s = bool(best_suffix)
            if p:
                has_prefix += 1
            if s:
                has_suffix += 1
            if p and s:
                has_both += 1
            if p or s:
                decomposed += 1

            if p or s:
                stem_start = len(best_prefix)
                stem_end = len(word) - len(best_suffix) if best_suffix else len(word)
                stem = word[stem_start:stem_end] if stem_start < stem_end else ''
                if len(examples) < 20:
                    examples.append({
                        'word': word,
                        'prefix': best_prefix,
                        'stem': stem,
                        'suffix': best_suffix,
                    })

        n = len(self._words)
        return {
            'total_words': n,
            'decomposed': decomposed,
            'has_prefix': has_prefix,
            'has_suffix': has_suffix,
            'has_both': has_both,
            'decomposition_rate': decomposed / max(1, n),
            'prefix_rate': has_prefix / max(1, n),
            'suffix_rate': has_suffix / max(1, n),
            'examples': examples,
        }

    def compute_morpheme_statistics(self) -> Dict:
        """
        Compute summary statistics about morphological structure.
        """
        self._ensure_words()
        paradigms = self.detect_paradigms()
        decomp = self.decompose_words()
        affixes = self.find_productive_affixes()

        words_in_paradigms = set()
        for p in paradigms:
            for m in p['members']:
                words_in_paradigms.add(m['word'])

        all_suffixes = []
        for p in paradigms:
            all_suffixes.extend(p['all_suffixes'])
        suffix_counts = Counter(all_suffixes)
        total_s = sum(suffix_counts.values())
        suffix_entropy = 0.0
        if total_s > 0:
            for count in suffix_counts.values():
                p = count / total_s
                if p > 0:
                    suffix_entropy -= p * math.log2(p)

        return {
            'n_paradigms': len(paradigms),
            'words_in_paradigms': len(words_in_paradigms),
            'paradigm_coverage': len(words_in_paradigms) / max(1, len(self._words)),
            'mean_paradigm_size': float(np.mean([p['n_members'] for p in paradigms])) if paradigms else 0,
            'max_paradigm_size': max((p['n_members'] for p in paradigms), default=0),
            'n_unique_suffixes_in_paradigms': len(suffix_counts),
            'suffix_entropy': suffix_entropy,
            'n_productive_prefixes': affixes['n_productive_prefixes'],
            'n_productive_suffixes': affixes['n_productive_suffixes'],
            'decomposition_rate': decomp['decomposition_rate'],
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run full morphological analysis."""
        affixes = self.find_productive_affixes()
        paradigms = self.detect_paradigms()
        decomp = self.decompose_words()
        stats = self.compute_morpheme_statistics()

        n_prod = affixes['n_productive_prefixes'] + affixes['n_productive_suffixes']
        paradigm_cov = stats['paradigm_coverage']
        decomp_rate = stats['decomposition_rate']

        if n_prod > 10 and paradigm_cov > 0.5:
            interpretation = (
                f'STRONG morphological structure: {n_prod} productive affixes, '
                f'{stats["n_paradigms"]} paradigms covering {paradigm_cov:.1%} of vocabulary. '
                f'Voynich words are compositional, not opaque codebook entries. '
                f'Word-level substitution cipher is likely wrong.'
            )
            morphology_confirmed = True
        elif n_prod > 5 and paradigm_cov > 0.3:
            interpretation = (
                f'MODERATE morphological structure: {n_prod} productive affixes, '
                f'{stats["n_paradigms"]} paradigms covering {paradigm_cov:.1%}. '
                f'Partial compositionality detected but not dominant.'
            )
            morphology_confirmed = False
        else:
            interpretation = (
                f'WEAK morphological structure: {n_prod} productive affixes, '
                f'{paradigm_cov:.1%} paradigm coverage. '
                f'Words appear to be mostly opaque tokens, consistent with codebook model.'
            )
            morphology_confirmed = False

        results = {
            'productive_affixes': affixes,
            'paradigms': paradigms[:30],
            'decomposition': decomp,
            'statistics': stats,
            'morphology_confirmed': morphology_confirmed,
            'interpretation': interpretation,
            'synthesis': {
                'n_productive_affixes': n_prod,
                'n_paradigms': stats['n_paradigms'],
                'paradigm_coverage': paradigm_cov,
                'decomposition_rate': decomp_rate,
                'morphology_confirmed': morphology_confirmed,
                'conclusion': interpretation,
            },
        }

        if verbose:
            print(f'\n  Morpheme Analyzer:')
            print(f'    Productive prefixes: {affixes["n_productive_prefixes"]}')
            print(f'    Productive suffixes: {affixes["n_productive_suffixes"]}')
            if affixes['productive_prefixes']:
                print(f'    --- Top Prefixes ---')
                for a in affixes['productive_prefixes'][:5]:
                    print(f'      "{a["prefix"]}" — {a["n_words"]} words '
                          f'(enrichment: {a["enrichment"]}×)')
            if affixes['productive_suffixes']:
                print(f'    --- Top Suffixes ---')
                for a in affixes['productive_suffixes'][:5]:
                    print(f'      "{a["suffix"]}" — {a["n_words"]} words '
                          f'(enrichment: {a["enrichment"]}×)')
            print(f'    Paradigms found: {stats["n_paradigms"]}')
            print(f'    Paradigm coverage: {paradigm_cov:.1%}')
            print(f'    Decomposition rate: {decomp_rate:.1%}')
            if paradigms:
                print(f'    --- Top Paradigms ---')
                for p in paradigms[:5]:
                    members_str = ', '.join(
                        f'{m["word"]}(-{m["suffix"]})' for m in p['members'][:5]
                    )
                    print(f'      "{p["stem"]}" — {p["n_members"]} members: '
                          f'{members_str}')
            print(f'    Interpretation: {interpretation}')

        return results
