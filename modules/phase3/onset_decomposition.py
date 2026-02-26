"""
Attack 2: Onset Is the Message
================================
Decomposes Language B words into [onset] + [body] structure,
testing whether the observed onsets encode a small symbol alphabet.

Onset decomposition: [modifier] x [base]
  base = {ch, sh, k, t}
  modifier = {null, l, qo, y, o}

7 unique onsets observed in Language B's 13-word vocabulary.
The full corpus test checks if Language A words fill more grid cells.

Priority: HIGH
"""

import sys
import os
import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase3.lang_b_profiler import (
    LanguageBProfiler, LANG_B_VOCABULARY, LANG_B_TARGETS
)
from data.voynich_corpus import (
    get_all_tokens, SAMPLE_CORPUS, ZODIAC_LABELS, SECTIONS
)


# Onset decomposition grid axes
BASES = ['ch', 'sh', 'k', 't']
MODIFIERS = ['', 'l', 'qo', 'y', 'o']

# Known Language B suffix patterns (bodies)
LANG_B_BODIES = ['edy', 'eedy', 'eey', 'aiin', 'ain', 'aiir']

# Classical planets for mapping hypothesis
CLASSICAL_PLANETS = ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']

# Zodiac ruling planets (traditional)
ZODIAC_RULING_PLANETS = {
    'Aries': 'Mars', 'Taurus': 'Venus', 'Gemini': 'Mercury',
    'Cancer': 'Moon', 'Leo': 'Sun', 'Virgo': 'Mercury',
    'Libra': 'Venus', 'Scorpio': 'Mars', 'Sagittarius': 'Jupiter',
    'Capricorn': 'Saturn', 'Aquarius': 'Saturn', 'Pisces': 'Jupiter',
}


class OnsetDecomposition:
    """
    Decomposes Language B words into onset+body components and tests
    whether the onset system encodes a small symbol alphabet.
    """

    def __init__(self, profiler: Optional[LanguageBProfiler] = None):
        self.profiler = profiler or LanguageBProfiler()
        self.profiler.extract_corpus()

    def decompose_word(self, word: str) -> Dict:
        """
        Decompose a single word into onset + body.

        First tries known Language B vocabulary, then falls back to
        suffix-stripping heuristic.
        """
        if word in LANG_B_VOCABULARY:
            info = LANG_B_VOCABULARY[word]
            onset = info['onset']
            body = info['body']
            family = info['family']
        else:
            # Heuristic: try known body patterns longest-first
            onset, body, family = '', word, 'unknown'
            for suffix in sorted(LANG_B_BODIES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) > len(suffix):
                    onset = word[:-len(suffix)]
                    body = suffix
                    if suffix in ('edy', 'eedy'):
                        family = 'edy'
                    elif suffix in ('aiin', 'ain', 'aiir'):
                        family = 'aiin'
                    else:
                        family = 'residual'
                    break

        # Decompose onset into modifier + base
        modifier, base = self._decompose_onset(onset)

        return {
            'word': word,
            'onset': onset,
            'body': body,
            'modifier': modifier,
            'base': base,
            'family': family,
            'fully_parsed': onset != '' or word in LANG_B_VOCABULARY,
        }

    def _decompose_onset(self, onset: str) -> Tuple[str, str]:
        """
        Decompose an onset into (modifier, base).

        Tries modifiers longest-first to find a valid (modifier, base) pair.
        """
        if not onset:
            return ('', '')

        # Try modifiers longest first
        for mod in sorted(MODIFIERS, key=len, reverse=True):
            if mod and onset.startswith(mod):
                remainder = onset[len(mod):]
                if remainder in BASES or remainder in ('ch', 'sh', 'k', 't'):
                    return (mod, remainder)

        # No modifier — onset itself might be a base
        if onset in BASES:
            return ('', onset)

        # Check if entire onset matches a known pattern
        # e.g., 'ot' = modifier 'o' + base 't'
        for mod in MODIFIERS:
            if mod and onset.startswith(mod):
                remainder = onset[len(mod):]
                if remainder:
                    return (mod, remainder)

        return ('', onset)

    def decompose_all_lang_b(self) -> List[Dict]:
        """Decompose all Language B tokens."""
        decompositions = []
        for token in self.profiler.lang_b_tokens:
            decompositions.append(self.decompose_word(token))
        return decompositions

    def build_onset_grid(self) -> Dict:
        """
        Build the modifier x base onset grid for Language B.

        For each (modifier, base) pair, count distinct words and total tokens.
        """
        decompositions = self.decompose_all_lang_b()

        grid = {}
        for mod in MODIFIERS:
            for base in BASES:
                grid[(mod, base)] = {'words': set(), 'count': 0}

        unparsed = []
        for d in decompositions:
            key = (d['modifier'], d['base'])
            if key in grid and d['base'] in BASES:
                grid[key]['words'].add(d['word'])
                grid[key]['count'] += 1
            elif d['onset']:  # Has onset but doesn't fit grid
                unparsed.append(d)

        filled = {k: {'words': sorted(v['words']), 'count': v['count']}
                  for k, v in grid.items() if v['count'] > 0}
        empty = [k for k, v in grid.items() if v['count'] == 0]

        return {
            'grid': {f'{k[0]}+{k[1]}' if k[0] else k[1]: v
                     for k, v in filled.items()},
            'n_filled_cells': len(filled),
            'n_empty_cells': len(empty),
            'total_possible': len(BASES) * len(MODIFIERS),
            'empty_cells': [f'{k[0]}+{k[1]}' if k[0] else k[1] for k in empty],
            'unparsed': unparsed,
        }

    def run_on_full_corpus(self) -> Dict:
        """
        Run onset decomposition on the FULL corpus (all tokens, both
        Language A and Language B).

        Checks if Language A words fill more grid cells, which would
        confirm the two languages have different onset structures.
        """
        all_tokens = get_all_tokens()
        lang_a_tokens = get_all_tokens(lang='A')
        lang_b_tokens = get_all_tokens(lang='B')

        # Decompose all tokens
        full_grid = defaultdict(lambda: {'words': set(), 'count': 0, 'lang_a': 0, 'lang_b': 0})

        for tok in all_tokens:
            d = self.decompose_word(tok)
            if d['base'] in BASES and d['fully_parsed']:
                key = (d['modifier'], d['base'])
                full_grid[key]['words'].add(tok)
                full_grid[key]['count'] += 1

        # Tag Language A vs B contributions
        for tok in lang_a_tokens:
            d = self.decompose_word(tok)
            if d['base'] in BASES and d['fully_parsed']:
                key = (d['modifier'], d['base'])
                full_grid[key]['lang_a'] += 1

        for tok in lang_b_tokens:
            d = self.decompose_word(tok)
            if d['base'] in BASES and d['fully_parsed']:
                key = (d['modifier'], d['base'])
                full_grid[key]['lang_b'] += 1

        # Language B grid
        lang_b_grid = self.build_onset_grid()

        # Count cells with at least one token
        full_filled = {k for k, v in full_grid.items() if v['count'] > 0}
        lang_b_only = {k for k, v in full_grid.items() if v['lang_b'] > 0 and v['lang_a'] == 0}
        lang_a_only = {k for k, v in full_grid.items() if v['lang_a'] > 0 and v['lang_b'] == 0}
        shared = {k for k, v in full_grid.items() if v['lang_a'] > 0 and v['lang_b'] > 0}

        # New combinations found in Language A that aren't in Language B
        new_in_a = lang_a_only - {k for k, v in full_grid.items() if v['lang_b'] > 0}

        # Parse success rate
        parseable = sum(1 for tok in all_tokens if self.decompose_word(tok)['fully_parsed'])
        parse_rate = parseable / len(all_tokens) if all_tokens else 0

        return {
            'full_corpus_size': len(all_tokens),
            'lang_a_size': len(lang_a_tokens),
            'lang_b_size': len(lang_b_tokens),
            'full_filled_cells': len(full_filled),
            'lang_b_only_cells': len(lang_b_only),
            'lang_a_only_cells': len(lang_a_only),
            'shared_cells': len(shared),
            'new_combinations_in_a': len(new_in_a),
            'onset_overlap': len(shared) / max(len(full_filled), 1),
            'parse_success_rate': parse_rate,
            'lang_b_grid': lang_b_grid,
        }

    def onset_entropy(self) -> Dict:
        """
        Compute the entropy of the onset distribution.

        If onsets encode an 8-symbol alphabet uniformly: H ≈ log2(8) = 3.0.
        """
        decompositions = self.decompose_all_lang_b()
        onset_counts = Counter(d['onset'] for d in decompositions if d['onset'])

        total = sum(onset_counts.values())
        if total == 0:
            return {'onset_entropy': 0, 'max_entropy': 0, 'efficiency': 0}

        probs = {onset: count / total for onset, count in onset_counts.items()}
        h = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        n_onsets = len(onset_counts)
        max_h = math.log2(n_onsets) if n_onsets > 1 else 0

        return {
            'onset_entropy': h,
            'max_entropy': max_h,
            'efficiency': h / max_h if max_h > 0 else 0,
            'n_unique_onsets': n_onsets,
            'onset_distribution': {k: round(v, 4) for k, v in
                                   sorted(probs.items(), key=lambda x: -x[1])},
            'onset_counts': dict(onset_counts.most_common()),
            'interpretation': (
                f'H(onset) = {h:.3f} bits (max {max_h:.3f} for {n_onsets} symbols). '
                f'Efficiency = {h/max_h:.1%}. '
                f'{"Near-uniform" if h/max_h > 0.9 else "Skewed"} onset usage.'
            ),
        }

    def test_planet_mapping(self) -> Dict:
        """
        Test if onsets could map to 7 classical planets + 1 wildcard.

        For zodiac pages with known signs, check if the dominant onset
        matches the ruling planet.
        """
        # Get zodiac folio data
        zodiac_folios = [f for f in self.profiler.extract_folio_level_data()
                         if 'zodiac_sign' in f]

        if not zodiac_folios:
            return {'verdict': 'NO_ZODIAC_DATA', 'consistent': False}

        # For each zodiac folio, compute onset distribution
        folio_onset_profiles = []
        for f in zodiac_folios:
            onset_counts = Counter()
            for tok in f['tokens']:
                d = self.decompose_word(tok)
                if d['onset']:
                    onset_counts[d['onset']] += 1

            dominant_onset = onset_counts.most_common(1)[0][0] if onset_counts else ''
            sign = f.get('zodiac_sign', '')
            planet = ZODIAC_RULING_PLANETS.get(sign, '')

            folio_onset_profiles.append({
                'folio': f['folio'],
                'sign': sign,
                'planet': planet,
                'dominant_onset': dominant_onset,
                'onset_distribution': dict(onset_counts),
            })

        # Check if each planet maps to a consistent onset
        planet_to_onsets = defaultdict(list)
        for p in folio_onset_profiles:
            if p['planet']:
                planet_to_onsets[p['planet']].append(p['dominant_onset'])

        # Consistency: does each planet always have the same dominant onset?
        consistent_planets = 0
        planet_assignments = {}
        for planet, onsets in planet_to_onsets.items():
            most_common = Counter(onsets).most_common(1)[0]
            planet_assignments[planet] = most_common[0]
            if most_common[1] == len(onsets):
                consistent_planets += 1

        # Are the planet assignments unique? (each onset maps to one planet)
        onset_to_planet = defaultdict(set)
        for planet, onset in planet_assignments.items():
            onset_to_planet[onset].add(planet)
        unique_mapping = all(len(planets) == 1 for planets in onset_to_planet.values())

        consistency_rate = consistent_planets / max(len(planet_to_onsets), 1)

        return {
            'verdict': 'CONSISTENT' if consistency_rate > 0.7 and unique_mapping else 'INCONSISTENT',
            'consistent': consistency_rate > 0.7 and unique_mapping,
            'consistency_rate': consistency_rate,
            'unique_mapping': unique_mapping,
            'planet_assignments': planet_assignments,
            'folio_profiles': folio_onset_profiles,
            'interpretation': (
                f'{consistent_planets}/{len(planet_to_onsets)} planets map to consistent onsets. '
                f'Mapping is {"unique" if unique_mapping else "NOT unique"}. '
                f'Planet hypothesis is {"CONSISTENT" if consistency_rate > 0.7 and unique_mapping else "INCONSISTENT"}.'
            ),
        }

    def test_direction_mapping(self) -> Dict:
        """
        Test if onsets in biological section correlate with spatial
        arrangement (8 cardinal/ordinal directions).

        Limited by available metadata — tests for systematic onset
        variation within the biological section.
        """
        # Get biological section folios
        bio_folios = [f for f in self.profiler.extract_folio_level_data()
                      if f['section'] == 'biological']

        if len(bio_folios) < 2:
            return {'verdict': 'INSUFFICIENT_DATA', 'consistent': False}

        # Check if onset distributions differ between biological folios
        folio_onsets = []
        for f in bio_folios:
            onset_counts = Counter()
            for tok in f['tokens']:
                d = self.decompose_word(tok)
                if d['onset']:
                    onset_counts[d['onset']] += 1
            folio_onsets.append({
                'folio': f['folio'],
                'onset_distribution': dict(onset_counts),
                'dominant': onset_counts.most_common(1)[0][0] if onset_counts else '',
            })

        # Check diversity of dominant onsets across folios
        dominant_onsets = [f['dominant'] for f in folio_onsets if f['dominant']]
        n_distinct = len(set(dominant_onsets))
        max_possible = min(len(folio_onsets), 8)

        return {
            'verdict': 'PLAUSIBLE' if n_distinct >= 3 else 'UNLIKELY',
            'consistent': n_distinct >= 3,
            'n_bio_folios': len(bio_folios),
            'n_distinct_dominant_onsets': n_distinct,
            'max_possible': max_possible,
            'folio_onsets': folio_onsets,
            'interpretation': (
                f'{n_distinct} distinct dominant onsets across {len(bio_folios)} '
                f'biological folios (max {max_possible}). '
                f'Direction mapping is {"plausible" if n_distinct >= 3 else "unlikely"}.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run all onset decomposition analyses."""
        if verbose:
            print('\n=== Attack 2: Onset Is the Message ===')

        results = {}

        # Word decompositions
        decompositions = self.decompose_all_lang_b()
        if verbose:
            print(f'\n  Decomposed {len(decompositions)} Language B tokens')
            for d in decompositions[:10]:
                print(f'    {d["word"]:12s} = [{d["modifier"]}]+[{d["base"]}] + {d["body"]} '
                      f'({d["family"]})')

        # Onset grid
        if verbose:
            print('\n  --- Onset Grid (Language B) ---')
        results['lang_b_grid'] = self.build_onset_grid()
        if verbose:
            print(f'  Filled cells: {results["lang_b_grid"]["n_filled_cells"]}/'
                  f'{results["lang_b_grid"]["total_possible"]}')
            for cell, info in results['lang_b_grid']['grid'].items():
                print(f'    {cell:8s}: {info["count"]:3d} tokens, '
                      f'words: {", ".join(info["words"])}')

        # Full corpus grid
        if verbose:
            print('\n  --- Full Corpus Onset Grid ---')
        results['full_corpus'] = self.run_on_full_corpus()
        if verbose:
            fc = results['full_corpus']
            print(f'  Full corpus: {fc["full_filled_cells"]} filled cells')
            print(f'  Lang B only: {fc["lang_b_only_cells"]}, '
                  f'Lang A only: {fc["lang_a_only_cells"]}, '
                  f'Shared: {fc["shared_cells"]}')
            print(f'  New combinations in A: {fc["new_combinations_in_a"]}')
            print(f'  Parse success rate: {fc["parse_success_rate"]:.1%}')

        # Onset entropy
        if verbose:
            print('\n  --- Onset Entropy ---')
        results['onset_entropy'] = self.onset_entropy()
        if verbose:
            print(f'  {results["onset_entropy"]["interpretation"]}')

        # Planet mapping
        if verbose:
            print('\n  --- Planet Mapping Test ---')
        results['planet_mapping'] = self.test_planet_mapping()
        if verbose:
            print(f'  {results["planet_mapping"]["interpretation"]}')

        # Direction mapping
        if verbose:
            print('\n  --- Direction Mapping Test ---')
        results['direction_mapping'] = self.test_direction_mapping()
        if verbose:
            print(f'  {results["direction_mapping"]["interpretation"]}')

        # Synthesis
        onset_ent = results['onset_entropy']
        results['synthesis'] = {
            'n_unique_onsets': onset_ent['n_unique_onsets'],
            'onset_entropy_bits': onset_ent['onset_entropy'],
            'information_per_word_from_onset': onset_ent['onset_entropy'],
            'total_onset_information_bits': onset_ent['onset_entropy'] * len(decompositions),
            'planet_consistent': results['planet_mapping'].get('consistent', False),
            'direction_plausible': results['direction_mapping'].get('consistent', False),
        }

        if verbose:
            s = results['synthesis']
            print(f'\n  Synthesis:')
            print(f'    {s["n_unique_onsets"]} onsets encoding '
                  f'{s["onset_entropy_bits"]:.2f} bits/word')
            print(f'    Total onset information: '
                  f'{s["total_onset_information_bits"]:.0f} bits')

        return results
