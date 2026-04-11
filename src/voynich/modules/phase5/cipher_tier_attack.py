"""
Cipher Tier Attack: Tier 2 Singleton Decryption
=================================================
Once Attack A produces a Tier 1 codebook mapping, the Tier 2 singletons
become a known-plaintext cipher problem. Each singleton sits in a context
of decoded Tier 1 words, heavily constraining its identity.

Three attack steps:
  1. Context Recovery: Extract ±3 decoded Tier 1 neighbors for each singleton.
  2. Character-Level Analysis: Frequency matching against Latin character stats.
  3. Pattern Dictionary Attack: Match character repeat patterns against a
     medieval Latin herbal word dictionary.

Phase 5  ·  Voynich Convergence Attack
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set

from voynich.modules.phase5.tier_splitter import TierSplitter
from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor
from voynich.modules.phase5.latin_corpus_expanded import EXPANDED_PLANT_NAMES

LATIN_CHAR_FREQUENCIES = {
    'a': 0.078, 'b': 0.015, 'c': 0.042, 'd': 0.035, 'e': 0.120,
    'f': 0.012, 'g': 0.018, 'h': 0.015, 'i': 0.095, 'l': 0.035,
    'm': 0.042, 'n': 0.065, 'o': 0.055, 'p': 0.028, 'q': 0.008,
    'r': 0.065, 's': 0.068, 't': 0.082, 'u': 0.065, 'v': 0.012,
    'x': 0.003, 'y': 0.002, 'z': 0.001,
}

LATIN_HERBAL_DICTIONARY = (
    list(EXPANDED_PLANT_NAMES) + [
        'artemisiae', 'artemisiam', 'malvae', 'malvam',
        'rutae', 'rutam', 'salviae', 'salviam',
        'rosae', 'rosam', 'rosarum',
        'camomillae', 'camomillam',
        'lavandulae', 'lavandulam',
        'plantaginis', 'plantaginem',
        'menthae', 'mentham',
        'boraginis', 'boraginem',
        'violae', 'violam',
        'hellebori', 'helleborum',
        'calendulae', 'calendulam',
        'petroselini', 'petroselinum',
        'hyperici', 'hypericum',
        'achilleae', 'achilleam',
        'papaveris', 'papaver',
        'rosmarini', 'rosmarinum',
        'nigellae', 'nigellam',
        'coriandri', 'coriandrum',
        'valerianae', 'valerianam',
        'verbenae', 'verbenam',
    ] + [
        'caput', 'capitis', 'capite',
        'venter', 'ventris', 'ventre',
        'stomachus', 'stomachi', 'stomacho',
        'hepar', 'hepatis', 'hepate',
        'splen', 'splenis', 'splene',
        'pulmo', 'pulmonis', 'pulmone',
        'ren', 'renis', 'rene', 'renum', 'renibus',
        'oculus', 'oculi', 'oculo', 'oculorum', 'oculis',
        'dens', 'dentis', 'dente', 'dentium', 'dentibus',
        'pectus', 'pectoris', 'pectore',
        'cor', 'cordis', 'corde',
        'cerebrum', 'cerebri', 'cerebro',
        'matrix', 'matricis', 'matrice',
        'vesica', 'vesicae', 'vesicam',
        'nervus', 'nervi', 'nervo', 'nervorum', 'nervis',
    ] + [
        'radix', 'radicem', 'radicis', 'radice',
        'folium', 'folia', 'foliorum', 'foliis',
        'semen', 'seminis', 'semine',
        'flos', 'floris', 'flore', 'flores', 'florum',
        'fructus', 'fructuum', 'fructibus',
        'cortex', 'corticis', 'cortice', 'corticem',
        'succus', 'succi', 'succo', 'succum',
        'pulvis', 'pulveris', 'pulvere', 'pulverem',
        'aqua', 'aquae', 'aquam',
        'vinum', 'vini', 'vino',
        'oleum', 'olei', 'oleo',
    ] + [
        'calidus', 'calida', 'calidum', 'calidi', 'calidae',
        'frigidus', 'frigida', 'frigidum', 'frigidi', 'frigidae',
        'siccus', 'sicca', 'siccum', 'sicci', 'siccae',
        'humidus', 'humida', 'humidum', 'humidi', 'humidae',
        'fortis', 'forte', 'fortior', 'fortissimus',
        'magnus', 'magna', 'magnum', 'maior',
        'parvus', 'parva', 'parvum', 'minor',
        'bonus', 'bona', 'bonum', 'melior', 'optimus',
    ]
)

def _compute_pattern_signature(word: str) -> str:
    """
    Compute the character repeat pattern of a word.

    Maps each unique character to a letter (A, B, C...) in order of
    first appearance. Reveals the internal repeat structure.

    Examples:
        'chotchol' → 'ABCDEACD' (c→A, h→B, o→C, t→D, c→A, h→B, o→C, l→E → 'ABCDABCE')
        'abcabc'   → 'ABCABC'
        'aabbcc'   → 'AABBCC'
    """
    mapping = {}
    next_label = 0
    pattern = []
    for char in word:
        if char not in mapping:
            mapping[char] = chr(65 + next_label)
            next_label += 1
        pattern.append(mapping[char])
    return ''.join(pattern)

class CipherTierAttack:
    """
    Attack the Tier 2 (singleton) words using context from decoded
    Tier 1 words and character-level cipher analysis.
    """

    def __init__(self, splitter: TierSplitter,
                 tier1_mapping: Dict[str, str],
                 extractor: LanguageAExtractor):
        self.splitter = splitter
        self.tier1_mapping = tier1_mapping
        self.extractor = extractor
        self._contexts = None
        self._singleton_chars = None

    def recover_contexts(self, context_window: int = 3) -> List[Dict]:
        """
        For each Tier 2 singleton, extract the ±N decoded Tier 1
        neighbors to constrain its identity.

        Returns list of dicts with:
            singleton, position, left_context, right_context,
            decoded_left, decoded_right, full_decoded_context
        """
        if self._contexts is not None:
            return self._contexts

        annotated = self.splitter.get_annotated_sequence()
        tokens = [t for t, _ in annotated]
        tiers = [tier for _, tier in annotated]

        contexts = []
        for i, (token, tier) in enumerate(annotated):
            if tier != 2:
                continue

            left_tokens = []
            left_decoded = []
            j = i - 1
            while j >= 0 and len(left_tokens) < context_window:
                if tiers[j] == 1:
                    left_tokens.insert(0, tokens[j])
                    left_decoded.insert(0, self.tier1_mapping.get(tokens[j], f'[{tokens[j]}]'))
                j -= 1

            right_tokens = []
            right_decoded = []
            j = i + 1
            while j < len(tokens) and len(right_tokens) < context_window:
                if tiers[j] == 1:
                    right_tokens.append(tokens[j])
                    right_decoded.append(self.tier1_mapping.get(tokens[j], f'[{tokens[j]}]'))
                j += 1

            full_decoded = (
                ' '.join(left_decoded) + f' [{token}] ' + ' '.join(right_decoded)
            )

            contexts.append({
                'singleton': token,
                'position': i,
                'left_context': left_tokens,
                'right_context': right_tokens,
                'decoded_left': left_decoded,
                'decoded_right': right_decoded,
                'full_decoded_context': full_decoded.strip(),
            })

        self._contexts = contexts
        return contexts

    def analyze_character_frequencies(self) -> Dict:
        """
        Analyze the character frequency distribution of all Tier 2 singletons
        and compare against Latin character frequencies.
        """
        tier2_tokens = self.splitter.get_tier2_tokens()

        char_counts = Counter()
        for token in tier2_tokens:
            char_counts.update(token)

        total_chars = sum(char_counts.values())
        voynich_freqs = {c: n / total_chars for c, n in char_counts.items()}

        h1 = -sum(p * math.log2(p) for p in voynich_freqs.values() if p > 0)

        common_chars = set(voynich_freqs.keys())
        latin_h1 = -sum(p * math.log2(p) for p in LATIN_CHAR_FREQUENCIES.values() if p > 0)

        lengths = [len(t) for t in tier2_tokens]
        mean_length = np.mean(lengths) if lengths else 0
        std_length = np.std(lengths) if lengths else 0

        return {
            'n_singletons': len(tier2_tokens),
            'total_characters': total_chars,
            'alphabet_size': len(char_counts),
            'character_h1': h1,
            'latin_h1': latin_h1,
            'h1_compatible': abs(h1 - latin_h1) < 0.5,
            'mean_word_length': float(mean_length),
            'std_word_length': float(std_length),
            'top_characters': char_counts.most_common(15),
            'voynich_char_freqs': dict(sorted(voynich_freqs.items(),
                                              key=lambda x: -x[1])[:15]),
        }

    def build_substitution_candidates(self) -> Dict[str, List[str]]:
        """
        Build candidate character mappings using frequency matching.

        Ranks Voynich characters and Latin characters by frequency,
        then proposes 3-5 Latin character candidates for each.
        """
        tier2_tokens = self.splitter.get_tier2_tokens()
        char_counts = Counter()
        for token in tier2_tokens:
            char_counts.update(token)

        v_ranked = [c for c, _ in char_counts.most_common()]
        l_ranked = sorted(LATIN_CHAR_FREQUENCIES.keys(),
                          key=lambda c: -LATIN_CHAR_FREQUENCIES[c])

        candidates = {}
        for i, v_char in enumerate(v_ranked):
            start = max(0, i - 2)
            end = min(len(l_ranked), i + 3)
            candidates[v_char] = l_ranked[start:end]

        return candidates

    def build_pattern_dictionary(self) -> Dict[str, List[str]]:
        """
        Build a dictionary mapping pattern signatures to Latin words.

        Under simple substitution, a Voynich word and its plaintext Latin
        equivalent must have the same repeat pattern. E.g., if 'aabbc' maps
        to 'eettr', both have pattern 'AABBC'.
        """
        pattern_dict = defaultdict(list)
        for word in LATIN_HERBAL_DICTIONARY:
            sig = _compute_pattern_signature(word)
            pattern_dict[sig].append(word)
        return dict(pattern_dict)

    def match_singletons_to_patterns(self, max_candidates: int = 10) -> List[Dict]:
        """
        For each Tier 2 singleton:
        1. Compute its pattern signature.
        2. Match against the Latin pattern dictionary.
        3. Filter by context (decoded Tier 1 neighbors).
        4. Rank candidates by character frequency compatibility.

        Returns list of dicts with singleton, pattern, candidates, best_match.
        """
        pattern_dict = self.build_pattern_dictionary()
        contexts = self.recover_contexts()

        singleton_context = {}
        for ctx in contexts:
            singleton_context[ctx['singleton']] = ctx

        tier2_types = self.splitter.get_tier2_types()
        matches = []

        for singleton in tier2_types:
            sig = _compute_pattern_signature(singleton)
            latin_candidates = pattern_dict.get(sig, [])

            context = singleton_context.get(singleton, {})
            decoded_context = context.get('full_decoded_context', '')

            scored_candidates = []
            for cand in latin_candidates[:max_candidates * 2]:
                score = 0.0
                if 'recipe' in decoded_context.lower() or 'contra' in decoded_context.lower():
                    if cand in EXPANDED_PLANT_NAMES:
                        score += 2.0
                    if len(cand) == len(singleton):
                        score += 1.0

                if len(cand) == len(singleton):
                    score += 3.0

                scored_candidates.append({
                    'latin_word': cand,
                    'score': score,
                    'length_match': len(cand) == len(singleton),
                })

            scored_candidates.sort(key=lambda x: -x['score'])
            top = scored_candidates[:max_candidates]

            matches.append({
                'singleton': singleton,
                'pattern': sig,
                'n_candidates': len(latin_candidates),
                'n_length_matched': sum(1 for c in top if c['length_match']),
                'top_candidates': top[:5],
                'context_preview': decoded_context[:100] if decoded_context else '',
            })

        return matches

    def run(self, verbose: bool = True) -> Dict:
        """Run the full cipher tier attack."""
        contexts = self.recover_contexts()
        char_analysis = self.analyze_character_frequencies()
        subst_candidates = self.build_substitution_candidates()
        pattern_matches = self.match_singletons_to_patterns()

        n_with_matches = sum(1 for m in pattern_matches if m['n_candidates'] > 0)
        n_with_length_match = sum(1 for m in pattern_matches
                                   if m['n_length_matched'] > 0)
        n_unique_match = sum(1 for m in pattern_matches
                              if m['n_candidates'] == 1)

        results = {
            'context_recovery': {
                'n_singletons_with_context': len(contexts),
                'sample_contexts': [
                    {
                        'singleton': c['singleton'],
                        'decoded_context': c['full_decoded_context'][:100],
                    }
                    for c in contexts[:10]
                ],
            },
            'character_analysis': char_analysis,
            'substitution_candidates': {
                'n_characters_mapped': len(subst_candidates),
                'sample': dict(list(subst_candidates.items())[:10]),
            },
            'pattern_matching': {
                'n_singletons': len(pattern_matches),
                'n_with_pattern_matches': n_with_matches,
                'n_with_length_match': n_with_length_match,
                'n_unique_match': n_unique_match,
                'match_rate': n_with_matches / max(1, len(pattern_matches)),
                'sample_matches': [
                    {
                        'singleton': m['singleton'],
                        'pattern': m['pattern'],
                        'n_candidates': m['n_candidates'],
                        'top': [c['latin_word'] for c in m['top_candidates'][:3]],
                    }
                    for m in pattern_matches[:10]
                ],
            },
            'synthesis': {
                'conclusion': (
                    f'Cipher tier attack: {len(contexts)} singletons analyzed. '
                    f'Character H1={char_analysis["character_h1"]:.2f} '
                    f'(Latin H1={char_analysis["latin_h1"]:.2f}, '
                    f'{"compatible" if char_analysis["h1_compatible"] else "incompatible"}). '
                    f'{n_with_matches}/{len(pattern_matches)} singletons have '
                    f'pattern-matched candidates, {n_unique_match} uniquely determined.'
                ),
            },
        }

        if verbose:
            print(f'\n  Cipher Tier Attack (Attack B):')
            print(f'    --- Step 1: Context Recovery ---')
            print(f'    Singletons with context: {len(contexts)}')
            if contexts:
                print(f'    Sample: {contexts[0]["full_decoded_context"][:80]}...')
            print(f'    --- Step 2: Character Analysis ---')
            print(f'    Alphabet size: {char_analysis["alphabet_size"]}')
            print(f'    Character H1: {char_analysis["character_h1"]:.3f} '
                  f'(Latin: {char_analysis["latin_h1"]:.3f})')
            print(f'    Compatible: {char_analysis["h1_compatible"]}')
            print(f'    Mean word length: {char_analysis["mean_word_length"]:.2f}')
            print(f'    --- Step 3: Pattern Dictionary Attack ---')
            print(f'    Singletons tested: {len(pattern_matches)}')
            print(f'    With pattern matches: {n_with_matches}')
            print(f'    With length match: {n_with_length_match}')
            print(f'    Uniquely determined: {n_unique_match}')
            if pattern_matches:
                m = pattern_matches[0]
                top_words = [c['latin_word'] for c in m['top_candidates'][:3]]
                print(f'    Sample: {m["singleton"]} (pattern {m["pattern"]}) '
                      f'→ {top_words}')

        return results
