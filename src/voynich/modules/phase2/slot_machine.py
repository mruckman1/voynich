"""
Model 3: Slot Machine (Combinatorial Word Assembly)
=====================================================
Voynich words are not atomic units but are assembled from independent
"slots", each drawn from a small set. The word "qokaiin" decomposes as:
  prefix(q) + connector(o) + root(k) + vowel_block(ai) + suffix(in)

Each slot independently encodes one dimension of information:
- prefix = topic category (e.g., humoral quality)
- root = specific item (e.g., plant identifier)
- suffix = preparation method (e.g., body part)

Total information per word = SUM of independent slot selections.

Historical plausibility: MODERATE
Predicted H2: 1.0–2.0
Priority: HIGH

Critical test: Slot mutual information < 0.3 bits for all slot pairs
AND H2/TTR/Zipf triple match.
"""

import random
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from voynich.modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS, TRIPLE_THRESHOLDS
from voynich.modules.naibbe_cipher import PREFIX_GLYPHS, MEDIAL_GLYPHS, SUFFIX_GLYPHS, ANY_GLYPHS

DEFAULT_SLOT_POOLS = {
    'prefix':      PREFIX_GLYPHS,
    'connector':   ['o', 'a', 'e', ''],
    'root':        MEDIAL_GLYPHS,
    'vowel_block': ['ai', 'oi', 'ee', 'a', 'o', 'e', 'i', ''],
    'suffix':      SUFFIX_GLYPHS + ['in', 'iin', 'y', 'dy', 'n', 'm', ''],
}

MI_THRESHOLD = 0.3

class SlotMachine(Phase2GenerativeModel):
    """
    Model 3: Combinatorial Word Assembly.

    Words are assembled from independent slots, each encoding one
    dimension of information. The slots are statistically independent,
    which explains the low H2 (cross-slot transitions are uninformative).
    """

    MODEL_NAME = 'slot_machine'
    MODEL_PRIORITY = 'HIGH'

    def __init__(self, n_slots: int = 4,
                 slot_sizes: Tuple = (5, 8, 10, 4),
                 slot_names: Tuple = ('prefix', 'connector', 'root', 'suffix'),
                 independence_noise: float = 0.0,
                 frequency_skew: float = 1.5,
                 seed: int = 42, **kwargs):
        params = {
            'n_slots': n_slots,
            'slot_sizes': tuple(slot_sizes),
            'slot_names': tuple(slot_names) if slot_names else None,
            'independence_noise': independence_noise,
            'frequency_skew': frequency_skew,
            'seed': seed,
        }
        super().__init__(**params)

        self.n_slots = n_slots
        self.slot_sizes = slot_sizes[:n_slots]
        self.slot_names = slot_names[:n_slots] if slot_names else [f'slot_{i}' for i in range(n_slots)]
        self.independence_noise = independence_noise
        self.frequency_skew = frequency_skew

        self.slot_inventories = []
        self.slot_weights = []
        self._build_slot_inventories()

    def _build_slot_inventories(self):
        """Build the inventory of glyph sequences for each slot."""
        for i in range(self.n_slots):
            name = self.slot_names[i] if i < len(self.slot_names) else f'slot_{i}'
            size = self.slot_sizes[i] if i < len(self.slot_sizes) else 5
            pool = DEFAULT_SLOT_POOLS.get(name, MEDIAL_GLYPHS + ANY_GLYPHS)

            inventory = self._generate_slot_items(pool, size)
            self.slot_inventories.append(inventory)

            ranks = list(range(1, len(inventory) + 1))
            weights = [1.0 / (r ** self.frequency_skew) for r in ranks]
            total = sum(weights)
            weights = [w / total for w in weights]
            self.slot_weights.append(weights)

    def _generate_slot_items(self, pool: list, size: int) -> List[str]:
        """Generate a list of glyph sequences for a single slot."""
        items = set()
        attempts = 0
        while len(items) < size and attempts < size * 10:
            length = self.rng.choices([1, 2, 3], weights=[3, 4, 2])[0]
            item = ''.join(self.rng.choice(pool) for _ in range(length)
                          if pool)
            if item and item not in items:
                items.add(item)
            attempts += 1

        result = sorted(items)
        while len(result) < size:
            c = self.rng.choice(pool) if pool else 'o'
            if c not in result:
                result.append(c)
            else:
                result.append(c + self.rng.choice(MEDIAL_GLYPHS))
        return result[:size]

    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Generate text by independently sampling from each slot.

        If plaintext is provided, its character content influences slot
        selections via a simple hash (simulating information encoding).
        Otherwise, slots are sampled independently with Zipf-like weights.
        """
        words = []
        plaintext_chars = [c.lower() for c in plaintext if c.isalpha()] if plaintext else []

        for word_idx in range(n_words):
            parts = []
            for slot_idx in range(self.n_slots):
                inventory = self.slot_inventories[slot_idx]
                weights = self.slot_weights[slot_idx]

                if plaintext_chars and word_idx < len(plaintext_chars):
                    char_val = ord(plaintext_chars[word_idx % len(plaintext_chars)]) - ord('a')
                    base_idx = (char_val * (slot_idx + 1)) % len(inventory)
                    if self.rng.random() < self.independence_noise:
                        item = self.rng.choices(inventory, weights=weights, k=1)[0]
                    else:
                        item = inventory[base_idx]
                else:
                    item = self.rng.choices(inventory, weights=weights, k=1)[0]

                parts.append(item)

            word = ''.join(parts)
            if word:
                words.append(word)

        return ' '.join(words)

    def _compute_slot_mutual_information(self, text: str) -> Dict:
        """
        Compute pairwise mutual information between all slot pairs
        in a corpus of generated words.

        This requires decomposing words back into slots, which we can do
        since we know the slot inventories. We use a greedy left-to-right
        parsing approach.

        Returns:
            {pair: (slot_i, slot_j), mi: float, ...}
        """
        tokens = text.split()
        slot_assignments = []
        for token in tokens:
            assignment = self._parse_word(token)
            if assignment and len(assignment) == self.n_slots:
                slot_assignments.append(assignment)

        if len(slot_assignments) < 20:
            return {'error': 'Too few parseable tokens', 'pairs': {}}

        n = len(slot_assignments)
        results = {}

        for i in range(self.n_slots):
            for j in range(i + 1, self.n_slots):
                joint = Counter()
                marginal_i = Counter()
                marginal_j = Counter()

                for assignment in slot_assignments:
                    si, sj = assignment[i], assignment[j]
                    joint[(si, sj)] += 1
                    marginal_i[si] += 1
                    marginal_j[sj] += 1

                mi = 0.0
                for (si, sj), count in joint.items():
                    p_joint = count / n
                    p_i = marginal_i[si] / n
                    p_j = marginal_j[sj] / n
                    if p_joint > 0 and p_i > 0 and p_j > 0:
                        mi += p_joint * math.log2(p_joint / (p_i * p_j))

                pair_name = f'{self.slot_names[i]}_{self.slot_names[j]}'
                results[pair_name] = {
                    'slot_i': i,
                    'slot_j': j,
                    'mi': mi,
                    'independent': mi < MI_THRESHOLD,
                }

        return {'pairs': results, 'n_parsed': len(slot_assignments)}

    def _parse_word(self, token: str) -> Optional[List[str]]:
        """
        Parse a word back into slot assignments using greedy matching.
        Returns list of slot values, or None if parsing fails.
        """
        assignment = []
        pos = 0

        for slot_idx in range(self.n_slots):
            inventory = self.slot_inventories[slot_idx]
            sorted_items = sorted(inventory, key=len, reverse=True)

            matched = False
            for item in sorted_items:
                if token[pos:pos+len(item)] == item:
                    assignment.append(item)
                    pos += len(item)
                    matched = True
                    break

            if not matched:
                return None

        return assignment if pos == len(token) else None

    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """Generate parameter sweep grid."""
        if resolution == 'coarse':
            configs = [
                (3, (5, 8, 4), ('prefix', 'root', 'suffix')),
                (4, (5, 4, 8, 4), ('prefix', 'connector', 'root', 'suffix')),
                (5, (5, 3, 8, 4, 5), ('prefix', 'connector', 'root', 'vowel_block', 'suffix')),
            ]
            noises = [0.0, 0.1]
            skews = [1.0, 1.5]
        elif resolution == 'medium':
            configs = [
                (3, (4, 6, 3), ('prefix', 'root', 'suffix')),
                (3, (6, 10, 5), ('prefix', 'root', 'suffix')),
                (4, (4, 3, 6, 3), ('prefix', 'connector', 'root', 'suffix')),
                (4, (5, 4, 8, 4), ('prefix', 'connector', 'root', 'suffix')),
                (4, (6, 5, 12, 5), ('prefix', 'connector', 'root', 'suffix')),
                (5, (4, 3, 6, 4, 4), ('prefix', 'connector', 'root', 'vowel_block', 'suffix')),
                (5, (5, 4, 10, 5, 5), ('prefix', 'connector', 'root', 'vowel_block', 'suffix')),
            ]
            noises = [0.0, 0.05, 0.1, 0.2]
            skews = [1.0, 1.5, 2.0]
        else:
            configs = [
                (3, (s1, s2, s3), ('prefix', 'root', 'suffix'))
                for s1 in [3, 5, 7] for s2 in [5, 8, 12] for s3 in [3, 5, 7]
            ] + [
                (4, (s1, s2, s3, s4), ('prefix', 'connector', 'root', 'suffix'))
                for s1 in [3, 5, 7] for s2 in [3, 4, 5] for s3 in [5, 8, 12] for s4 in [3, 5, 7]
            ]
            noises = [0.0, 0.05, 0.1, 0.15, 0.2]
            skews = [0.8, 1.0, 1.2, 1.5, 2.0]

        grid = []
        for n_slots, sizes, names in configs:
            for noise in noises:
                for skew in skews:
                    grid.append({
                        'n_slots': n_slots,
                        'slot_sizes': sizes,
                        'slot_names': names,
                        'independence_noise': noise,
                        'frequency_skew': skew,
                        'seed': 42,
                    })
        return grid

    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Critical test: slot independence AND triple match.
        """
        entropy = generated_profile.get('entropy', {})
        zipf = generated_profile.get('zipf', {})

        h2 = entropy.get('H2', 0.0)
        ttr = zipf.get('type_token_ratio', 0.0)
        zipf_exp = zipf.get('zipf_exponent', 0.0)

        h2_match = abs(h2 - VOYNICH_TARGETS['H2']) < TRIPLE_THRESHOLDS['H2']
        ttr_match = abs(ttr - VOYNICH_TARGETS['type_token_ratio']) < TRIPLE_THRESHOLDS['TTR']
        zipf_match = abs(zipf_exp - VOYNICH_TARGETS['zipf_exponent']) < TRIPLE_THRESHOLDS['zipf_exponent']
        triple_match = h2_match and ttr_match and zipf_match

        return {
            'passes': triple_match,
            'description': (
                f'Triple: H2={h2:.4f} ({"PASS" if h2_match else "FAIL"}), '
                f'TTR={ttr:.4f} ({"PASS" if ttr_match else "FAIL"}), '
                f'Zipf={zipf_exp:.4f} ({"PASS" if zipf_match else "FAIL"}). '
                f'MI test requires separate run_mi_test() call.'
            ),
            'details': {
                'H2': h2, 'TTR': ttr, 'zipf_exponent': zipf_exp,
                'triple_match': triple_match,
                'mi_test_pending': True,
            },
        }

    def run_mi_test(self, text: str) -> Dict:
        """
        Run the mutual information independence test on generated text.
        Separate from critical_test because it needs the raw text.
        """
        mi_results = self._compute_slot_mutual_information(text)
        pairs = mi_results.get('pairs', {})

        all_independent = all(
            p.get('independent', False) for p in pairs.values()
        )
        max_mi = max((p.get('mi', 0) for p in pairs.values()), default=0)

        return {
            'all_independent': all_independent,
            'max_mi': max_mi,
            'threshold': MI_THRESHOLD,
            'pairs': pairs,
            'n_parsed': mi_results.get('n_parsed', 0),
        }
