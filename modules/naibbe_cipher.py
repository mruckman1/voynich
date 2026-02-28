"""
Naibbe Cipher Implementation
==============================
A historically plausible multi-table, combinatorial cipher system that
can produce ciphertext with Voynich-like statistical properties.

Based on the 2025 Cryptologia paper by Michael A. Greshko, this implements
the core cipher mechanism using:
- Multi-table substitution (unigram and bigram)
- Structured randomization via dice/card selection
- Positional constraints that force glyphs to word-initial/medial/final positions
- Word-boundary generation that preserves Zipfian distribution

The cipher is parameterized so we can search the parameter space (Strategy 1).
"""

import random
import string
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

PREFIX_GLYPHS = list('dskqtpf')
MEDIAL_GLYPHS = list('oaielchr')
SUFFIX_GLYPHS = list('nym')
ANY_GLYPHS = list('oaeichlr')

BIGRAM_UNITS = [
    'sh', 'ch', 'ol', 'or', 'al', 'ar', 'dy', 'ey',
    'qo', 'ok', 'ot', 'in', 'an', 'ai', 'oi', 'ee',
]

FULL_ALPHABET = sorted(set(
    PREFIX_GLYPHS + MEDIAL_GLYPHS + SUFFIX_GLYPHS + ANY_GLYPHS
))

class SubstitutionTable:
    """
    A single substitution table mapping plaintext units to Voynich glyphs.
    
    The table maps:
    - Individual plaintext letters → Voynich glyph sequences (unigrams)
    - Plaintext bigrams → Voynich glyph sequences (bigrams)
    
    Positional constraints are enforced: certain output glyphs are only
    allowed at word-initial or word-final positions.
    """

    def __init__(self, table_id: int, seed: int = None):
        self.table_id = table_id
        self.rng = random.Random(seed if seed is not None else table_id)
        self.unigram_map: Dict[str, List[str]] = {}
        self.bigram_map: Dict[str, List[str]] = {}
        self._build()

    def _build(self):
        """Build the substitution mappings."""
        latin_chars = list('abcdefghilmnopqrstuvx')

        for char in latin_chars:
            n_variants = self.rng.randint(1, 3)
            variants = []
            for _ in range(n_variants):
                glyph_len = self.rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
                glyph = self._random_glyph_sequence(glyph_len)
                variants.append(glyph)
            self.unigram_map[char] = variants

        latin_bigrams = [
            'qu', 'th', 'ch', 'an', 'en', 'in', 'on', 'er', 'ar',
            'es', 'is', 'us', 'um', 'am', 'em', 'at', 'et', 'it',
            'ut', 'ab', 'ad', 'ex', 'de', 'co', 'pr', 're', 'st',
            'nt', 'ri', 'ra', 'ti', 'ta', 'tu', 'ma', 'me', 'mi',
            'na', 'ne', 'ni', 'la', 'le', 'li', 'sa', 'se', 'si',
        ]
        for bg in latin_bigrams:
            n_variants = self.rng.randint(1, 2)
            variants = []
            for _ in range(n_variants):
                glyph_len = self.rng.choices([1, 2, 3], weights=[2, 5, 3])[0]
                glyph = self._random_glyph_sequence(glyph_len)
                variants.append(glyph)
            self.bigram_map[bg] = variants

    def _random_glyph_sequence(self, length: int) -> str:
        """Generate a random Voynich glyph sequence of given length."""
        if length == 1:
            return self.rng.choice(MEDIAL_GLYPHS + ANY_GLYPHS)
        elif length == 2:
            return (self.rng.choice(ANY_GLYPHS + MEDIAL_GLYPHS) +
                    self.rng.choice(ANY_GLYPHS + MEDIAL_GLYPHS))
        else:
            middle = ''.join(self.rng.choice(MEDIAL_GLYPHS)
                             for _ in range(length - 2))
            return (self.rng.choice(ANY_GLYPHS) + middle +
                    self.rng.choice(ANY_GLYPHS))

    def substitute_unigram(self, char: str) -> str:
        """Look up a single character substitution."""
        char = char.lower()
        if char in self.unigram_map:
            return self.rng.choice(self.unigram_map[char])
        return self.rng.choice(MEDIAL_GLYPHS)

    def substitute_bigram(self, bigram: str) -> Optional[str]:
        """Look up a bigram substitution. Returns None if not in table."""
        bigram = bigram.lower()
        if bigram in self.bigram_map:
            return self.rng.choice(self.bigram_map[bigram])
        return None

class NaibbeCipher:
    """
    The Naibbe cipher: a multi-table combinatorial cipher with structured
    randomization that produces Voynich-like statistical output.
    
    Parameters:
    -----------
    n_tables : int
        Number of substitution tables (more tables = more variety)
    bigram_probability : float
        Probability of attempting bigram substitution vs unigram (0-1)
    word_length_range : tuple
        (min, max) output word length before forced word break
    prefix_probability : float
        Probability of prepending a prefix glyph to an output word
    suffix_probability : float
        Probability of appending a suffix glyph to an output word
    dice_sides : int
        Number of sides on the randomization die (table selection)
    seed : int
        Random seed for reproducibility
    """

    def __init__(self,
                 n_tables: int = 4,
                 bigram_probability: float = 0.4,
                 word_length_range: Tuple[int, int] = (3, 8),
                 prefix_probability: float = 0.35,
                 suffix_probability: float = 0.45,
                 dice_sides: int = 6,
                 seed: int = 42):

        self.n_tables = n_tables
        self.bigram_prob = bigram_probability
        self.word_min, self.word_max = word_length_range
        self.prefix_prob = prefix_probability
        self.suffix_prob = suffix_probability
        self.dice_sides = dice_sides
        self.seed = seed
        self.rng = random.Random(seed)

        self.tables = [
            SubstitutionTable(i, seed=seed * 100 + i)
            for i in range(n_tables)
        ]

        self.stats = defaultdict(int)

    def get_params(self) -> Dict:
        """Return the current parameter set as a dictionary."""
        return {
            'n_tables': self.n_tables,
            'bigram_probability': self.bigram_prob,
            'word_length_range': (self.word_min, self.word_max),
            'prefix_probability': self.prefix_prob,
            'suffix_probability': self.suffix_prob,
            'dice_sides': self.dice_sides,
            'seed': self.seed,
        }

    def _roll_dice(self) -> int:
        """Simulate a dice roll to select a table."""
        return self.rng.randint(0, self.dice_sides - 1) % self.n_tables

    def _should_use_bigram(self) -> bool:
        """Decide whether to attempt bigram substitution."""
        return self.rng.random() < self.bigram_prob

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string using the Naibbe cipher.
        
        The encryption process:
        1. Normalize plaintext (lowercase, strip non-alpha)
        2. Process characters left-to-right, choosing bigram or unigram sub
        3. Accumulate output glyphs into a word buffer
        4. Apply word-boundary rules (length constraints + positional glyphs)
        5. Output space-separated Voynich-like tokens
        """
        self.stats = defaultdict(int)

        plain = ''.join(c.lower() for c in plaintext if c.isalpha())
        if not plain:
            return ''

        output_words = []
        current_word_glyphs = []
        current_word_len = 0
        target_word_len = self.rng.randint(self.word_min, self.word_max)

        i = 0
        while i < len(plain):
            table_idx = self._roll_dice()
            table = self.tables[table_idx]

            glyph = None
            if i + 1 < len(plain) and self._should_use_bigram():
                bigram = plain[i:i + 2]
                glyph = table.substitute_bigram(bigram)
                if glyph:
                    i += 2
                    self.stats['bigram_subs'] += 1

            if glyph is None:
                glyph = table.substitute_unigram(plain[i])
                i += 1
                self.stats['unigram_subs'] += 1

            current_word_glyphs.append(glyph)
            current_word_len += len(glyph)

            if current_word_len >= target_word_len:
                word = self._finalize_word(current_word_glyphs)
                output_words.append(word)
                current_word_glyphs = []
                current_word_len = 0
                target_word_len = self.rng.randint(self.word_min, self.word_max)
                self.stats['words_emitted'] += 1

        if current_word_glyphs:
            word = self._finalize_word(current_word_glyphs)
            output_words.append(word)
            self.stats['words_emitted'] += 1

        return ' '.join(output_words)

    def _finalize_word(self, glyphs: List[str]) -> str:
        """
        Finalize a word by joining glyphs and applying positional constraints.
        Optionally prepends prefix glyphs and appends suffix glyphs.
        """
        core = ''.join(glyphs)

        if self.rng.random() < self.prefix_prob:
            prefix = self.rng.choice(PREFIX_GLYPHS)
            core = prefix + core
            self.stats['prefixes_added'] += 1

        if self.rng.random() < self.suffix_prob:
            suffix = self.rng.choice(SUFFIX_GLYPHS)
            core = core + suffix
            self.stats['suffixes_added'] += 1

        while len(core) < self.word_min:
            core += self.rng.choice(MEDIAL_GLYPHS)

        if len(core) > self.word_max + 2:
            core = core[:self.word_max + 2]

        return core

    def encrypt_medical_corpus(self, terms: List[str]) -> List[Tuple[str, str]]:
        """
        Encrypt a list of medical terms/phrases and return (plain, cipher) pairs.
        Useful for generating crib material for the known-plaintext attack.
        """
        pairs = []
        for term in terms:
            cipher = self.encrypt(term)
            pairs.append((term, cipher))
        return pairs

def generate_parameter_grid(resolution: str = 'medium') -> List[Dict]:
    """
    Generate a grid of Naibbe cipher parameter combinations to search.
    
    Resolution levels:
    - 'coarse': ~100 combinations (quick screening)
    - 'medium': ~1000 combinations (balanced)
    - 'fine':   ~5000 combinations (exhaustive)
    """
    if resolution == 'coarse':
        n_tables_range = [2, 4, 6]
        bigram_prob_range = [0.2, 0.4, 0.6]
        word_len_ranges = [(3, 6), (3, 8), (4, 9)]
        prefix_probs = [0.2, 0.35, 0.5]
        suffix_probs = [0.3, 0.45, 0.6]
        seeds = [42, 137]
    elif resolution == 'medium':
        n_tables_range = [2, 3, 4, 5, 6, 8]
        bigram_prob_range = [0.15, 0.25, 0.35, 0.45, 0.55]
        word_len_ranges = [(2, 6), (3, 7), (3, 8), (4, 9), (4, 10)]
        prefix_probs = [0.15, 0.25, 0.35, 0.45, 0.55]
        suffix_probs = [0.2, 0.35, 0.45, 0.55, 0.65]
        seeds = [42]
    else:
        n_tables_range = list(range(2, 10))
        bigram_prob_range = [i / 20 for i in range(1, 15)]
        word_len_ranges = [(2, 5), (2, 6), (3, 6), (3, 7), (3, 8),
                           (4, 8), (4, 9), (4, 10), (5, 10)]
        prefix_probs = [i / 20 for i in range(1, 15)]
        suffix_probs = [i / 20 for i in range(2, 16)]
        seeds = [42]

    grid = []
    for nt in n_tables_range:
        for bp in bigram_prob_range:
            for wl in word_len_ranges:
                for pp in prefix_probs:
                    for sp in suffix_probs:
                        for s in seeds:
                            grid.append({
                                'n_tables': nt,
                                'bigram_probability': bp,
                                'word_length_range': wl,
                                'prefix_probability': pp,
                                'suffix_probability': sp,
                                'dice_sides': 6,
                                'seed': s,
                            })

    return grid

def demo():
    """Run a quick demonstration of the Naibbe cipher."""
    cipher = NaibbeCipher(
        n_tables=4,
        bigram_probability=0.4,
        word_length_range=(3, 8),
        prefix_probability=0.35,
        suffix_probability=0.45,
        seed=42,
    )

    test_phrases = [
        "recipe artemisia cum aqua",
        "contra suffocationem matricis",
        "balneum cum herbis calidis",
        "accipe ruta et contere cum melle",
        "fomenta calida super ventrem",
        "unguentum ad provocandum menstrua",
    ]

    print("=" * 70)
    print("NAIBBE CIPHER DEMONSTRATION")
    print("=" * 70)
    for phrase in test_phrases:
        encrypted = cipher.encrypt(phrase)
        print(f"\n  PLAIN:  {phrase}")
        print(f"  CIPHER: {encrypted}")

    print(f"\n  Stats: {dict(cipher.stats)}")
    print("=" * 70)

if __name__ == '__main__':
    demo()
