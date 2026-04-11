"""
Phase 2: Super-Character Generative Models
============================================
Tests six generative models that operate at encoding units larger than
individual characters, following the Phase 1 finding that all character-level
ciphers are statistically incompatible with the Voynich Manuscript.

Models:
  1. VerboseCipher       — plaintext letter → ciphertext word
  2. SyllabaryCode       — plaintext syllable → ciphertext word
  3. SlotMachine         — combinatorial word assembly from independent slots
  4. SteganographicCarrier — carrier text with hidden deviation channel
  5. GrammarInduction    — stochastic grammar generation
  6. GlyphDecomposition  — re-alphabetization analysis
"""

from voynich.modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS
from voynich.modules.phase2.verbose_cipher import VerboseCipher
from voynich.modules.phase2.syllabary_code import SyllabaryCode
from voynich.modules.phase2.slot_machine import SlotMachine
from voynich.modules.phase2.steganographic_carrier import SteganographicCarrier
from voynich.modules.phase2.grammar_induction import GrammarInduction
from voynich.modules.phase2.glyph_decomposition import GlyphDecomposition

__all__ = [
    'Phase2GenerativeModel', 'VOYNICH_TARGETS',
    'VerboseCipher', 'SyllabaryCode', 'SlotMachine',
    'SteganographicCarrier', 'GrammarInduction', 'GlyphDecomposition',
]
