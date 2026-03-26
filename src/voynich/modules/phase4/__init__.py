"""
Phase 4: Language A Decryption Attack
======================================
Phase 3 proved Language B is a 13-word notation system. Phase 4 attacks
Language A (57 words, H2=1.487) using the word-level codebook hypothesis.

Key insight: deltaH2 = 0 proves Voynich words are NOT character-level
encryptions -- they are generated at the WORD level.

Models:
  1. WholeWordCodebook         -- Each Voynich word = one plaintext word
  2. NomenclatorModel          -- Top-25 codebook + character-cipher for rest
  3. SemanticCompressionModel  -- Words as slot markers in topic templates

Approaches:
  1. BotanicalKnownPlaintext   -- Plant identifications constrain codebook
  2. SuccessorAlphabetAttack   -- Transition matrix matching (Hungarian/SA)
  3. EntropyGradientAnalysis   -- H2 gradient by page position
  4. MultiLanguageSourceTest   -- Test multiple source languages
"""

from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor, LANG_A_FULL_TARGETS
from voynich.modules.phase4.latin_herbal_corpus import LatinHerbalCorpus
from voynich.modules.phase4.model_a1_codebook import WholeWordCodebook
from voynich.modules.phase4.model_a2_nomenclator import NomenclatorModel
from voynich.modules.phase4.model_a3_semantic_compression import SemanticCompressionModel
from voynich.modules.phase4.known_plaintext_botanical import BotanicalKnownPlaintext
from voynich.modules.phase4.successor_alphabet import SuccessorAlphabetAttack
from voynich.modules.phase4.entropy_gradient_page import EntropyGradientAnalysis
from voynich.modules.phase4.multi_language_source import MultiLanguageSourceTest

__all__ = [
    'LanguageAExtractor', 'LANG_A_FULL_TARGETS',
    'LatinHerbalCorpus',
    'WholeWordCodebook', 'NomenclatorModel', 'SemanticCompressionModel',
    'BotanicalKnownPlaintext', 'SuccessorAlphabetAttack',
    'EntropyGradientAnalysis', 'MultiLanguageSourceTest',
]
