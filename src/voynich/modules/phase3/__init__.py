"""
Phase 3: Language B First
==========================
Exploits Language B's extreme regularity (13-word vocabulary, H2=0.74)
as the entry point for the Voynich attack.

Attacks:
  1. TwoPatternAttack    -- edy/aiin family semantic correlation
  2. OnsetDecomposition   -- 4x5 onset grid analysis
  3. LangBGenerator       -- Markov chain synthetic generator
  4. LangAReprofiler      -- Re-run null framework on Language A alone
  5. HybridModel          -- Language A cipher + Language B notation
"""

from voynich.modules.phase3.lang_b_profiler import LanguageBProfiler, LANG_B_TARGETS
from voynich.modules.phase3.two_pattern_attack import TwoPatternAttack
from voynich.modules.phase3.onset_decomposition import OnsetDecomposition
from voynich.modules.phase3.lang_b_generator import LanguageBGenerator
from voynich.modules.phase3.lang_a_reprofiling import LanguageAReprofiler, LANG_A_TARGETS
from voynich.modules.phase3.hybrid_model import HybridModel

__all__ = [
    'LanguageBProfiler', 'LANG_B_TARGETS',
    'TwoPatternAttack', 'OnsetDecomposition', 'LanguageBGenerator',
    'LanguageAReprofiler', 'LANG_A_TARGETS', 'HybridModel',
]
