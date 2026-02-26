"""
Phase 6: Three Recovery Paths After SAA Failure
=================================================
Phase 5's SAA produced zero recognizable Latin phrases. Phase 6 runs three
parallel investigations to diagnose why and find a path forward.

Path A: Fix the SAA (improved corpus + bijection enforcement + inverted weights)
Path B: Homophonic hypothesis (distributional similarity → merge → reduced SAA)
Path C: Morphological hypothesis (prefix/suffix decomposition + boundary analysis)

February 2026  ·  Voynich Convergence Attack  ·  Phase 6
"""

from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.fixed_saa import FixedSAA
from modules.phase6.homophone_detector import HomophoneDetector
from modules.phase6.homophone_merger import HomophoneMerger
from modules.phase6.reduced_saa import ReducedSAA
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from modules.phase6.boundary_analyzer import BoundaryAnalyzer

__all__ = [
    'ImprovedLatinCorpus', 'FixedSAA',
    'HomophoneDetector', 'HomophoneMerger', 'ReducedSAA',
    'MorphemeAnalyzer', 'BoundaryAnalyzer',
]
