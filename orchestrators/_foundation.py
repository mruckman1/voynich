"""Shared initialization chains for phase orchestrators.

Phases 7-12 all construct the same pipeline of extractors, splitters,
morphemers, and Latin corpus instances. This module provides factory
functions that eliminate the 6 copies of that initialization code.
"""
from typing import Optional

from modules.phase4.lang_a_extractor import LanguageAExtractor
from modules.phase5.tier_splitter import TierSplitter
from modules.phase6.improved_latin_corpus import ImprovedLatinCorpus
from modules.phase6.morpheme_analyzer import MorphemeAnalyzer
from modules.phase7.voynich_morphemer import VoynichMorphemer

class FoundationContext:
    """Container for the standard pipeline objects shared across phases 6-12.

    Avoids passing 5-8 individual objects between functions.
    """
    __slots__ = (
        'extractor', 'splitter', 'morpheme_analyzer', 'p6_morph_results',
        'voynich_morphemer', 'latin_corpus', 'latin_tokens',
    )

    def __init__(
        self,
        extractor: LanguageAExtractor,
        splitter: TierSplitter,
        morpheme_analyzer: Optional[MorphemeAnalyzer] = None,
        p6_morph_results: Optional[dict] = None,
        voynich_morphemer: Optional[VoynichMorphemer] = None,
        latin_corpus: Optional[ImprovedLatinCorpus] = None,
        latin_tokens: Optional[list] = None,
    ):
        self.extractor = extractor
        self.splitter = splitter
        self.morpheme_analyzer = morpheme_analyzer
        self.p6_morph_results = p6_morph_results
        self.voynich_morphemer = voynich_morphemer
        self.latin_corpus = latin_corpus
        self.latin_tokens = latin_tokens

def build_base_context(verbose: bool = False) -> FoundationContext:
    """Build the minimal context: extractor + splitter.

    Used by phases that don't need morphology (e.g. 5, 6).
    """
    extractor = LanguageAExtractor(verbose=verbose)
    splitter = TierSplitter(extractor)
    splitter.split()
    return FoundationContext(extractor=extractor, splitter=splitter)

def build_morphological_context(
    verbose: bool = False,
    latin_corpus_tokens: int = 30_000,
) -> FoundationContext:
    """Build the full context: extractor + splitter + morphology + Latin corpus.

    This is the 8-line block that phases 7, 8, 9, 10, 11, 12 all duplicate:
        extractor = LanguageAExtractor(verbose=False)
        splitter = TierSplitter(extractor)
        splitter.split()
        m_analyzer = MorphemeAnalyzer(splitter)
        p6_morph = m_analyzer.run(verbose=False)
        v_morph = VoynichMorphemer(splitter, p6_morph)
        v_morph.process_corpus()
        l_corpus = ImprovedLatinCorpus(target_tokens=N, verbose=False)
    """
    ctx = build_base_context(verbose=verbose)

    m_analyzer = MorphemeAnalyzer(ctx.splitter)
    p6_morph = m_analyzer.run(verbose=False)

    v_morph = VoynichMorphemer(ctx.splitter, p6_morph)
    v_morph.process_corpus()

    l_corpus = ImprovedLatinCorpus(target_tokens=latin_corpus_tokens, verbose=verbose)
    l_tokens = l_corpus.get_tokens()

    ctx.morpheme_analyzer = m_analyzer
    ctx.p6_morph_results = p6_morph
    ctx.voynich_morphemer = v_morph
    ctx.latin_corpus = l_corpus
    ctx.latin_tokens = l_tokens

    return ctx
