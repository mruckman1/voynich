"""
Phase 13, Module 13.2: Deterministic English Glosser
=====================================================
Converts Phase 12's decoded Latin text into a literal English translation
using dictionary lookup + lightweight Latin inflection rules.

No LLM. No neural model. Pure 1:1 dictionary + morphological rules.

The result is intentionally literal ("take bones of toad and boil in water
hot...") to preserve cryptographic traceability.

Phase 13  ·  Voynich Convergence Attack
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple

FUNCTION_GLOSS = {
    'et': 'and', 'sed': 'but', 'vel': 'or', 'aut': 'or',
    'quia': 'because', 'quae': 'which', 'quod': 'that',
    'si': 'if', 'non': 'not', 'sive': 'or-if',
    'est': 'is', 'habet': 'has', 'valet': 'is-effective',
    'fiat': 'let-be-made', 'sit': 'let-be',
    'bibe': 'drink', 'coque': 'boil', 'fac': 'make',
    'da': 'give', 'tere': 'grind', 'recipe': 'take',
    'accipe': 'take', 'misce': 'mix', 'contere': 'grind',
    'destilla': 'distill', 'pone': 'place', 'unge': 'anoint',
    'lava': 'wash', 'nota': 'note', 'bis': 'twice',
    'hoc': 'this', 'hac': 'this', 'eos': 'them',
    'item': 'likewise', 'similiter': 'similarly',
    'quoque': 'also', 'ideo': 'therefore',
}

PREPOSITION_MAP = {
    'in': 'in', 'cum': 'with', 'per': 'through', 'ad': 'to',
    'de': 'of', 'pro': 'for', 'contra': 'against',
    'super': 'upon', 'sub': 'under', 'ex': 'from', 'ab': 'from',
}

INFLECTION_RULES: List[Tuple[str, str, str]] = [
    ('orum', 'us',  'gen_pl'),
    ('arum', 'a',   'gen_pl'),
    ('ium',  'is',  'gen_pl'),
    ('ibus', 'is',  'abl_pl'),
    ('onis', 'o',   'gen'),
    ('inis', 'en',  'gen'),
    ('icis', 'ix',  'gen'),
    ('oris', 'us',  'gen'),
    ('eris', 'us',  'gen'),
    ('itis', 'ut',  'gen'),
    ('is',   'is',  'gen_3rd'),
    ('ae',   'a',   'gen_fem'),
    ('i',    'us',  'gen_masc'),
    ('am',   'a',   'acc_fem'),
    ('um',   'us',  'acc'),
    ('em',   'is',  'acc_3rd'),
    ('o',    'us',  'abl_masc'),
    ('e',    'is',  'abl_3rd'),
    ('es',   'is',  'nom_pl'),
    ('a',    'um',  'nom_pl_neut'),
]

class LatinInflectionEngine:
    """Lightweight Latin morphological analysis for glossing.

    Attempts to identify the base form of an inflected Latin word
    and determine its grammatical role for English rendering.
    """

    def __init__(self, glossary: Dict[str, Dict]):
        self.glossary = glossary

    def lookup(self, latin_word: str) -> Optional[Tuple[str, str]]:
        """Look up a Latin word and return (english_gloss, grammatical_role).

        Tries:
          1. Direct dictionary hit
          2. Function word hit
          3. Inflection rule stripping → dictionary hit
          4. Return None if not found
        """
        word = latin_word.lower().strip()

        if word in self.glossary:
            return (self.glossary[word]['en'], 'direct')

        if word in FUNCTION_GLOSS:
            return (FUNCTION_GLOSS[word], 'function')

        for suffix, replacement, role in INFLECTION_RULES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[:-len(suffix)] + replacement
                if base in self.glossary:
                    return (self.glossary[base]['en'], role)
                stem = word[:-len(suffix)]
                for candidate_base, entry in self.glossary.items():
                    if candidate_base.startswith(stem) and len(candidate_base) <= len(stem) + 3:
                        return (entry['en'], role)

        return None

    def gloss_word(
        self, latin_word: str, prev_word: Optional[str], next_word: Optional[str]
    ) -> str:
        """Produce a contextual English gloss for a Latin word.

        For genitive forms: "of [noun]"
        For ablative after 'cum': "with [noun]"
        For accusative: just the noun
        For unresolved brackets: pass through unchanged
        """
        if latin_word.startswith('[') or latin_word.startswith('<'):
            return latin_word

        result = self.lookup(latin_word)
        if result is None:
            return f'[{latin_word}]'

        english, role = result

        if role.startswith('gen'):
            return f'of-{english}'
        elif role.startswith('abl'):
            if prev_word and prev_word.lower() in PREPOSITION_MAP:
                return english
            return f'in/with-{english}'
        elif role.startswith('acc'):
            return english
        else:
            return english

class EnglishGlosser:
    """Translates Phase 12 decoded Latin text into literal English.

    Each folio is glossed word-by-word. The result is intentionally
    "translation-ese" to preserve the 1:1 correspondence with the
    Latin and the underlying cryptographic chain.
    """

    def __init__(self, glossary_path: str):
        with open(glossary_path, 'r') as f:
            data = json.load(f)
        self.glossary = data.get('entries', data)
        self.engine = LatinInflectionEngine(self.glossary)

    def gloss_folio(self, latin_text: str) -> str:
        """Gloss a full folio's Latin text into English."""
        words = latin_text.split()
        glossed = []
        for i, word in enumerate(words):
            prev_word = words[i - 1] if i > 0 else None
            next_word = words[i + 1] if i < len(words) - 1 else None
            glossed.append(self.engine.gloss_word(word, prev_word, next_word))
        return ' '.join(glossed)

    def gloss_all(self, final_translations: Dict[str, str]) -> Dict[str, str]:
        """Gloss all folios. Returns {folio_id: english_text}."""
        return {
            folio: self.gloss_folio(text)
            for folio, text in final_translations.items()
        }

def run_english_glosser(
    phase12_data: Dict,
    glossary_path: str,
    output_path: str,
    verbose: bool = False,
) -> Dict:
    """Top-level entry point for Module 13.2.

    Args:
        phase12_data: Phase 12/13 translation data dict with 'final_translations'
        glossary_path: Path to data/english_glossary.json
        output_path: Path for output English translations JSON

    Returns:
        Dict with: folios_glossed, total_words, glossed_rate, sample_output
    """
    glosser = EnglishGlosser(glossary_path)
    final_translations = phase12_data.get('final_translations', {})

    english = glosser.gloss_all(final_translations)

    total_words = 0
    glossed_words = 0
    for text in english.values():
        words = text.split()
        total_words += len(words)
        glossed_words += sum(1 for w in words if not w.startswith('['))

    glossed_rate = glossed_words / max(1, total_words)

    output = {
        'english_translations': english,
        'metrics': {
            'folios_glossed': len(english),
            'total_words': total_words,
            'glossed_words': glossed_words,
            'glossed_rate': glossed_rate,
        },
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    if verbose:
        for folio, text in list(english.items())[:2]:
            print(f'  [{folio}]: {text[:200]}...')

    return output['metrics']
