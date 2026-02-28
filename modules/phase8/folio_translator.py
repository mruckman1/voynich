from collections import defaultdict
from modules.phase4.lang_a_extractor import LanguageAExtractor

class FolioTranslator:
    """Applies the translation pipeline to actual Voynich Folios."""

    def __init__(self, extractor: LanguageAExtractor, v_morph, stem_map: dict, synthesizer):
        self.by_folio = extractor.extract_lang_a_by_folio()
        self.v_morph = v_morph
        self.stem_map = stem_map
        self.synth = synthesizer

    def translate_all_folios(self) -> dict:
        translations = {}

        for folio, tokens in self.by_folio.items():
            translated_words = []

            for token in tokens:
                try:
                    prefix, v_stem, v_suffix = self.v_morph._strip_affixes(token)

                    l_stem = self.stem_map.get(v_stem)

                    if l_stem:
                        final_latin_word = self.synth.synthesize(l_stem, v_suffix)
                        translated_words.append(final_latin_word)
                    else:
                        translated_words.append(f"<{token}>")
                except Exception:
                    translated_words.append(f"[{token}]")

            translations[folio] = ' '.join(translated_words)

        return translations
