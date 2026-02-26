class MorphologicalSynthesizer:
    """Takes Latin stems and grammatical suffixes and merges them applying Latin phonetic rules."""

    def __init__(self, affix_map: dict):
        self.v_suffix_to_l_suffix = affix_map

    def synthesize(self, latin_stem: str, voynich_suffix: str) -> str:
        if not latin_stem:
            return ""

        l_suffix = self.v_suffix_to_l_suffix.get(voynich_suffix, "")

        # Grammatical padding (Nulls)
        if l_suffix == "":
            return latin_stem

        # Latin Inflection rules (Simplified Medieval Herbal rules)
        if latin_stem.endswith('e') and l_suffix.startswith('e'):
            return latin_stem[:-1] + l_suffix # vale + em = valem

        if latin_stem.endswith('a') and l_suffix == 'a':
            return latin_stem # frigida + a = frigida

        if latin_stem.endswith('i') and l_suffix.startswith('i'):
            return latin_stem[:-1] + l_suffix

        # Third person verbs (vale- + -t = valet)
        if l_suffix == "t" and not latin_stem.endswith(('a', 'e', 'i', 'o', 'u')):
            return latin_stem + "it"

        return latin_stem + l_suffix
