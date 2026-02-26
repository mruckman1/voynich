from collections import defaultdict
from modules.phase7.voynich_morphemer import VoynichMorphemer
from modules.phase7.latin_morphology import LatinMorphologyParser

class AffixAligner:
    """Takes the translated stems and deduces grammatical affix translations."""

    def __init__(self, v_morph: VoynichMorphemer, l_morph: LatinMorphologyParser, stem_map: dict):
        self.v_parsed = v_morph.parsed_tokens
        self.l_parsed = l_morph.parsed_tokens
        self.stem_map = stem_map

    def run(self):
        # Determine what Latin suffixes most commonly attach to the translated stems
        l_stem_to_suf = defaultdict(lambda: defaultdict(int))
        for token in self.l_parsed:
            l_stem_to_suf[token['stem']][token['suffix']] += 1

        v_suf_to_l_suf = defaultdict(lambda: defaultdict(int))

        decoded_text = []

        for token in self.v_parsed:
            v_stem = token['stem']
            v_suf = token['suffix']

            l_stem = self.stem_map.get(v_stem)

            if l_stem and v_suf:
                # Based on the Latin corpus, what suffix usually goes on this stem?
                if l_stem_to_suf[l_stem]:
                    best_l_suf = max(l_stem_to_suf[l_stem].items(), key=lambda x: x[1])[0]
                    v_suf_to_l_suf[v_suf][best_l_suf] += 1

            # Build visualization
            dec_stem = l_stem if l_stem else f"[{v_stem}]"
            dec_suf = f"-{v_suf}" if v_suf else ""
            decoded_text.append(f"{dec_stem}{dec_suf}")

        # Final affix map
        affix_map = {}
        for v_s, counts in v_suf_to_l_suf.items():
            affix_map[v_s] = max(counts.items(), key=lambda x: x[1])[0]

        return {
            'n_voynich_suffixes_mapped': len(affix_map),
            'affix_map': affix_map,
            'decoded_sample': ' '.join(decoded_text[:100])
        }
