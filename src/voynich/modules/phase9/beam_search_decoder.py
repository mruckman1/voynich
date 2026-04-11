import math

class SyllabicBeamSearch:
    """Uses Beam Search to decode Voynich as a continuous stream of Latin Syllables."""

    def __init__(self, voynich_tokens: list, syllable_transitions: dict, sigla_constraints: dict, beam_width: int = 15):
        self.v_tokens = voynich_tokens
        self.transitions = syllable_transitions
        self.sigla = sigla_constraints
        self.beam_width = beam_width

    def _get_candidates(self, v_token: str) -> list:
        candidates = []
        parsed = False

        for pref, l_prefs in self.sigla.items():
            if v_token.startswith(pref):
                for suf, l_sufs in self.sigla.items():
                    if v_token.endswith(suf) and len(v_token) >= len(pref) + len(suf):
                        for lp in l_prefs:
                            for ls in l_sufs:
                                candidates.append(lp + ls)
                        parsed = True

        if not parsed:
            candidates.append(v_token[0] + "e")

        return sorted(set(candidates))

    def decode(self, max_tokens: int = 200):
        beam = [(0.0, "", "<SPACE>")]

        for i, token in enumerate(self.v_tokens[:max_tokens]):
            new_beam = []
            candidates = self._get_candidates(token)

            for log_prob, text, last_syl in beam:
                for cand in candidates:
                    trans_prob = self.transitions.get(last_syl, {}).get(cand, 0.0001)
                    score = log_prob + math.log(trans_prob)

                    space_prob = self.transitions.get(cand, {}).get("<SPACE>", 0.2)
                    new_text = text + cand + (" " if space_prob > 0.1 else "")

                    new_beam.append((score, new_text, cand))

            new_beam.sort(key=lambda x: (-x[0], x[1]))
            beam = new_beam[:self.beam_width]

        best_translation = beam[0][1]

        return best_translation, {
            "tokens_processed": max_tokens,
            "beam_width_used": self.beam_width,
            "final_log_probability": beam[0][0]
        }
