"""
Phase 12, Module 12.3: Syntactic Scaffolder
=============================================
Tags remaining bracketed tokens ([ykal], <ckhar>) with Part-of-Speech
constraints derived from Voynich suffix-to-Latin inflection mappings.

This transforms the bracketed unknowns from opaque stems into
grammatically constrained slots, enabling the deterministic n-gram
mask solver (Module 12.4) to filter candidates by POS.

Phase 12  ·  Voynich Convergence Attack
"""

import re
from collections import Counter
from typing import Dict, Tuple

from modules.phase7.voynich_morphemer import VoynichMorphemer


# Voynich suffix → Latin POS constraint mapping
# Derived from Phase 7's affix-to-inflection analysis
SUFFIX_POS_MAP = {
    'iin': 'VERB_3P',       # -t / -nt (3rd person verb endings)
    'in':  'NOUN_ACC',      # -um / -em (accusative nouns)
    'dy':  'NOUN_FEM',      # -ae / -a (feminine nominative/genitive)
    'ey':  'NOUN_NOM',      # -es / -er (nominative/agent nouns)
    'y':   'ADJ',           # -a / -i / -e (adjective agreement)
    'l':   'NOUN',          # -al / -il (neuter nouns)
    'r':   'NOUN_AGT',      # -or / -er (agent nouns)
    'm':   'NOUN_ACC',      # -am / -um (accusative)
    's':   'NOUN_PL',       # -as / -os / -es (plural)
}

# Regex to match bracketed tokens: [word] or <word>
BRACKET_RE = re.compile(r'\[([^\]]+)\]|<([^>]+)>')


class SyntacticScaffolder:
    """
    Annotates bracketed (unresolved) tokens with POS tags based on
    their Voynich suffix, producing structured slots for the n-gram solver.

    Example: "[ykal]" → "[ykal_VERB_3P]" if the original Voynich token
    had the suffix 'iin'.
    """

    def __init__(self, v_morphemer: VoynichMorphemer):
        """
        Args:
            v_morphemer: Phase 7 VoynichMorphemer for affix stripping
        """
        self.v_morphemer = v_morphemer

    def _get_pos_tag(self, token: str) -> str:
        """
        Determine the POS tag for a bracketed Voynich token.

        Args:
            token: The raw Voynich token (without brackets)

        Returns:
            POS tag string (e.g., 'VERB_3P', 'NOUN_ACC') or 'UNK'
        """
        _, _, suffix = self.v_morphemer._strip_affixes(token)

        if suffix and suffix in SUFFIX_POS_MAP:
            return SUFFIX_POS_MAP[suffix]

        return 'UNK'

    def scaffold(self, decoded_text: str) -> str:
        """
        Replace bracketed tokens with POS-tagged versions.

        Transforms:
            "facies [ykal] et oleo"
        Into:
            "facies [ykal_VERB_3P] et oleo"

        Args:
            decoded_text: Decoded Latin text with bracketed unknowns

        Returns:
            Scaffolded text with POS-annotated brackets
        """
        def _replace_bracket(match):
            # Extract the token from either [token] or <token>
            square_token = match.group(1)
            angle_token = match.group(2)

            if square_token is not None:
                token = square_token
                pos = self._get_pos_tag(token)
                return f"[{token}_{pos}]"
            else:
                token = angle_token
                pos = self._get_pos_tag(token)
                return f"<{token}_{pos}>"

        return BRACKET_RE.sub(_replace_bracket, decoded_text)

    def get_bracket_stats(self, text: str) -> Dict:
        """
        Count bracketed tokens by POS type.

        Args:
            text: Scaffolded text with POS-tagged brackets

        Returns:
            Dict with counts per POS tag and totals
        """
        # Match POS-tagged brackets: [token_POS] or <token_POS>
        tagged_re = re.compile(r'\[([^_\]]+)_([A-Z_]+)\]|<([^_>]+)_([A-Z_]+)>')

        pos_counts = Counter()
        total_square = 0
        total_angle = 0

        for match in tagged_re.finditer(text):
            if match.group(1) is not None:
                pos_counts[match.group(2)] += 1
                total_square += 1
            else:
                pos_counts[match.group(4)] += 1
                total_angle += 1

        return {
            'total_brackets': total_square + total_angle,
            'square_brackets': total_square,
            'angle_brackets': total_angle,
            'by_pos': dict(pos_counts),
        }
