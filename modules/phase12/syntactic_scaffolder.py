"""
Phase 12, Module 12.3: Syntactic Scaffolder
=============================================
Tags remaining bracketed tokens ([ykal], <ckhar>) with Part-of-Speech
constraints derived from Voynich suffix-to-Latin inflection mappings.

This transforms the bracketed unknowns from opaque stems into
grammatically constrained slots, enabling the deterministic n-gram
mask solver (Module 12.4) to filter candidates by POS.

Also provides a rule-based Latin POS tagger and POS-to-POS transition
matrix builder for the syntactic veto system (Academic Fortification).

Phase 12  ·  Voynich Convergence Attack
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

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

# ─── Latin POS Tagger: Closed-class word sets ───────────────────────
LATIN_PREPOSITIONS = frozenset({
    'in', 'cum', 'per', 'ad', 'de', 'pro', 'contra', 'sub', 'super',
    'ex', 'ab', 'sine', 'inter', 'ante', 'post', 'apud', 'circa',
    'praeter', 'propter', 'trans', 'ultra', 'infra', 'supra',
})

LATIN_CONJUNCTIONS = frozenset({
    'et', 'sed', 'aut', 'vel', 'atque', 'nec', 'neque', 'quod',
    'quia', 'nam', 'enim', 'ergo', 'igitur', 'tamen', 'autem',
    'sive', 'seu', 'ac', 'nisi', 'si', 'quam', 'ut', 'sicut',
})

LATIN_ADVERBS = frozenset({
    'bene', 'male', 'valde', 'satis', 'nimis', 'semper', 'numquam',
    'tunc', 'inde', 'ibi', 'ubi', 'non', 'iam', 'adhuc', 'item',
    'similiter', 'maxime', 'optime', 'melius', 'fortiter', 'statim',
})

LATIN_DETERMINERS = frozenset({
    'hic', 'haec', 'hoc', 'ille', 'illa', 'illud', 'iste', 'ista',
    'istud', 'qui', 'quae', 'quod', 'is', 'ea', 'id', 'idem',
    'eadem', 'omnis', 'omne', 'omnia', 'totus', 'tota', 'totum',
    'nullus', 'nulla', 'nullum', 'alius', 'alia', 'aliud',
    'quidam', 'quaedam', 'quoddam', 'aliquis', 'aliqua', 'aliquod',
})

# Latin verb endings (3rd person forms, imperatives, infinitives)
_VERB_ENDINGS = (
    'at', 'et', 'it', 'ut', 'nt', 'unt', 'ent', 'ant', 'unt',
    'atur', 'etur', 'itur', 'are', 'ere', 'ire', 'ari', 'eri', 'iri',
    'avit', 'evit', 'ivit', 'atur', 'antur', 'entur', 'untur',
)

# Latin adjective endings
_ADJ_ENDINGS = (
    'us', 'a', 'um', 'is', 'e', 'em', 'am', 'ior', 'ius',
    'alis', 'aris', 'ilis', 'anus', 'inus', 'ivus', 'osus',
)

# Coarse POS categories for transition matrix
POS_CATEGORIES = ['NOUN', 'VERB', 'ADJ', 'PREP', 'CONJ', 'ADV', 'DET', 'OTHER']


class LatinPOSTagger:
    """
    Rule-based Latin POS tagger using closed-class word sets and
    morphological suffix heuristics. Designed for medieval Latin
    herbal texts — not a full parser, but sufficient to build a
    valid POS transition matrix for the syntactic veto.
    """

    def tag(self, word: str) -> str:
        """
        Assign a coarse POS tag to a Latin word.

        Priority: closed-class lookup > verb endings > adjective endings > NOUN.
        """
        w = word.lower().strip()
        if not w:
            return 'OTHER'

        # 1. Closed-class sets (highest priority)
        if w in LATIN_PREPOSITIONS:
            return 'PREP'
        if w in LATIN_CONJUNCTIONS:
            return 'CONJ'
        if w in LATIN_ADVERBS:
            return 'ADV'
        if w in LATIN_DETERMINERS:
            return 'DET'

        # 2. Verb endings (before adjective — many overlap)
        if any(w.endswith(end) for end in _VERB_ENDINGS) and len(w) > 3:
            return 'VERB'

        # 3. Adjective endings (common Latin adj morphology)
        if any(w.endswith(end) for end in _ADJ_ENDINGS) and len(w) > 3:
            return 'ADJ'

        # 4. Default: NOUN (most open-class Latin words are nouns in herbal texts)
        return 'NOUN'

    def tag_tokens(self, tokens: List[str]) -> List[str]:
        """Tag a sequence of Latin tokens."""
        return [self.tag(t) for t in tokens]


def build_pos_transition_matrix(
    corpus_tokens: List[str],
    tagger: Optional[LatinPOSTagger] = None,
) -> Tuple[np.ndarray, List[str], LatinPOSTagger]:
    """
    Build a POS-to-POS bigram transition matrix from a Latin corpus.

    Args:
        corpus_tokens: List of Latin word tokens (e.g., from ImprovedLatinCorpus)
        tagger: Optional pre-built tagger (creates one if None)

    Returns:
        (matrix, pos_vocab, tagger) where:
            matrix: shape (N, N) normalized transition matrix P(POS_j | POS_i)
            pos_vocab: list of POS category strings matching matrix indices
            tagger: the LatinPOSTagger instance used
    """
    if tagger is None:
        tagger = LatinPOSTagger()

    pos_vocab = list(POS_CATEGORIES)
    pos_to_idx = {p: i for i, p in enumerate(pos_vocab)}
    n = len(pos_vocab)

    # Tag corpus
    tags = tagger.tag_tokens(corpus_tokens)

    # Count bigram transitions
    counts = np.zeros((n, n), dtype=float)
    for i in range(len(tags) - 1):
        t_from = tags[i]
        t_to = tags[i + 1]
        if t_from in pos_to_idx and t_to in pos_to_idx:
            counts[pos_to_idx[t_from]][pos_to_idx[t_to]] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = counts / row_sums

    return matrix, pos_vocab, tagger


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
