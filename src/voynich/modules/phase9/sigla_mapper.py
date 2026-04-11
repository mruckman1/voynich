class SiglaMapper:
    """Maps Voynich phonotactics to established Medieval Latin Tironian abbreviations."""

    def __init__(self, v_tokens: list, l_syllables: list):
        self.v_tokens = v_tokens
        self.l_syllables = l_syllables

    def generate_mappings(self) -> dict:
        """
        Hard-constraints based on known medieval Sigla formats.
        Voynich gallows (t, p, k, f) heavily mirror Latin 'p' and 'q' abbreviations (pro/per/pre/quod).
        Voynich flourishes (-iin, -dy, -m) heavily mirror Latin suspension terminals (-us, -um, -tionem).
        """
        constraints = {
            "qo": ["con", "qu", "com"],
            "ch": ["ca", "ce", "co"],
            "sh": ["si", "se", "sa"],
            "d": ["de", "di"],

            "iin": ["us", "um", "is"],
            "dy": ["ae", "ti", "ur"],
            "ey": ["es", "em", "et"],
            "m": ["rum", "num"],
            "r": ["er", "ar", "or"],
            "l": ["al", "el", "il"],

            "a": ["a"], "o": ["o"], "e": ["e"], "y": ["i"]
        }

        return constraints
