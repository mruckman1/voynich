"""
Model 5: Self-Referential Text (Grammar Induction)
=====================================================
The Voynich is generated text produced by a mechanical template system
(stochastic grammar). The question: can a compact grammar (<20 rules,
<50 symbols) reproduce ALL 17 constraints simultaneously?

Historical plausibility: MODERATE
Predicted H2: 1.0–1.5
Priority: MEDIUM

Critical test: Grammar with <20 rules and <50 symbols generates text
that matches all 17 Voynich constraints within their tolerances.
"""

import sys
import os
import random
import math
import copy
from collections import Counter
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS
from modules.statistical_analysis import full_statistical_profile, profile_distance
from modules.naibbe_cipher import MEDIAL_GLYPHS, PREFIX_GLYPHS, SUFFIX_GLYPHS


# ============================================================================
# PCFG REPRESENTATION
# ============================================================================

class ProductionRule:
    """A single PCFG production rule: LHS -> RHS with probability."""

    __slots__ = ['lhs', 'rhs', 'probability']

    def __init__(self, lhs: str, rhs: List[str], probability: float = 1.0):
        self.lhs = lhs           # Non-terminal symbol
        self.rhs = rhs           # List of symbols (terminal or non-terminal)
        self.probability = probability

    def to_dict(self) -> Dict:
        return {
            'lhs': self.lhs,
            'rhs': self.rhs,
            'probability': self.probability,
        }


class StochasticGrammar:
    """A probabilistic context-free grammar for generating Voynich-like text."""

    def __init__(self, rules: List[ProductionRule] = None,
                 terminals: set = None, seed: int = 42):
        self.rules = rules or []
        self.terminals = terminals or set()
        self.rng = random.Random(seed)
        self._rules_by_lhs = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild the LHS lookup index."""
        self._rules_by_lhs = {}
        for rule in self.rules:
            if rule.lhs not in self._rules_by_lhs:
                self._rules_by_lhs[rule.lhs] = []
            self._rules_by_lhs[rule.lhs].append(rule)

    def add_rule(self, lhs: str, rhs: List[str], probability: float = 1.0):
        rule = ProductionRule(lhs, rhs, probability)
        self.rules.append(rule)
        if lhs not in self._rules_by_lhs:
            self._rules_by_lhs[lhs] = []
        self._rules_by_lhs[lhs].append(rule)

    def generate_word(self, start: str = 'W', max_depth: int = 10) -> str:
        """Generate a single word by expanding from start symbol."""
        return self._expand(start, max_depth)

    def _expand(self, symbol: str, depth: int) -> str:
        if depth <= 0 or symbol in self.terminals:
            return symbol

        rules = self._rules_by_lhs.get(symbol, [])
        if not rules:
            return symbol

        # Choose rule probabilistically
        probs = [r.probability for r in rules]
        total = sum(probs)
        if total <= 0:
            return symbol
        probs = [p / total for p in probs]

        chosen = self.rng.choices(rules, weights=probs, k=1)[0]

        # Expand each symbol in RHS
        parts = []
        for s in chosen.rhs:
            parts.append(self._expand(s, depth - 1))
        return ''.join(parts)

    def generate_text(self, n_words: int = 500) -> str:
        """Generate n_words by repeatedly expanding from start symbol."""
        words = []
        for _ in range(n_words):
            word = self.generate_word()
            if word:
                words.append(word)
        return ' '.join(words)

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def n_symbols(self) -> int:
        nonterminals = set(r.lhs for r in self.rules)
        all_rhs = set(s for r in self.rules for s in r.rhs)
        return len(nonterminals | all_rhs | self.terminals)

    def to_dict(self) -> Dict:
        return {
            'n_rules': self.n_rules,
            'n_symbols': self.n_symbols,
            'terminals': sorted(self.terminals),
            'rules': [r.to_dict() for r in self.rules],
        }


# ============================================================================
# GRAMMAR INDUCTION VIA EVOLUTIONARY SEARCH
# ============================================================================

def _create_initial_grammar(rng: random.Random, max_rules: int,
                            max_symbols: int) -> StochasticGrammar:
    """
    Create a random initial grammar seeded with Voynich glyph knowledge.

    Structure:
      W -> [prefix?] [body] [suffix?]   (word structure)
      body -> M | M M | M M M           (medial sequences)
      prefix -> one of PREFIX_GLYPHS
      suffix -> one of SUFFIX_GLYPHS
    """
    terminals = set(PREFIX_GLYPHS + MEDIAL_GLYPHS + SUFFIX_GLYPHS)
    # Limit terminals to stay under budget
    if len(terminals) > max_symbols // 2:
        terminals = set(list(terminals)[:max_symbols // 2])

    grammar = StochasticGrammar(terminals=terminals, seed=rng.randint(0, 10000))

    # Word-level rules
    grammar.add_rule('W', ['P', 'B', 'S'], 0.4)
    grammar.add_rule('W', ['B', 'S'], 0.3)
    grammar.add_rule('W', ['P', 'B'], 0.2)
    grammar.add_rule('W', ['B'], 0.1)

    # Prefix rules
    for g in PREFIX_GLYPHS:
        if g in terminals:
            grammar.add_rule('P', [g], 1.0 / len(PREFIX_GLYPHS))

    # Suffix rules
    for g in SUFFIX_GLYPHS:
        if g in terminals:
            grammar.add_rule('S', [g], 1.0 / len(SUFFIX_GLYPHS))

    # Body rules (medial sequences of 1-4 characters)
    grammar.add_rule('B', ['M'], 0.2)
    grammar.add_rule('B', ['M', 'M'], 0.35)
    grammar.add_rule('B', ['M', 'M', 'M'], 0.3)
    grammar.add_rule('B', ['M', 'M', 'M', 'M'], 0.15)

    # Medial character rules
    medial_in_terminals = [g for g in MEDIAL_GLYPHS if g in terminals]
    for g in medial_in_terminals:
        grammar.add_rule('M', [g], 1.0 / max(len(medial_in_terminals), 1))

    # Add some random additional rules up to budget
    nonterminals = ['W', 'P', 'B', 'S', 'M']
    while grammar.n_rules < max_rules and len(nonterminals) < 8:
        # Create new non-terminal
        new_nt = f'N{len(nonterminals)}'
        nonterminals.append(new_nt)
        # Add rules for it
        n_expansions = rng.randint(2, 4)
        for _ in range(n_expansions):
            rhs_len = rng.randint(1, 3)
            rhs = [rng.choice(list(terminals) + nonterminals[:5]) for _ in range(rhs_len)]
            grammar.add_rule(new_nt, rhs, rng.random())
        # Link it from body
        grammar.add_rule('B', [new_nt], rng.random() * 0.3)

    grammar._rebuild_index()
    return grammar


def _mutate_grammar(grammar: StochasticGrammar, rng: random.Random,
                    max_rules: int, max_symbols: int) -> StochasticGrammar:
    """Apply a random mutation to the grammar."""
    new_grammar = StochasticGrammar(
        rules=[copy.copy(r) for r in grammar.rules],
        terminals=set(grammar.terminals),
        seed=rng.randint(0, 10000)
    )

    mutation_type = rng.choice([
        'adjust_prob', 'adjust_prob', 'adjust_prob',  # Most common
        'add_rule', 'remove_rule', 'modify_rhs',
    ])

    if mutation_type == 'adjust_prob' and new_grammar.rules:
        # Adjust a random rule's probability
        rule = rng.choice(new_grammar.rules)
        rule.probability = max(0.01, rule.probability + rng.gauss(0, 0.1))

    elif mutation_type == 'add_rule' and new_grammar.n_rules < max_rules:
        # Add a new rule
        lhs_options = list(set(r.lhs for r in new_grammar.rules))
        if lhs_options:
            lhs = rng.choice(lhs_options)
            rhs_len = rng.randint(1, 3)
            all_symbols = list(new_grammar.terminals) + lhs_options
            rhs = [rng.choice(all_symbols) for _ in range(rhs_len)]
            new_grammar.add_rule(lhs, rhs, rng.random())

    elif mutation_type == 'remove_rule' and len(new_grammar.rules) > 5:
        # Remove a random rule (but keep at least 5)
        idx = rng.randint(0, len(new_grammar.rules) - 1)
        new_grammar.rules.pop(idx)

    elif mutation_type == 'modify_rhs' and new_grammar.rules:
        # Change one symbol in a rule's RHS
        rule = rng.choice(new_grammar.rules)
        if rule.rhs:
            all_symbols = list(new_grammar.terminals) + list(set(r.lhs for r in new_grammar.rules))
            idx = rng.randint(0, len(rule.rhs) - 1)
            rule.rhs[idx] = rng.choice(all_symbols)

    new_grammar._rebuild_index()
    return new_grammar


# ============================================================================
# MODEL CLASS
# ============================================================================

class GrammarInduction(Phase2GenerativeModel):
    """
    Model 5: Grammar Induction.

    Uses evolutionary search to find a compact stochastic grammar
    that generates text matching the Voynich's statistical fingerprint.
    """

    MODEL_NAME = 'grammar_induction'
    MODEL_PRIORITY = 'MEDIUM'

    def __init__(self, max_rules: int = 20, max_symbols: int = 50,
                 evolution_generations: int = 200,
                 population_size: int = 20,
                 seed: int = 42, **kwargs):
        params = {
            'max_rules': max_rules,
            'max_symbols': max_symbols,
            'evolution_generations': evolution_generations,
            'population_size': population_size,
            'seed': seed,
        }
        super().__init__(**params)

        self.max_rules = max_rules
        self.max_symbols = max_symbols
        self.evolution_generations = evolution_generations
        self.population_size = population_size

        self.best_grammar = None
        self.best_score = float('inf')

    def generate(self, plaintext: str = '', n_words: int = 500) -> str:
        """
        Generate text from the current best grammar.
        If no grammar has been induced, creates a random one.
        """
        if self.best_grammar is None:
            self.best_grammar = _create_initial_grammar(
                self.rng, self.max_rules, self.max_symbols
            )
        return self.best_grammar.generate_text(n_words)

    def induce_grammar(self, verbose: bool = False) -> Dict:
        """
        Run evolutionary search to find a grammar that produces
        Voynich-like text.

        Returns:
            {best_grammar: Dict, best_distance: float,
             generations: int, improvement_history: [...]}
        """
        # Build target profile for scoring
        voynich_profile = {
            'entropy': {
                'H1': VOYNICH_TARGETS['H1'],
                'H2': VOYNICH_TARGETS['H2'],
                'H3': VOYNICH_TARGETS['H3'],
            },
            'zipf': {
                'zipf_exponent': VOYNICH_TARGETS['zipf_exponent'],
                'type_token_ratio': VOYNICH_TARGETS['type_token_ratio'],
            },
            'mean_word_length': VOYNICH_TARGETS['mean_word_length'],
            'positional_entropy': {},
        }

        # Initialize population
        population = []
        for _ in range(self.population_size):
            grammar = _create_initial_grammar(
                random.Random(self.rng.randint(0, 100000)),
                self.max_rules, self.max_symbols
            )
            population.append(grammar)

        history = []
        best_ever_score = float('inf')
        best_ever_grammar = None

        for gen in range(self.evolution_generations):
            # Evaluate population
            scored = []
            for grammar in population:
                text = grammar.generate_text(500)
                if not text or len(text.split()) < 20:
                    scored.append((float('inf'), grammar))
                    continue
                profile = full_statistical_profile(text, 'grammar_output')
                dist = profile_distance(profile, voynich_profile)
                scored.append((dist, grammar))

            scored.sort(key=lambda x: x[0])

            # Track best
            if scored[0][0] < best_ever_score:
                best_ever_score = scored[0][0]
                best_ever_grammar = scored[0][1]

            if verbose and gen % 20 == 0:
                print(f'  [grammar] Gen {gen}: best distance = {scored[0][0]:.4f} '
                      f'(best ever = {best_ever_score:.4f})')

            history.append({
                'generation': gen,
                'best_distance': scored[0][0],
                'median_distance': scored[len(scored)//2][0],
            })

            # Selection: keep top half
            survivors = [g for _, g in scored[:self.population_size // 2]]

            # Reproduction: mutate survivors
            new_population = list(survivors)
            while len(new_population) < self.population_size:
                parent = self.rng.choice(survivors)
                child = _mutate_grammar(
                    parent, random.Random(self.rng.randint(0, 100000)),
                    self.max_rules, self.max_symbols
                )
                new_population.append(child)

            population = new_population

        self.best_grammar = best_ever_grammar
        self.best_score = best_ever_score

        return {
            'best_grammar': best_ever_grammar.to_dict() if best_ever_grammar else None,
            'best_distance': best_ever_score,
            'generations': self.evolution_generations,
            'improvement_history': history,
        }

    def parameter_grid(self, resolution: str = 'medium') -> List[Dict]:
        """Generate parameter sweep grid."""
        if resolution == 'coarse':
            rules = [10, 20]
            symbols = [30, 50]
            gens = [100]
        elif resolution == 'medium':
            rules = [10, 15, 20, 25]
            symbols = [30, 40, 50]
            gens = [100, 300]
        else:
            rules = [10, 15, 20, 25, 30]
            symbols = [30, 40, 50, 60]
            gens = [100, 300, 500, 1000]

        grid = []
        for r in rules:
            for s in symbols:
                for g in gens:
                    grid.append({
                        'max_rules': r,
                        'max_symbols': s,
                        'evolution_generations': g,
                        'population_size': 20,
                        'seed': 42,
                    })
        return grid

    def critical_test(self, generated_profile: Dict) -> Dict:
        """
        Critical test: grammar complexity AND constraint matching.

        A grammar with <20 rules and <50 symbols that matches all 17
        constraints means the "generated text" hypothesis cannot be rejected.
        """
        if self.best_grammar is None:
            return {
                'passes': False,
                'description': 'No grammar induced yet. Call induce_grammar() first.',
                'details': {},
            }

        n_rules = self.best_grammar.n_rules
        n_symbols = self.best_grammar.n_symbols
        complexity_ok = n_rules <= self.max_rules and n_symbols <= self.max_symbols

        # Check distance
        distance_ok = self.best_score < 2.0

        passes = complexity_ok and distance_ok

        return {
            'passes': passes,
            'description': (
                f'Grammar: {n_rules} rules, {n_symbols} symbols '
                f'(budget: {self.max_rules}/{self.max_symbols}). '
                f'Distance: {self.best_score:.4f}. '
                f'{"PASS" if passes else "FAIL"}'
            ),
            'details': {
                'n_rules': n_rules,
                'n_symbols': n_symbols,
                'max_rules': self.max_rules,
                'max_symbols': self.max_symbols,
                'complexity_ok': complexity_ok,
                'best_distance': self.best_score,
                'distance_ok': distance_ok,
            },
        }
