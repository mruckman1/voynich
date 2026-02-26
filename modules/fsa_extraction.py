"""
Track 2: Finite State Automaton Extraction
=============================================
Extracts a formal grammar of Voynichese as a minimum-state FSA, then compares
Language A vs Language B topologies. The FSA state count directly measures how
much structure the cipher preserves.

Simple substitution inherits the source language's FSA. Polyalphabetic ciphers
flatten it. The Voynich FSA tells us whether we're looking at a
structure-preserving or structure-destroying cipher.
"""

import sys
import os
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.voynich_corpus import get_all_tokens, get_section_text, SAMPLE_CORPUS


# ============================================================================
# PREFIX TREE AUTOMATON
# ============================================================================

class PrefixTreeAutomaton:
    """Builds a prefix tree (trie) from a set of words."""

    def __init__(self):
        self.transitions: Dict[int, Dict[str, int]] = defaultdict(dict)
        self.accept_states: Set[int] = set()
        self.next_state = 1  # state 0 is start
        self.state_freq: Dict[int, int] = defaultdict(int)

    def add_word(self, word: str):
        """Add a word to the prefix tree."""
        state = 0
        for ch in word:
            if ch not in self.transitions[state]:
                self.transitions[state][ch] = self.next_state
                self.next_state += 1
            state = self.transitions[state][ch]
            self.state_freq[state] += 1
        self.accept_states.add(state)

    @property
    def n_states(self):
        return self.next_state

    @property
    def n_transitions(self):
        return sum(len(t) for t in self.transitions.values())


# ============================================================================
# FSA EXTRACTION VIA STATE MERGING (ALERGIA-INSPIRED)
# ============================================================================

class VoynichFSA:
    """
    Extracts a minimum-state FSA from Voynichese using a state-merging
    approach inspired by the Alergia algorithm.
    """

    def __init__(self, max_states: int = 50, merge_threshold: float = 0.1):
        self.max_states = max_states
        self.merge_threshold = merge_threshold

    def build_fsa(self, tokens: List[str]) -> Dict:
        """
        Build a character-level FSA from tokens via state merging.

        1. Build prefix tree automaton (PTA) from all tokens
        2. Merge compatible states using frequency-based compatibility test
        3. Return minimized FSA
        """
        # Step 1: Build PTA
        pta = PrefixTreeAutomaton()
        for token in tokens:
            if token:
                pta.add_word(token)

        # Step 2: Convert to mergeable representation
        transitions = dict(pta.transitions)
        accept_states = set(pta.accept_states)
        state_freq = dict(pta.state_freq)
        n_states = pta.n_states

        # Collect alphabet
        alphabet = set()
        for trans in transitions.values():
            alphabet.update(trans.keys())

        # Step 3: State merging — merge states with similar transition profiles
        # Group states by depth (distance from root)
        state_depth = {}
        queue = [(0, 0)]
        visited = {0}
        while queue:
            state, depth = queue.pop(0)
            state_depth[state] = depth
            for ch, next_state in transitions.get(state, {}).items():
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, depth + 1))

        # Merge states at the same depth with compatible transitions
        merge_map = {}  # old_state -> canonical_state
        depth_groups = defaultdict(list)
        for state, depth in state_depth.items():
            depth_groups[depth].append(state)

        for depth, states in sorted(depth_groups.items()):
            canonical = {}  # transition_signature -> canonical_state
            for state in states:
                # Compute transition signature
                trans = transitions.get(state, {})
                is_accept = state in accept_states
                sig_parts = [f"{'A' if is_accept else 'N'}"]
                for ch in sorted(alphabet):
                    if ch in trans:
                        sig_parts.append(ch)
                sig = '|'.join(sig_parts)

                if sig in canonical and len(merge_map) + n_states - len(canonical) > self.max_states:
                    merge_map[state] = canonical[sig]
                else:
                    canonical[sig] = state

        # Apply merges
        merged_transitions: Dict[int, Dict[str, int]] = defaultdict(dict)
        merged_accept = set()

        def resolve(s):
            while s in merge_map:
                s = merge_map[s]
            return s

        active_states = set()
        for state, trans in transitions.items():
            rs = resolve(state)
            active_states.add(rs)
            for ch, next_state in trans.items():
                rn = resolve(next_state)
                merged_transitions[rs][ch] = rn
                active_states.add(rn)
            if state in accept_states:
                merged_accept.add(rs)

        # Reindex states to be contiguous
        state_list = sorted(active_states)
        reindex = {s: i for i, s in enumerate(state_list)}

        final_transitions = {}
        final_accept = set()
        for state, trans in merged_transitions.items():
            ri = reindex.get(state, state)
            final_transitions[ri] = {}
            for ch, next_state in trans.items():
                rj = reindex.get(next_state, next_state)
                final_transitions[ri][ch] = rj
            if state in merged_accept:
                final_accept.add(ri)

        return {
            'states': len(state_list),
            'transitions': final_transitions,
            'accept_states': final_accept,
            'alphabet': sorted(alphabet),
            'original_pta_states': n_states,
            'compression_ratio': len(state_list) / max(1, n_states),
        }

    def minimize_fsa(self, fsa: Dict) -> Dict:
        """
        Minimize FSA using partition refinement (Myhill-Nerode).
        """
        states = set(range(fsa['states']))
        accept = fsa['accept_states']
        non_accept = states - accept
        alphabet = fsa['alphabet']
        transitions = fsa['transitions']

        # Initial partition: accept vs non-accept
        partitions = []
        if accept:
            partitions.append(frozenset(accept))
        if non_accept:
            partitions.append(frozenset(non_accept))

        changed = True
        iterations = 0
        while changed and iterations < 100:
            changed = False
            iterations += 1
            new_partitions = []
            for partition in partitions:
                # Try to split this partition
                split = self._try_split(partition, partitions, transitions, alphabet)
                if len(split) > 1:
                    changed = True
                new_partitions.extend(split)
            partitions = new_partitions

        # Build minimized FSA
        state_to_partition = {}
        for i, partition in enumerate(partitions):
            for state in partition:
                state_to_partition[state] = i

        min_transitions = {}
        min_accept = set()
        for i, partition in enumerate(partitions):
            rep = next(iter(partition))
            min_transitions[i] = {}
            for ch in alphabet:
                next_state = transitions.get(rep, {}).get(ch)
                if next_state is not None and next_state in state_to_partition:
                    min_transitions[i][ch] = state_to_partition[next_state]
            if rep in accept:
                min_accept.add(i)

        return {
            'states': len(partitions),
            'transitions': min_transitions,
            'accept_states': min_accept,
            'alphabet': alphabet,
            'original_states': fsa['states'],
            'minimized': True,
        }

    def _try_split(self, partition, all_partitions, transitions, alphabet):
        """Try to split a partition based on transition targets."""
        if len(partition) <= 1:
            return [partition]

        partition_map = {}
        for p_idx, p in enumerate(all_partitions):
            for s in p:
                partition_map[s] = p_idx

        for ch in alphabet:
            groups = defaultdict(set)
            for state in partition:
                target = transitions.get(state, {}).get(ch)
                if target is not None and target in partition_map:
                    groups[partition_map[target]].add(state)
                else:
                    groups[-1].add(state)

            if len(groups) > 1:
                return [frozenset(g) for g in groups.values()]

        return [partition]

    def fsa_topology_signature(self, fsa: Dict) -> Dict:
        """Extract topology metrics from an FSA."""
        n_states = fsa['states']
        transitions = fsa['transitions']
        accept = fsa['accept_states']

        n_transitions = sum(len(t) for t in transitions.values())

        # Branching factor
        branching_factors = [len(t) for t in transitions.values() if t]
        avg_branching = np.mean(branching_factors) if branching_factors else 0

        # Depth (longest path from start)
        depth = self._compute_depth(transitions, n_states)

        # Cycle detection
        has_cycles = self._detect_cycles(transitions, n_states)

        return {
            'n_states': n_states,
            'n_transitions': n_transitions,
            'n_accept_states': len(accept),
            'alphabet_size': len(fsa.get('alphabet', [])),
            'avg_branching_factor': float(avg_branching),
            'max_branching_factor': max(branching_factors) if branching_factors else 0,
            'depth': depth,
            'has_cycles': has_cycles,
            'transition_density': n_transitions / max(1, n_states * len(fsa.get('alphabet', []))),
        }

    def _compute_depth(self, transitions: Dict, n_states: int) -> int:
        """BFS depth from state 0."""
        visited = {0}
        queue = [(0, 0)]
        max_depth = 0
        while queue:
            state, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for ch, next_state in transitions.get(state, {}).items():
                if next_state not in visited and next_state < n_states:
                    visited.add(next_state)
                    queue.append((next_state, depth + 1))
        return max_depth

    def _detect_cycles(self, transitions: Dict, n_states: int) -> bool:
        """Check for cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)

        def dfs(state):
            color[state] = GRAY
            for ch, next_state in transitions.get(state, {}).items():
                if next_state < n_states:
                    if color[next_state] == GRAY:
                        return True
                    if color[next_state] == WHITE and dfs(next_state):
                        return True
            color[state] = BLACK
            return False

        return dfs(0)

    def compare_topologies(self, fsa_a: Dict, fsa_b: Dict) -> Dict:
        """Compare two FSA topologies (e.g., Language A vs B)."""
        sig_a = self.fsa_topology_signature(fsa_a)
        sig_b = self.fsa_topology_signature(fsa_b)

        return {
            'topology_a': sig_a,
            'topology_b': sig_b,
            'state_count_ratio': sig_a['n_states'] / max(1, sig_b['n_states']),
            'branching_ratio': sig_a['avg_branching_factor'] / max(0.01, sig_b['avg_branching_factor']),
            'depth_ratio': sig_a['depth'] / max(1, sig_b['depth']),
            'density_ratio': sig_a['transition_density'] / max(0.001, sig_b['transition_density']),
            'same_cycle_behavior': sig_a['has_cycles'] == sig_b['has_cycles'],
            'structural_similarity': self._structural_similarity(sig_a, sig_b),
        }

    def _structural_similarity(self, sig_a: Dict, sig_b: Dict) -> float:
        """Compute overall structural similarity between two FSA topologies."""
        features = ['n_states', 'avg_branching_factor', 'depth', 'transition_density']
        diffs = []
        for f in features:
            va = sig_a.get(f, 0)
            vb = sig_b.get(f, 0)
            max_val = max(abs(va), abs(vb), 1)
            diffs.append(abs(va - vb) / max_val)
        return 1.0 - np.mean(diffs)


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

def run(verbose: bool = True) -> Dict:
    """
    Run FSA extraction and comparison.

    Returns:
        Dict with FSAs for Language A, Language B, and overall;
        topology comparison; cipher type classification.
    """
    extractor = VoynichFSA(max_states=50)

    if verbose:
        print("\n" + "=" * 70)
        print("TRACK 2: FINITE STATE AUTOMATON EXTRACTION")
        print("=" * 70)

    # Extract tokens by language
    all_tokens = get_all_tokens()

    lang_a_tokens = []
    lang_b_tokens = []
    for folio, data in SAMPLE_CORPUS.items():
        lang = data.get('lang', '')
        for line in data.get('text', []):
            tokens = line.split()
            if lang == 'A':
                lang_a_tokens.extend(tokens)
            elif lang == 'B':
                lang_b_tokens.extend(tokens)

    # Build FSAs
    if verbose:
        print(f"\n  Building FSA for overall corpus ({len(all_tokens)} tokens)...")
    fsa_overall = extractor.build_fsa(all_tokens)
    fsa_overall_min = extractor.minimize_fsa(fsa_overall)

    if verbose:
        print(f"    PTA states: {fsa_overall['original_pta_states']}")
        print(f"    Merged states: {fsa_overall['states']}")
        print(f"    Minimized states: {fsa_overall_min['states']}")

    fsa_a = None
    fsa_a_min = None
    if lang_a_tokens:
        if verbose:
            print(f"\n  Building FSA for Language A ({len(lang_a_tokens)} tokens)...")
        fsa_a = extractor.build_fsa(lang_a_tokens)
        fsa_a_min = extractor.minimize_fsa(fsa_a)
        if verbose:
            print(f"    Merged: {fsa_a['states']}  Minimized: {fsa_a_min['states']}")

    fsa_b = None
    fsa_b_min = None
    if lang_b_tokens:
        if verbose:
            print(f"\n  Building FSA for Language B ({len(lang_b_tokens)} tokens)...")
        fsa_b = extractor.build_fsa(lang_b_tokens)
        fsa_b_min = extractor.minimize_fsa(fsa_b)
        if verbose:
            print(f"    Merged: {fsa_b['states']}  Minimized: {fsa_b_min['states']}")

    # Topology signatures
    overall_sig = extractor.fsa_topology_signature(fsa_overall_min)

    # Compare A vs B
    comparison = None
    if fsa_a_min and fsa_b_min:
        if verbose:
            print("\n  Comparing Language A vs Language B topologies...")
        comparison = extractor.compare_topologies(fsa_a_min, fsa_b_min)

        if verbose:
            print(f"    State count ratio (A/B): {comparison['state_count_ratio']:.2f}")
            print(f"    Branching ratio (A/B): {comparison['branching_ratio']:.2f}")
            print(f"    Structural similarity: {comparison['structural_similarity']:.3f}")

    # Classify cipher type
    cipher_type = 'unknown'
    if overall_sig['n_states'] < 15:
        cipher_type = 'structure_preserving'
        interpretation = ('Low state count suggests the cipher preserves significant '
                          'source-language structure (consistent with simple substitution '
                          'or structure-preserving cipher).')
    elif overall_sig['n_states'] > 35:
        cipher_type = 'structure_destroying'
        interpretation = ('High state count suggests the cipher destroys source-language '
                          'structure (consistent with polyalphabetic or randomizing cipher).')
    else:
        cipher_type = 'intermediate'
        interpretation = ('Intermediate state count: cipher partially preserves structure. '
                          'Consistent with a cipher that has some structure-preserving '
                          'and some randomizing components (e.g., Naibbe-type).')

    # Language A vs B interpretation
    lang_interpretation = None
    if comparison:
        sim = comparison['structural_similarity']
        if sim > 0.8:
            lang_interpretation = ('Language A and B have SIMILAR FSA topologies. '
                                   'They likely represent the same underlying language '
                                   'with different cipher parameters or scribal conventions.')
        elif sim > 0.5:
            lang_interpretation = ('Language A and B have MODERATELY different FSA topologies. '
                                   'They may represent different dialects, registers, '
                                   'or content types of the same language.')
        else:
            lang_interpretation = ('Language A and B have DIFFERENT FSA topologies. '
                                   'They may represent different source languages '
                                   'or fundamentally different cipher mechanisms.')

    results = {
        'track': 'fsa_extraction',
        'track_number': 2,
        'overall': {
            'topology': overall_sig,
            'n_minimized_states': fsa_overall_min['states'],
        },
        'language_a': {
            'topology': extractor.fsa_topology_signature(fsa_a_min) if fsa_a_min else None,
            'n_minimized_states': fsa_a_min['states'] if fsa_a_min else None,
            'n_tokens': len(lang_a_tokens),
        },
        'language_b': {
            'topology': extractor.fsa_topology_signature(fsa_b_min) if fsa_b_min else None,
            'n_minimized_states': fsa_b_min['states'] if fsa_b_min else None,
            'n_tokens': len(lang_b_tokens),
        },
        'comparison': comparison,
        'cipher_type': cipher_type,
        'cipher_type_interpretation': interpretation,
        'language_interpretation': lang_interpretation,
    }

    if verbose:
        print("\n" + "─" * 70)
        print("FSA EXTRACTION SUMMARY")
        print("─" * 70)
        print(f"  Overall minimized states: {fsa_overall_min['states']}")
        print(f"  Cipher type: {cipher_type}")
        print(f"  {interpretation}")
        if lang_interpretation:
            print(f"\n  {lang_interpretation}")

    return results
