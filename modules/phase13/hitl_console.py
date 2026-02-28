"""
Phase 13, Module 13.3: Human-in-the-Loop Console
==================================================
Interactive CLI for resolving [UNRESOLVED] tokens with human judgment.

Iterates through Phase 12's final_translations, stopping at each
[token_UNRESOLVED] tag. For each:
  1. Displays the surrounding decoded Latin context
  2. Queries the FuzzySkeletonizer for viable candidates per skeleton + POS
  3. Presents ranked candidates with scores
  4. Accepts human selection or skip

Human decisions are saved to hitl_overrides.json and applied ONLY at
rendering time — the base algorithm remains untouched.

Phase 13  ·  Voynich Convergence Attack
"""

import json
import re
import os
import sys
from typing import Dict, List, Optional, Tuple

UNRESOLVED_RE = re.compile(r'\[([^_\]]+)_(UNRESOLVED|VERB_3P|NOUN_\w+|ADJ|UNK)\]')

class CandidateProvider:
    """Provides Latin word candidates for UNRESOLVED tokens using
    the Phase 12 FuzzySkeletonizer and LatinPhoneticSkeletonizer."""

    def __init__(self, fuzzy_skeletonizer, latin_skeletonizer):
        self.f_skel = fuzzy_skeletonizer
        self.l_skel = latin_skeletonizer

    def get_candidates(
        self,
        voynich_token: str,
        max_candidates: int = 10,
    ) -> List[Tuple[str, float, str]]:
        """Get ranked Latin word candidates for a Voynich token.

        Returns:
            List of (latin_word, skeleton_weight, skeleton_string) tuples,
            sorted by weight descending.
        """
        skeleton_candidates = self.f_skel.get_skeleton_candidates(voynich_token)
        if not skeleton_candidates:
            return []

        results = []
        seen = set()
        for skeleton, weight in skeleton_candidates:
            latin_words = self.l_skel.skeleton_index.get(skeleton, [])
            for lword in latin_words:
                if lword not in seen:
                    seen.add(lword)
                    results.append((lword, weight, skeleton))

        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:max_candidates]

class HITLSession:
    """Manages an interactive HITL session for resolving unresolved tokens."""

    def __init__(
        self,
        final_translations: Dict[str, str],
        candidate_provider: CandidateProvider,
        overrides_path: str,
    ):
        self.translations = final_translations
        self.provider = candidate_provider
        self.overrides_path = overrides_path
        self.overrides: Dict[str, Dict[str, str]] = {}
        if os.path.exists(overrides_path):
            with open(overrides_path, 'r') as f:
                self.overrides = json.load(f)
        self.selections_made = 0
        self.tokens_reviewed = 0
        self.skips = 0

    def _save_overrides(self) -> None:
        with open(self.overrides_path, 'w') as f:
            json.dump(self.overrides, f, indent=2)

    def _display_context(self, words: List[str], position: int, window: int = 5) -> None:
        """Print the surrounding context with the unresolved token highlighted."""
        start = max(0, position - window)
        end = min(len(words), position + window + 1)
        parts = []
        for i in range(start, end):
            if i == position:
                parts.append(f'\033[91m{words[i]}\033[0m')
            else:
                parts.append(words[i])
        print(f'  Context: {" ".join(parts)}')

    def _display_candidates(self, candidates: List[Tuple[str, float, str]]) -> None:
        """Print numbered candidate list with scores."""
        if not candidates:
            print('  No candidates found.')
            return
        for i, (word, weight, skeleton) in enumerate(candidates, 1):
            print(f'    {i}. {word:20s}  (skeleton: {skeleton}, weight: {weight:.2f})')

    def run(self, folio_filter: Optional[str] = None) -> Dict:
        """Run the interactive HITL session.

        Commands:
          [number] — select that candidate
          s/skip   — skip this token
          q/quit   — save and exit
          u/undo   — undo last selection in this folio
          ?/help   — show commands
        """
        if not sys.stdin.isatty():
            print('  HITL console requires an interactive terminal. Skipping.')
            return {
                'tokens_reviewed': 0, 'selections_made': 0,
                'skips': 0, 'interactive': False,
            }

        print('\n' + '=' * 60)
        print('HUMAN-IN-THE-LOOP RESOLUTION CONSOLE')
        print('=' * 60)
        print('Commands: [number]=select, s=skip, q=quit, u=undo, ?=help\n')

        last_folio = None
        last_tag = None

        folios = list(self.translations.items())
        if folio_filter:
            folios = [(f, t) for f, t in folios if f == folio_filter]

        for folio_id, text in folios:
            words = text.split()
            print(f'\n--- Folio {folio_id} ---')

            for i, word in enumerate(words):
                match = UNRESOLVED_RE.match(word)
                if not match:
                    continue

                v_token = match.group(1)
                pos_tag = match.group(2)
                tag = f'[{v_token}_{pos_tag}]'

                if folio_id in self.overrides and tag in self.overrides[folio_id]:
                    continue

                self.tokens_reviewed += 1
                print(f'\n  Token: {tag}  (position {i})')
                self._display_context(words, i)

                candidates = self.provider.get_candidates(v_token)
                self._display_candidates(candidates)

                while True:
                    try:
                        choice = input('  > ').strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = 'q'

                    if choice in ('q', 'quit'):
                        self._save_overrides()
                        print(f'\n  Saved {self.selections_made} overrides to {self.overrides_path}')
                        return self._stats()

                    if choice in ('s', 'skip', ''):
                        self.skips += 1
                        break

                    if choice in ('u', 'undo'):
                        if last_folio and last_tag:
                            if last_folio in self.overrides and last_tag in self.overrides[last_folio]:
                                del self.overrides[last_folio][last_tag]
                                self.selections_made -= 1
                                print(f'  Undone: {last_tag} in {last_folio}')
                        else:
                            print('  Nothing to undo.')
                        continue

                    if choice in ('?', 'help'):
                        print('  [number]=select candidate, s=skip, q=quit, u=undo, ?=help')
                        continue

                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(candidates):
                            selected = candidates[idx][0]
                            if folio_id not in self.overrides:
                                self.overrides[folio_id] = {}
                            self.overrides[folio_id][tag] = selected
                            self.selections_made += 1
                            last_folio = folio_id
                            last_tag = tag
                            print(f'  → Selected: {selected}')
                            break
                        else:
                            print(f'  Invalid number. Choose 1-{len(candidates)}')
                    except ValueError:
                        print('  Invalid input. Type a number, s, q, u, or ?')

        self._save_overrides()
        print(f'\n  Session complete. {self.selections_made} overrides saved to {self.overrides_path}')
        return self._stats()

    def _stats(self) -> Dict:
        return {
            'tokens_reviewed': self.tokens_reviewed,
            'selections_made': self.selections_made,
            'skips': self.skips,
            'interactive': True,
        }

def apply_overrides(
    final_translations: Dict[str, str],
    overrides_path: str,
) -> Dict[str, str]:
    """Apply HITL overrides to the final translations without modifying
    the base algorithm output.

    Returns:
        New dict with overrides applied (original dict unchanged)
    """
    if not os.path.exists(overrides_path):
        return dict(final_translations)

    with open(overrides_path, 'r') as f:
        overrides = json.load(f)

    result = {}
    for folio_id, text in final_translations.items():
        folio_overrides = overrides.get(folio_id, {})
        for tag, replacement in folio_overrides.items():
            text = text.replace(tag, replacement)
        result[folio_id] = text
    return result

def run_hitl_console(
    phase12_data: Dict,
    overrides_path: str,
    fuzzy_skeletonizer,
    latin_skeletonizer,
    verbose: bool = False,
    folio: Optional[str] = None,
) -> Dict:
    """Top-level entry point for Module 13.3.

    Requires interactive terminal. If stdin is not a TTY,
    returns without starting the session.
    """
    final_translations = phase12_data.get('final_translations', {})
    provider = CandidateProvider(fuzzy_skeletonizer, latin_skeletonizer)
    session = HITLSession(final_translations, provider, overrides_path)
    return session.run(folio_filter=folio)
