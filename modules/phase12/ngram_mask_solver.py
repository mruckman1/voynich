"""
Phase 12, Module 12.4: Deterministic N-Gram Mask Solver
========================================================
Resolves bracketed (unresolved) tokens using Latin bigram/trigram
transition matrices from Phase 5/6. Every resolution is mathematically
provable; ambiguous cases are left honestly bracketed.

Algorithm for each bracketed token [token_POS]:
  1. Target isolation: extract w_prev and w_next from context
  2. Candidate generation: skeleton lookup via FuzzySkeletonizer
  3. POS filtering: restrict candidates by Latin inflection patterns
  4. Trigram scoring: P(candidate | w_prev) * P(w_next | candidate)
  5. Humoral multiplier: boost contextually appropriate candidates
  6. Strict thresholding: lock in only if best >= 3x second-best

If the math can't prove it, it stays as [token_UNRESOLVED].

Phase 12  ·  Voynich Convergence Attack
"""

import re
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer, dynamic_levenshtein_threshold

import Levenshtein

from data.botanical_identifications import PLANT_IDS


# Latin inflection endings by POS category
# Used to filter skeleton-matched candidates by grammatical constraint
LATIN_POS_ENDINGS: Dict[str, List[str]] = {
    'VERB_3P': ['t', 'nt', 'it', 'at', 'et', 'ut', 'unt', 'ent', 'ant'],
    'NOUN_ACC': ['um', 'am', 'em', 'im', 'em'],
    'NOUN_FEM': ['a', 'ae', 'am', 'arum', 'as', 'is'],
    'NOUN_NOM': ['us', 'a', 'um', 'es', 'er', 'is', 'or', 'io'],
    'ADJ': ['us', 'a', 'um', 'is', 'e', 'es', 'i', 'em', 'am'],
    'NOUN': ['um', 'al', 'il', 'en', 'ur', 'us', 'er'],
    'NOUN_AGT': ['or', 'er', 'ar', 'rix'],
    'NOUN_PL': ['es', 'as', 'os', 'us', 'i', 'ae', 'ia', 'a'],
    'UNK': [],  # No filtering for unknown POS
}

# Regex for POS-tagged brackets
TAGGED_BRACKET_RE = re.compile(r'\[([^_\]]+)_([A-Z_]+)\]|<([^_>]+)_([A-Z_]+)>')


class NgramMaskSolver:
    """
    Deterministic resolver for bracketed tokens using n-gram transition
    probabilities. Every resolved word has a mathematical proof chain;
    ambiguous words stay bracketed.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        vocab: List[str],
        latin_skeletonizer: LatinPhoneticSkeletonizer,
        fuzzy_skeletonizer: FuzzySkeletonizer,
        humoral_vocab: Optional[Dict[str, List[str]]] = None,
        min_confidence_ratio: float = 3.0,
    ):
        """
        Args:
            transition_matrix: Word-level transition matrix P(w_j | w_i)
                from ImprovedLatinCorpus.build_transition_matrix()
            vocab: Ordered vocabulary list matching matrix dimensions
            latin_skeletonizer: Phase 11 Latin skeleton index
            fuzzy_skeletonizer: Phase 12 branching Voynich skeletonizer
            humoral_vocab: Dict mapping humoral categories to word lists
            min_confidence_ratio: Minimum ratio best/second-best to resolve
                a bracket (default 3.0x)
        """
        self.matrix = transition_matrix
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.l_skel = latin_skeletonizer
        self.f_skel = fuzzy_skeletonizer
        self.humoral_vocab = humoral_vocab or {}
        self.min_confidence_ratio = min_confidence_ratio

    def _get_transition_prob(self, w_from: str, w_to: str) -> float:
        """Look up P(w_to | w_from) from the transition matrix."""
        if w_from not in self.word_to_idx or w_to not in self.word_to_idx:
            return 0.0
        i = self.word_to_idx[w_from]
        j = self.word_to_idx[w_to]
        return float(self.matrix[i][j])

    def _candidate_matches_pos(self, word: str, pos_tag: str) -> bool:
        """Check if a Latin word's ending matches the required POS."""
        if pos_tag == 'UNK':
            return True  # No POS constraint, allow all

        endings = LATIN_POS_ENDINGS.get(pos_tag, [])
        if not endings:
            return True  # Unknown POS tag, no filter

        return any(word.endswith(ending) for ending in endings)

    def _get_candidates_for_token(self, voynich_token: str) -> List[str]:
        """
        Generate Latin word candidates for a Voynich token using
        the fuzzy skeletonizer and skeleton index.

        Returns list of unique Latin words across all skeleton branches.
        """
        # Strip any existing POS tag
        token = voynich_token.split('_')[0] if '_' in voynich_token else voynich_token

        # Get skeleton candidates
        skeleton_candidates = self.f_skel.get_skeleton_candidates(token)
        if not skeleton_candidates:
            return []

        all_candidates: Set[str] = set()

        for target_skel, _ in skeleton_candidates:
            # Exact match
            words = self.l_skel.skeleton_index.get(target_skel, [])
            if words:
                all_candidates.update(words)
                continue

            # Fuzzy match with dynamic threshold
            max_dist = dynamic_levenshtein_threshold(target_skel)
            if max_dist > 0:
                for l_skel in self.l_skel.skeleton_index.keys():
                    dist = Levenshtein.distance(target_skel, l_skel)
                    if dist <= max_dist:
                        all_candidates.update(self.l_skel.skeleton_index[l_skel])

        return list(all_candidates)

    def solve(
        self,
        decoded_words: List[str],
        folio_id: str = None,
    ) -> Tuple[List[str], List[Dict]]:
        """
        Attempt to resolve all bracketed tokens in a decoded word sequence.

        Args:
            decoded_words: List of decoded words (mix of Latin words and
                bracketed tokens like '[ykal_VERB_3P]')
            folio_id: Optional folio ID for humoral boosting

        Returns:
            (resolved_words, resolution_log) where:
                resolved_words: Updated word list with resolved or UNRESOLVED tags
                resolution_log: List of dicts with resolution details
        """
        # Determine humoral boost set for this folio
        humoral_boost_words: Set[str] = set()
        if folio_id and folio_id in PLANT_IDS:
            humoral_cat = PLANT_IDS[folio_id].get('humoral', '')
            if humoral_cat in self.humoral_vocab:
                humoral_boost_words = set(self.humoral_vocab[humoral_cat])

        result = list(decoded_words)
        log = []

        for i, word in enumerate(decoded_words):
            # Check if this is a bracketed token
            match = TAGGED_BRACKET_RE.match(word)
            if not match:
                continue

            # Extract token and POS tag
            if match.group(1) is not None:
                voynich_token = match.group(1)
                pos_tag = match.group(2)
                bracket_type = 'square'
            else:
                voynich_token = match.group(3)
                pos_tag = match.group(4)
                bracket_type = 'angle'

            # 1. Find non-bracketed context words
            w_prev = None
            for j in range(i - 1, -1, -1):
                if not TAGGED_BRACKET_RE.match(decoded_words[j]):
                    w_prev = decoded_words[j]
                    break

            w_next = None
            for j in range(i + 1, len(decoded_words)):
                if not TAGGED_BRACKET_RE.match(decoded_words[j]):
                    w_next = decoded_words[j]
                    break

            # 2. Generate candidates from skeleton
            candidates = self._get_candidates_for_token(voynich_token)
            if not candidates:
                tag = f"[{voynich_token}_UNRESOLVED]" if bracket_type == 'square' else f"<{voynich_token}_UNRESOLVED>"
                result[i] = tag
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'no_candidates', 'resolved': None,
                })
                continue

            # 3. POS filtering
            pos_filtered = [c for c in candidates if self._candidate_matches_pos(c, pos_tag)]
            if not pos_filtered:
                pos_filtered = candidates  # Fall back to unfiltered if POS is too strict

            # 4. Trigram probability scoring
            scored = []
            for candidate in pos_filtered:
                score = 0.0

                if w_prev is not None:
                    p_forward = self._get_transition_prob(w_prev, candidate)
                    score += p_forward

                if w_next is not None:
                    p_backward = self._get_transition_prob(candidate, w_next)
                    score += p_backward

                # 5. Humoral multiplier
                if candidate in humoral_boost_words:
                    score *= 3.0

                scored.append((score, candidate))

            # Sort descending
            scored.sort(key=lambda x: (-x[0], x[1]))

            # 6. Strict thresholding
            if not scored or scored[0][0] == 0.0:
                # No transition probability for any candidate
                tag = f"[{voynich_token}_UNRESOLVED]" if bracket_type == 'square' else f"<{voynich_token}_UNRESOLVED>"
                result[i] = tag
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'zero_probability', 'resolved': None,
                    'top_candidates': [(s, w) for s, w in scored[:3]],
                })
                continue

            best_score, best_word = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0

            if second_score > 0 and best_score / second_score < self.min_confidence_ratio:
                # Ambiguous — scores too close
                tag = f"[{voynich_token}_UNRESOLVED]" if bracket_type == 'square' else f"<{voynich_token}_UNRESOLVED>"
                result[i] = tag
                ratio = best_score / second_score if second_score > 0 else float('inf')
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'ambiguous',
                    'ratio': round(ratio, 2),
                    'top_candidates': [(round(s, 6), w) for s, w in scored[:3]],
                    'resolved': None,
                })
            else:
                # Confident resolution
                result[i] = best_word
                ratio = best_score / second_score if second_score > 0 else float('inf')
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'resolved',
                    'resolved': best_word,
                    'score': round(best_score, 6),
                    'ratio': round(ratio, 2) if ratio != float('inf') else 'inf',
                })

        return result, log

    def solve_folio(
        self,
        decoded_text: str,
        folio_id: str = None,
    ) -> Tuple[str, Dict]:
        """
        Resolve bracketed tokens in a folio's decoded text.

        Args:
            decoded_text: Scaffolded text with POS-tagged brackets
            folio_id: Folio identifier for humoral boosting

        Returns:
            (resolved_text, stats_dict)
        """
        words = decoded_text.split()
        resolved_words, log = self.solve(words, folio_id=folio_id)

        # Compute stats
        total_brackets = sum(1 for entry in log)
        resolved_count = sum(1 for entry in log if entry['status'] == 'resolved')
        unresolved_count = total_brackets - resolved_count

        stats = {
            'total_brackets': total_brackets,
            'resolved': resolved_count,
            'unresolved': unresolved_count,
            'resolution_rate': resolved_count / max(1, total_brackets),
            'confidence_scores': [
                {
                    'token': entry['token'],
                    'winner': entry.get('resolved'),
                    'ratio': entry.get('ratio'),
                    'status': entry['status'],
                }
                for entry in log
            ],
        }

        return ' '.join(resolved_words), stats
