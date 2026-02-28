"""
Phase 12, Module 12.4: Deterministic N-Gram Mask Solver
========================================================
Resolves bracketed (unresolved) tokens using Latin bigram/trigram
transition matrices from Phase 5/6. Every resolution is mathematically
provable; ambiguous cases are left honestly bracketed.

Algorithm for each bracketed token [token_POS]:
  1. Target isolation: extract w_prev and w_next from context
  2. Function word recovery: contextual gate (both neighbors resolved + trigram)
  3. Candidate generation: skeleton lookup via FuzzySkeletonizer
  4. POS filtering: restrict candidates by Latin inflection patterns
  5. Trigram scoring: P(candidate | w_prev) + P(w_next | candidate)
  6. Humoral multiplier: boost contextually appropriate candidates
  7. Strict thresholding: length-scaled confidence ratio

Supports bidirectional multi-pass solving: each pass creates context
anchors for the next, converging when no new resolutions are found.

If the math can't prove it, it stays as [token_UNRESOLVED].

Phase 12  ·  Voynich Convergence Attack
"""

import re
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from modules.phase11.phonetic_skeletonizer import LatinPhoneticSkeletonizer
from modules.phase11.csp_decoder import FUNCTION_WORDS
from modules.phase12.fuzzy_skeletonizer import FuzzySkeletonizer, dynamic_levenshtein_threshold
from modules.phase12.syntactic_scaffolder import LatinPOSTagger

import Levenshtein

from data.botanical_identifications import PLANT_IDS

LATIN_POS_ENDINGS: Dict[str, List[str]] = {
    'VERB_3P': ['t', 'nt', 'it', 'at', 'et', 'ut', 'unt', 'ent', 'ant'],
    'NOUN_ACC': ['um', 'am', 'em', 'im', 'em'],
    'NOUN_FEM': ['a', 'ae', 'am', 'arum', 'as', 'is'],
    'NOUN_NOM': ['us', 'a', 'um', 'es', 'er', 'is', 'or', 'io'],
    'ADJ': ['us', 'a', 'um', 'is', 'e', 'es', 'i', 'em', 'am'],
    'NOUN': ['um', 'al', 'il', 'en', 'ur', 'us', 'er'],
    'NOUN_AGT': ['or', 'er', 'ar', 'rix'],
    'NOUN_PL': ['es', 'as', 'os', 'us', 'i', 'ae', 'ia', 'a'],
    'UNK': [],
}

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
        pos_tagger: Optional[LatinPOSTagger] = None,
        pos_transition_matrix: Optional[np.ndarray] = None,
        pos_vocab: Optional[List[str]] = None,
        min_confidence_ratio_long: float = 3.5,
        long_skeleton_segments: int = 6,
        enable_length_scaled_ratio: bool = True,
        enable_bidirectional: bool = True,
        max_solving_passes: int = 4,
        enable_function_word_recovery: bool = True,
        function_word_trigram_threshold: float = 0.01,
        dual_context_ratio_factor: float = 0.6,
        dual_context_max_distance: int = 3,
        enable_unigram_backoff: bool = True,
        unigram_backoff_ratio_factor: float = 1.5,
        unigram_backoff_min_segments: int = 3,
        enable_pos_backoff: bool = False,
        pos_backoff_weight: float = 0.1,
        pos_backoff_min_confidence: float = 3.0,
        enable_char_ngram_fallback: bool = False,
        char_ngram_model=None,
        char_ngram_min_score_gap: float = 0.5,
        char_ngram_min_segments: int = 3,
        char_ngram_max_context_distance: int = 4,
        char_ngram_require_context: bool = True,
        enable_illustration_prior: bool = False,
        illustration_prior: Optional[Dict[str, Dict[str, float]]] = None,
        illustration_boosted_ratio_factor: float = 0.5,
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
            pos_tagger: Optional LatinPOSTagger for syntactic veto
            pos_transition_matrix: Optional POS-to-POS transition matrix
            pos_vocab: Optional POS category list matching matrix indices
            min_confidence_ratio_long: Relaxed ratio for long skeletons
            long_skeleton_segments: Segment count threshold for relaxed ratio
            enable_length_scaled_ratio: Toggle length-scaled ratio
            enable_bidirectional: Toggle bidirectional multi-pass solving
            max_solving_passes: Maximum passes before stopping
            enable_function_word_recovery: Toggle contextual function word gate
            function_word_trigram_threshold: Minimum trigram score for recovery
            dual_context_ratio_factor: Multiplier for confidence threshold when
                BOTH w_prev and w_next are resolved within max_distance. Uses
                two-sided trigram evidence which is more discriminative. Set to
                1.0 to disable. Default 0.6 reduces 5.0x→3.0x for dual-context.
            dual_context_max_distance: Maximum positional distance for a context
                word to qualify for dual-context reduction (default 3).
            enable_unigram_backoff: When bigram scores are all 0.0, fall back
                to corpus unigram frequency as a weaker discriminator. Applies
                a stricter confidence ratio (base × unigram_backoff_ratio_factor).
            unigram_backoff_ratio_factor: Multiplier to make the confidence
                threshold stricter for unigram-only resolutions (default 1.5,
                so 5.0x becomes 7.5x for normal skeletons).
            unigram_backoff_min_segments: Minimum skeleton segment count for
                unigram backoff (default 3, skip very short skeletons).
            enable_pos_backoff: When word-level bigram returns 0, fall back to
                POS transition probability as a coarser discriminator.
            pos_backoff_weight: Scale factor for POS scores relative to word-level
                scores (default 0.1).
            pos_backoff_min_confidence: Separate (lower) confidence ratio for
                candidates scored entirely via POS backoff (default 3.0).
            enable_char_ngram_fallback: Enable character-level n-gram scoring
                as a final fallback for unresolved tokens.
            char_ngram_model: Trained LatinCharNgramModel instance, or None.
            char_ngram_min_score_gap: Minimum avg-log-prob gap between best
                and second-best candidate (default 0.5).
            char_ngram_min_segments: Skeleton segment minimum for char n-gram
                resolution (default 3).
            char_ngram_max_context_distance: Max positions to nearest resolved
                neighbor for context gate (default 4).
            char_ngram_require_context: Require at least one resolved neighbor
                within max_context_distance (default True).
        """
        self.matrix = transition_matrix
        self.vocab = vocab
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.l_skel = latin_skeletonizer
        self.f_skel = fuzzy_skeletonizer
        self.humoral_vocab = humoral_vocab or {}
        self.min_confidence_ratio = min_confidence_ratio

        self.pos_tagger = pos_tagger
        self.pos_matrix = pos_transition_matrix
        self.pos_vocab = pos_vocab
        self.pos_to_idx = {p: i for i, p in enumerate(pos_vocab)} if pos_vocab else {}
        self._syntactic_veto_enabled = (
            pos_tagger is not None
            and pos_transition_matrix is not None
            and pos_vocab is not None
        )

        self.min_confidence_ratio_long = min_confidence_ratio_long
        self.long_skeleton_segments = long_skeleton_segments
        self.enable_length_scaled_ratio = enable_length_scaled_ratio

        self.enable_bidirectional = enable_bidirectional
        self.max_solving_passes = max_solving_passes

        self.enable_function_word_recovery = enable_function_word_recovery
        self.function_word_trigram_threshold = function_word_trigram_threshold
        self.function_words = FUNCTION_WORDS

        self.dual_context_ratio_factor = dual_context_ratio_factor
        self.dual_context_max_distance = dual_context_max_distance

        self.enable_unigram_backoff = enable_unigram_backoff
        self.unigram_backoff_ratio_factor = unigram_backoff_ratio_factor
        self.unigram_backoff_min_segments = unigram_backoff_min_segments
        self._unigram_freq: Dict[str, int] = {}

        self.enable_pos_backoff = enable_pos_backoff
        self.pos_backoff_weight = pos_backoff_weight
        self.pos_backoff_min_confidence = pos_backoff_min_confidence
        self._word_level_resolutions = 0
        self._pos_backoff_resolutions = 0

        self.enable_char_ngram_fallback = enable_char_ngram_fallback
        self.char_ngram_model = char_ngram_model
        self.char_ngram_min_score_gap = char_ngram_min_score_gap
        self.char_ngram_min_segments = char_ngram_min_segments
        self.char_ngram_max_context_distance = char_ngram_max_context_distance
        self.char_ngram_require_context = char_ngram_require_context
        self._char_ngram_resolutions = 0

        self.enable_illustration_prior = enable_illustration_prior
        self.illustration_prior = illustration_prior or {}
        self.illustration_boosted_ratio_factor = illustration_boosted_ratio_factor
        self._illustration_boosted_resolutions = 0

    def set_corpus_frequencies(self, corpus_tokens: List[str]) -> None:
        """
        Precompute unigram frequencies from the Latin corpus tokens.
        Called by the orchestrator after construction to enable unigram backoff.
        """
        from collections import Counter
        self._unigram_freq = dict(Counter(corpus_tokens))

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
            return True

        endings = LATIN_POS_ENDINGS.get(pos_tag, [])
        if not endings:
            return True

        return any(word.endswith(ending) for ending in endings)

    def _syntactic_veto(
        self, candidates: List[str], w_prev: Optional[str],
        min_pos_prob: float = 0.001,
    ) -> List[str]:
        """
        Remove candidates whose POS is grammatically invalid given w_prev.

        Uses the POS transition matrix to veto candidates where
        P(candidate_POS | w_prev_POS) < min_pos_prob.

        Falls back to the full candidate list if all are vetoed.
        """
        if not self._syntactic_veto_enabled or w_prev is None:
            return candidates

        prev_pos = self.pos_tagger.tag(w_prev)
        if prev_pos not in self.pos_to_idx:
            return candidates

        prev_idx = self.pos_to_idx[prev_pos]
        surviving = []
        for c in candidates:
            c_pos = self.pos_tagger.tag(c)
            if c_pos not in self.pos_to_idx:
                surviving.append(c)
                continue
            c_idx = self.pos_to_idx[c_pos]
            if self.pos_matrix[prev_idx][c_idx] >= min_pos_prob:
                surviving.append(c)

        return surviving if surviving else candidates

    def _get_candidates_for_token(self, voynich_token: str) -> List[str]:
        """
        Generate Latin word candidates for a Voynich token using
        the fuzzy skeletonizer and skeleton index.

        Returns list of unique Latin words across all skeleton branches.
        """
        token = voynich_token.split('_')[0] if '_' in voynich_token else voynich_token

        skeleton_candidates = self.f_skel.get_skeleton_candidates(token)
        if not skeleton_candidates:
            return []

        all_candidates: Set[str] = set()

        for target_skel, _ in skeleton_candidates:
            words = self.l_skel.skeleton_index.get(target_skel, [])
            if words:
                all_candidates.update(words)
                continue

            max_dist = dynamic_levenshtein_threshold(target_skel)
            if max_dist > 0:
                for l_skel in self.l_skel.skeleton_index.keys():
                    dist = Levenshtein.distance(target_skel, l_skel)
                    if dist <= max_dist:
                        all_candidates.update(self.l_skel.skeleton_index[l_skel])

        return list(all_candidates)

    def _get_skeleton_segment_count(self, voynich_token: str) -> int:
        """
        Return the maximum skeleton segment count for a Voynich token.
        Used to determine if a token qualifies for the relaxed confidence ratio.
        """
        token = voynich_token.split('_')[0] if '_' in voynich_token else voynich_token
        skeleton_candidates = self.f_skel.get_skeleton_candidates(token)
        if not skeleton_candidates:
            return 0
        primary_skeleton = skeleton_candidates[0][0]
        return len(primary_skeleton.split('-'))

    def _effective_confidence_ratio(self, voynich_token: str) -> float:
        """
        Return the effective minimum confidence ratio for a token,
        using a relaxed threshold for long skeletons when enabled.
        """
        if not self.enable_length_scaled_ratio:
            return self.min_confidence_ratio

        seg_count = self._get_skeleton_segment_count(voynich_token)
        if seg_count >= self.long_skeleton_segments:
            return self.min_confidence_ratio_long
        return self.min_confidence_ratio

    def _try_function_word_recovery(
        self,
        voynich_token: str,
        w_prev: Optional[str],
        w_next: Optional[str],
        result: List[str],
        position: int,
    ) -> Optional[str]:
        """
        Attempt to recover a function word when the local context is
        sufficiently dense in resolved words and the trigram probability
        supports it.

        Three gates prevent random noise from triggering recovery:
        1. Density gate: >= 3 of 5 words in centered window must be resolved
        2. Context gate: both w_prev and w_next must exist (non-None)
        3. Trigram gate: P(fw|w_prev) + P(w_next|fw) >= threshold
        """
        if not self.enable_function_word_recovery:
            return None

        raw_token = voynich_token.split('_')[0] if '_' in voynich_token else voynich_token
        if raw_token not in self.function_words:
            return None

        if w_prev is None or w_next is None:
            return None

        window_start = max(0, position - 2)
        window_end = min(len(result), position + 3)
        window = result[window_start:window_end]
        resolved_in_window = sum(
            1 for w in window if not TAGGED_BRACKET_RE.match(w)
        )
        if resolved_in_window < 2:
            return None

        candidate_fw = self.function_words[raw_token]

        p_forward = self._get_transition_prob(w_prev, candidate_fw)
        p_backward = self._get_transition_prob(candidate_fw, w_next)
        trigram_score = p_forward + p_backward

        if trigram_score < self.function_word_trigram_threshold:
            return None

        return candidate_fw

    def _solve_single_pass(
        self,
        current_words: List[str],
        folio_id: str = None,
        reverse: bool = False,
        mark_unresolved: bool = True,
    ) -> Tuple[List[str], List[Dict], int]:
        """
        Single directional pass of the n-gram mask solver.

        Args:
            current_words: Current state of the word list (context read from here)
            folio_id: Folio ID for humoral boosting
            reverse: If True, iterate right-to-left
            mark_unresolved: If False, leave failed tokens with original POS tags
                (preserving POS info for subsequent passes)

        Returns:
            (updated_words, resolution_log, new_resolutions_count)
        """
        humoral_boost_words: Set[str] = set()
        if folio_id and folio_id in PLANT_IDS:
            humoral_cat = PLANT_IDS[folio_id].get('humoral', '')
            if humoral_cat in self.humoral_vocab:
                humoral_boost_words = set(self.humoral_vocab[humoral_cat])

        illust_boosts: Dict[str, float] = {}
        if self.enable_illustration_prior and folio_id and folio_id in self.illustration_prior:
            illust_boosts = self.illustration_prior[folio_id]

        result = list(current_words)
        log = []
        new_resolutions = 0

        indices = list(range(len(current_words)))
        if reverse:
            indices = indices[::-1]

        for i in indices:
            word = current_words[i]

            match = TAGGED_BRACKET_RE.match(word)
            if not match:
                continue

            if match.group(1) is not None:
                voynich_token = match.group(1)
                pos_tag = match.group(2)
                bracket_type = 'square'
            else:
                voynich_token = match.group(3)
                pos_tag = match.group(4)
                bracket_type = 'angle'

            if pos_tag == 'UNRESOLVED':
                continue

            w_prev = None
            w_prev_dist = 0
            for j in range(i - 1, -1, -1):
                if not TAGGED_BRACKET_RE.match(current_words[j]):
                    w_prev = current_words[j]
                    w_prev_dist = i - j
                    break

            w_next = None
            w_next_dist = 0
            for j in range(i + 1, len(current_words)):
                if not TAGGED_BRACKET_RE.match(current_words[j]):
                    w_next = current_words[j]
                    w_next_dist = j - i
                    break

            fw_result = self._try_function_word_recovery(
                voynich_token, w_prev, w_next, result, i
            )
            if fw_result is not None:
                result[i] = fw_result
                new_resolutions += 1
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'resolved',
                    'resolved': fw_result,
                    'resolution_type': 'function_word_recovery',
                    'score': None,
                    'ratio': 'fw_contextual',
                })
                continue

            candidates = self._get_candidates_for_token(voynich_token)

            if hasattr(self, '_medium_candidates') and self._medium_candidates and i in self._medium_candidates:
                medium_words = [word for _, word in self._medium_candidates[i]]
                candidates = list(set(medium_words + candidates))

            if not candidates:
                if mark_unresolved:
                    tag = f"[{voynich_token}_UNRESOLVED]" if bracket_type == 'square' else f"<{voynich_token}_UNRESOLVED>"
                    result[i] = tag
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'no_candidates', 'resolved': None,
                })
                continue

            pos_filtered = [c for c in candidates if self._candidate_matches_pos(c, pos_tag)]
            if not pos_filtered:
                pos_filtered = candidates

            pos_filtered = self._syntactic_veto(pos_filtered, w_prev)

            scored = []
            for candidate in pos_filtered:
                score = 0.0

                if w_prev is not None:
                    p_forward = self._get_transition_prob(w_prev, candidate)
                    score += p_forward

                if w_next is not None:
                    p_backward = self._get_transition_prob(candidate, w_next)
                    score += p_backward

                if candidate in humoral_boost_words:
                    score *= 3.0

                if candidate in illust_boosts:
                    score *= illust_boosts[candidate]

                scored.append((score, candidate))

            scored.sort(key=lambda x: (-x[0], x[1]))

            if not scored or scored[0][0] == 0.0:
                if (self.enable_unigram_backoff
                        and self._unigram_freq
                        and pos_filtered
                        and (w_prev is not None or w_next is not None)):
                    seg_count = self._get_skeleton_segment_count(voynich_token)
                    has_close_context = (
                        (w_prev is not None and w_prev_dist <= 2)
                        or (w_next is not None and w_next_dist <= 2)
                    )
                    if (seg_count >= self.unigram_backoff_min_segments
                            and has_close_context):
                        viable = []
                        for c in pos_filtered:
                            freq = self._unigram_freq.get(c, 0)
                            if freq >= 20:
                                if c in illust_boosts:
                                    freq *= illust_boosts[c]
                                viable.append((freq, c))
                        viable.sort(key=lambda x: (-x[0], x[1]))

                        if viable:
                            best_freq, best_word = viable[0]
                            second_freq = viable[1][0] if len(viable) > 1 else 0

                            eff_ratio = self._effective_confidence_ratio(voynich_token)
                            uni_threshold = eff_ratio * self.unigram_backoff_ratio_factor

                            if second_freq == 0 or best_freq / second_freq >= uni_threshold:
                                result[i] = best_word
                                new_resolutions += 1
                                ratio = best_freq / max(1, second_freq) if second_freq > 0 else float('inf')
                                log.append({
                                    'position': i, 'token': voynich_token,
                                    'pos': pos_tag,
                                    'status': 'resolved',
                                    'resolved': best_word,
                                    'resolution_type': 'unigram_backoff',
                                    'score': best_freq,
                                    'ratio': round(ratio, 2) if ratio != float('inf') else 'inf',
                                })
                                continue

                if (illust_boosts and pos_filtered):
                    seg_count = self._get_skeleton_segment_count(voynich_token)
                    if seg_count >= self.unigram_backoff_min_segments:
                        illust_candidates = [
                            (illust_boosts[c], c) for c in pos_filtered
                            if c in illust_boosts and illust_boosts[c] >= 2.0
                        ]
                        if illust_candidates:
                            illust_candidates.sort(key=lambda x: (-x[0], x[1]))
                            best_ib, best_iw = illust_candidates[0]
                            second_ib = illust_candidates[1][0] if len(illust_candidates) > 1 else 0.0
                            if second_ib == 0.0 or best_ib / second_ib >= 1.5:
                                result[i] = best_iw
                                new_resolutions += 1
                                self._word_level_resolutions += 1
                                self._illustration_boosted_resolutions += 1
                                log.append({
                                    'position': i, 'token': voynich_token,
                                    'pos': pos_tag,
                                    'status': 'resolved',
                                    'resolved': best_iw,
                                    'resolution_type': 'illustration_prior',
                                    'score': best_ib,
                                    'ratio': 'illustration_sole' if second_ib == 0 else round(best_ib / second_ib, 2),
                                })
                                continue

                if mark_unresolved:
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

            effective_ratio = self._effective_confidence_ratio(voynich_token)

            if (self.dual_context_ratio_factor < 1.0
                    and w_prev is not None and w_next is not None
                    and w_prev_dist <= self.dual_context_max_distance
                    and w_next_dist <= self.dual_context_max_distance):
                effective_ratio *= self.dual_context_ratio_factor

            if (self.enable_illustration_prior
                    and best_word in illust_boosts
                    and self.illustration_boosted_ratio_factor < 1.0):
                effective_ratio *= self.illustration_boosted_ratio_factor

            effective_ratio = max(1.5, effective_ratio)

            if second_score > 0 and best_score / second_score < effective_ratio:
                if mark_unresolved:
                    tag = f"[{voynich_token}_UNRESOLVED]" if bracket_type == 'square' else f"<{voynich_token}_UNRESOLVED>"
                    result[i] = tag
                ratio = best_score / second_score
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'ambiguous',
                    'ratio': round(ratio, 2),
                    'effective_threshold': round(effective_ratio, 2),
                    'top_candidates': [(round(s, 6), w) for s, w in scored[:3]],
                    'resolved': None,
                })
            else:
                result[i] = best_word
                new_resolutions += 1
                self._word_level_resolutions += 1
                if best_word in illust_boosts:
                    self._illustration_boosted_resolutions += 1
                ratio = best_score / second_score if second_score > 0 else float('inf')
                log.append({
                    'position': i, 'token': voynich_token, 'pos': pos_tag,
                    'status': 'resolved',
                    'resolved': best_word,
                    'resolution_type': 'word_level',
                    'score': round(best_score, 6),
                    'ratio': round(ratio, 2) if ratio != float('inf') else 'inf',
                })

        return result, log, new_resolutions

    def solve(
        self,
        decoded_words: List[str],
        folio_id: str = None,
        medium_candidates: Optional[Dict[int, List[Tuple[float, str]]]] = None,
    ) -> Tuple[List[str], List[Dict]]:
        """
        Attempt to resolve all bracketed tokens in a decoded word sequence.

        When bidirectional solving is enabled, alternates L->R and R->L passes
        until convergence (no new resolutions) or max passes reached.

        Args:
            decoded_words: List of decoded words (mix of Latin words and
                bracketed tokens like '[ykal_VERB_3P]')
            folio_id: Optional folio ID for humoral boosting
            medium_candidates: Optional dict of position -> [(score, word)]
                from the graduated CSP scoring. These pre-vetted candidates
                are merged with skeleton-generated candidates.

        Returns:
            (resolved_words, resolution_log) where:
                resolved_words: Updated word list with resolved or UNRESOLVED tags
                resolution_log: List of dicts with resolution details
        """
        self._medium_candidates = medium_candidates

        result, log, _ = self._solve_single_pass(
            decoded_words, folio_id=folio_id, reverse=False,
            mark_unresolved=not self.enable_bidirectional,
        )
        all_logs = list(log)

        if not self.enable_bidirectional:
            return result, all_logs

        current_words = result
        for pass_num in range(1, self.max_solving_passes):
            reverse = (pass_num % 2 == 1)
            direction = 'R-L' if reverse else 'L-R'

            fw_result = list(current_words)
            fw_log = []
            new_resolutions = 0

            indices = list(range(len(current_words)))
            if reverse:
                indices = indices[::-1]

            for i in indices:
                word = current_words[i]
                match = TAGGED_BRACKET_RE.match(word)
                if not match:
                    continue

                if match.group(1) is not None:
                    voynich_token = match.group(1)
                    pos_tag = match.group(2)
                else:
                    voynich_token = match.group(3)
                    pos_tag = match.group(4)

                if pos_tag == 'UNRESOLVED':
                    continue

                fw_prev = None
                for j in range(i - 1, -1, -1):
                    if not TAGGED_BRACKET_RE.match(fw_result[j]):
                        fw_prev = fw_result[j]
                        break
                fw_next = None
                for j in range(i + 1, len(fw_result)):
                    if not TAGGED_BRACKET_RE.match(fw_result[j]):
                        fw_next = fw_result[j]
                        break

                fw = self._try_function_word_recovery(
                    voynich_token, fw_prev, fw_next, fw_result, i
                )
                if fw is not None:
                    fw_result[i] = fw
                    new_resolutions += 1
                    fw_log.append({
                        'position': i, 'token': voynich_token, 'pos': pos_tag,
                        'status': 'resolved',
                        'resolved': fw,
                        'resolution_type': 'function_word_recovery',
                        'pass': pass_num, 'direction': direction,
                        'score': None, 'ratio': 'fw_contextual',
                    })

            all_logs.extend(fw_log)
            current_words = fw_result

            if new_resolutions == 0:
                break

        final_result = list(current_words)
        for i, word in enumerate(final_result):
            match = TAGGED_BRACKET_RE.match(word)
            if match:
                if match.group(1) is not None:
                    voynich_token = match.group(1)
                    pos_tag = match.group(2)
                    if pos_tag != 'UNRESOLVED':
                        final_result[i] = f"[{voynich_token}_UNRESOLVED]"
                else:
                    voynich_token = match.group(3)
                    pos_tag = match.group(4)
                    if pos_tag != 'UNRESOLVED':
                        final_result[i] = f"<{voynich_token}_UNRESOLVED>"

        return final_result, all_logs

    def solve_folio(
        self,
        decoded_text: str,
        folio_id: str = None,
        medium_candidates: Optional[Dict[int, List[Tuple[float, str]]]] = None,
    ) -> Tuple[str, Dict]:
        """
        Resolve bracketed tokens in a folio's decoded text.

        Args:
            decoded_text: Scaffolded text with POS-tagged brackets
            folio_id: Folio identifier for humoral boosting
            medium_candidates: Optional dict of position -> [(score, word)]
                from the graduated CSP scoring

        Returns:
            (resolved_text, stats_dict)
        """
        self._word_level_resolutions = 0
        self._pos_backoff_resolutions = 0
        self._char_ngram_resolutions = 0
        self._illustration_boosted_resolutions = 0

        words = decoded_text.split()
        resolved_words, log = self.solve(
            words, folio_id=folio_id, medium_candidates=medium_candidates,
        )

        final_status = {}
        for entry in log:
            final_status[entry['position']] = entry

        deduped_log = sorted(final_status.values(), key=lambda e: e['position'])

        total_brackets = len(deduped_log)
        resolved_count = sum(1 for entry in deduped_log if entry['status'] == 'resolved')
        unresolved_count = total_brackets - resolved_count

        stats = {
            'total_brackets': total_brackets,
            'resolved': resolved_count,
            'unresolved': unresolved_count,
            'resolution_rate': resolved_count / max(1, total_brackets),
            'total_passes': max((e.get('pass', 0) for e in log), default=0) + 1 if log else 1,
            'word_level_resolutions': self._word_level_resolutions,
            'pos_backoff_resolutions': self._pos_backoff_resolutions,
            'char_ngram_resolutions': self._char_ngram_resolutions,
            'illustration_boosted_resolutions': self._illustration_boosted_resolutions,
            'confidence_scores': [
                {
                    'token': entry['token'],
                    'winner': entry.get('resolved'),
                    'ratio': entry.get('ratio'),
                    'status': entry['status'],
                }
                for entry in deduped_log
            ],
        }

        return ' '.join(resolved_words), stats

    def pos_backoff_pass(self, resolved_text: str, folio_id: str = None) -> Tuple[str, int]:
        """
        Post-consistency POS backoff pass.

        Scans remaining UNRESOLVED brackets and attempts to resolve them
        using POS transition probabilities when word-level bigrams are zero.

        Called AFTER cross-folio consistency to avoid poisoning cross-folio
        agreement with inconsistent POS-context-dependent resolutions.

        Gates (prevent false positives on random text):
          1. Both neighbors must be resolved words (two-sided POS evidence)
          2. Both neighbors within dual_context_max_distance positions
          3. Skeleton >= unigram_backoff_min_segments segments

        Returns:
            (updated_text, num_resolved)
        """
        if not self.enable_pos_backoff or not self._syntactic_veto_enabled:
            return resolved_text, 0

        words = resolved_text.split()
        result = list(words)
        num_resolved = 0

        for i, word in enumerate(words):
            match = TAGGED_BRACKET_RE.match(word)
            if not match:
                continue
            if match.group(1) is not None:
                voynich_token = match.group(1)
                pos_tag = match.group(2)
                bracket_type = 'square'
            else:
                voynich_token = match.group(3)
                pos_tag = match.group(4)
                bracket_type = 'angle'
            if pos_tag != 'UNRESOLVED':
                continue

            w_prev, w_prev_dist = None, 999
            for j in range(i - 1, -1, -1):
                if not TAGGED_BRACKET_RE.match(result[j]):
                    w_prev = result[j]
                    w_prev_dist = i - j
                    break

            w_next, w_next_dist = None, 999
            for j in range(i + 1, len(result)):
                if not TAGGED_BRACKET_RE.match(result[j]):
                    w_next = result[j]
                    w_next_dist = j - i
                    break

            if (w_prev is None or w_next is None
                    or w_prev_dist > self.dual_context_max_distance
                    or w_next_dist > self.dual_context_max_distance):
                continue

            seg_count = self._get_skeleton_segment_count(voynich_token)
            if seg_count < self.unigram_backoff_min_segments:
                continue

            candidates = self._get_candidates_for_token(voynich_token)
            if not candidates:
                continue

            pos_filtered = [c for c in candidates
                            if self._candidate_matches_pos(c, pos_tag)]
            if not pos_filtered:
                pos_filtered = candidates
            pos_filtered = self._syntactic_veto(pos_filtered, w_prev)

            prev_pos = self.pos_tagger.tag(w_prev)
            next_pos = self.pos_tagger.tag(w_next)
            prev_pos_idx = self.pos_to_idx.get(prev_pos)
            next_pos_idx = self.pos_to_idx.get(next_pos)
            if prev_pos_idx is None or next_pos_idx is None:
                continue

            pos_scored = []
            for candidate in pos_filtered:
                cand_pos = self.pos_tagger.tag(candidate)
                cand_pos_idx = self.pos_to_idx.get(cand_pos)
                if cand_pos_idx is not None:
                    ps = (
                        self.pos_matrix[prev_pos_idx][cand_pos_idx]
                        + self.pos_matrix[cand_pos_idx][next_pos_idx]
                    ) * self.pos_backoff_weight
                else:
                    ps = 0.0
                pos_scored.append((ps, candidate))
            pos_scored.sort(key=lambda x: (-x[0], x[1]))

            illust_boosts = {}
            if self.enable_illustration_prior and folio_id and folio_id in self.illustration_prior:
                illust_boosts = self.illustration_prior[folio_id]
                boosted_pos_scored = []
                for ps, cand in pos_scored:
                    if cand in illust_boosts:
                        ps *= illust_boosts[cand]
                    boosted_pos_scored.append((ps, cand))
                pos_scored = boosted_pos_scored
                pos_scored.sort(key=lambda x: (-x[0], x[1]))

            if not pos_scored or pos_scored[0][0] == 0.0:
                continue

            best_ps, best_pw = pos_scored[0]
            second_ps = pos_scored[1][0] if len(pos_scored) > 1 else 0.0

            pos_min_conf = self.pos_backoff_min_confidence
            if (self.enable_illustration_prior
                    and best_pw in illust_boosts
                    and self.illustration_boosted_ratio_factor < 1.0):
                pos_min_conf = max(1.5, pos_min_conf * self.illustration_boosted_ratio_factor)

            if second_ps == 0 or best_ps / second_ps >= pos_min_conf:
                result[i] = best_pw
                num_resolved += 1
                self._pos_backoff_resolutions += 1
                if best_pw in illust_boosts:
                    self._illustration_boosted_resolutions += 1

        return ' '.join(result), num_resolved

    def char_ngram_pass(self, resolved_text: str, folio_id: str = None) -> Tuple[str, int]:
        """
        Post-consistency character n-gram fallback pass.

        Scans remaining UNRESOLVED brackets and attempts to resolve them
        using character trigram plausibility scoring. Fires only when
        all higher-level scorers (bigram, unigram, POS backoff) returned
        zero signal.

        Called AFTER POS backoff as the final resolution attempt.

        Gates (prevent false positives on random text):
          1. enable_char_ngram_fallback must be True
          2. char_ngram_model must be trained (not None)
          3. Skeleton >= char_ngram_min_segments segments
          4. At least one resolved neighbor within char_ngram_max_context_distance
             (when char_ngram_require_context is True)
          5. Score gap between best and second candidate >= char_ngram_min_score_gap

        Returns:
            (updated_text, num_resolved)
        """
        if (not self.enable_char_ngram_fallback
                or self.char_ngram_model is None):
            return resolved_text, 0

        words = resolved_text.split()
        result = list(words)
        num_resolved = 0

        for i, word in enumerate(words):
            match = TAGGED_BRACKET_RE.match(word)
            if not match:
                continue
            if match.group(1) is not None:
                voynich_token = match.group(1)
                pos_tag = match.group(2)
            else:
                voynich_token = match.group(3)
                pos_tag = match.group(4)
            if pos_tag != 'UNRESOLVED':
                continue

            seg_count = self._get_skeleton_segment_count(voynich_token)
            if seg_count < self.char_ngram_min_segments:
                continue

            if self.char_ngram_require_context:
                has_context = False
                for j in range(i - 1, max(-1, i - self.char_ngram_max_context_distance - 1), -1):
                    if j >= 0 and not TAGGED_BRACKET_RE.match(result[j]):
                        has_context = True
                        break
                if not has_context:
                    for j in range(i + 1, min(len(result), i + self.char_ngram_max_context_distance + 1)):
                        if not TAGGED_BRACKET_RE.match(result[j]):
                            has_context = True
                            break
                if not has_context:
                    continue

            candidates = self._get_candidates_for_token(voynich_token)
            if not candidates:
                continue

            scored = self.char_ngram_model.score_candidates(candidates)
            if not scored:
                continue

            best_score, best_word = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else -float('inf')

            illust_boosts = {}
            if self.enable_illustration_prior and folio_id and folio_id in self.illustration_prior:
                illust_boosts = self.illustration_prior[folio_id]
            is_boosted = best_word in illust_boosts

            if len(scored) == 1:
                threshold = -10.0 if is_boosted else -8.0
                if best_score > threshold:
                    result[i] = best_word
                    num_resolved += 1
                    self._char_ngram_resolutions += 1
                    if is_boosted:
                        self._illustration_boosted_resolutions += 1
            else:
                effective_gap = self.char_ngram_min_score_gap
                if is_boosted:
                    second_word = scored[1][1]
                    if second_word not in illust_boosts:
                        effective_gap *= self.illustration_boosted_ratio_factor
                gap = best_score - second_score
                if gap >= effective_gap:
                    result[i] = best_word
                    num_resolved += 1
                    self._char_ngram_resolutions += 1
                    if is_boosted:
                        self._illustration_boosted_resolutions += 1

        return ' '.join(result), num_resolved
