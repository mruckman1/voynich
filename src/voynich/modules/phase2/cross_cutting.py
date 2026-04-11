"""
Cross-Cutting Investigations
===============================
Three investigations that apply to all six Phase 2 models and exploit
specific anomalies from the Phase 1 pipeline results.

A) Language A/B Inversion — run all models separately on each language
B) qo- Word Predictions — test each model's prediction about qo- words
C) Information-Theoretic Reverse Engineering — derive minimum FSM
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Optional

from voynich.core.stats import (
    full_statistical_profile, profile_distance, compute_all_entropy,
    first_order_entropy, conditional_entropy
)
from voynich.modules.phase2.base_model import Phase2GenerativeModel, VOYNICH_TARGETS
from voynich.core.voynich_corpus import (
    get_all_tokens, get_section_text, SECTIONS, SAMPLE_CORPUS
)

class LanguageABSplitter:
    """
    Split Voynich corpus by Currier language and run models separately.

    Language A (herbal_a, Scribe 1) and Language B (herbal_b, astronomical,
    biological, pharmaceutical, recipes, Scribes 2-5) have structurally
    inverted properties: glyphs 'd' and 'l' swap positional roles,
    FSA state counts differ dramatically (46 vs 22).
    """

    def split_corpus(self) -> Tuple[str, str]:
        """
        Split the Voynich corpus into Language A and Language B text.

        Returns:
            (lang_a_text, lang_b_text)
        """
        lang_a_sections = [name for name, info in SECTIONS.items()
                          if info.get('currier_lang') == 'A']
        lang_b_sections = [name for name, info in SECTIONS.items()
                          if info.get('currier_lang') == 'B']

        lang_a_text = ' '.join(
            get_section_text(s) for s in lang_a_sections
            if get_section_text(s)
        )
        lang_b_text = ' '.join(
            get_section_text(s) for s in lang_b_sections
            if get_section_text(s)
        )

        return lang_a_text.strip(), lang_b_text.strip()

    def compute_split_profiles(self) -> Dict:
        """Compute statistical profiles for Language A and B separately."""
        lang_a, lang_b = self.split_corpus()

        profile_a = full_statistical_profile(lang_a, 'language_a') if lang_a else {}
        profile_b = full_statistical_profile(lang_b, 'language_b') if lang_b else {}

        comparison = {}
        if profile_a and profile_b:
            comparison = {
                'distance': profile_distance(profile_a, profile_b),
                'H2_diff': (profile_a.get('entropy', {}).get('H2', 0) -
                          profile_b.get('entropy', {}).get('H2', 0)),
                'TTR_diff': (profile_a.get('zipf', {}).get('type_token_ratio', 0) -
                           profile_b.get('zipf', {}).get('type_token_ratio', 0)),
                'word_length_diff': (profile_a.get('mean_word_length', 0) -
                                   profile_b.get('mean_word_length', 0)),
            }

        return {
            'language_a': {
                'profile': profile_a,
                'n_tokens': len(lang_a.split()) if lang_a else 0,
                'sections': [n for n, i in SECTIONS.items() if i.get('currier_lang') == 'A'],
            },
            'language_b': {
                'profile': profile_b,
                'n_tokens': len(lang_b.split()) if lang_b else 0,
                'sections': [n for n, i in SECTIONS.items() if i.get('currier_lang') == 'B'],
            },
            'comparison': comparison,
        }

    def run_model_split(self, model_class, model_params: Dict,
                        plaintext: str = '', verbose: bool = False) -> Dict:
        """
        Run a Phase 2 model separately on Language A and Language B.

        Parameters:
            model_class: The Phase2GenerativeModel subclass
            model_params: Parameters for model initialization
            plaintext: Source plaintext for generation
            verbose: Print details

        Returns:
            {lang_a: {profile, score}, lang_b: {profile, score},
             comparison: {distance, ...}}
        """
        lang_a_text, lang_b_text = self.split_corpus()
        results = {}

        for lang_name, lang_text in [('lang_a', lang_a_text), ('lang_b', lang_b_text)]:
            model = model_class(**model_params)

            generated = model.generate(plaintext=plaintext, n_words=300)
            if not generated:
                results[lang_name] = {'error': 'No output generated'}
                continue

            gen_profile = model.get_profile(generated)
            lang_profile = full_statistical_profile(lang_text, lang_name)

            score = model.quick_score(gen_profile)
            dist = profile_distance(gen_profile, lang_profile)

            results[lang_name] = {
                'generated_profile': gen_profile.get('entropy', {}),
                'target_profile': lang_profile.get('entropy', {}),
                'distance_to_target': dist,
                'quick_score': score,
            }

            if verbose:
                print(f'  {lang_name}: distance={dist:.4f}, H2={score.get("H2", 0):.4f}')

        dist_a = results.get('lang_a', {}).get('distance_to_target', float('inf'))
        dist_b = results.get('lang_b', {}).get('distance_to_target', float('inf'))
        results['comparison'] = {
            'better_match': 'lang_a' if dist_a < dist_b else 'lang_b',
            'distance_ratio': dist_a / max(dist_b, 0.001),
        }

        return results

class QoPredictionTester:
    """
    Test each model's specific prediction about qo- words.

    The pipeline found qo- words cluster at paragraph ENDS (48.9% in Q4),
    not beginnings. This contradicts the article/demonstrative hypothesis
    and provides discriminating tests for each model.
    """

    def __init__(self):
        self.qo_words = []
        self.all_tokens = []
        self._extract_qo_data()

    def _extract_qo_data(self):
        """Extract qo- word data from the corpus."""
        self.all_tokens = get_all_tokens()
        self.qo_words = [t for t in self.all_tokens if t.startswith('qo')]

    def verbose_cipher_prediction(self) -> Dict:
        """
        Under verbose cipher: qo-words encode a specific high-frequency letter.
        Test: do all qo-words map to the same 1-2 plaintext letters?

        If true, the qo- words should have a frequency consistent with
        those letters in the source language.
        """
        qo_freq = len(self.qo_words) / max(len(self.all_tokens), 1)

        best_letter_match = None
        best_diff = float('inf')
        from voynich.modules.phase2.verbose_cipher import LATIN_LETTER_FREQ
        for letter, freq in LATIN_LETTER_FREQ.items():
            diff = abs(qo_freq - freq)
            if diff < best_diff:
                best_diff = diff
                best_letter_match = letter

        qo_suffixes = Counter(w[2:] for w in self.qo_words if len(w) > 2)
        n_unique_suffixes = len(qo_suffixes)

        return {
            'prediction': 'qo-words encode 1-2 high-frequency letters as homophones',
            'qo_frequency': qo_freq,
            'best_letter_match': best_letter_match,
            'letter_frequency': LATIN_LETTER_FREQ.get(best_letter_match, 0),
            'frequency_diff': best_diff,
            'n_unique_suffixes': n_unique_suffixes,
            'top_suffixes': qo_suffixes.most_common(10),
            'consistent': best_diff < 0.05 and n_unique_suffixes <= 10,
        }

    def syllabary_prediction(self) -> Dict:
        """
        Under syllabary: qo-words encode a common Latin syllable.
        Candidate: "que" (enclitic, appears at word ends = end-biased).

        Test: qo- frequency should match a top-frequency syllable.
        """
        qo_freq = len(self.qo_words) / max(len(self.all_tokens), 1)

        from voynich.core.latin_syllables import LATIN_SYLLABLES
        best_syl = None
        best_diff = float('inf')
        for syl, freq in LATIN_SYLLABLES.items():
            diff = abs(qo_freq - freq)
            if diff < best_diff:
                best_diff = diff
                best_syl = syl

        que_freq = LATIN_SYLLABLES.get('que', 0)
        que_match = abs(qo_freq - que_freq) < 0.03

        return {
            'prediction': 'qo-words encode common Latin syllable (possibly enclitic "que")',
            'qo_frequency': qo_freq,
            'best_syllable_match': best_syl,
            'best_syllable_freq': LATIN_SYLLABLES.get(best_syl, 0),
            'que_hypothesis': {
                'que_frequency': que_freq,
                'frequency_match': que_match,
                'end_bias_consistent': True,
            },
            'consistent': que_match,
        }

    def slot_machine_prediction(self) -> Dict:
        """
        Under slot machine: "qo" occupies the first slot (prefix slot value).
        Test: qo-words should have the same root/suffix distribution as
        non-qo words (slots are independent).
        """
        qo_suffixes = Counter(w[2:] for w in self.qo_words if len(w) > 2)
        non_qo = [t for t in self.all_tokens if not t.startswith('qo') and len(t) > 2]
        non_qo_suffixes = Counter(t[2:] for t in non_qo)

        qo_total = sum(qo_suffixes.values())
        non_qo_total = sum(non_qo_suffixes.values())

        if qo_total == 0 or non_qo_total == 0:
            return {
                'prediction': '"qo" is a prefix slot value; suffix distribution independent',
                'consistent': False,
                'error': 'Insufficient data',
            }

        all_suffixes = set(qo_suffixes.keys()) | set(non_qo_suffixes.keys())
        p = [(qo_suffixes.get(s, 0) + 1e-10) / (qo_total + len(all_suffixes) * 1e-10)
             for s in all_suffixes]
        q = [(non_qo_suffixes.get(s, 0) + 1e-10) / (non_qo_total + len(all_suffixes) * 1e-10)
             for s in all_suffixes]

        import numpy as np
        p_arr = np.array(p)
        q_arr = np.array(q)
        m = 0.5 * (p_arr + q_arr)
        js = 0.5 * np.sum(p_arr * np.log2(p_arr / m)) + 0.5 * np.sum(q_arr * np.log2(q_arr / m))

        return {
            'prediction': '"qo" is a prefix slot value; suffix distribution independent',
            'suffix_js_divergence': float(js),
            'distributions_similar': js < 0.5,
            'n_qo_suffixes': len(qo_suffixes),
            'n_non_qo_suffixes': len(non_qo_suffixes),
            'consistent': js < 0.5,
        }

    def run_all_predictions(self, verbose: bool = False) -> Dict:
        """Run all model predictions about qo- words."""
        results = {
            'verbose_cipher': self.verbose_cipher_prediction(),
            'syllabary_code': self.syllabary_prediction(),
            'slot_machine': self.slot_machine_prediction(),
        }

        if verbose:
            for model, result in results.items():
                status = 'CONSISTENT' if result.get('consistent', False) else 'INCONSISTENT'
                print(f'  {model:25s}: {status} — {result.get("prediction", "")}')

        return results

class MinimalFSMDerivation:
    """
    Derive the minimum finite-state machine from the 17 constraints.

    Given H2 = 1.41: each word has 2^1.41 ≈ 2.66 effective choices.
    Given TTR = 0.164: ~50-100 active words carry most of the text.
    Given Zipf = 1.24: steep power law in frequency distribution.

    Together: a system with ~50-100 states and ~150 transitions.
    The minimum FSM size gives the Kolmogorov complexity upper bound.
    """

    def derive_from_constraints(self) -> Dict:
        """
        Derive the minimum FSM that could produce Voynich-like text.

        Returns:
            {estimated_states, estimated_transitions, effective_choices,
             information_per_word, total_information_bits,
             equivalent_latin_words, kolmogorov_bound_kb}
        """
        h2 = VOYNICH_TARGETS['H2']
        ttr = VOYNICH_TARGETS['type_token_ratio']
        zipf_exp = VOYNICH_TARGETS['zipf_exponent']

        effective_choices = 2 ** h2

        total_tokens = 36238
        vocab_size = int(total_tokens * ttr)

        alpha = zipf_exp
        if alpha != 1.0:
            core_vocab = int((0.8 ** (1.0 / max(1.0 - alpha, 0.01))) * vocab_size)
        else:
            core_vocab = int(vocab_size * 0.2)
        core_vocab = max(10, min(core_vocab, 200))

        estimated_states = core_vocab
        estimated_transitions = int(estimated_states * effective_choices)

        information_per_word = h2
        total_info_bits = information_per_word * total_tokens

        bits_per_latin_word = 22.5
        equivalent_latin_words = int(total_info_bits / bits_per_latin_word)

        kolmogorov_bound_kb = total_info_bits / (8 * 1024)

        return {
            'effective_choices_per_position': round(effective_choices, 2),
            'estimated_vocabulary': vocab_size,
            'core_vocabulary': core_vocab,
            'estimated_states': estimated_states,
            'estimated_transitions': estimated_transitions,
            'information_per_word_bits': round(information_per_word, 4),
            'total_information_bits': int(total_info_bits),
            'equivalent_latin_words': equivalent_latin_words,
            'kolmogorov_bound_kb': round(kolmogorov_bound_kb, 2),
            'interpretation': (
                f'The Voynich encodes at most {total_info_bits:.0f} bits '
                f'≈ {kolmogorov_bound_kb:.1f} KB. '
                f'A medieval herbal of {equivalent_latin_words} Latin words '
                f'would be {"plausible" if 2000 <= equivalent_latin_words <= 10000 else "implausible"} '
                f'for a manuscript of this size.'
            ),
        }

    def compare_to_models(self, model_outputs: Dict[str, Dict]) -> Dict:
        """
        Compare the minimum FSM derivation to each model's effective complexity.

        Parameters:
            model_outputs: {model_name: {profile, text, ...}}

        Returns:
            {model_name: {model_effective_states, ratio_to_minimum, compatible}}
        """
        min_fsm = self.derive_from_constraints()
        min_states = min_fsm['estimated_states']

        comparisons = {}
        for model_name, output in model_outputs.items():
            profile = output.get('profile', {})
            vocab_size = profile.get('zipf', {}).get('vocabulary_size', 0)
            h2 = profile.get('entropy', {}).get('H2', 0)

            model_effective_states = vocab_size if vocab_size > 0 else min_states
            ratio = model_effective_states / max(min_states, 1)

            comparisons[model_name] = {
                'model_effective_states': model_effective_states,
                'minimum_states': min_states,
                'ratio': round(ratio, 2),
                'model_H2': round(h2, 4),
                'compatible': 0.3 <= ratio <= 3.0,
            }

        return comparisons

def run_cross_cutting(model_outputs: Optional[Dict] = None,
                      verbose: bool = True) -> Dict:
    """
    Run all three cross-cutting investigations.

    Parameters:
        model_outputs: {model_name: {profile: Dict, text: str}}
            Required for Investigation C. Optional for A and B.
        verbose: Print progress

    Returns:
        {language_ab: {...}, qo_predictions: {...}, minimal_fsm: {...}}
    """
    results = {}

    if verbose:
        print('\n--- Investigation A: Language A/B Inversion ---')
    splitter = LanguageABSplitter()
    results['language_ab'] = splitter.compute_split_profiles()
    if verbose:
        comp = results['language_ab'].get('comparison', {})
        print(f'  A/B distance: {comp.get("distance", "N/A")}')
        print(f'  H2 diff: {comp.get("H2_diff", "N/A")}')

    if verbose:
        print('\n--- Investigation B: qo- Word Predictions ---')
    qo_tester = QoPredictionTester()
    results['qo_predictions'] = qo_tester.run_all_predictions(verbose=verbose)

    if verbose:
        print('\n--- Investigation C: Minimal FSM Derivation ---')
    fsm = MinimalFSMDerivation()
    results['minimal_fsm'] = fsm.derive_from_constraints()
    if verbose:
        info = results['minimal_fsm']
        print(f'  Effective choices per position: {info["effective_choices_per_position"]}')
        print(f'  Core vocabulary: {info["core_vocabulary"]} words')
        print(f'  Total information: {info["total_information_bits"]} bits '
              f'({info["kolmogorov_bound_kb"]} KB)')
        print(f'  Equivalent Latin text: ~{info["equivalent_latin_words"]} words')

    if model_outputs:
        results['model_comparisons'] = fsm.compare_to_models(model_outputs)

    return results
