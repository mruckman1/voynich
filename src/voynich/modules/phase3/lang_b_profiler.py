"""
Language B Profiler
====================
Extracts Language B sub-corpus and computes detailed statistical profile.
Provides LANG_B_TARGETS constants and word family classification used
by all Phase 3 attacks.

Language B has 13 word types across 227 tokens with H2=0.74 — extreme
regularity that makes it the ideal entry point for decryption.
"""

import math
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

from voynich.modules.phase2.cross_cutting import LanguageABSplitter
from voynich.core.stats import (
    full_statistical_profile, compute_all_entropy, zipf_analysis,
    bigram_transition_matrix, word_positional_entropy
)
from voynich.core.voynich_corpus import (
    get_all_tokens, SAMPLE_CORPUS, ZODIAC_LABELS, SECTIONS
)

LANG_B_TARGETS = {
    'H1': 3.491,
    'H2': 0.741,
    'H3': 0.549,
    'zipf_exponent': 1.583,
    'type_token_ratio': 0.057,
    'vocabulary_size': 13,
    'total_tokens': 227,
    'mean_word_length': 5.49,
}

LANG_B_VOCABULARY = {
    'chedy':    {'family': 'edy', 'onset': 'ch',  'body': 'edy'},
    'shedy':    {'family': 'edy', 'onset': 'sh',  'body': 'edy'},
    'otedy':    {'family': 'edy', 'onset': 'ot',  'body': 'edy'},
    'lchedy':   {'family': 'edy', 'onset': 'lch', 'body': 'edy'},
    'qokedy':   {'family': 'edy', 'onset': 'qok', 'body': 'edy'},
    'qokeedy':  {'family': 'edy', 'onset': 'qok', 'body': 'eedy'},
    'lkeedy':   {'family': 'edy', 'onset': 'lk',  'body': 'eedy'},
    'ykeedy':   {'family': 'edy', 'onset': 'yk',  'body': 'eedy'},
    'qokaiin':  {'family': 'aiin', 'onset': 'qok', 'body': 'aiin'},
    'otaiin':   {'family': 'aiin', 'onset': 'ot',  'body': 'aiin'},
    'qokain':   {'family': 'aiin', 'onset': 'qok', 'body': 'ain'},
    'otaiir':   {'family': 'aiin', 'onset': 'ot',  'body': 'aiir'},
    'qokeey':   {'family': 'residual', 'onset': 'qok', 'body': 'eey'},
}

class LanguageBProfiler:
    """
    Extracts Language B sub-corpus, computes word-level statistics,
    and classifies words into families for downstream attacks.
    """

    def __init__(self):
        self.splitter = LanguageABSplitter()
        self.lang_b_text = ''
        self.lang_b_tokens = []
        self.lang_a_text = ''
        self.word_freq = Counter()
        self.word_families = {}
        self._extracted = False

    def extract_corpus(self) -> Tuple[str, List[str]]:
        """Extract Language B text and tokens from the corpus."""
        if self._extracted:
            return self.lang_b_text, self.lang_b_tokens

        self.lang_a_text, self.lang_b_text = self.splitter.split_corpus()
        self.lang_b_tokens = self.lang_b_text.split()
        self.word_freq = Counter(self.lang_b_tokens)
        self._extracted = True
        return self.lang_b_text, self.lang_b_tokens

    def compute_profile(self) -> Dict:
        """Compute full statistical profile for Language B."""
        self.extract_corpus()
        return full_statistical_profile(self.lang_b_text, 'language_b')

    def classify_word_families(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Classify Language B vocabulary into edy, aiin, and residual families.

        Returns:
            {'edy': [('chedy', 51), ...], 'aiin': [('qokaiin', 19), ...],
             'residual': [('qokeey', 2)]}
        """
        self.extract_corpus()

        families = defaultdict(list)
        for word, count in self.word_freq.most_common():
            if word in LANG_B_VOCABULARY:
                family = LANG_B_VOCABULARY[word]['family']
            elif word.endswith(('edy', 'eedy', 'dy')):
                family = 'edy'
            elif word.endswith(('aiin', 'ain', 'aiir')):
                family = 'aiin'
            else:
                family = 'residual'
            families[family].append((word, count))

        self.word_families = dict(families)
        return self.word_families

    def compute_family_statistics(self) -> Dict:
        """
        Compute token counts and proportions for each family.
        """
        if not self.word_families:
            self.classify_word_families()

        total = len(self.lang_b_tokens)
        stats = {}
        for family, words in self.word_families.items():
            family_tokens = sum(count for _, count in words)
            stats[family] = {
                'tokens': family_tokens,
                'proportion': family_tokens / total if total > 0 else 0,
                'types': len(words),
                'words': words,
            }

        return stats

    def extract_folio_level_data(self) -> List[Dict]:
        """
        Extract per-folio Language B data with metadata.

        Returns list of dicts with folio, section, zodiac info, and
        per-family token counts.
        """
        self.extract_corpus()
        if not self.word_families:
            self.classify_word_families()

        edy_words = set(w for w, _ in self.word_families.get('edy', []))
        aiin_words = set(w for w, _ in self.word_families.get('aiin', []))

        folio_data = []
        for folio_id, data in SAMPLE_CORPUS.items():
            if data['lang'] != 'B':
                continue

            tokens = []
            for line in data['text']:
                tokens.extend(line.split())

            edy_count = sum(1 for t in tokens if t in edy_words)
            aiin_count = sum(1 for t in tokens if t in aiin_words)
            residual_count = len(tokens) - edy_count - aiin_count

            entry = {
                'folio': folio_id,
                'section': data['section'],
                'scribe': data['scribe'],
                'tokens': tokens,
                'n_tokens': len(tokens),
                'edy_count': edy_count,
                'aiin_count': aiin_count,
                'residual_count': residual_count,
                'edy_ratio': edy_count / len(tokens) if tokens else 0,
                'aiin_ratio': aiin_count / len(tokens) if tokens else 0,
            }

            if folio_id in ZODIAC_LABELS:
                zl = ZODIAC_LABELS[folio_id]
                entry['zodiac_sign'] = zl.get('zodiac_sign', '')
                entry['body_part'] = zl.get('body_part', '')
                entry['month_label'] = zl.get('month_label', '')

            folio_data.append(entry)

        return folio_data

    def compute_word_transition_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build word-level bigram transition matrix for Language B.

        Skips transitions across folio boundaries.

        Returns:
            (matrix, vocabulary_list) where matrix[i][j] = P(word_j | word_i)
        """
        self.extract_corpus()
        vocabulary = sorted(self.word_freq.keys())
        word_to_idx = {w: i for i, w in enumerate(vocabulary)}
        n = len(vocabulary)
        counts = np.zeros((n, n), dtype=float)

        for folio_id, data in SAMPLE_CORPUS.items():
            if data['lang'] != 'B':
                continue
            folio_tokens = []
            for line in data['text']:
                folio_tokens.extend(line.split())

            for i in range(len(folio_tokens) - 1):
                w1, w2 = folio_tokens[i], folio_tokens[i + 1]
                if w1 in word_to_idx and w2 in word_to_idx:
                    counts[word_to_idx[w1]][word_to_idx[w2]] += 1

        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = counts / row_sums

        return matrix, vocabulary

    def run(self, verbose: bool = True) -> Dict:
        """Run full Language B profiling. Returns results dict."""
        self.extract_corpus()

        if verbose:
            print('\n=== Language B Profiler ===')
            print(f'  Tokens: {len(self.lang_b_tokens)}')
            print(f'  Vocabulary: {len(self.word_freq)} types')

        profile = self.compute_profile()
        if verbose:
            ent = profile.get('entropy', {})
            zipf = profile.get('zipf', {})
            print(f'  H1={ent.get("H1", 0):.4f}  '
                  f'H2={ent.get("H2", 0):.4f}  '
                  f'H3={ent.get("H3", 0):.4f}')
            print(f'  Zipf={zipf.get("zipf_exponent", 0):.4f}  '
                  f'TTR={zipf.get("type_token_ratio", 0):.4f}')

        family_stats = self.compute_family_statistics()
        if verbose:
            for fam, stats in family_stats.items():
                print(f'  Family {fam}: {stats["tokens"]} tokens '
                      f'({stats["proportion"]:.1%}), {stats["types"]} types')

        folio_data = self.extract_folio_level_data()
        if verbose:
            zodiac_folios = [f for f in folio_data if 'zodiac_sign' in f]
            print(f'  Language B folios: {len(folio_data)} '
                  f'({len(zodiac_folios)} zodiac)')

        trans_matrix, vocab = self.compute_word_transition_matrix()
        n_nonzero = int(np.sum(trans_matrix > 0))
        if verbose:
            print(f'  Transition matrix: {len(vocab)}x{len(vocab)}, '
                  f'{n_nonzero} non-zero entries')

        word_table = []
        for word, count in self.word_freq.most_common():
            info = LANG_B_VOCABULARY.get(word, {})
            word_table.append({
                'word': word,
                'count': count,
                'proportion': count / len(self.lang_b_tokens),
                'family': info.get('family', 'unknown'),
                'onset': info.get('onset', ''),
                'body': info.get('body', ''),
            })

        if verbose:
            print('\n  Word Frequency Table:')
            print(f'  {"Word":<12} {"Count":>5} {"Prop":>6} {"Family":<8} {"Onset":<5} {"Body":<6}')
            print(f'  {"-"*47}')
            for w in word_table:
                print(f'  {w["word"]:<12} {w["count"]:>5} {w["proportion"]:>6.1%} '
                      f'{w["family"]:<8} {w["onset"]:<5} {w["body"]:<6}')

        return {
            'profile': profile,
            'word_families': family_stats,
            'folio_data': [
                {k: v for k, v in f.items() if k != 'tokens'}
                for f in folio_data
            ],
            'transition_matrix': trans_matrix.tolist(),
            'vocabulary': vocab,
            'n_nonzero_transitions': n_nonzero,
            'word_table': word_table,
            'targets_match': {
                'H2_within_0.01': abs(
                    profile.get('entropy', {}).get('H2', 0) - LANG_B_TARGETS['H2']
                ) < 0.01,
                'vocabulary_exact': len(self.word_freq) == LANG_B_TARGETS['vocabulary_size'],
            },
        }
