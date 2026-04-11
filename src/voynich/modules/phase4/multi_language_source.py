"""
Approach 4: Multi-Language Source Testing
==========================================
Tests Hebrew, Arabic, Spanish, Catalan, Czech as source languages
alongside Latin, Italian, and German.

For each language: generate medieval medical text, compute word-level
H2/TTR/Zipf, and compare to Language A targets. If a language matches
better than Latin, it becomes the primary candidate for the SAA attack.
"""

import math
import random
import numpy as np
from collections import Counter
from typing import Dict, List, Optional

from voynich.core.stats import (
    word_conditional_entropy, zipf_analysis,
)
from voynich.modules.phase4.lang_a_extractor import LanguageAExtractor, LANG_A_FULL_TARGETS
from voynich.modules.phase4.latin_herbal_corpus import LatinHerbalCorpus
from voynich.core.medieval_text_templates import (
    generate_italian_text, generate_german_text,
)

HEBREW_MEDICAL_VOCAB = {
    'high_freq': [
        'esev', 'shoresh', 'aleh', 'perach', 'zera', 'mayim', 'shemen',
        'avak', 'cham', 'kar', 'yavesh', 'lach', 'guf', 'dam',
        'refuah', 'segulah', 'koach', 'teva', 'choli', 'keev',
        'rosh', 'beten', 'chazeh', 'kevah', 'kaved', 'kilyah',
        'kach', 'arev', 'bashal', 'shteh', 'sim', 'rechatz',
    ],
    'medium_freq': [
        'laanah', 'chalamit', 'pegam', 'marvah', 'naanah', 'babunag',
        'rozmarin', 'ezov', 'gad', 'luach', 'leshonit',
        'petza', 'kadachat', 'shiuul', 'sam', 'dever', 'shechin',
        'mirchah', 'rikahat', 'tavshil', 'retiyah', 'ktor',
    ],
}

ARABIC_MEDICAL_VOCAB = {
    'high_freq': [
        'ushb', 'jidhr', 'waraq', 'zahr', 'bazr', 'ma', 'zayt',
        'mashuq', 'harr', 'barid', 'yabis', 'ratb', 'jism', 'damm',
        'dawa', 'ilaj', 'quwwa', 'tabi', 'marad', 'waja',
        'ras', 'batn', 'sadr', 'mida', 'kabid', 'kulya',
        'khudh', 'ukhlut', 'utbukh', 'ushrib', 'da', 'ughsil',
    ],
    'medium_freq': [
        'afsintin', 'khubbazi', 'sadhab', 'maramiyyah', 'nana', 'babunaj',
        'iklil', 'khuzama', 'shiih', 'lisaan', 'hummayd',
        'jurh', 'humma', 'sual', 'summ', 'taun', 'khuraj',
        'marham', 'sharab', 'matbukh', 'lasuq', 'bakhur',
    ],
}

SPANISH_MEDICAL_VOCAB = {
    'high_freq': [
        'hierba', 'raiz', 'hoja', 'flor', 'simiente', 'agua', 'aceite',
        'polvo', 'caliente', 'frio', 'seco', 'humedo', 'cuerpo', 'sangre',
        'medicina', 'remedio', 'virtud', 'natura', 'enfermedad', 'dolor',
        'cabeza', 'vientre', 'pecho', 'estomago', 'higado', 'rinon',
        'toma', 'mezcla', 'cuece', 'bebe', 'pon', 'lava',
    ],
    'medium_freq': [
        'artemisa', 'malva', 'ruda', 'salvia', 'menta', 'manzanilla',
        'romero', 'lavanda', 'ajenjo', 'borraja', 'llanten',
        'herida', 'fiebre', 'tos', 'veneno', 'peste', 'apostema',
        'ungento', 'jarabe', 'cocimiento', 'emplasto', 'sahumerio',
    ],
}

CATALAN_MEDICAL_VOCAB = {
    'high_freq': [
        'herba', 'arrel', 'fulla', 'flor', 'llavor', 'aigua', 'oli',
        'pols', 'calent', 'fred', 'sec', 'humit', 'cos', 'sang',
        'medicina', 'remei', 'virtut', 'natura', 'malaltia', 'dolor',
        'cap', 'ventre', 'pit', 'estomac', 'fetge', 'ronyo',
        'pren', 'barreja', 'bull', 'beu', 'posa', 'renta',
    ],
    'medium_freq': [
        'artemisa', 'malva', 'ruda', 'salvia', 'menta', 'camamilla',
        'romani', 'espigol', 'donzell', 'borraina', 'plantatge',
        'ferida', 'febre', 'tos', 'veri', 'pesta', 'apostema',
        'unguent', 'xarop', 'decoccio', 'emplastre', 'fumigacio',
    ],
}

CZECH_MEDICAL_VOCAB = {
    'high_freq': [
        'bylina', 'koren', 'list', 'kvet', 'semeno', 'voda', 'olej',
        'prasek', 'teply', 'studeny', 'suchy', 'vlhky', 'telo', 'krev',
        'lek', 'pomoc', 'sila', 'priroda', 'nemoc', 'bolest',
        'hlava', 'bricho', 'prsa', 'zaludek', 'jatra', 'ledvina',
        'vezmi', 'smichej', 'var', 'pij', 'poloz', 'umyj',
    ],
    'medium_freq': [
        'pelynek', 'slez', 'routa', 'salvej', 'mata', 'heřmanek',
        'rozmaryn', 'levandule', 'blen', 'brutnák', 'jitrocel',
        'rana', 'horecka', 'kasel', 'jed', 'mor', 'vred',
        'mast', 'sirup', 'odvar', 'naplastr', 'vykurovani',
    ],
}

SOURCE_LANGUAGE_VOCABS = {
    'latin': None,
    'italian': None,
    'german': None,
    'hebrew': HEBREW_MEDICAL_VOCAB,
    'arabic': ARABIC_MEDICAL_VOCAB,
    'spanish': SPANISH_MEDICAL_VOCAB,
    'catalan': CATALAN_MEDICAL_VOCAB,
    'czech': CZECH_MEDICAL_VOCAB,
}

def _generate_from_vocab(vocab: Dict, n_words: int = 500,
                          seed: int = 42) -> str:
    """Generate synthetic medical text from a vocabulary dict."""
    rng = random.Random(seed)
    high = vocab['high_freq']
    med = vocab['medium_freq']
    words = []
    for _ in range(n_words):
        if rng.random() < 0.6:
            words.append(rng.choice(high))
        else:
            words.append(rng.choice(med))
    return ' '.join(words)

class MultiLanguageSourceTest:
    """
    Test multiple source languages as candidates for Language A's
    plaintext language.

    For each language: compute word-level H2, TTR, Zipf exponent
    and compare to Language A targets.
    """

    def __init__(self, extractor: LanguageAExtractor,
                 latin_corpus: LatinHerbalCorpus):
        self.extractor = extractor
        self.latin_corpus = latin_corpus

    def generate_source_text(self, language: str, n_words: int = 500) -> str:
        """Generate or load medieval medical text for a given language."""
        if language == 'latin':
            return self.latin_corpus.get_corpus()
        elif language == 'italian':
            return generate_italian_text(n_words=n_words)
        elif language == 'german':
            return generate_german_text(n_words=n_words)
        elif language in SOURCE_LANGUAGE_VOCABS:
            vocab = SOURCE_LANGUAGE_VOCABS[language]
            if vocab is not None:
                return _generate_from_vocab(vocab, n_words=n_words)
        return ''

    def compute_word_level_metrics(self, text: str) -> Dict:
        """Compute word-level H2, TTR, and Zipf exponent."""
        tokens = text.split()
        if len(tokens) < 10:
            return {'error': 'Text too short'}

        h2_word = word_conditional_entropy(tokens, order=1)
        zipf = zipf_analysis(tokens)

        return {
            'H2_word': h2_word,
            'TTR': zipf['type_token_ratio'],
            'zipf_exponent': zipf['zipf_exponent'],
            'vocabulary_size': zipf['vocabulary_size'],
            'total_tokens': len(tokens),
        }

    def compare_to_lang_a(self, metrics: Dict) -> Dict:
        """Compute distance between a language's metrics and Language A targets."""
        profile = self.extractor.compute_full_profile()

        target_h2 = profile['entropy']['H2']
        target_ttr = profile['zipf']['type_token_ratio']
        target_zipf = profile['zipf']['zipf_exponent']

        if 'error' in metrics:
            return {'error': metrics['error'], 'distance': float('inf')}

        h2_delta = abs(metrics['H2_word'] - target_h2)
        ttr_delta = abs(metrics['TTR'] - target_ttr)
        zipf_delta = abs(metrics['zipf_exponent'] - target_zipf)

        distance = math.sqrt(
            3.0 * h2_delta ** 2 +
            1.5 * ttr_delta ** 2 +
            2.0 * zipf_delta ** 2
        )

        return {
            'H2_delta': h2_delta,
            'TTR_delta': ttr_delta,
            'zipf_delta': zipf_delta,
            'distance': distance,
            'h2_within_tolerance': h2_delta < 0.2,
        }

    def rank_source_languages(self) -> Dict:
        """
        Rank all candidate source languages by compatibility with
        Language A targets.
        """
        results = {}

        for language in SOURCE_LANGUAGE_VOCABS:
            text = self.generate_source_text(language)
            if not text:
                results[language] = {'error': 'No text generated'}
                continue

            metrics = self.compute_word_level_metrics(text)
            comparison = self.compare_to_lang_a(metrics)

            results[language] = {
                'metrics': metrics,
                'comparison': comparison,
            }

        ranked = sorted(
            [(lang, data) for lang, data in results.items()
             if 'error' not in data.get('comparison', {})],
            key=lambda x: x[1]['comparison']['distance']
        )

        return {
            'results': results,
            'ranking': [(lang, data['comparison']['distance'])
                        for lang, data in ranked],
            'best_match': ranked[0][0] if ranked else 'unknown',
            'best_distance': ranked[0][1]['comparison']['distance'] if ranked else float('inf'),
        }

    def _synthesize(self, ranking_result: Dict) -> Dict:
        """Combine results into synthesis."""
        ranking = ranking_result.get('ranking', [])
        best = ranking_result.get('best_match', 'unknown')
        best_dist = ranking_result.get('best_distance', float('inf'))

        best_data = ranking_result['results'].get(best, {})
        h2_match = best_data.get('comparison', {}).get('h2_within_tolerance', False)

        return {
            'best_source_language': best,
            'best_distance': best_dist,
            'h2_within_tolerance': h2_match,
            'ranking': ranking[:5],
            'conclusion': (
                f'Best source language: {best} (distance={best_dist:.3f}). '
                f'H2 within tolerance: {h2_match}. '
                f'Top 3: {", ".join(l for l, _ in ranking[:3])}.'
            ),
        }

    def run(self, verbose: bool = True) -> Dict:
        """Run multi-language source testing."""
        ranking_result = self.rank_source_languages()
        synthesis = self._synthesize(ranking_result)

        results = {
            'languages_tested': list(SOURCE_LANGUAGE_VOCABS.keys()),
            'n_languages': len(SOURCE_LANGUAGE_VOCABS),
            'language_results': {},
            'synthesis': synthesis,
        }

        for lang, data in ranking_result['results'].items():
            if 'error' in data:
                results['language_results'][lang] = {'error': data['error']}
            else:
                results['language_results'][lang] = {
                    'H2_word': data['metrics'].get('H2_word'),
                    'TTR': data['metrics'].get('TTR'),
                    'zipf_exponent': data['metrics'].get('zipf_exponent'),
                    'distance': data['comparison'].get('distance'),
                    'h2_within_tolerance': data['comparison'].get('h2_within_tolerance'),
                }

        if verbose:
            print(f'\n  Approach 4: Multi-Language Source Testing')
            print(f'    Languages tested: {len(SOURCE_LANGUAGE_VOCABS)}')
            for lang, dist in ranking_result.get('ranking', []):
                lang_data = ranking_result['results'][lang]
                h2 = lang_data['metrics'].get('H2_word', 0)
                print(f'    {lang:12s}: H2_word={h2:.3f}, distance={dist:.3f}')
            print(f'    --- Synthesis ---')
            print(f'    {synthesis["conclusion"]}')

        return results
