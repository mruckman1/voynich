"""
Semantic Coherence Analyzer for Phase 14.

Measures whether decoded folio text forms coherent medieval medical
recipes or is semantically random.

Four metrics:
  1. Medical vocabulary rate — fraction of decoded words in medical fields
  2. Semantic field concentration — Shannon entropy of field distribution
  3. Recipe template matching — sequences matching known recipe patterns
  4. Collocational coherence — adjacent word-pair field plausibility

Each metric is compared against null distributions (shuffle, random vocab,
cross-folio swap) to assess statistical significance.

Phase 14  ·  Voynich Convergence Attack
"""

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from voynich.core.semantic_fields import (
    SEMANTIC_FIELDS, MEDICAL_FIELDS, MEDICAL_VOCABULARY,
    WORD_TO_FIELDS, get_fields, get_primary_field,
)


# ── Recipe template patterns ────────────────────────────────────────
# Ordered field sequences that commonly appear in medieval medical texts.
# Matching is done on the compressed field sequence (brackets removed).

RECIPE_TEMPLATES = [
    # Classic: take plant, prepare with medium, apply for indication
    ['PREPARATION', 'PLANT', 'MEDIUM'],
    ['PLANT', 'PREPARATION', 'MEDIUM'],
    ['PREPARATION', 'PLANT', 'MEDIUM', 'APPLICATION'],
    ['PREPARATION', 'PLANT', 'MEDIUM', 'INDICATION'],

    # Abbreviated: plant + preparation + indication
    ['PLANT', 'PREPARATION', 'INDICATION'],
    ['PREPARATION', 'PLANT', 'INDICATION'],

    # Humoral description: plant + quality + quality
    ['PLANT', 'HUMORAL', 'HUMORAL'],
    ['PLANT', 'HUMORAL', 'HUMORAL', 'HUMORAL'],

    # Application: prepare + apply + body part
    ['PREPARATION', 'APPLICATION', 'BODY_PART'],
    ['APPLICATION', 'BODY_PART'],
    ['PREPARATION', 'MEDIUM', 'APPLICATION'],

    # Dosage: take + amount + plant
    ['PREPARATION', 'DOSAGE', 'PLANT'],
    ['DOSAGE', 'PLANT', 'PREPARATION'],
    ['DOSAGE', 'PLANT', 'MEDIUM'],

    # Indication clause: connective + indication
    ['CONNECTIVE', 'INDICATION'],
    ['CONNECTIVE', 'INDICATION', 'BODY_PART'],

    # Compound: mix plant with plant in medium
    ['PREPARATION', 'PLANT', 'CONNECTIVE', 'PLANT'],
    ['PLANT', 'CONNECTIVE', 'PLANT', 'MEDIUM'],

    # Property description
    ['PLANT', 'PROPERTY'],
    ['PLANT', 'CONNECTIVE', 'PROPERTY'],
    ['PROPERTY', 'CONNECTIVE', 'INDICATION'],

    # Ingredient combinations
    ['PLANT', 'CONNECTIVE', 'INGREDIENT'],
    ['PREPARATION', 'PLANT', 'CONNECTIVE', 'INGREDIENT'],

    # Medium + body target
    ['MEDIUM', 'APPLICATION', 'BODY_PART'],
    ['MEDIUM', 'PLANT', 'APPLICATION'],
]

# ── Collocation plausibility ────────────────────────────────────────
# Field pairs that commonly co-occur in medieval medical recipes.

PLAUSIBLE_COLLOCATIONS = {
    ('PREPARATION', 'PLANT'),
    ('PLANT', 'PREPARATION'),
    ('PREPARATION', 'MEDIUM'),
    ('MEDIUM', 'PREPARATION'),
    ('CONNECTIVE', 'MEDIUM'),
    ('CONNECTIVE', 'INDICATION'),
    ('CONNECTIVE', 'PLANT'),
    ('CONNECTIVE', 'BODY_PART'),
    ('CONNECTIVE', 'PROPERTY'),
    ('APPLICATION', 'BODY_PART'),
    ('APPLICATION', 'MEDIUM'),
    ('HUMORAL', 'HUMORAL'),
    ('PLANT', 'HUMORAL'),
    ('HUMORAL', 'PLANT'),
    ('DOSAGE', 'PLANT'),
    ('DOSAGE', 'MEDIUM'),
    ('MEDIUM', 'PLANT'),
    ('PLANT', 'CONNECTIVE'),
    ('BODY_PART', 'INDICATION'),
    ('INDICATION', 'BODY_PART'),
    ('PREPARATION', 'CONNECTIVE'),
    ('APPLICATION', 'CONNECTIVE'),
    ('CONNECTIVE', 'PREPARATION'),
    ('CONNECTIVE', 'APPLICATION'),
    ('PLANT', 'PROPERTY'),
    ('PROPERTY', 'PLANT'),
    ('PROPERTY', 'CONNECTIVE'),
    ('PLANT', 'MEDIUM'),
    ('PLANT', 'INGREDIENT'),
    ('INGREDIENT', 'MEDIUM'),
    ('INGREDIENT', 'CONNECTIVE'),
    ('PREPARATION', 'INGREDIENT'),
    ('MEDIUM', 'CONNECTIVE'),
    ('MEDIUM', 'APPLICATION'),
    ('TEMPORAL', 'CONNECTIVE'),
    ('CONNECTIVE', 'TEMPORAL'),
    ('DOSAGE', 'CONNECTIVE'),
    ('CONNECTIVE', 'DOSAGE'),
    ('PREPARATION', 'APPLICATION'),
    ('INDICATION', 'CONNECTIVE'),
}

IMPLAUSIBLE_COLLOCATIONS = {
    ('DOSAGE', 'DOSAGE'),
    ('APPLICATION', 'APPLICATION'),
    ('TEMPORAL', 'TEMPORAL'),
}

# Minimum resolved words for a folio to be analyzed
MIN_RESOLVED_WORDS = 5


# Non-medical Latin words for random baseline.  When sampling for the
# medical_rate null distribution, we draw 50/50 from medical vocab and
# this list, giving a ~50% medical baseline rate.
_GENERAL_LATIN_WORDS = [
    # Religious (Vulgate)
    'deus', 'dominus', 'filius', 'ecclesia', 'baptismus',
    'evangelium', 'apostolus', 'propheta', 'sacerdos', 'episcopus',
    'peccatum', 'gratia', 'fides', 'caritas', 'spes', 'resurrectio',
    'crux', 'salvator', 'angelus', 'diabolus', 'caelum', 'infernus',
    'paradisus', 'anima', 'oratio', 'psalmus', 'sacrificium',
    'testamentum', 'verbum', 'amen', 'alleluia', 'pater', 'mater',
    'christus', 'sanctus', 'beatus', 'martyr', 'confessor',
    'virgo', 'monachus', 'abbas', 'prior', 'claustrum',
    'missa', 'communio', 'confessio', 'paenitentia', 'absolutio',
    'benedictio', 'exorcismus', 'sacramentum', 'mysterium',
    # Legal
    'lex', 'ius', 'iudex', 'iudicium', 'testimonium',
    'crimen', 'poena', 'reus', 'advocatus', 'sententia', 'edictum',
    'rescriptum', 'constitutio', 'servitus', 'libertas', 'dominium',
    'possessio', 'hereditas', 'contractus', 'obligatio', 'delictum',
    'actio', 'exceptio', 'appellatio', 'magistratus', 'consul',
    'praetor', 'senator', 'tribunus', 'imperator', 'provincia',
    'decretum', 'canon', 'privilegium', 'immunitas', 'feudum',
    'census', 'tributum', 'vectigal', 'usura', 'mutuum',
    # Philosophy / theology
    'philosophia', 'veritas', 'ratio', 'intellectus', 'voluntas',
    'animus', 'cognitio', 'scientia', 'sapientia', 'opinio',
    'argumentum', 'conclusio', 'praemissa', 'definitio', 'genus',
    'differentia', 'accidens', 'universale', 'particulare',
    'necessarium', 'possibile', 'contingens', 'causa', 'effectus',
    'potentia', 'actus', 'motus', 'spatium', 'infinitum',
    'essentia', 'existentia', 'substantia', 'attributum',
    'praedicamentum', 'categoria', 'propositio', 'syllogismus',
    # Everyday / agriculture / architecture
    'domus', 'villa', 'ager', 'campus', 'silva', 'flumen',
    'mons', 'via', 'porta', 'murus', 'turris', 'urbs', 'civitas',
    'populus', 'exercitus', 'bellum', 'pax', 'gladius', 'scutum',
    'equus', 'canis', 'bos', 'ovis', 'gallina', 'piscis',
    'panis', 'vestis', 'pecunia', 'pretium', 'merx',
    'navis', 'portus', 'iter', 'nox', 'annus', 'mensis',
    'rex', 'regina', 'miles', 'servus', 'liber', 'homo',
    'femina', 'puer', 'puella', 'frater', 'soror', 'amicus',
    'fenestra', 'tectum', 'columna', 'arcus', 'fornix',
    'atrium', 'cubiculum', 'hortus', 'fons', 'puteus',
    'faber', 'artifex', 'mercator', 'agricola', 'pastor',
    'piscator', 'textor', 'sutor', 'carpentarius',
    'ferrum', 'aurum', 'argentum', 'lignum', 'lapis',
    'lana', 'linum', 'seta', 'corium', 'pellis',
    'triticum', 'hordeum', 'avena', 'vitis', 'oliva',
    'ficus', 'pomum', 'nux', 'pirum', 'malum',
    'cervus', 'lupus', 'ursus', 'vulpes', 'lepus',
    # Astronomy / cosmology (non-medical)
    'stella', 'planeta', 'orbis', 'sphaera', 'eclipsis',
    'aequinoctium', 'solstitium', 'zodiacus', 'cometa',
    'firmamentum', 'polus', 'circulus', 'epicyclus',
    'caelestis', 'mundanus', 'terrestris', 'sublunaris',
    # Grammar / education
    'grammatica', 'rhetorica', 'dialectica', 'arithmetica',
    'geometria', 'musica', 'astronomia', 'trivium', 'quadrivium',
    'magister', 'discipulus', 'lectio', 'disputatio', 'quaestio',
    'littera', 'syllaba', 'nomen', 'pronomen', 'adverbium',
    'coniunctio', 'praepositio', 'participium', 'declinatio',
    'coniugatio', 'constructio', 'figurae', 'tropus', 'metaphora',
    # Abstract / administrative
    'regnum', 'potestas', 'auctoritas', 'dignitas', 'honor',
    'gloria', 'fama', 'nobilitas', 'virtus', 'fortitudo',
    'prudentia', 'iustitia', 'temperantia', 'moderatio',
    'concordia', 'discordia', 'rebellio', 'seditio', 'foedus',
    'legatio', 'nuntius', 'epistola', 'diploma', 'sigillum',
]


class SemanticCoherenceAnalyzer:
    """Analyze semantic coherence of Phase 12 decoded text."""

    def __init__(self, translations, per_folio_stats):
        """
        Args:
            translations: {folio_id: decoded_text_string} from Phase 12
            per_folio_stats: {folio_id: {language, section, ...}}
        """
        self.translations = translations
        self.per_folio_stats = per_folio_stats
        # Medical vocabulary list (for entropy/concentration baseline)
        self._medical_vocab_list = sorted(WORD_TO_FIELDS.keys())
        # Non-medical vocabulary list (for medical_rate baseline)
        self._nonmedical_vocab_list = list(_GENERAL_LATIN_WORDS)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_resolved(text):
        """Extract resolved (non-bracketed) words from decoded text."""
        return [w for w in text.split()
                if not w.startswith('[') and not w.startswith('<')]

    @staticmethod
    def _extract_word_positions(text):
        """Return list of (position, word, is_resolved) tuples."""
        result = []
        for i, w in enumerate(text.split()):
            resolved = not w.startswith('[') and not w.startswith('<')
            result.append((i, w, resolved))
        return result

    # ── Metric 1: Medical Vocabulary Rate ───────────────────────────

    def medical_vocabulary_rate(self, decoded_text):
        """Fraction of resolved words belonging to any medical field.

        Returns dict with medical_rate, field_breakdown, totals,
        or None if too few resolved words.
        """
        words = self._extract_resolved(decoded_text)
        if len(words) < MIN_RESOLVED_WORDS:
            return None

        field_counts = Counter()
        medical_count = 0
        for word in words:
            fields = get_fields(word)
            if 'UNKNOWN' not in fields:
                # Count as medical if any field is in MEDICAL_FIELDS
                if fields & MEDICAL_FIELDS:
                    medical_count += 1
            for f in fields:
                field_counts[f] += 1

        total = len(words)
        return {
            'medical_rate': medical_count / total,
            'field_breakdown': {f: c / total for f, c in field_counts.most_common()},
            'total_resolved': total,
            'medical_words': medical_count,
            'unknown_words': field_counts.get('UNKNOWN', 0),
        }

    # ── Metric 2: Semantic Field Concentration (Entropy) ────────────

    def field_concentration(self, decoded_text):
        """Shannon entropy of semantic field distribution.

        Low entropy = words concentrated in few fields (coherent).
        High entropy = uniform across fields (random).
        Normalized to 0–1 scale.
        """
        words = self._extract_resolved(decoded_text)
        if len(words) < MIN_RESOLVED_WORDS:
            return None

        field_counts = Counter()
        for word in words:
            primary = get_primary_field(word)
            if primary != 'UNKNOWN':
                field_counts[primary] += 1

        total = sum(field_counts.values())
        if total == 0:
            return None

        entropy = 0.0
        for count in field_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        max_entropy = math.log2(len(SEMANTIC_FIELDS))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        return {
            'entropy': round(entropy, 4),
            'normalized_entropy': round(normalized, 4),
            'dominant_fields': field_counts.most_common(5),
            'classified_words': total,
        }

    # ── Metric 3: Recipe Template Matching ──────────────────────────

    def template_matching(self, decoded_text, max_gap=2):
        """Check if resolved-word field sequences match recipe templates.

        Uses subsequence matching with gaps: template elements can be
        separated by up to `max_gap` non-matching classified words in
        the compressed field sequence. Only templates of length >= 3
        are used (2-field templates match too easily by chance).

        Works on compressed field sequence (brackets removed) but
        tracks original positions for coverage calculation.
        """
        words = decoded_text.split()

        # Build compressed field sequence from resolved words only
        field_seq = []
        for i, word in enumerate(words):
            if not word.startswith('[') and not word.startswith('<'):
                primary = get_primary_field(word)
                if primary != 'UNKNOWN':
                    field_seq.append((i, word, primary))

        if len(field_seq) < 3:
            return {'matches': [], 'match_count': 0, 'coverage': 0.0,
                    'total_classified': len(field_seq)}

        # Only use templates of length >= 3 for statistical validity
        templates_3plus = [t for t in RECIPE_TEMPLATES if len(t) >= 3]

        matches = []
        for template in templates_3plus:
            # Subsequence matching with gap tolerance
            for start in range(len(field_seq)):
                matched_indices = self._match_template_with_gaps(
                    field_seq, start, template, max_gap)
                if matched_indices is not None:
                    matched = [(field_seq[j][0], field_seq[j][1],
                                field_seq[j][2]) for j in matched_indices]
                    matches.append({
                        'template': template,
                        'position': field_seq[start][0],
                        'words': [(w, f) for _, w, f in matched],
                        'indices': matched_indices,
                    })

        # Coverage: fraction of classified words in at least one match
        matched_positions = set()
        for m in matches:
            matched_positions.update(m['indices'])

        coverage = len(matched_positions) / len(field_seq) if field_seq else 0

        # Count templates by type
        template_counts = Counter()
        for m in matches:
            template_counts[tuple(m['template'])] += 1

        return {
            'match_count': len(matches),
            'coverage': round(coverage, 4),
            'total_classified': len(field_seq),
            'top_templates': [
                {'template': list(t), 'count': c}
                for t, c in template_counts.most_common(10)
            ],
        }

    @staticmethod
    def _match_template_with_gaps(field_seq, start, template, max_gap):
        """Try to match a template starting at position `start` in field_seq.

        Each template element must match within max_gap positions of the
        previous match. Returns list of matched indices or None.
        """
        if start >= len(field_seq):
            return None
        if field_seq[start][2] != template[0]:
            return None

        matched = [start]
        pos = start
        for t_idx in range(1, len(template)):
            found = False
            for gap in range(1, max_gap + 2):  # +2 because gap=0 means next
                next_pos = pos + gap
                if next_pos >= len(field_seq):
                    break
                if field_seq[next_pos][2] == template[t_idx]:
                    matched.append(next_pos)
                    pos = next_pos
                    found = True
                    break
            if not found:
                return None
        return matched

    # ── Metric 4: Collocational Coherence ───────────────────────────

    def collocational_coherence(self, decoded_text, window=5):
        """Check field-pair plausibility for nearby resolved words.

        For each pair of resolved words within `window` token positions,
        check whether their field pair is plausible, implausible, or neutral.
        """
        words = decoded_text.split()
        resolved = []
        for i, word in enumerate(words):
            if not word.startswith('[') and not word.startswith('<'):
                primary = get_primary_field(word)
                if primary != 'UNKNOWN':
                    resolved.append((i, word, primary))

        plausible = 0
        implausible = 0
        neutral = 0
        pairs_checked = 0

        for idx in range(len(resolved)):
            for jdx in range(idx + 1, len(resolved)):
                pos_i = resolved[idx][0]
                pos_j = resolved[jdx][0]
                if pos_j - pos_i > window:
                    break
                pairs_checked += 1
                pair = (resolved[idx][2], resolved[jdx][2])
                if pair in PLAUSIBLE_COLLOCATIONS:
                    plausible += 1
                elif pair in IMPLAUSIBLE_COLLOCATIONS:
                    implausible += 1
                else:
                    neutral += 1

        if pairs_checked == 0:
            return None

        return {
            'plausible_rate': round(plausible / pairs_checked, 4),
            'implausible_rate': round(implausible / pairs_checked, 4),
            'neutral_rate': round(neutral / pairs_checked, 4),
            'pairs_checked': pairs_checked,
            'plausible_count': plausible,
            'implausible_count': implausible,
        }

    # ── Null distribution builders ──────────────────────────────────

    def _shuffle_baseline(self, decoded_text, rng):
        """Shuffle resolved word positions (brackets stay fixed)."""
        words = decoded_text.split()
        resolved_indices = [i for i, w in enumerate(words)
                           if not w.startswith('[') and not w.startswith('<')]
        resolved_words = [words[i] for i in resolved_indices]
        rng.shuffle(resolved_words)
        shuffled = list(words)
        for idx, word in zip(resolved_indices, resolved_words):
            shuffled[idx] = word
        return ' '.join(shuffled)

    def _random_vocab_baseline(self, decoded_text, rng, general=False):
        """Replace each resolved word with a random vocab entry.

        Args:
            general: if True, 50/50 sample from medical + non-medical vocab.
                     if False, sample from classified medical vocabulary.
        """
        words = decoded_text.split()
        result = []
        for w in words:
            if not w.startswith('[') and not w.startswith('<'):
                if general:
                    # 50/50 coin flip: medical vs non-medical word
                    if rng.random() < 0.5:
                        result.append(rng.choice(self._medical_vocab_list))
                    else:
                        result.append(rng.choice(self._nonmedical_vocab_list))
                else:
                    result.append(rng.choice(self._medical_vocab_list))
            else:
                result.append(w)
        return ' '.join(result)

    def _cross_folio_baseline(self, decoded_text, donor_text):
        """Place donor folio's resolved words into this folio's brackets.

        If donor has fewer resolved words, cycle. If more, truncate.
        """
        words = decoded_text.split()
        donor_resolved = self._extract_resolved(donor_text)
        if not donor_resolved:
            return decoded_text

        result = []
        j = 0
        for w in words:
            if not w.startswith('[') and not w.startswith('<'):
                result.append(donor_resolved[j % len(donor_resolved)])
                j += 1
            else:
                result.append(w)
        return ' '.join(result)

    def build_null_distribution(self, folio_id, method='shuffle',
                                n_trials=1000, metric='medical_rate'):
        """Generate null distribution for a folio.

        Args:
            method: 'shuffle', 'random_vocab', 'random_general', or 'cross_folio'
            metric: which metric to compute ('medical_rate', 'entropy',
                    'template_coverage', 'collocation_plausible')

        Returns list of null metric values.
        """
        text = self.translations.get(folio_id, '')
        if not text:
            return []

        # For cross-folio, pick donor folios
        other_folios = [f for f in self.translations if f != folio_id]

        null_scores = []
        for trial in range(n_trials):
            rng = random.Random(42 + trial)

            if method == 'shuffle':
                null_text = self._shuffle_baseline(text, rng)
            elif method == 'random_vocab':
                null_text = self._random_vocab_baseline(text, rng, general=False)
            elif method == 'random_general':
                null_text = self._random_vocab_baseline(text, rng, general=True)
            elif method == 'cross_folio':
                donor = other_folios[trial % len(other_folios)]
                null_text = self._cross_folio_baseline(
                    text, self.translations[donor])
            else:
                raise ValueError(f'Unknown method: {method}')

            score = self._compute_metric(null_text, metric)
            if score is not None:
                null_scores.append(score)

        return null_scores

    def _compute_metric(self, text, metric):
        """Compute a single scalar metric value from decoded text."""
        if metric == 'medical_rate':
            result = self.medical_vocabulary_rate(text)
            return result['medical_rate'] if result else None
        elif metric == 'entropy':
            result = self.field_concentration(text)
            return result['normalized_entropy'] if result else None
        elif metric == 'template_coverage':
            result = self.template_matching(text)
            return result['coverage'] if result else None
        elif metric == 'collocation_plausible':
            result = self.collocational_coherence(text)
            return result['plausible_rate'] if result else None
        return None

    # ── Per-folio analysis ──────────────────────────────────────────

    def analyze_folio(self, folio_id):
        """Run all four metrics on one folio."""
        text = self.translations.get(folio_id, '')
        if not text:
            return None

        resolved = self._extract_resolved(text)
        if len(resolved) < MIN_RESOLVED_WORDS:
            return None

        meta = self.per_folio_stats.get(folio_id, {})

        return {
            'folio_id': folio_id,
            'language': meta.get('language', '?'),
            'section': meta.get('section', '?'),
            'total_words': len(text.split()),
            'resolved_words': len(resolved),
            'vocabulary': self.medical_vocabulary_rate(text),
            'concentration': self.field_concentration(text),
            'templates': self.template_matching(text),
            'collocations': self.collocational_coherence(text),
        }

    # ── Corpus-wide analysis ────────────────────────────────────────

    def analyze_all(self, folio_limit=None):
        """Analyze all folios. Return aggregate + per-folio results."""
        folio_ids = sorted(self.translations.keys())
        if folio_limit:
            folio_ids = folio_ids[:folio_limit]

        per_folio = {}
        for fid in folio_ids:
            result = self.analyze_folio(fid)
            if result is not None:
                per_folio[fid] = result

        if not per_folio:
            return {'error': 'No folios with sufficient resolved words'}

        # Aggregate metrics
        med_rates = [r['vocabulary']['medical_rate']
                     for r in per_folio.values() if r['vocabulary']]
        entropies = [r['concentration']['normalized_entropy']
                     for r in per_folio.values() if r['concentration']]
        coverages = [r['templates']['coverage']
                     for r in per_folio.values() if r['templates']]
        plausibles = [r['collocations']['plausible_rate']
                      for r in per_folio.values() if r['collocations']]

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        # Aggregate field counts
        total_field_counts = Counter()
        total_classified = 0
        for r in per_folio.values():
            if r['vocabulary']:
                for f, frac in r['vocabulary']['field_breakdown'].items():
                    total_field_counts[f] += frac * r['vocabulary']['total_resolved']
                total_classified += r['vocabulary']['total_resolved']

        field_pcts = {}
        if total_classified > 0:
            field_pcts = {f: round(c / total_classified, 4)
                          for f, c in total_field_counts.most_common()}

        # Top templates across all folios
        all_template_counts = Counter()
        for r in per_folio.values():
            if r['templates']:
                for t_info in r['templates']['top_templates']:
                    all_template_counts[tuple(t_info['template'])] += t_info['count']

        summary = {
            'folios_analyzed': len(per_folio),
            'folios_skipped': len(folio_ids) - len(per_folio),
            'overall_medical_rate': round(_mean(med_rates), 4),
            'overall_entropy': round(_mean(entropies), 4),
            'overall_template_coverage': round(_mean(coverages), 4),
            'overall_collocation_plausible': round(_mean(plausibles), 4),
            'field_distribution': field_pcts,
            'top_templates': [
                {'template': list(t), 'count': c}
                for t, c in all_template_counts.most_common(10)
            ],
        }

        # Group by section
        by_section = {}
        for r in per_folio.values():
            sec = r['section']
            if sec not in by_section:
                by_section[sec] = {'rates': [], 'entropies': [],
                                   'coverages': [], 'plausibles': [],
                                   'folios': 0}
            by_section[sec]['folios'] += 1
            if r['vocabulary']:
                by_section[sec]['rates'].append(r['vocabulary']['medical_rate'])
            if r['concentration']:
                by_section[sec]['entropies'].append(
                    r['concentration']['normalized_entropy'])
            if r['templates']:
                by_section[sec]['coverages'].append(r['templates']['coverage'])
            if r['collocations']:
                by_section[sec]['plausibles'].append(
                    r['collocations']['plausible_rate'])

        section_summary = {}
        for sec, data in sorted(by_section.items()):
            section_summary[sec] = {
                'folios': data['folios'],
                'medical_rate': round(_mean(data['rates']), 4),
                'entropy': round(_mean(data['entropies']), 4),
                'template_coverage': round(_mean(data['coverages']), 4),
                'collocation_plausible': round(_mean(data['plausibles']), 4),
            }

        # Group by language
        by_lang = {}
        for r in per_folio.values():
            lang = r['language']
            if lang not in by_lang:
                by_lang[lang] = {'rates': [], 'entropies': [],
                                 'coverages': [], 'plausibles': [],
                                 'folios': 0}
            by_lang[lang]['folios'] += 1
            if r['vocabulary']:
                by_lang[lang]['rates'].append(r['vocabulary']['medical_rate'])
            if r['concentration']:
                by_lang[lang]['entropies'].append(
                    r['concentration']['normalized_entropy'])
            if r['templates']:
                by_lang[lang]['coverages'].append(r['templates']['coverage'])
            if r['collocations']:
                by_lang[lang]['plausibles'].append(
                    r['collocations']['plausible_rate'])

        lang_summary = {}
        for lang, data in sorted(by_lang.items()):
            lang_summary[lang] = {
                'folios': data['folios'],
                'medical_rate': round(_mean(data['rates']), 4),
                'entropy': round(_mean(data['entropies']), 4),
                'template_coverage': round(_mean(data['coverages']), 4),
                'collocation_plausible': round(_mean(data['plausibles']), 4),
            }

        return {
            'summary': summary,
            'by_section': section_summary,
            'by_language': lang_summary,
            'per_folio': per_folio,
        }

    # ── Significance testing ────────────────────────────────────────

    def compute_significance(self, folio_limit=None, n_trials=1000,
                             verbose=False):
        """Run null distributions and compute p-values for each metric.

        Tests:
        - medical_rate: real vs random_general (general Latin incl. non-medical)
        - entropy: real vs random_vocab (random medical vocab → high entropy)
        - template_coverage: real vs shuffle (same words, random order)
        - collocation_plausible: real vs shuffle (same words, random order)

        Returns significance dict with p-values and effect sizes.
        """
        folio_ids = sorted(self.translations.keys())
        if folio_limit:
            folio_ids = folio_ids[:folio_limit]

        # Filter to analyzable folios
        analyzable = []
        for fid in folio_ids:
            text = self.translations.get(fid, '')
            resolved = self._extract_resolved(text)
            if len(resolved) >= MIN_RESOLVED_WORDS:
                analyzable.append(fid)

        if not analyzable:
            return {'error': 'No analyzable folios'}

        tests = [
            ('medical_rate', 'random_general', 'higher'),
            ('entropy', 'random_vocab', 'lower'),
            ('template_coverage', 'shuffle', 'higher'),
            ('collocation_plausible', 'shuffle', 'higher'),
        ]

        results = {}
        for metric, method, direction in tests:
            real_scores = []
            null_all = []  # per-folio null distributions
            p_values = []

            for i, fid in enumerate(analyzable):
                if verbose and (i + 1) % 50 == 0:
                    print(f'    Significance: {metric} — '
                          f'{i + 1}/{len(analyzable)} folios...')

                real = self._compute_metric(self.translations[fid], metric)
                if real is None:
                    continue

                nulls = self.build_null_distribution(
                    fid, method=method, n_trials=n_trials, metric=metric)
                if not nulls:
                    continue

                real_scores.append(real)
                null_all.append(nulls)

                # Per-folio p-value
                if direction == 'higher':
                    p = sum(1 for n in nulls if n >= real) / len(nulls)
                else:  # lower is better (entropy)
                    p = sum(1 for n in nulls if n <= real) / len(nulls)
                p_values.append(p)

            if not real_scores:
                results[metric] = {'error': 'insufficient data'}
                continue

            real_mean = sum(real_scores) / len(real_scores)
            null_mean = sum(
                sum(ns) / len(ns) for ns in null_all
            ) / len(null_all)
            mean_p = sum(p_values) / len(p_values) if p_values else 1.0
            effect = real_mean - null_mean

            results[metric] = {
                'real_mean': round(real_mean, 4),
                'null_mean': round(null_mean, 4),
                'effect_size': round(effect, 4),
                'mean_p_value': round(mean_p, 4),
                'folios_tested': len(p_values),
                'significant_005': mean_p < 0.05,
                'direction': direction,
                'null_method': method,
            }

        return results

    # ── Language B diagnostic ───────────────────────────────────────

    def language_b_diagnostic(self):
        """Compare Language A vs B decoded vocabulary distributions.

        Checks whether B decodes the same words or a restricted subset.
        """
        lang_a_words = Counter()
        lang_b_words = Counter()
        lang_a_fields = Counter()
        lang_b_fields = Counter()

        for fid, text in self.translations.items():
            meta = self.per_folio_stats.get(fid, {})
            lang = meta.get('language', '?')
            resolved = self._extract_resolved(text)

            counter = lang_a_words if lang == 'A' else lang_b_words
            field_counter = lang_a_fields if lang == 'A' else lang_b_fields

            for w in resolved:
                counter[w] += 1
                primary = get_primary_field(w)
                field_counter[primary] += 1

        # Vocabulary overlap
        a_types = set(lang_a_words.keys())
        b_types = set(lang_b_words.keys())
        overlap = a_types & b_types
        a_only = a_types - b_types
        b_only = b_types - a_types

        # Field distribution comparison
        a_total = sum(lang_a_fields.values())
        b_total = sum(lang_b_fields.values())
        field_comparison = {}
        all_fields = set(lang_a_fields.keys()) | set(lang_b_fields.keys())
        for f in sorted(all_fields):
            field_comparison[f] = {
                'lang_a_pct': round(lang_a_fields[f] / a_total, 4) if a_total else 0,
                'lang_b_pct': round(lang_b_fields[f] / b_total, 4) if b_total else 0,
            }

        return {
            'lang_a_types': len(a_types),
            'lang_a_tokens': sum(lang_a_words.values()),
            'lang_b_types': len(b_types),
            'lang_b_tokens': sum(lang_b_words.values()),
            'overlap_types': len(overlap),
            'a_only_types': len(a_only),
            'b_only_types': len(b_only),
            'jaccard_similarity': round(
                len(overlap) / len(a_types | b_types), 4
            ) if (a_types | b_types) else 0,
            'field_comparison': field_comparison,
            'a_top_words': lang_a_words.most_common(20),
            'b_top_words': lang_b_words.most_common(20),
        }
