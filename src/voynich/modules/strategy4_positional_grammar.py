"""
Strategy 4: Positional Glyph Grammar Extraction
==================================================
Treat the positional restrictions of Voynich glyphs not as a cipher artifact
but as a deliberate MORPHOLOGICAL ENCODING LAYER.

If the cipher was designed to preserve grammatical structure:
- Prefix glyphs → grammatical markers (tense, case, imperative)
- Root glyphs → semantic cipher content (high entropy, medial position)
- Suffix glyphs → quantity, body-part classifiers, dosage markers

This module:
1. Extracts rigorous positional class assignments for all glyphs
2. Tests whether prefix distributions correlate with section type
3. Isolates the "semantic core" from the "grammatical wrapper"
4. Identifies candidate functional morphemes vs content morphemes
"""

import sys
import os
import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Optional


from voynich.core.stats import (
    positional_glyph_distribution, classify_glyphs_by_position,
    word_positional_entropy, first_order_entropy,
    compute_all_entropy, char_frequencies
)
from voynich.core.voynich_corpus import (
    SAMPLE_CORPUS, EVA_GLYPHS, SECTIONS, get_all_tokens,
    get_section_text, tokenize
)

def extract_glyph_classes(threshold: float = 0.65) -> Dict:
    """
    Extract positional classes for all glyphs in the corpus.
    
    Returns a detailed report including:
    - Classification (P/S/M/A) for each glyph
    - Positional distribution percentages
    - Confidence score for each classification
    """
    all_tokens = get_all_tokens()
    dist = positional_glyph_distribution(all_tokens)

    glyph_report = {}
    for char in sorted(dist.keys()):
        counts = dist[char]
        total = sum(counts.values())
        if total == 0:
            continue

        ratios = {k: v / total for k, v in counts.items()}

        max_pos = max(ratios, key=ratios.get)
        max_ratio = ratios[max_pos]

        if ratios.get('initial', 0) >= threshold:
            cls = 'PREFIX'
        elif ratios.get('final', 0) >= threshold:
            cls = 'SUFFIX'
        elif ratios.get('medial', 0) >= threshold:
            cls = 'MEDIAL'
        else:
            cls = 'ANY'

        second_max = sorted(ratios.values(), reverse=True)[1] if len(ratios) > 1 else 0
        confidence = max_ratio - second_max

        glyph_report[char] = {
            'class': cls,
            'initial_pct': round(ratios.get('initial', 0) * 100, 1),
            'medial_pct': round(ratios.get('medial', 0) * 100, 1),
            'final_pct': round(ratios.get('final', 0) * 100, 1),
            'singleton_pct': round(ratios.get('singleton', 0) * 100, 1),
            'total_occurrences': total,
            'confidence': round(confidence, 3),
            'dominant_position': max_pos,
        }

    return glyph_report

def decompose_word(word: str, glyph_classes: Dict) -> Dict:
    """
    Decompose a Voynich word into prefix, root, and suffix components
    based on positional glyph classes.
    
    Algorithm:
    1. Scan from left: accumulate PREFIX-class glyphs → prefix
    2. Scan from right: accumulate SUFFIX-class glyphs → suffix
    3. Everything in between → root (semantic core)
    """
    if not word:
        return {'prefix': '', 'root': '', 'suffix': '', 'original': word}

    chars = list(word)
    prefix = []
    suffix = []

    for c in chars:
        cls = glyph_classes.get(c, {}).get('class', 'ANY')
        if cls == 'PREFIX':
            prefix.append(c)
        else:
            break

    for c in reversed(chars):
        cls = glyph_classes.get(c, {}).get('class', 'ANY')
        if cls == 'SUFFIX':
            suffix.insert(0, c)
        else:
            break

    prefix_len = len(prefix)
    suffix_len = len(suffix)
    root_chars = chars[prefix_len:len(chars) - suffix_len if suffix_len > 0 else len(chars)]

    return {
        'prefix': ''.join(prefix),
        'root': ''.join(root_chars),
        'suffix': ''.join(suffix),
        'original': word,
    }

def decompose_corpus(glyph_classes: Optional[Dict] = None) -> List[Dict]:
    """Decompose all tokens in the corpus into prefix-root-suffix."""
    if glyph_classes is None:
        glyph_classes = extract_glyph_classes()

    tokens = get_all_tokens()
    return [decompose_word(t, glyph_classes) for t in tokens]

def prefix_section_correlation(glyph_classes: Optional[Dict] = None) -> Dict:
    """
    Test whether prefix glyph distributions differ between sections.
    
    Hypothesis: If prefixes encode grammatical information, their distribution
    should shift between descriptive sections (herbal: "this plant is...") and
    instructional sections (recipe: "take X, grind Y...").
    
    Returns chi-squared test results and distribution tables.
    """
    if glyph_classes is None:
        glyph_classes = extract_glyph_classes()

    section_prefixes = {}
    section_suffixes = {}
    section_roots = {}

    for section in ['herbal_a', 'herbal_b', 'pharmaceutical', 'recipes',
                    'biological', 'astronomical']:
        tokens = get_all_tokens(section=section)
        if not tokens:
            continue

        prefixes = []
        suffixes = []
        roots = []

        for t in tokens:
            d = decompose_word(t, glyph_classes)
            if d['prefix']:
                prefixes.append(d['prefix'])
            if d['suffix']:
                suffixes.append(d['suffix'])
            if d['root']:
                roots.append(d['root'])

        section_prefixes[section] = Counter(prefixes)
        section_suffixes[section] = Counter(suffixes)
        section_roots[section] = Counter(roots)

    all_prefixes = set()
    for counts in section_prefixes.values():
        all_prefixes.update(counts.keys())

    prefix_table = {}
    for section in section_prefixes:
        total = sum(section_prefixes[section].values())
        prefix_table[section] = {}
        for p in sorted(all_prefixes):
            count = section_prefixes[section].get(p, 0)
            prefix_table[section][p] = {
                'count': count,
                'pct': round(count / total * 100, 1) if total > 0 else 0,
            }

    prefix_entropy = {}
    for section, counts in section_prefixes.items():
        total = sum(counts.values())
        if total > 0:
            probs = [c / total for c in counts.values()]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
            prefix_entropy[section] = round(h, 4)

    suffix_entropy = {}
    for section, counts in section_suffixes.items():
        total = sum(counts.values())
        if total > 0:
            probs = [c / total for c in counts.values()]
            h = -sum(p * math.log2(p) for p in probs if p > 0)
            suffix_entropy[section] = round(h, 4)

    root_uniqueness = {}
    for section, counts in section_roots.items():
        total = sum(counts.values())
        unique = len(counts)
        root_uniqueness[section] = {
            'total_roots': total,
            'unique_roots': unique,
            'type_token_ratio': round(unique / total, 4) if total > 0 else 0,
        }

    return {
        'prefix_table': prefix_table,
        'prefix_entropy_by_section': prefix_entropy,
        'suffix_entropy_by_section': suffix_entropy,
        'root_uniqueness': root_uniqueness,
        'all_prefixes': sorted(all_prefixes),
        'interpretation': _interpret_prefix_correlation(prefix_entropy),
    }

def _interpret_prefix_correlation(prefix_entropy: Dict) -> str:
    """Generate human-readable interpretation of prefix-section correlation."""
    if not prefix_entropy:
        return "Insufficient data for interpretation."

    values = list(prefix_entropy.values())
    variance = sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)

    if variance > 0.3:
        return ("STRONG CORRELATION: Prefix distributions differ significantly "
                "across sections, suggesting prefixes carry grammatical or "
                "functional information tied to content type (descriptive vs "
                "instructional vs anatomical).")
    elif variance > 0.1:
        return ("MODERATE CORRELATION: Some variation in prefix usage across "
                "sections detected. Prefixes may carry partial grammatical "
                "information.")
    else:
        return ("WEAK CORRELATION: Prefix distributions are relatively uniform "
                "across sections. Prefixes may be primarily cipher artifacts "
                "rather than grammatical markers.")

def isolate_semantic_cores(glyph_classes: Optional[Dict] = None) -> Dict:
    """
    Strip prefixes and suffixes from all words to isolate the
    "semantic core" — the portion most likely to carry cipher content.
    
    Then analyze whether the core alone has different statistical properties
    than the full words (which it should, if prefix/suffix are grammatical).
    """
    if glyph_classes is None:
        glyph_classes = extract_glyph_classes()

    decomposed = decompose_corpus(glyph_classes)

    cores = [d['root'] for d in decomposed if d['root']]
    core_text = ' '.join(cores)

    full_text = ' '.join(get_all_tokens())

    core_stats = compute_all_entropy(core_text)
    full_stats = compute_all_entropy(full_text)

    core_freq = Counter(cores)
    full_freq = Counter(get_all_tokens())

    return {
        'core_entropy': core_stats,
        'full_entropy': full_stats,
        'entropy_delta': {
            'H1': round(core_stats['H1'] - full_stats['H1'], 4),
            'H2': round(core_stats['H2'] - full_stats['H2'], 4),
            'H3': round(core_stats['H3'] - full_stats['H3'], 4),
        },
        'core_vocab_size': len(core_freq),
        'full_vocab_size': len(full_freq),
        'core_top_20': core_freq.most_common(20),
        'prefix_removed_pct': round(
            sum(1 for d in decomposed if d['prefix']) / max(1, len(decomposed)) * 100, 1
        ),
        'suffix_removed_pct': round(
            sum(1 for d in decomposed if d['suffix']) / max(1, len(decomposed)) * 100, 1
        ),
        'interpretation': _interpret_core_isolation(core_stats, full_stats),
    }

def _interpret_core_isolation(core: Dict, full: Dict) -> str:
    """Interpret the effect of stripping affixes."""
    h2_diff = core['H2'] - full['H2']
    if h2_diff > 0.2:
        return ("SIGNIFICANT: Removing positional affixes INCREASES entropy, "
                "meaning the affixes were suppressing entropy. This is consistent "
                "with affixes being low-entropy grammatical markers and cores "
                "carrying higher-entropy cipher content.")
    elif h2_diff > 0.05:
        return ("MODERATE: Some entropy increase when stripping affixes. "
                "Partial grammatical encoding is plausible.")
    elif h2_diff < -0.1:
        return ("INVERSE: Removing affixes DECREASES entropy. This suggests "
                "the positional constraints are integral to the cipher mechanism "
                "rather than a separate grammatical layer.")
    else:
        return ("MINIMAL: Little entropy change from stripping affixes. "
                "The positional classes may not form a separable grammatical layer.")

def identify_functional_morphemes(glyph_classes: Optional[Dict] = None) -> Dict:
    """
    Identify candidate functional morphemes (grammatical words) vs content
    morphemes in the Voynich text.
    
    Functional morphemes in natural languages are:
    - Very high frequency
    - Short
    - Positionally constrained
    - Uniformly distributed across sections (not topic-specific)
    
    Content morphemes are:
    - Section-specific
    - Moderate frequency
    - Variable length
    - Concentrated in particular sections
    """
    if glyph_classes is None:
        glyph_classes = extract_glyph_classes()

    all_tokens = get_all_tokens()
    overall_freq = Counter(all_tokens)

    section_freqs = {}
    for section in SECTIONS:
        tokens = get_all_tokens(section=section)
        if tokens:
            section_freqs[section] = Counter(tokens)

    scores = {}
    for word, count in overall_freq.items():
        freq_score = math.log(count + 1)

        len_score = 1.0 / (len(word) + 1)

        sections_present = sum(
            1 for sf in section_freqs.values() if word in sf
        )
        uniformity = sections_present / max(1, len(section_freqs))

        max_section_pct = 0
        for sf in section_freqs.values():
            total = sum(sf.values())
            if total > 0:
                pct = sf.get(word, 0) / total
                max_section_pct = max(max_section_pct, pct)

        functional_score = freq_score * len_score * uniformity
        content_score = freq_score * (1 - uniformity) * max_section_pct * len(word)

        d = decompose_word(word, glyph_classes)

        scores[word] = {
            'count': count,
            'length': len(word),
            'functional_score': round(functional_score, 4),
            'content_score': round(content_score, 4),
            'sections_present': sections_present,
            'uniformity': round(uniformity, 3),
            'has_prefix': bool(d['prefix']),
            'has_suffix': bool(d['suffix']),
            'root': d['root'],
            'classification': 'FUNCTIONAL' if functional_score > content_score
                             else 'CONTENT',
        }

    functional = sorted(
        [(w, s) for w, s in scores.items() if s['classification'] == 'FUNCTIONAL'],
        key=lambda x: x[1]['functional_score'], reverse=True
    )
    content = sorted(
        [(w, s) for w, s in scores.items() if s['classification'] == 'CONTENT'],
        key=lambda x: x[1]['content_score'], reverse=True
    )

    return {
        'functional_morphemes': functional[:30],
        'content_morphemes': content[:30],
        'total_functional': len(functional),
        'total_content': len(content),
        'ratio': round(len(functional) / max(1, len(content)), 3),
    }

def run(verbose: bool = True) -> Dict:
    """Run all positional glyph grammar analyses."""
    if verbose:
        print("=" * 70)
        print("STRATEGY 4: POSITIONAL GLYPH GRAMMAR EXTRACTION")
        print("=" * 70)

    results = {}

    if verbose:
        print("\n[1/4] Extracting positional glyph classes...")
    glyph_classes = extract_glyph_classes()
    results['glyph_classes'] = glyph_classes
    if verbose:
        for char in sorted(glyph_classes.keys()):
            g = glyph_classes[char]
            print(f"  '{char}': {g['class']:8s} "
                  f"(init={g['initial_pct']:5.1f}% med={g['medial_pct']:5.1f}% "
                  f"fin={g['final_pct']:5.1f}% conf={g['confidence']:.3f} "
                  f"n={g['total_occurrences']})")

    if verbose:
        print("\n[2/4] Testing prefix-section correlation...")
    prefix_corr = prefix_section_correlation(glyph_classes)
    results['prefix_correlation'] = prefix_corr
    if verbose:
        print(f"  Prefix entropy by section:")
        for section, h in prefix_corr['prefix_entropy_by_section'].items():
            print(f"    {section:20s}: H={h:.4f}")
        print(f"\n  Root uniqueness:")
        for section, data in prefix_corr['root_uniqueness'].items():
            print(f"    {section:20s}: TTR={data['type_token_ratio']:.4f} "
                  f"({data['unique_roots']}/{data['total_roots']})")
        print(f"\n  Interpretation: {prefix_corr['interpretation']}")

    if verbose:
        print("\n[3/4] Isolating semantic cores...")
    core_analysis = isolate_semantic_cores(glyph_classes)
    results['core_isolation'] = core_analysis
    if verbose:
        print(f"  Full text entropy: H1={core_analysis['full_entropy']['H1']:.4f} "
              f"H2={core_analysis['full_entropy']['H2']:.4f}")
        print(f"  Core-only entropy: H1={core_analysis['core_entropy']['H1']:.4f} "
              f"H2={core_analysis['core_entropy']['H2']:.4f}")
        print(f"  Delta: ΔH1={core_analysis['entropy_delta']['H1']:+.4f} "
              f"ΔH2={core_analysis['entropy_delta']['H2']:+.4f}")
        print(f"  Words with prefix stripped: {core_analysis['prefix_removed_pct']}%")
        print(f"  Words with suffix stripped: {core_analysis['suffix_removed_pct']}%")
        print(f"\n  Interpretation: {core_analysis['interpretation']}")

    if verbose:
        print("\n[4/4] Identifying functional vs content morphemes...")
    morphemes = identify_functional_morphemes(glyph_classes)
    results['morphemes'] = morphemes
    if verbose:
        print(f"  Functional morphemes: {morphemes['total_functional']}")
        print(f"  Content morphemes: {morphemes['total_content']}")
        print(f"  Ratio: {morphemes['ratio']:.3f}")
        print(f"\n  Top functional candidates:")
        for word, data in morphemes['functional_morphemes'][:10]:
            print(f"    '{word}' (n={data['count']}, len={data['length']}, "
                  f"uniformity={data['uniformity']:.2f})")
        print(f"\n  Top content candidates:")
        for word, data in morphemes['content_morphemes'][:10]:
            print(f"    '{word}' (n={data['count']}, len={data['length']}, "
                  f"sections={data['sections_present']})")

    return results

if __name__ == '__main__':
    run()
