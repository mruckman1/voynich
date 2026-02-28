"""
Voynich Convergence Attack — MAX SETTINGS
===========================================
Run all strategies at maximum resolution using the full ZL corpus.

Usage:
    PYTHONHASHSEED=0 uv run python run_max.py

Prerequisites:
    1. Download the corpus:
       curl https://www.voynich.nu/data/ZL_ivtff_2b.txt -o data/corpus/ZL_ivtff_2b.txt

    2. (Optional) Additional transliterations:
       curl https://www.voynich.nu/data/IT_ivtff_1a.txt -o data/corpus/IT_ivtff_1a.txt
       curl https://www.voynich.nu/data/RF_ivtff_1b_Eva.txt -o data/corpus/RF_ivtff_1b_Eva.txt
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ivtff_parser import load_corpus, VoynichCorpus
from data.voynich_corpus import (
    SAMPLE_CORPUS, HARTLIEB_MEDICAL_VOCAB, ZODIAC_LABELS,
    LATIN_RECIPE_FORMULAS
)
from modules.statistical_analysis import (
    full_statistical_profile, profile_distance, compute_all_entropy,
    bigram_transition_matrix, compare_bigram_matrices,
    zipf_analysis, word_positional_entropy, classify_glyphs_by_position,
    positional_glyph_distribution
)
from modules.naibbe_cipher import NaibbeCipher, generate_parameter_grid
from modules.strategy1_parameter_search import (
    generate_medical_plaintext, generate_section_specific_plaintext,
)
from modules.strategy2_scribe_seams import (
    cross_scribe_vocabulary, first_word_patterns,
    scribe_positional_divergence,
)
from modules.strategy3_binding_reconstruction import (
    compute_sequential_consistency, RECONSTRUCTED_ORDERS,
)
from modules.strategy4_positional_grammar import (
    extract_glyph_classes, decompose_word, prefix_section_correlation,
    isolate_semantic_cores, identify_functional_morphemes,
)
from modules.strategy5_zodiac_attack import (
    zodiac_known_plaintext_attack, generate_zodiac_plaintext,
)

def run_max():
    """Run the full convergence attack at maximum settings."""
    start = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║" + " VOYNICH CONVERGENCE ATTACK — MAXIMUM SETTINGS ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"  {datetime.now().isoformat()}\n")

    print("=" * 70)
    print("PHASE 0: LOADING FULL CORPUS")
    print("=" * 70)

    try:
        corpus = load_corpus('data/corpus', verbose=True)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("\n  Please download the corpus first:")
        print("    mkdir -p data/corpus")
        print("    curl https://www.voynich.nu/data/ZL_ivtff_2b.txt -o data/corpus/ZL_ivtff_2b.txt")
        sys.exit(1)

    summary = corpus.summary()
    print(f"\n  Corpus loaded: {summary['total_tokens']} tokens across {summary['total_pages']} pages")
    print(f"  Unique tokens: {summary['unique_tokens']} (TTR: {summary['type_token_ratio']:.4f})")
    print(f"  Languages: {summary['languages']}")
    print(f"  Hands: {summary['hands']}")
    print(f"  Sections: {summary['sections']}")

    results = {'corpus_summary': summary}

    print("\n" + "=" * 70)
    print("PHASE 1: COMPREHENSIVE STATISTICAL PROFILING")
    print("=" * 70)

    profiles = {}

    all_text = corpus.get_text(paragraph_only=True)
    profiles['overall'] = full_statistical_profile(all_text, 'Full-Voynich')
    e = profiles['overall']['entropy']
    print(f"\n  Overall: H1={e['H1']:.4f}  H2={e['H2']:.4f}  H3={e['H3']:.4f}")
    z = profiles['overall']['zipf']
    print(f"  Zipf exponent: {z['zipf_exponent']:.4f}  R²={z['r_squared']:.4f}")
    print(f"  Tokens: {z['total_tokens']}  Types: {z['vocabulary_size']}  TTR: {z['type_token_ratio']:.4f}")

    for lang in ['A', 'B']:
        text = corpus.get_text(language=lang)
        if text.strip():
            profiles[f'lang_{lang}'] = full_statistical_profile(text, f'Language-{lang}')
            e = profiles[f'lang_{lang}']['entropy']
            print(f"  Language {lang}: H1={e['H1']:.4f}  H2={e['H2']:.4f}  H3={e['H3']:.4f}  "
                  f"({len(text.split())} tokens)")

    for hand in range(1, 6):
        text = corpus.get_text(hand=hand)
        if text.strip():
            profiles[f'hand_{hand}'] = full_statistical_profile(text, f'Hand-{hand}')
            e = profiles[f'hand_{hand}']['entropy']
            print(f"  Hand {hand}: H1={e['H1']:.4f}  H2={e['H2']:.4f}  H3={e['H3']:.4f}  "
                  f"({len(text.split())} tokens)")

    for section in ['herbal', 'astronomical', 'biological', 'cosmological',
                    'pharmaceutical', 'recipes', 'zodiac', 'text_only']:
        text = corpus.get_text(section=section)
        if text.strip() and len(text.split()) > 20:
            profiles[section] = full_statistical_profile(text, f'Section-{section}')
            e = profiles[section]['entropy']
            print(f"  {section:20s}: H1={e['H1']:.4f}  H2={e['H2']:.4f}  H3={e['H3']:.4f}  "
                  f"({len(text.split())} tokens)")

    results['profiles'] = {k: v['entropy'] for k, v in profiles.items()}

    print("\n" + "=" * 70)
    print("PHASE 2: NAIBBE PARAMETER SEARCH (MEDIUM resolution)")
    print("=" * 70)
    print("  (Use 'fine' in generate_parameter_grid() for exhaustive search)")
    print("  (Medium ≈ 1000 combinations, Fine ≈ 5000+, takes several minutes)")

    target_prof = profiles['overall']
    plaintext = generate_medical_plaintext(n_words=800)
    grid = generate_parameter_grid('medium')
    print(f"  Testing {len(grid)} parameter combinations against full Voynich profile...")

    param_results = []
    t0 = time.time()
    progress_step = max(1, len(grid) // 20)

    for idx, params in enumerate(grid):
        if idx % progress_step == 0:
            pct = idx / len(grid) * 100
            print(f"    {pct:.0f}%...", end=' ', flush=True)

        try:
            cipher = NaibbeCipher(**params)
            ct = cipher.encrypt(plaintext)
            if not ct.strip():
                continue

            cp = full_statistical_profile(ct, f'p{idx}')
            dist = profile_distance(cp, target_prof)

            param_results.append({
                'params': params,
                'distance': round(dist, 6),
                'entropy': cp['entropy'],
                'zipf_exp': cp['zipf'].get('zipf_exponent', 0),
            })
        except Exception:
            continue

    print(f"\n  Done in {time.time()-t0:.1f}s, {len(param_results)} valid results")
    param_results.sort(key=lambda r: r['distance'])

    print("\n  Top 5 parameter sets:")
    for i, r in enumerate(param_results[:5]):
        e = r['entropy']
        p = r['params']
        print(f"    #{i+1} dist={r['distance']:.4f}  H1={e['H1']:.3f} H2={e['H2']:.3f} H3={e['H3']:.3f}  "
              f"tables={p['n_tables']} bigram={p['bigram_probability']:.2f} "
              f"prefix={p['prefix_probability']:.2f} suffix={p['suffix_probability']:.2f}")

    results['strategy1_top_params'] = param_results[:20]

    print("\n" + "=" * 70)
    print("PHASE 3: SCRIBE SEAM ANALYSIS (full corpus)")
    print("=" * 70)

    transitions = corpus.get_scribe_transitions()
    print(f"  Found {len(transitions)} scribe transitions in corpus")

    seam_results = []
    for p1, p2 in transitions:
        t1 = p1.all_text
        t2 = p2.all_text
        if len(t1) < 20 or len(t2) < 20:
            continue

        e1 = compute_all_entropy(t1)
        e2 = compute_all_entropy(t2)
        h2_jump = abs(e2['H2'] - e1['H2'])

        tok1 = set(t1.split()[-15:])
        tok2 = set(t2.split()[:15])
        shared = tok1 & tok2

        cold_start = h2_jump > 0.3
        seam_results.append({
            'transition': f'{p1.folio}(H{p1.hand}) → {p2.folio}(H{p2.hand})',
            'H2_jump': round(h2_jump, 4),
            'cold_start': cold_start,
            'shared_boundary_tokens': len(shared),
            'shared_words': sorted(shared)[:10],
        })
        status = "⚡ COLD START" if cold_start else "  normal"
        print(f"  {p1.folio}(H{p1.hand}) → {p2.folio}(H{p2.hand}): "
              f"ΔH2={h2_jump:.4f} shared={len(shared)} {status}")

    results['strategy2_seams'] = seam_results

    print("\n  Cross-scribe vocabulary:")
    hand_vocabs = {}
    for h in range(1, 6):
        tokens = corpus.get_tokens(hand=h)
        if tokens:
            hand_vocabs[h] = set(tokens)
            print(f"    Hand {h}: {len(hand_vocabs[h])} unique types from {len(tokens)} tokens")

    for h1 in sorted(hand_vocabs):
        for h2 in sorted(hand_vocabs):
            if h1 >= h2:
                continue
            shared = hand_vocabs[h1] & hand_vocabs[h2]
            total = hand_vocabs[h1] | hand_vocabs[h2]
            jaccard = len(shared) / len(total) if total else 0
            print(f"    H{h1} vs H{h2}: Jaccard={jaccard:.3f} shared={len(shared)}")

    print("\n" + "=" * 70)
    print("PHASE 4: SEQUENTIAL STATE ANALYSIS (full corpus)")
    print("=" * 70)

    folio_mats = {}
    ordered_pages = corpus.get_folio_sequence()

    for page in ordered_pages:
        text = page.paragraph_text
        if len(text.replace(' ', '')) > 50:
            mat, alph = bigram_transition_matrix(text)
            folio_mats[page.folio] = (mat, alph, page)

    print(f"  Computing pairwise bigram distances for {len(folio_mats)} folios...")

    import math
    folio_keys = list(folio_mats.keys())
    distances = []
    sample_step = max(1, len(folio_keys) // 100)

    for i in range(0, len(folio_keys), sample_step):
        for j in range(i + 1, min(i + 20, len(folio_keys))):
            f1, f2 = folio_keys[i], folio_keys[j]
            m1, a1, _ = folio_mats[f1]
            m2, a2, _ = folio_mats[f2]
            bg_dist = compare_bigram_matrices(m1, m2, a1, a2)
            if bg_dist != float('inf'):
                distances.append((abs(j - i), bg_dist))

    if len(distances) >= 3:
        seq_d, bg_d = zip(*distances)
        mean_s = sum(seq_d) / len(seq_d)
        mean_b = sum(bg_d) / len(bg_d)
        cov = sum((s - mean_s) * (b - mean_b) for s, b in distances) / len(distances)
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in seq_d) / len(seq_d))
        std_b = math.sqrt(sum((b - mean_b) ** 2 for b in bg_d) / len(bg_d))
        corr = cov / (std_s * std_b) if std_s > 0 and std_b > 0 else 0
        print(f"  Sequential correlation: {corr:.4f} ({'STATE DETECTED' if abs(corr) > 0.3 else 'no state detected'})")
        results['strategy3_correlation'] = round(corr, 4)
    else:
        print("  Insufficient data for correlation")
        results['strategy3_correlation'] = None

    print("\n" + "=" * 70)
    print("PHASE 5: POSITIONAL GRAMMAR EXTRACTION (full corpus)")
    print("=" * 70)

    all_tokens = corpus.get_tokens(paragraph_only=True)
    all_text = ' '.join(all_tokens)

    glyph_dist = positional_glyph_distribution(all_tokens)
    glyph_classes = {}
    print("\n  Glyph positional classes (full corpus):")
    for char in sorted(glyph_dist.keys()):
        counts = glyph_dist[char]
        total = sum(counts.values())
        if total < 10:
            continue
        ratios = {k: v / total for k, v in counts.items()}

        if ratios.get('initial', 0) >= 0.65:
            cls = 'PREFIX'
        elif ratios.get('final', 0) >= 0.65:
            cls = 'SUFFIX'
        elif ratios.get('medial', 0) >= 0.65:
            cls = 'MEDIAL'
        else:
            cls = 'ANY'

        glyph_classes[char] = {
            'class': cls,
            'initial': round(ratios.get('initial', 0) * 100, 1),
            'medial': round(ratios.get('medial', 0) * 100, 1),
            'final': round(ratios.get('final', 0) * 100, 1),
            'n': total,
        }
        print(f"    '{char}': {cls:8s}  init={ratios.get('initial',0)*100:5.1f}%  "
              f"med={ratios.get('medial',0)*100:5.1f}%  fin={ratios.get('final',0)*100:5.1f}%  (n={total})")

    full_entropy = compute_all_entropy(all_text)

    gc_simple = {}
    for char, data in glyph_classes.items():
        gc_simple[char] = data

    cores = []
    for tok in all_tokens:
        d = decompose_word(tok, gc_simple)
        if d['root']:
            cores.append(d['root'])

    core_text = ' '.join(cores)
    core_entropy = compute_all_entropy(core_text)

    print(f"\n  Full text entropy:  H1={full_entropy['H1']:.4f}  H2={full_entropy['H2']:.4f}  H3={full_entropy['H3']:.4f}")
    print(f"  Core-only entropy:  H1={core_entropy['H1']:.4f}  H2={core_entropy['H2']:.4f}  H3={core_entropy['H3']:.4f}")
    dh2 = core_entropy['H2'] - full_entropy['H2']
    print(f"  ΔH2 = {dh2:+.4f} ({'AFFIXES SUPPRESS ENTROPY' if dh2 > 0.05 else 'minimal effect'})")

    results['strategy4'] = {
        'glyph_classes': glyph_classes,
        'full_entropy': full_entropy,
        'core_entropy': core_entropy,
        'delta_H2': round(dh2, 4),
    }

    pos_ent = word_positional_entropy(all_tokens)
    print(f"\n  Positional entropy (character position within word):")
    for pos, h in sorted(pos_ent.items()):
        bar = "█" * int(h * 10)
        print(f"    {pos}: H={h:.4f}  {bar}")

    results['strategy4']['positional_entropy'] = pos_ent

    print("\n" + "=" * 70)
    print("PHASE 6: ZODIAC KNOWN-PLAINTEXT ATTACK (full corpus)")
    print("=" * 70)

    zodiac_results = {}
    for folio, zdata in ZODIAC_LABELS.items():
        page = corpus.get_page(folio)
        if page and page.all_text.strip():
            voynich_text = page.all_text
            sign = zdata['zodiac_sign']
            plains = generate_zodiac_plaintext(sign)

            print(f"\n  {folio} ({sign} / {zdata['month_label']}):")
            print(f"    Voynich text: {voynich_text[:80]}...")
            print(f"    Voynich tokens: {len(voynich_text.split())}")

            v_prof = full_statistical_profile(voynich_text, f'V-{folio}')

            best_dist = float('inf')
            best_plain = ''
            best_cipher = ''

            for pr in param_results[:10]:
                for plain in plains[:3]:
                    try:
                        c = NaibbeCipher(**pr['params'])
                        ct = c.encrypt(plain)
                        if not ct.strip():
                            continue
                        cp = full_statistical_profile(ct, 'tmp')
                        d = profile_distance(cp, v_prof)
                        if d < best_dist:
                            best_dist = d
                            best_plain = plain
                            best_cipher = ct
                    except Exception:
                        continue

            if best_dist < float('inf'):
                print(f"    Best match: dist={best_dist:.4f}")
                print(f"    Plaintext:  {best_plain[:60]}")
                print(f"    Cipher:     {best_cipher[:60]}")
                zodiac_results[folio] = {
                    'sign': sign,
                    'distance': round(best_dist, 4),
                    'plaintext': best_plain,
                }

    results['strategy5_zodiac'] = zodiac_results

    elapsed = time.time() - start

    print("\n" + "═" * 70)
    print("  CONVERGENCE SYNTHESIS")
    print("═" * 70)

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Corpus: {summary['total_tokens']} tokens, {summary['total_pages']} pages")
    print(f"\n  KEY FINDINGS:")

    if results.get('strategy3_correlation') and abs(results['strategy3_correlation']) > 0.3:
        print(f"  ★ Sequential state DETECTED (r={results['strategy3_correlation']:.4f})")
    if results['strategy4']['delta_H2'] > 0.05:
        print(f"  ★ Positional affixes suppress entropy (ΔH2={results['strategy4']['delta_H2']:+.4f})")

    cold_starts = sum(1 for s in seam_results if s.get('cold_start'))
    if cold_starts:
        print(f"  ◆ {cold_starts} cold-start patterns at scribe transitions")

    if param_results:
        top = param_results[0]
        print(f"  ◆ Best cipher params: tables={top['params']['n_tables']}, "
              f"bigram_p={top['params']['bigram_probability']:.2f}, "
              f"prefix_p={top['params']['prefix_probability']:.2f}, "
              f"suffix_p={top['params']['suffix_probability']:.2f}")

    os.makedirs('output', exist_ok=True)
    report_path = 'output/max_convergence_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full report: {report_path}")

def run_phased_tracks(verbose: bool = True):
    """
    Run the 10 new research tracks (Phases 1-4) independently.
    This is a lightweight runner that doesn't require the full IVTFF corpus.
    """
    from modules import (
        null_framework, word_length,
        positional_shape, fsa_extraction, nmf_analysis, error_model,
        qo_analysis, label_analysis, paragraph_analysis, entropy_gradient,
        constraint_model as cm_module, candidate_search as cs_module,
    )

    start = time.time()
    results = {}

    print("╔" + "═" * 68 + "╗")
    print("║" + " VOYNICH PHASED RESEARCH TRACKS ".center(68) + "║")
    print("║" + " 10 Tracks × 4 Phases ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n" + "▓" * 70)
    print("  PHASE 1: STATISTICAL FOUNDATIONS")
    print("▓" * 70)

    try:
        results['null_framework'] = null_framework.run(verbose=verbose, n_samples=100)
    except Exception as e:
        print(f"  [ERROR] Track 10 (Null Framework): {e}")
        results['null_framework'] = {'error': str(e)}

    try:
        results['word_length'] = word_length.run(verbose=verbose)
    except Exception as e:
        print(f"  [ERROR] Track 4 (Word Length): {e}")
        results['word_length'] = {'error': str(e)}

    print("\n" + "▓" * 70)
    print("  PHASE 2: STRUCTURAL FINGERPRINTING")
    print("▓" * 70)

    tracks_p2 = [
        ('positional_shape', positional_shape, 'Track 1'),
        ('fsa_extraction', fsa_extraction, 'Track 2'),
        ('nmf_analysis', nmf_analysis, 'Track 7'),
        ('error_model', error_model, 'Track 3'),
    ]
    for key, mod, label in tracks_p2:
        try:
            results[key] = mod.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            results[key] = {'error': str(e)}

    print("\n" + "▓" * 70)
    print("  PHASE 3: PLAINTEXT ANCHORS")
    print("▓" * 70)

    tracks_p3 = [
        ('qo_analysis', qo_analysis, 'Track 5'),
        ('label_analysis', label_analysis, 'Track 8'),
        ('paragraph_analysis', paragraph_analysis, 'Track 6'),
        ('entropy_gradient', entropy_gradient, 'Track 9'),
    ]
    for key, mod, label in tracks_p3:
        try:
            results[key] = mod.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            results[key] = {'error': str(e)}

    print("\n" + "▓" * 70)
    print("  PHASE 4: CONSTRAINT INTEGRATION & DECRYPTION")
    print("▓" * 70)

    try:
        cm_result = cm_module.run(verbose=verbose, phase_results=results)
        results['constraint_model'] = cm_result

        anchor_pairs = results.get('label_analysis', {}).get('candidate_pairs', [])
        model_obj = cm_result.get('model')

        results['candidate_search'] = cs_module.run(
            verbose=verbose,
            constraint_model=model_obj,
            anchor_pairs=anchor_pairs,
            n_candidates=50,
        )
    except Exception as e:
        print(f"  [ERROR] Phase 4: {e}")

    elapsed = time.time() - start
    print(f"\n  Total runtime: {elapsed:.1f}s")

    os.makedirs('output', exist_ok=True)
    report_path = 'output/phased_tracks_report.json'
    try:
        save_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                save_results[k] = {sk: sv for sk, sv in v.items()
                                   if sk != 'model'}
            else:
                save_results[k] = v
        with open(report_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"  Report saved to: {report_path}")
    except Exception as e:
        print(f"  [WARN] Could not save: {e}")

    return results

if __name__ == '__main__':
    import sys as _sys
    if '--phased' in _sys.argv or '-p' in _sys.argv:
        run_phased_tracks()
    elif '--phase2' in _sys.argv or '-p2' in _sys.argv:
        from orchestrators.phase2 import run_phase2_attack
        run_phase2_attack(verbose=True)
    else:
        run_max()
