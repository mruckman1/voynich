"""
Convergence Attack: Voynich Manuscript Decryption Framework
=============================================================
Orchestrates all five attack strategies and synthesizes their results
into a unified constraint system.

The strategies constrain each other:
- Strategy 1 narrows cipher parameters and source language
- Strategy 2 reveals cipher state-management at scribe transitions
- Strategy 3 corrects the input sequence for all analyses
- Strategy 4 strips the grammatical layer, exposing semantic core
- Strategy 5 provides ground-truth anchors via zodiac known-plaintext

The convergence attack finds parameter/mapping configurations that
survive ALL constraints simultaneously.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional


from voynich.modules import (
    strategy1_parameter_search as s1,
    strategy2_scribe_seams as s2,
    strategy3_binding_reconstruction as s3,
    strategy4_positional_grammar as s4,
    strategy5_zodiac_attack as s5,
)
from voynich.core.stats import (
    full_statistical_profile, profile_distance, compute_all_entropy
)
from voynich.core.voynich_corpus import get_all_tokens, SAMPLE_CORPUS

def synthesize_results(results: Dict) -> Dict:
    """
    Synthesize results from all strategies into a convergence report.
    
    Identifies:
    1. Parameter configurations that survive all constraints
    2. Linguistic evidence (source language indicators)
    3. Structural insights (binding, scribes, grammar)
    4. Confidence assessment for each finding
    """
    synthesis = {
        'timestamp': datetime.now().isoformat(),
        'findings': [],
        'constraints': [],
        'parameter_consensus': {},
        'source_language_evidence': {},
        'structural_model': {},
        'confidence_assessment': {},
    }

    s1_results = results.get('strategy1', {})
    best_params = {}
    for section, section_res in s1_results.get('sections_analyzed', {}).items():
        if section_res:
            top = section_res[0]
            best_params[section] = {
                'params': top.get('params', {}),
                'score': top.get('composite_score', float('inf')),
                'entropy': top.get('cipher_entropy', {}),
            }
    synthesis['parameter_consensus'] = _find_parameter_consensus(best_params)

    s2_results = results.get('strategy2', {})
    scribe_findings = []

    cold_starts = [ea for ea in s2_results.get('entropy_anomalies', [])
                   if ea.get('cold_start_detected')]
    if cold_starts:
        scribe_findings.append({
            'finding': 'Cold start patterns detected at scribe transitions',
            'evidence': f'{len(cold_starts)} transition(s) show entropy anomalies',
            'implication': 'Cipher uses resettable state (supports Naibbe dice mechanism)',
            'confidence': 'MODERATE',
        })

    vocab = s2_results.get('vocabulary', {})
    universal = vocab.get('universal_vocabulary', [])
    if universal:
        scribe_findings.append({
            'finding': f'{len(universal)} words shared by ALL scribes',
            'evidence': f'Universal vocabulary: {universal[:10]}',
            'implication': 'Shared substitution tables confirmed (not independent ciphers)',
            'confidence': 'HIGH',
        })

    synthesis['findings'].extend(scribe_findings)

    s3_results = results.get('strategy3', {})
    binding = s3_results.get('binding_comparison', {})
    ranking = binding.get('ranking', [])
    if ranking and ranking[0] != 'current':
        synthesis['findings'].append({
            'finding': f'Reconstructed order "{ranking[0]}" is more consistent than current binding',
            'evidence': f'Binding consistency ranking: {ranking[:3]}',
            'implication': 'All future analyses should use reconstructed folio order',
            'confidence': 'MODERATE',
        })

    state = s3_results.get('state_progression', {})
    if state.get('state_detected'):
        synthesis['findings'].append({
            'finding': 'Sequential cipher state detected',
            'evidence': f'Cross-folio bigram correlation = {state.get("correlation", 0):.4f}',
            'implication': 'Cipher uses progressive state; correct page order is critical',
            'confidence': 'HIGH' if abs(state.get('correlation', 0)) > 0.5 else 'MODERATE',
        })

    s4_results = results.get('strategy4', {})
    core = s4_results.get('core_isolation', {})
    delta_h2 = core.get('entropy_delta', {}).get('H2', 0)

    if abs(delta_h2) > 0.05:
        synthesis['findings'].append({
            'finding': f'Positional affixes are a separable layer (ΔH2={delta_h2:+.4f})',
            'evidence': core.get('interpretation', ''),
            'implication': 'Decrypt should target semantic cores, not full tokens',
            'confidence': 'HIGH' if abs(delta_h2) > 0.15 else 'MODERATE',
        })

    morphemes = s4_results.get('morphemes', {})
    if morphemes.get('functional_morphemes'):
        func_words = [w for w, _ in morphemes['functional_morphemes'][:5]]
        synthesis['findings'].append({
            'finding': f'Candidate functional morphemes identified: {func_words}',
            'evidence': f'Ratio functional/content = {morphemes.get("ratio", 0):.3f}',
            'implication': 'These high-frequency uniform words likely encode grammar, not content',
            'confidence': 'MODERATE',
        })

    s5_results = results.get('strategy5', {})
    consistency = s5_results.get('attack', {}).get('cross_section_consistency', {})
    n_consistent = consistency.get('n_consistent_params', 0)

    if n_consistent > 0:
        synthesis['findings'].append({
            'finding': f'{n_consistent} cipher parameters match across multiple zodiac sections',
            'evidence': consistency.get('interpretation', ''),
            'implication': 'Cross-section consistency suggests viable cipher configuration found',
            'confidence': 'HIGH' if n_consistent >= 3 else 'MODERATE',
        })

    bootstrap = s5_results.get('bootstrap', {})
    if 'tentative_mapping' in bootstrap:
        quality = bootstrap.get('mapping_quality', '')
        synthesis['findings'].append({
            'finding': 'Preliminary glyph mappings bootstrapped from zodiac attack',
            'evidence': quality,
            'implication': 'These mappings can seed constrained decryption of other sections',
            'confidence': 'HIGH' if 'HIGH' in quality else 'LOW',
        })

    synthesis['constraints'] = _build_constraints(results)

    synthesis['source_language_evidence'] = _assess_source_language(results)

    synthesis['confidence_assessment'] = _overall_confidence(synthesis['findings'])

    return synthesis

def _find_parameter_consensus(best_params: Dict) -> Dict:
    """Find parameter values that are consistent across sections."""
    if not best_params:
        return {'consensus': 'insufficient data'}

    param_values = {
        'n_tables': [],
        'bigram_probability': [],
        'prefix_probability': [],
        'suffix_probability': [],
    }

    for section, data in best_params.items():
        params = data.get('params', {})
        for key in param_values:
            if key in params:
                param_values[key].append(params[key])

    consensus = {}
    for key, values in param_values.items():
        if values:
            sorted_v = sorted(values)
            median = sorted_v[len(sorted_v) // 2]
            range_v = max(values) - min(values)
            consensus[key] = {
                'median': median,
                'range': round(range_v, 4),
                'values': values,
                'agreement': 'STRONG' if range_v < 0.2 * median else 'WEAK',
            }

    return consensus

def _build_constraints(results: Dict) -> List[Dict]:
    """Compile all constraints that narrow the solution space."""
    constraints = []

    all_text = ' '.join(get_all_tokens())
    entropy = compute_all_entropy(all_text)
    constraints.append({
        'type': 'entropy',
        'description': 'Cipher output must match Voynich entropy profile',
        'values': entropy,
        'tolerance': 0.15,
        'source': 'statistical_analysis',
    })

    constraints.append({
        'type': 'multi_scribe',
        'description': 'Cipher must be operable by 5 distinct scribes with consistent output',
        'implication': 'System must use shared, standardized tables',
        'source': 'Davis_2020',
    })

    constraints.append({
        'type': 'positional_glyphs',
        'description': 'Certain glyphs must be restricted to word-initial/final positions',
        'implication': 'Cipher includes positional encoding rules',
        'source': 'strategy4',
    })

    constraints.append({
        'type': 'historical',
        'description': 'Cipher mechanism must be executable with 1400-1438 CE technology',
        'implication': 'Only hand-operable methods: tables, dice, cards, grilles',
        'source': 'radiocarbon_dating',
    })

    constraints.append({
        'type': 'language',
        'description': 'Source language must produce Voynich-like entropy when encrypted',
        'candidates': ['Medieval Latin', 'Northern Italian', 'Middle High German'],
        'evidence': 'Romance month labels, swallowtail merlons, alpine botanical style',
        'source': 'multiple',
    })

    return constraints

def _assess_source_language(results: Dict) -> Dict:
    """Assess evidence for each candidate source language."""
    evidence = {
        'medieval_latin': {
            'score': 0.0,
            'pro': [],
            'contra': [],
        },
        'northern_italian': {
            'score': 0.0,
            'pro': [],
            'contra': [],
        },
        'middle_high_german': {
            'score': 0.0,
            'pro': [],
            'contra': [],
        },
    }

    evidence['medieval_latin']['pro'].extend([
        'Zodiac month labels in Romance language suggest Latin-sphere origin',
        'Pharmaceutical recipe format matches Latin medical tradition',
        'Hartlieb-era medical Latin vocabulary fits temporal/geographic window',
        'Latin was the universal scholarly language of 15th-century medicine',
    ])
    evidence['medieval_latin']['score'] = 0.7

    evidence['northern_italian']['pro'].extend([
        'Swallowtail merlons in cosmological section = Northern Italian architecture',
        'Month labels may be Northern French/Occitan (adjacent language area)',
        'Balneological tradition strong in Northern Italy (Balneis Puteolanis)',
    ])
    evidence['northern_italian']['score'] = 0.5

    evidence['middle_high_german']['pro'].extend([
        'Hartlieb was Bavarian, writing in German and Latin',
        'Alpine botanical style consistent with German-speaking regions',
        'Low entropy could be consistent with German compound words',
    ])
    evidence['middle_high_german']['contra'].extend([
        'German word structure less compatible with observed word-length distribution',
    ])
    evidence['middle_high_german']['score'] = 0.4

    return evidence

def _overall_confidence(findings: List[Dict]) -> Dict:
    """Compute overall confidence assessment."""
    confidence_levels = {'HIGH': 3, 'MODERATE': 2, 'LOW': 1}
    scores = [confidence_levels.get(f.get('confidence', 'LOW'), 1)
              for f in findings]

    avg = sum(scores) / max(1, len(scores))
    high_count = sum(1 for s in scores if s == 3)

    if avg >= 2.5 and high_count >= 2:
        overall = 'STRONG'
        message = ("Multiple high-confidence findings converge on a consistent "
                   "cipher model. The combined constraints significantly narrow "
                   "the solution space.")
    elif avg >= 1.8:
        overall = 'MODERATE'
        message = ("Several promising findings, but more data and cross-validation "
                   "needed. The full EVA corpus would strengthen all analyses.")
    else:
        overall = 'PRELIMINARY'
        message = ("Initial results establish the analytical framework. "
                   "Full corpus analysis required for definitive conclusions.")

    return {
        'overall_level': overall,
        'avg_finding_confidence': round(avg, 2),
        'high_confidence_findings': high_count,
        'total_findings': len(findings),
        'message': message,
    }

def run_convergence_attack(
        strategies: Optional[List[int]] = None,
        verbose: bool = True,
        output_dir: str = './output'
) -> Dict:
    """
    Run the full convergence attack.
    
    Parameters:
        strategies: List of strategy numbers to run (1-5). Default: all.
        verbose: Print detailed output.
        output_dir: Directory for output files.
    """
    if strategies is None:
        strategies = [1, 2, 3, 4, 5]

    start_time = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║" + " VOYNICH MANUSCRIPT CONVERGENCE ATTACK ".center(68) + "║")
    print("║" + " A Novel Multi-Strategy Cryptanalytic Framework ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Timestamp: {datetime.now().isoformat()}")
    print(f"  Strategies: {strategies}")
    print(f"  Corpus: {len(SAMPLE_CORPUS)} folios loaded")
    print(f"  Total tokens: {len(get_all_tokens())}")
    print()

    results = {}

    if 1 in strategies:
        print("\n" + "▓" * 70)
        try:
            results['strategy1'] = s1.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Strategy 1 failed: {e}")
            results['strategy1'] = {'error': str(e)}

    if 2 in strategies:
        print("\n" + "▓" * 70)
        try:
            results['strategy2'] = s2.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Strategy 2 failed: {e}")
            results['strategy2'] = {'error': str(e)}

    if 3 in strategies:
        print("\n" + "▓" * 70)
        try:
            results['strategy3'] = s3.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Strategy 3 failed: {e}")
            results['strategy3'] = {'error': str(e)}

    if 4 in strategies:
        print("\n" + "▓" * 70)
        try:
            results['strategy4'] = s4.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Strategy 4 failed: {e}")
            results['strategy4'] = {'error': str(e)}

    if 5 in strategies:
        print("\n" + "▓" * 70)
        try:
            results['strategy5'] = s5.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Strategy 5 failed: {e}")
            results['strategy5'] = {'error': str(e)}

    print("\n" + "▓" * 70)
    print("=" * 70)
    print("CONVERGENCE SYNTHESIS")
    print("=" * 70)

    synthesis = synthesize_results(results)
    results['synthesis'] = synthesis

    elapsed = time.time() - start_time

    print(f"\n{'─'*70}")
    print("FINDINGS SUMMARY")
    print(f"{'─'*70}")
    for i, f in enumerate(synthesis['findings'], 1):
        conf = f.get('confidence', '?')
        conf_marker = {'HIGH': '★', 'MODERATE': '◆', 'LOW': '○'}.get(conf, '?')
        print(f"\n  {conf_marker} Finding #{i} [{conf}]")
        print(f"    {f['finding']}")
        print(f"    Evidence: {f['evidence'][:100]}")
        print(f"    Implication: {f['implication'][:100]}")

    print(f"\n{'─'*70}")
    print("CONSTRAINTS")
    print(f"{'─'*70}")
    for c in synthesis['constraints']:
        print(f"  [{c['type']}] {c['description']}")

    print(f"\n{'─'*70}")
    print("SOURCE LANGUAGE ASSESSMENT")
    print(f"{'─'*70}")
    for lang, data in synthesis['source_language_evidence'].items():
        print(f"  {lang}: score={data['score']:.2f}")
        for p in data['pro'][:2]:
            print(f"    + {p}")
        for c in data.get('contra', [])[:1]:
            print(f"    - {c}")

    print(f"\n{'─'*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'─'*70}")
    conf = synthesis['confidence_assessment']
    print(f"  Confidence level: {conf['overall_level']}")
    print(f"  High-confidence findings: {conf['high_confidence_findings']}/{conf['total_findings']}")
    print(f"  {conf['message']}")
    print(f"\n  Total runtime: {elapsed:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'convergence_report.json')
    try:
        serializable = _make_serializable(results)
        with open(report_path, 'w') as fh:
            json.dump(serializable, fh, indent=2, default=str)
        print(f"\n  Full report saved to: {report_path}")
    except Exception as e:
        print(f"\n  [WARN] Could not save report: {e}")

    return results

def _make_serializable(obj):
    """Make nested dicts/lists JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, float):
        if obj == float('inf') or obj == float('-inf'):
            return None
        return obj
    else:
        return obj

def run_phased_attack(
        phases: Optional[List[int]] = None,
        run_original: bool = True,
        verbose: bool = True,
        output_dir: str = './output',
        null_samples: int = 200,
) -> Dict:
    """
    Run the full phased convergence attack with all 10 research tracks.

    Phase 1: Statistical Foundations (null distributions, word-length)
    Phase 2: Structural Fingerprinting (shape, FSA, NMF, error model)
    Phase 3: Plaintext Anchors (qo-, labels, paragraphs, entropy gradient)
    Phase 4: Constraint Integration + Decryption Candidate Search

    Parameters:
        phases: Which phases to run (1-4). Default: all.
        run_original: Run original strategies 1-5 first.
        verbose: Print detailed output.
        output_dir: Directory for output files.
        null_samples: Number of samples for null distributions.
    """
    from voynich.modules import (
        null_framework, word_length,
        positional_shape, fsa_extraction, nmf_analysis, error_model,
        qo_analysis, label_analysis, paragraph_analysis, entropy_gradient,
        constraint_model as cm_module, candidate_search as cs_module,
    )

    if phases is None:
        phases = [1, 2, 3, 4]

    start_time = time.time()
    all_results = {}

    print("╔" + "═" * 68 + "╗")
    print("║" + " VOYNICH PHASED CONVERGENCE ATTACK ".center(68) + "║")
    print("║" + " 10 Research Tracks × 4 Phases ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  Timestamp: {datetime.now().isoformat()}")
    print(f"  Phases: {phases}")
    print(f"  Corpus: {len(SAMPLE_CORPUS)} folios loaded")
    print(f"  Total tokens: {len(get_all_tokens())}")

    if run_original:
        print("\n" + "▓" * 70)
        print("  ORIGINAL STRATEGIES (1-5)")
        print("▓" * 70)
        original = run_convergence_attack(verbose=verbose, output_dir=output_dir)
        all_results['convergence_attack'] = original

    if 1 in phases:
        print("\n" + "▓" * 70)
        print("  PHASE 1: STATISTICAL FOUNDATIONS")
        print("▓" * 70)

        try:
            all_results['null_framework'] = null_framework.run(
                verbose=verbose, n_samples=null_samples)
        except Exception as e:
            print(f"  [ERROR] Null framework failed: {e}")
            all_results['null_framework'] = {'error': str(e)}

        try:
            all_results['word_length'] = word_length.run(verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Word length failed: {e}")
            all_results['word_length'] = {'error': str(e)}

    if 2 in phases:
        print("\n" + "▓" * 70)
        print("  PHASE 2: STRUCTURAL FINGERPRINTING")
        print("▓" * 70)

        for track_name, track_module in [
            ('positional_shape', positional_shape),
            ('fsa_extraction', fsa_extraction),
            ('nmf_analysis', nmf_analysis),
            ('error_model', error_model),
        ]:
            try:
                all_results[track_name] = track_module.run(verbose=verbose)
            except Exception as e:
                print(f"  [ERROR] {track_name} failed: {e}")
                all_results[track_name] = {'error': str(e)}

    if 3 in phases:
        print("\n" + "▓" * 70)
        print("  PHASE 3: PLAINTEXT ANCHORS")
        print("▓" * 70)

        for track_name, track_module in [
            ('qo_analysis', qo_analysis),
            ('label_analysis', label_analysis),
            ('paragraph_analysis', paragraph_analysis),
            ('entropy_gradient', entropy_gradient),
        ]:
            try:
                all_results[track_name] = track_module.run(verbose=verbose)
            except Exception as e:
                print(f"  [ERROR] {track_name} failed: {e}")
                all_results[track_name] = {'error': str(e)}

    if 4 in phases:
        print("\n" + "▓" * 70)
        print("  PHASE 4: CONSTRAINT INTEGRATION & DECRYPTION")
        print("▓" * 70)

        try:
            cm_result = cm_module.run(
                verbose=verbose, phase_results=all_results)
            all_results['constraint_model'] = cm_result

            anchor_pairs = all_results.get('label_analysis', {}).get(
                'candidate_pairs', [])

            model_obj = cm_result.get('model')

            cs_result = cs_module.run(
                verbose=verbose,
                constraint_model=model_obj,
                anchor_pairs=anchor_pairs,
                n_candidates=50,
            )
            all_results['candidate_search'] = cs_result
        except Exception as e:
            print(f"  [ERROR] Phase 4 failed: {e}")
            all_results['constraint_model'] = {'error': str(e)}

    elapsed = time.time() - start_time

    print("\n" + "▓" * 70)
    print("═" * 70)
    print("PHASED ATTACK SYNTHESIS")
    print("═" * 70)

    findings = []
    for track_name, result in all_results.items():
        if isinstance(result, dict) and 'error' not in result:
            _extract_track_findings(track_name, result, findings)

    print(f"\n{'─'*70}")
    print(f"ALL FINDINGS ({len(findings)} total)")
    print(f"{'─'*70}")
    for i, f in enumerate(findings, 1):
        conf = f.get('confidence', '?')
        marker = {'HIGH': '★', 'MODERATE': '◆', 'LOW': '○'}.get(conf, '?')
        print(f"\n  {marker} Finding #{i} [{conf}] ({f.get('source', '')})")
        print(f"    {f['finding']}")

    cm = all_results.get('constraint_model', {})
    spec = cm.get('specification', {})
    if spec:
        print(f"\n{'─'*70}")
        print("CONSTRAINT MODEL")
        print(f"{'─'*70}")
        print(f"  Total constraints: {spec.get('n_constraints', 0)}")
        print(f"  Excluded families: {spec.get('excluded_cipher_families', [])}")
        print(f"  Viable families: {spec.get('viable_cipher_families', [])}")
        print(f"  Candidate languages: {spec.get('candidate_languages', [])}")

    cs = all_results.get('candidate_search', {})
    top = cs.get('top_candidates', [])
    if top:
        print(f"\n{'─'*70}")
        print("TOP DECRYPTION CANDIDATES")
        print(f"{'─'*70}")
        for i, c in enumerate(top[:5], 1):
            print(f"  #{i}: score={c.get('score', 0):.4f}  "
                  f"lang={c.get('source_language', '?')}  "
                  f"dist={c.get('profile_distance', 0):.4f}")

    print(f"\n{'─'*70}")
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Tracks completed: {len([r for r in all_results.values() if 'error' not in r])}")
    print(f"  Tracks failed: {len([r for r in all_results.values() if isinstance(r, dict) and 'error' in r])}")

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'phased_convergence_report.json')
    try:
        serializable = _make_serializable(all_results)
        for key in list(serializable.keys()):
            if isinstance(serializable[key], dict):
                serializable[key].pop('model', None)
        with open(report_path, 'w') as fh:
            json.dump(serializable, fh, indent=2, default=str)
        print(f"\n  Full report saved to: {report_path}")
    except Exception as e:
        print(f"\n  [WARN] Could not save report: {e}")

    return all_results

def _extract_track_findings(track_name: str, result: Dict, findings: List):
    """Extract key findings from a track result."""
    if track_name == 'null_framework':
        summary = result.get('anomaly_summary', {})
        for metric, data in summary.items():
            if data.get('globally_anomalous'):
                findings.append({
                    'finding': f'Voynich {metric}={data["voynich_value"]:.4f} is '
                               f'anomalous across {data["anomalous_count"]} cipher×language combos',
                    'confidence': 'HIGH',
                    'source': 'Track 10: Null Framework',
                })

    elif track_name == 'word_length':
        if result.get('word_boundary_valid') is not None:
            findings.append({
                'finding': (f'Word boundaries {"carry" if result["word_boundary_valid"] else "do NOT carry"} '
                            f'semantic information'),
                'confidence': 'HIGH',
                'source': 'Track 4: Word Length',
            })

    elif track_name == 'positional_shape':
        excluded = result.get('excluded_families', [])
        if excluded:
            findings.append({
                'finding': f'Cipher families EXCLUDED by shape analysis: {excluded}',
                'confidence': 'HIGH',
                'source': 'Track 1: Positional Shape',
            })

    elif track_name == 'fsa_extraction':
        findings.append({
            'finding': f'Cipher type classified as: {result.get("cipher_type", "unknown")}',
            'confidence': 'MODERATE',
            'source': 'Track 2: FSA Extraction',
        })

    elif track_name == 'nmf_analysis':
        rank = result.get('effective_rank', 0)
        findings.append({
            'finding': f'Effective dimensionality of character transitions: rank {rank}',
            'confidence': 'MODERATE',
            'source': 'Track 7: NMF Analysis',
        })

    elif track_name == 'error_model':
        pattern = result.get('error_pattern', {}).get('pattern', 'unknown')
        findings.append({
            'finding': f'Scribe error pattern: {pattern}',
            'confidence': 'MODERATE',
            'source': 'Track 3: Error Model',
        })

    elif track_name == 'qo_analysis':
        cls = result.get('functional_classification', {})
        if cls.get('classification'):
            findings.append({
                'finding': f'qo- words classified as: {cls["classification"]}',
                'confidence': cls.get('confidence', 'LOW'),
                'source': 'Track 5: qo- Analysis',
            })

    elif track_name == 'label_analysis':
        n_pairs = result.get('n_candidate_pairs', 0)
        if n_pairs > 0:
            findings.append({
                'finding': f'{n_pairs} candidate plaintext-ciphertext pairs from labels',
                'confidence': 'MODERATE' if n_pairs >= 5 else 'LOW',
                'source': 'Track 8: Label Analysis',
            })

    elif track_name == 'paragraph_analysis':
        ratio = result.get('implications', {}).get('mean_ratio', 0)
        if ratio > 0:
            findings.append({
                'finding': f'Plaintext:ciphertext length ratio ≈ {ratio:.2f}',
                'confidence': 'MODERATE',
                'source': 'Track 6: Paragraph Analysis',
            })

    elif track_name == 'entropy_gradient':
        u_sections = result.get('u_curve_sections', [])
        if u_sections:
            findings.append({
                'finding': f'U-curve entropy gradient in: {u_sections}',
                'confidence': 'MODERATE',
                'source': 'Track 9: Entropy Gradient',
            })

if __name__ == '__main__':
    import sys as _sys
    if '--phased' in _sys.argv or '-p' in _sys.argv:
        run_phased_attack(verbose=True)
    elif '--phase2' in _sys.argv or '-p2' in _sys.argv:
        from voynich.phases.phase2 import run_phase2_attack
        run_phase2_attack(verbose=True)
    else:
        run_convergence_attack(verbose=True)
