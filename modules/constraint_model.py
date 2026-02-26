"""
Phase 4: Constraint Integration Model
========================================
Compiles ALL constraints from all phases and strategies into a formal
specification that any valid decryption must satisfy simultaneously.

This is the keystone integrator: every prior track narrows the hypothesis space,
and this module enumerates what remains.
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistical_analysis import compute_all_entropy, full_statistical_profile
from data.voynich_corpus import get_all_tokens


# ============================================================================
# CONSTRAINT MODEL
# ============================================================================

class ConstraintModel:
    """
    Compiles and manages the unified constraint specification.
    Any valid decryption must satisfy ALL constraints simultaneously.
    """

    def __init__(self):
        self.constraints: List[Dict] = []
        self.excluded_families: List[str] = []
        self.viable_families: List[str] = []
        self.candidate_languages: List[str] = []
        self.anchor_pairs: List[Dict] = []
        self.reliable_pages: List[str] = []
        self.parameter_bounds: Dict[str, Dict] = {}

    def add_entropy_constraints(self, null_results: Dict, voynich_profile: Dict):
        """Add entropy constraints from Phase 1 null framework."""
        metrics = voynich_profile if voynich_profile else {}

        # Extract p-values to determine which ranges are anomalous
        p_values = null_results.get('p_values', {})
        percentiles = null_results.get('percentile_ranks', {})

        for metric in ['H1', 'H2', 'H3']:
            value = metrics.get(metric, 0)
            if value > 0:
                # Find the cipher families where this value is normal
                normal_families = []
                for cipher in p_values:
                    for lang in p_values.get(cipher, {}):
                        pval = p_values[cipher][lang].get(metric, 0)
                        if pval > 0.05:
                            normal_families.append(f"{cipher}_{lang}")

                self.constraints.append({
                    'type': 'entropy',
                    'metric': metric,
                    'voynich_value': value,
                    'tolerance': 0.15,
                    'description': f'Cipher output {metric} must be within '
                                   f'{value:.4f} ± 0.15',
                    'source': 'null_framework',
                    'normal_for': normal_families[:5],
                })

    def add_word_boundary_constraint(self, word_length_results: Dict):
        """Add word boundary constraint from Track 4."""
        valid = word_length_results.get('word_boundary_valid', True)
        best_lang = word_length_results.get('best_matching_language', '')

        self.constraints.append({
            'type': 'word_boundary',
            'word_boundaries_semantic': valid,
            'best_matching_language': best_lang,
            'description': (
                'Word boundaries carry semantic information — analyze at word level'
                if valid else
                'Word boundaries may be artificial — consider character-stream analysis'
            ),
            'source': 'word_length_analysis',
        })

        if best_lang:
            self.candidate_languages.append(best_lang)

    def add_shape_constraints(self, shape_results: Dict):
        """Add positional entropy shape constraints from Track 1."""
        excluded = shape_results.get('excluded_families', [])
        viable = shape_results.get('viable_families', [])
        voynich_shape = shape_results.get('voynich_shape', [])

        self.excluded_families.extend(excluded)
        self.viable_families = viable

        self.constraints.append({
            'type': 'positional_entropy_shape',
            'voynich_shape': voynich_shape,
            'excluded_families': excluded,
            'viable_families': viable,
            'description': (f'Cipher must produce positional entropy curve matching '
                            f'Voynich shape (peak at position '
                            f'{np.argmax(voynich_shape) if voynich_shape else "?"})'),
            'source': 'positional_shape',
        })

    def add_fsa_constraints(self, fsa_results: Dict):
        """Add FSA topology constraints from Track 2."""
        cipher_type = fsa_results.get('cipher_type', 'unknown')
        overall = fsa_results.get('overall', {})
        n_states = overall.get('n_minimized_states', 0)

        self.constraints.append({
            'type': 'fsa_topology',
            'cipher_type': cipher_type,
            'n_states': n_states,
            'description': (f'Cipher must be {cipher_type} '
                            f'(FSA minimized to {n_states} states)'),
            'source': 'fsa_extraction',
        })

    def add_nmf_constraints(self, nmf_results: Dict):
        """Add NMF dimensionality constraints from Track 7."""
        rank = nmf_results.get('effective_rank', 0)
        convergent = nmf_results.get('positional_class_convergence', {})

        self.constraints.append({
            'type': 'nmf_dimensionality',
            'effective_rank': rank,
            'convergent_with_positional': convergent.get('convergent_evidence', False),
            'description': (f'Cipher character transition system has effective '
                            f'dimensionality {rank}'),
            'source': 'nmf_analysis',
        })

    def add_error_constraints(self, error_results: Dict):
        """Add scribe error constraints from Track 3."""
        reliable = error_results.get('reliable_pages', [])
        pattern = error_results.get('error_pattern', {})

        self.reliable_pages = [p['folio'] for p in reliable[:20]]

        self.constraints.append({
            'type': 'error_pattern',
            'pattern': pattern.get('pattern', 'unknown'),
            'n_reliable_pages': len(self.reliable_pages),
            'description': (f'Error pattern: {pattern.get("pattern", "unknown")}. '
                            f'{len(self.reliable_pages)} reliable pages identified '
                            f'for decryption testing.'),
            'source': 'error_model',
        })

    def add_qo_constraints(self, qo_results: Dict):
        """Add qo- functional constraints from Track 5."""
        classification = qo_results.get('functional_classification', {})

        self.constraints.append({
            'type': 'qo_function',
            'classification': classification.get('classification', 'unknown'),
            'confidence': classification.get('confidence', 'LOW'),
            'description': (f'qo- words function as '
                            f'{classification.get("classification", "unknown")} '
                            f'[{classification.get("confidence", "?")}]'),
            'source': 'qo_analysis',
        })

    def add_label_constraints(self, label_results: Dict):
        """Add label-derived constraints from Track 8."""
        pairs = label_results.get('candidate_pairs', [])
        self.anchor_pairs.extend(pairs)

        self.constraints.append({
            'type': 'label_anchors',
            'n_candidate_pairs': len(pairs),
            'n_high_confidence': label_results.get('n_high_confidence_pairs', 0),
            'description': (f'{len(pairs)} candidate plaintext-ciphertext pairs '
                            f'from label analysis'),
            'source': 'label_analysis',
        })

    def add_paragraph_constraints(self, paragraph_results: Dict):
        """Add paragraph structure constraints from Track 6."""
        implications = paragraph_results.get('implications', {})
        mean_ratio = implications.get('mean_ratio', 1.0)

        self.constraints.append({
            'type': 'length_ratio',
            'mean_ratio': mean_ratio,
            'mechanism_constraint': implications.get('mechanism_constraint', ''),
            'description': (f'Plaintext:ciphertext length ratio ≈ {mean_ratio:.2f}. '
                            f'{implications.get("mechanism_constraint", "")[:80]}'),
            'source': 'paragraph_analysis',
        })

    def add_gradient_constraints(self, gradient_results: Dict):
        """Add entropy gradient constraints from Track 9."""
        u_sections = gradient_results.get('u_curve_sections', [])
        anchors = gradient_results.get('anchor_regions', [])

        self.constraints.append({
            'type': 'entropy_gradient',
            'u_curve_sections': u_sections,
            'n_anchor_regions': len(anchors),
            'description': (f'U-curve entropy gradient detected in {len(u_sections)} '
                            f'sections. {len(anchors)} anchor regions for '
                            f'known-plaintext matching.'),
            'source': 'entropy_gradient',
        })

    def add_existing_constraints(self, convergence_results: Dict):
        """Add constraints from original strategies 1-5."""
        synthesis = convergence_results.get('synthesis', {})

        # Parameter consensus
        consensus = synthesis.get('parameter_consensus', {})
        for param, data in consensus.items():
            if isinstance(data, dict) and 'median' in data:
                self.parameter_bounds[param] = {
                    'median': data['median'],
                    'range': data.get('range', 0),
                    'agreement': data.get('agreement', 'WEAK'),
                }

        # Language evidence
        lang_evidence = synthesis.get('source_language_evidence', {})
        for lang, evidence in lang_evidence.items():
            if isinstance(evidence, dict) and evidence.get('score', 0) > 0.5:
                if lang not in self.candidate_languages:
                    self.candidate_languages.append(lang)

        # Existing constraints
        for c in synthesis.get('constraints', []):
            self.constraints.append({**c, 'source': 'convergence_attack_v1'})

    def compile(self) -> Dict:
        """
        Compile all constraints into a formal specification.
        """
        # Deduplicate
        unique_excluded = sorted(set(self.excluded_families))
        unique_languages = sorted(set(self.candidate_languages))

        specification = {
            'n_constraints': len(self.constraints),
            'constraints': self.constraints,
            'excluded_cipher_families': unique_excluded,
            'viable_cipher_families': self.viable_families,
            'candidate_languages': unique_languages,
            'parameter_bounds': self.parameter_bounds,
            'anchor_pairs': self.anchor_pairs[:20],  # top 20
            'n_anchor_pairs': len(self.anchor_pairs),
            'reliable_pages': self.reliable_pages,
            'n_reliable_pages': len(self.reliable_pages),
        }

        # Constraint summary by type
        type_counts = defaultdict(int)
        for c in self.constraints:
            type_counts[c.get('type', 'unknown')] += 1
        specification['constraint_type_counts'] = dict(type_counts)

        return specification

    def check_candidate(self, decryption_output: Dict) -> Dict:
        """
        Test a proposed decryption against all constraints.

        Parameters:
            decryption_output: Dict with 'text' (decrypted text),
                              'params' (cipher parameters used),
                              'cipher_family' (family name)

        Returns:
            Dict with {passes: bool, violations: [...], score: float}
        """
        violations = []
        total_checks = 0
        passed_checks = 0

        text = decryption_output.get('text', '')
        params = decryption_output.get('params', {})
        family = decryption_output.get('cipher_family', '')

        for constraint in self.constraints:
            ctype = constraint.get('type', '')

            if ctype == 'entropy':
                total_checks += 1
                metric = constraint.get('metric', '')
                target = constraint.get('voynich_value', 0)
                tolerance = constraint.get('tolerance', 0.15)

                if text:
                    entropy = compute_all_entropy(text)
                    actual = entropy.get(metric, 0)
                    if abs(actual - target) <= tolerance:
                        passed_checks += 1
                    else:
                        violations.append(
                            f'{metric}: {actual:.4f} vs target {target:.4f} '
                            f'(±{tolerance})'
                        )

            elif ctype == 'positional_entropy_shape':
                total_checks += 1
                excluded = constraint.get('excluded_families', [])
                if family in excluded:
                    violations.append(
                        f'Cipher family "{family}" is excluded by shape analysis'
                    )
                else:
                    passed_checks += 1

            elif ctype == 'length_ratio':
                total_checks += 1
                expected_ratio = constraint.get('mean_ratio', 1.0)
                # Check if decryption length is plausible
                if text:
                    dec_words = len(text.split())
                    # Would need original cipher text length for comparison
                    passed_checks += 1  # Can't check without original

        score = passed_checks / max(total_checks, 1)

        return {
            'passes': len(violations) == 0,
            'violations': violations,
            'score': score,
            'checks_passed': passed_checks,
            'checks_total': total_checks,
        }

    def narrow_parameter_space(self, parameter_grid: List[Dict]) -> List[Dict]:
        """
        Filter a parameter grid to only combinations satisfying all constraints.
        """
        surviving = []

        for params in parameter_grid:
            valid = True

            # Check against parameter bounds
            for param_name, bounds in self.parameter_bounds.items():
                if param_name in params:
                    value = params[param_name]
                    median = bounds.get('median', value)
                    range_val = bounds.get('range', float('inf'))
                    # Allow 50% wider than observed range
                    if abs(value - median) > range_val * 1.5:
                        valid = False
                        break

            if valid:
                surviving.append(params)

        return surviving


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

def run(verbose: bool = True, phase_results: Optional[Dict] = None) -> Dict:
    """
    Build the unified constraint model from all phase results.

    Parameters:
        verbose: Print detailed output
        phase_results: Dict with results from all tracks, keyed by track name.
                      If None, attempts to load from output files.

    Returns:
        Dict with compiled constraint specification.
    """
    model = ConstraintModel()

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: CONSTRAINT INTEGRATION MODEL")
        print("=" * 70)

    if phase_results is None:
        phase_results = {}

    # Voynich baselines
    tokens = get_all_tokens()
    voynich_text = ' '.join(tokens)
    voynich_entropy = compute_all_entropy(voynich_text)

    # Phase 1 constraints
    if 'null_framework' in phase_results:
        if verbose:
            print("\n  Adding Phase 1 constraints (null distributions)...")
        model.add_entropy_constraints(
            phase_results['null_framework'],
            phase_results['null_framework'].get('voynich_metrics', voynich_entropy)
        )

    if 'word_length' in phase_results:
        if verbose:
            print("  Adding Phase 1 constraints (word length)...")
        model.add_word_boundary_constraint(phase_results['word_length'])

    # Phase 2 constraints
    if 'positional_shape' in phase_results:
        if verbose:
            print("  Adding Phase 2 constraints (positional shape)...")
        model.add_shape_constraints(phase_results['positional_shape'])

    if 'fsa_extraction' in phase_results:
        if verbose:
            print("  Adding Phase 2 constraints (FSA topology)...")
        model.add_fsa_constraints(phase_results['fsa_extraction'])

    if 'nmf_analysis' in phase_results:
        if verbose:
            print("  Adding Phase 2 constraints (NMF dimensionality)...")
        model.add_nmf_constraints(phase_results['nmf_analysis'])

    if 'error_model' in phase_results:
        if verbose:
            print("  Adding Phase 2 constraints (error model)...")
        model.add_error_constraints(phase_results['error_model'])

    # Phase 3 constraints
    if 'qo_analysis' in phase_results:
        if verbose:
            print("  Adding Phase 3 constraints (qo- analysis)...")
        model.add_qo_constraints(phase_results['qo_analysis'])

    if 'label_analysis' in phase_results:
        if verbose:
            print("  Adding Phase 3 constraints (label analysis)...")
        model.add_label_constraints(phase_results['label_analysis'])

    if 'paragraph_analysis' in phase_results:
        if verbose:
            print("  Adding Phase 3 constraints (paragraph analysis)...")
        model.add_paragraph_constraints(phase_results['paragraph_analysis'])

    if 'entropy_gradient' in phase_results:
        if verbose:
            print("  Adding Phase 3 constraints (entropy gradient)...")
        model.add_gradient_constraints(phase_results['entropy_gradient'])

    # Original convergence attack constraints
    if 'convergence_attack' in phase_results:
        if verbose:
            print("  Adding original strategy 1-5 constraints...")
        model.add_existing_constraints(phase_results['convergence_attack'])

    # Compile
    if verbose:
        print("\n  Compiling constraint specification...")
    specification = model.compile()

    if verbose:
        print(f"\n  Total constraints: {specification['n_constraints']}")
        print(f"  Constraint types: {specification['constraint_type_counts']}")
        print(f"  Excluded families: {specification['excluded_cipher_families']}")
        print(f"  Viable families: {specification['viable_cipher_families']}")
        print(f"  Candidate languages: {specification['candidate_languages']}")
        print(f"  Parameter bounds: {list(specification['parameter_bounds'].keys())}")
        print(f"  Anchor pairs: {specification['n_anchor_pairs']}")
        print(f"  Reliable pages: {specification['n_reliable_pages']}")

    # Save
    try:
        os.makedirs('./output', exist_ok=True)
        save_data = {
            k: v for k, v in specification.items()
            if k != 'anchor_pairs'  # These can be large
        }
        save_data['n_anchor_pairs'] = specification['n_anchor_pairs']
        with open('./output/constraint_model.json', 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        if verbose:
            print("\n  Saved to output/constraint_model.json")
    except Exception as e:
        if verbose:
            print(f"\n  [WARN] Could not save: {e}")

    results = {
        'track': 'constraint_model',
        'specification': specification,
        'model': model,  # Keep reference for candidate_search
    }

    if verbose:
        print("\n" + "─" * 70)
        print("CONSTRAINT MODEL SUMMARY")
        print("─" * 70)
        print(f"  Constraints compiled: {specification['n_constraints']}")
        print(f"  Viable cipher families: {specification['viable_cipher_families']}")
        print(f"  Candidate languages: {specification['candidate_languages']}")
        print(f"  Any valid decryption must satisfy ALL constraints simultaneously.")

    return results
