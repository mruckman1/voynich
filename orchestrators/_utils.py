"""Shared utility functions for all phase orchestrators."""
import json
import os
from datetime import datetime
from typing import Any, Dict

def vprint(verbose, *args, **kwargs):
    """Print only when verbose is True."""
    if verbose:
        print(*args, **kwargs)

def save_json(filepath: str, data: Any) -> None:
    """Save results to JSON, handling non-serializable types.

    Consolidates the duplicated _save_json / inline json.dump calls
    from convergence_attack_p2 through p12. Uses the superset of all
    edge-case handlers found across the original files.
    """
    def default_handler(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, float):
            if obj != obj:
                return None
            if obj == float('inf') or obj == float('-inf'):
                return str(obj)
        return str(obj)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=default_handler)

def ensure_output_dir(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)

def make_results_header(**extra_fields) -> Dict:
    """Create the standard results dict header with timestamp."""
    result = {'timestamp': datetime.now().isoformat()}
    result.update(extra_fields)
    return result

PHASE_DESCRIPTIONS = {
    1:  '5-Strategy Convergence Attack',
    2:  'Super-Character Generative Models',
    3:  'Language B First',
    4:  'Language A Decryption Attack',
    5:  'Splitting the Nomenclator',
    6:  'Three Recovery Paths',
    7:  'Morphological Sub-Word Attack',
    8:  'Viterbi Translation Engine',
    9:  'Syllabic & Sigla Translation Engine',
    10: 'Dictionary-Guided Trigram Decoder',
    11: 'Phonetic Constraint Satisfaction Decoder',
    12:    'Contextual Reconstruction & Deterministic Mask Solving',
    '12.5': 'Adversarial Defense Suite',
    13:    'Scholarly Synthesis & Presentation',
}

def build_combined_report(phase_results: Dict[int, Dict], output_dir: str) -> str:
    """Build and save a combined report from all phases that were run.

    Each phase entry includes a description and the full result data.
    Returns the filepath of the saved report.
    """
    ensure_output_dir(output_dir)

    phases_run = sorted(phase_results.keys(),
                        key=lambda x: (isinstance(x, str), str(x)))
    total_elapsed = sum(
        r.get('elapsed_seconds', 0) for r in phase_results.values()
    )

    report = {
        'timestamp': datetime.now().isoformat(),
        'phases_run': phases_run,
        'total_elapsed_seconds': round(total_elapsed, 2),
    }

    for phase_num in phases_run:
        entry = {'description': PHASE_DESCRIPTIONS.get(phase_num, f'Phase {phase_num}')}
        entry.update(phase_results[phase_num])
        report[f'phase_{phase_num}'] = entry

    filepath = os.path.join(output_dir, 'combined_report.json')
    save_json(filepath, report)
    return filepath
