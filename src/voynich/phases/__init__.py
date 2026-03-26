"""Phase orchestrator registry.

Provides a single import point and lazy-loaded registry so that
cli.py and convergence_attack.py can dispatch to any phase by
number without importing all modules eagerly.
"""
from typing import Callable, Dict, Union

_PHASE_REGISTRY = {
    1:      ('voynich.phases.phase1',      'run_convergence_attack'),
    2:      ('voynich.phases.phase2',      'run_phase2_attack'),
    3:      ('voynich.phases.phase3',      'run_phase3_attack'),
    4:      ('voynich.phases.phase4',      'run_phase4_attack'),
    5:      ('voynich.phases.phase5',      'run_phase5_attack'),
    6:      ('voynich.phases.phase6',      'run_phase6_attack'),
    7:      ('voynich.phases.phase7',      'run_phase7_attack'),
    8:      ('voynich.phases.phase8',      'run_phase8_translation'),
    9:      ('voynich.phases.phase9',      'run_phase9_attack'),
    10:     ('voynich.phases.phase10',     'run_phase10_final_translation'),
    11:     ('voynich.phases.phase11',     'run_phase11_csp_translation'),
    12:     ('voynich.phases.phase12',     'run_phase12_reconstruction'),
    '12.5': ('voynich.phases.phase12_5',   'run_phase12_5_adversarial'),
    13:     ('voynich.phases.phase13',     'run_phase13_synthesis'),
    14:     ('voynich.phases.phase14',     'run_phase14'),
    'robustness': ('voynich.phases.robustness', 'run_robustness_tests'),
}

def get_phase_runner(phase_num: Union[int, str]) -> Callable:
    """Return the run function for a given phase, importing lazily.

    Accepts int (e.g. 12) or str (e.g. '12.5') phase keys.
    """
    if phase_num not in _PHASE_REGISTRY:
        key = str(phase_num)
        if key not in _PHASE_REGISTRY:
            raise ValueError(
                f'Unknown phase: {phase_num}. '
                f'Valid: {sorted(_PHASE_REGISTRY, key=lambda x: (isinstance(x, str), str(x)))}'
            )
        phase_num = key
    module_path, func_name = _PHASE_REGISTRY[phase_num]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)

def _sort_key(key):
    """Sort phase keys numerically: ints by value, strings by float value."""
    if isinstance(key, (int, float)):
        return float(key)
    try:
        return float(key)
    except (ValueError, TypeError):
        return float('inf')

def list_phases() -> Dict[Union[int, str], str]:
    """Return {phase_key: module_path} for all registered phases."""
    return {k: v[0] for k, v in sorted(
        _PHASE_REGISTRY.items(),
        key=lambda x: _sort_key(x[0]),
    )}
