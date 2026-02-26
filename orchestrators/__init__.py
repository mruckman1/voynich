"""Phase orchestrator registry.

Provides a single import point and lazy-loaded registry so that
cli.py and convergence_attack.py can dispatch to any phase by
number without importing all modules eagerly.
"""
from typing import Callable, Dict

# Lazy registry: maps phase number -> (module_path, function_name)
_PHASE_REGISTRY = {
    2:  ('orchestrators.phase2',  'run_phase2_attack'),
    3:  ('orchestrators.phase3',  'run_phase3_attack'),
    4:  ('orchestrators.phase4',  'run_phase4_attack'),
    5:  ('orchestrators.phase5',  'run_phase5_attack'),
    6:  ('orchestrators.phase6',  'run_phase6_attack'),
    7:  ('orchestrators.phase7',  'run_phase7_attack'),
    8:  ('orchestrators.phase8',  'run_phase8_translation'),
    9:  ('orchestrators.phase9',  'run_phase9_attack'),
    10: ('orchestrators.phase10', 'run_phase10_final_translation'),
    11: ('orchestrators.phase11', 'run_phase11_csp_translation'),
    12: ('orchestrators.phase12', 'run_phase12_reconstruction'),
    13: ('orchestrators.phase13', 'run_phase13_synthesis'),
}


def get_phase_runner(phase_num: int) -> Callable:
    """Return the run function for a given phase, importing lazily."""
    if phase_num not in _PHASE_REGISTRY:
        raise ValueError(f'Unknown phase: {phase_num}. Valid: {sorted(_PHASE_REGISTRY)}')
    module_path, func_name = _PHASE_REGISTRY[phase_num]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def list_phases() -> Dict[int, str]:
    """Return {phase_num: module_path} for all registered phases."""
    return {k: v[0] for k, v in sorted(_PHASE_REGISTRY.items())}
