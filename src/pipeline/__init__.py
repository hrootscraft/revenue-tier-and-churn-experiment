"""Pipeline package — module aliases for run_pipeline.py imports."""

from importlib import import_module as _im
import sys as _sys
from pathlib import Path as _Path

# Allow `from pipeline.step_01 import ...` to resolve to 01_generate_clients etc.
_aliases = {
    "pipeline.step_01": "pipeline.01_generate_clients",
    "pipeline.step_02": "pipeline.02_generate_activity",
    "pipeline.step_03": "pipeline.03_assign_treatment",
    "pipeline.step_04": "pipeline.04_simulate_outcomes",
    "pipeline.step_05": "pipeline.05_build_experiment_table",
}

for _alias, _real in _aliases.items():
    if _alias not in _sys.modules:
        _mod = _im(_real)
        _sys.modules[_alias] = _mod
