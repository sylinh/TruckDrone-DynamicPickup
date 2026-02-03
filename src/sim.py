"""
Thin wrapper ?? gi? API c?: cli/gp import run_episode, validate_routes t? sim.py.
Logic ?? t?ch trong sim_core.py, sim_validate.py ?? d? b?o tr?.
"""

from .sim_core import run_episode
from .sim_validate import validate_routes

__all__ = ["run_episode", "validate_routes"]
