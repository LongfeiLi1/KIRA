"""Preparation-only high-pass validation. No simulator or optimizer launcher."""
from .contract import explicit_hp_contract, expected_hp_variables, build_hp_matrix_deck
from .evidence import inspect_hp_linear_evidence, highpass_step_metrics
