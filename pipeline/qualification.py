"""Sallen-Key model qualification and independent endpoint requirements."""
import copy

def _status(value):
    return 'qualified' if value is True else 'unqualified' if value is False else 'unavailable'

def contract(family, *, measurement_valid, model_trusted, physical_gates, allowed_actions):
    if family != 'sallen_key':
        raise ValueError('Unsupported physical family')
    return {'schema_version': 'kira.family_qualification.v2', 'family': family, 'ota_qualification': {'applicable': True, 'status': 'backend_authority'}, 'repair_model': {'applicable': True, 'status': _status(model_trusted), 'value': model_trusted, 'required_for': ['rc_step', 'coupled_step']}, 'measurement': {'valid': measurement_valid, 'stage_validity_is_endpoint_acceptance': False}, 'physical_checks': {'gates': copy.deepcopy(physical_gates), 'empty_or_unavailable_is_pass': False}, 'independent_endpoint': {'required_for_success': True, 'tool': 'finish', 'currently_offered': 'finish' in allowed_actions, 'requires': ['fresh endpoint measurement', 'current candidate and execution receipts', 'independent physical verifier']}, 'stop': {'available': 'stop' in allowed_actions, 'meaning': 'stop without claiming success'}}

def card(family):
    if family != 'sallen_key':
        raise ValueError('Unsupported physical family')
    return {'id': 'sallen-key-contract', 'text': 'Model-guided repair requires the native qualification and feasibility gates. An offered finish action still needs fresh independent physical measurements and receipts. Model predictions and unavailable residuals are not physical acceptance.'}
