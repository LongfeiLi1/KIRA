"""Reuse an injected frozen old Full backend, factor installer and original validator."""
import copy
import json
FACTORS = ('no_diagnostic_repair', 'fixed_allocation_targets', 'fixed_rule_controller')
SCHEMAS = {'calibrate': {}, 'probe': {'axis': 'string', 'direction': 'number'}, 'rc_step': {'radius': 'string'}, 'coupled_step': {'radius': 'string'}, 'finish': {}, 'stop': {}}

class NativeSession:
    is_mock = False

    def __init__(self, native, *, install, native_validator, receipt_verifier, family_verifier, run_id, installed_arm=None, module_install=None, source_check=None):
        self.is_mock = bool(getattr(native,'is_mock',False))
        self.installed_arm = installed_arm
        self.module_install = module_install
        self.source_check = source_check
        self.terminal = None
        self.native = native
        self.install = install
        self.native_validator = native_validator
        self.receipt_verifier = receipt_verifier
        self.family_verifier = family_verifier
        self.run_id = run_id
        self.native_state = None

    def configure_arm(self, arm):
        if self.installed_arm is not None:
            if arm != self.installed_arm:
                raise ValueError('native arm already installed exactly once')
            return
        if arm not in ('full_on',) + FACTORS:
            raise ValueError('unknown ablation')
        self.install(self.native, {key: arm == key for key in FACTORS})

    def initialize(self):
        return self.native.initialize()

    def snapshot(self, step):
        self.native_state = self.native.snapshot(step)
        state = copy.deepcopy(self.native_state)
        state['allowed_actions'] = list(self.native_state['allowed_actions'])
        state['tool_interface'] = {name: {'schema': SCHEMAS[name]} for name in state['allowed_actions']}
        for name in ('rc_step', 'coupled_step'):
            if name in state['tool_interface']:
                state['tool_interface'][name]['allowed_values'] = {'radius': ['normal', 'small']}
        if 'probe' in state['tool_interface']:
            if self.native.case['type_id'] == 'T1':
                from backends.lowpass_contracts import AXES
                axes = sorted(AXES)
            else:
                axes = sorted(self.native_state['parameter_names'])
            state['tool_interface']['probe']['allowed_values'] = {'axis': axes, 'direction': [-1, 1]}
        from .residuals import components
        disabled = self.installed_arm == 'no_diagnosis' or getattr(self, 'arm', None) == 'no_diagnostic_repair'
        if not disabled:
            state['three_residuals'] = components(state.get('diagnostics'), family='sallen_key')
        from .qualification import contract, card
        diagnostic = state.get('diagnostics') or {}
        observed = state.get('observation') or {}
        state['qualification'] = contract('sallen_key', measurement_valid=observed.get('measured'), model_trusted=diagnostic.get('model_trusted'), physical_gates=observed.get('gates') or {}, allowed_actions=state['allowed_actions'])
        state['family_contract_note'] = card('sallen_key')['text']
        state['ablation_arm'] = self.installed_arm or getattr(self, 'arm', None)
        state['initialization_role'] = 'original fixed neutral start' if self.installed_arm == 'no_initialization' else 'analytic passive design and original device defaults; no solution lookup'
        state['diagnosis_role'] = 'total-only measured view; safety qualification retained' if disabled else 'current measured decomposition and model qualification; tools retain numeric authority'
        state['llm_role'] = 'unchanged full-action deterministic controller' if self.installed_arm == 'deterministic' else 'choose exactly one allowed action using current evidence and feedback'
        return state

    def validate(self, decision, state):
        if decision['action']['name'] == 'stop':
            return
        candidate = copy.deepcopy(decision)
        candidate['state_id'] = self.native_state['state_id']
        self.validated = self.native_validator(json.dumps(candidate), self.native_state)

    def execute(self, name, args):
        if self.source_check:
            self.source_check()
        return self.native.execute(self.validated)

    def finish(self, args):
        self.terminal = copy.deepcopy(self.native.finish())
        return copy.deepcopy(self.terminal)

    def verify_receipts(self, result):
        return self.receipt_verifier(result) is True

    def verify_endpoint(self, result, task):
        return self.family_verifier(result, task) is True

    def persist_trace(self, trace):
        from pathlib import Path
        if hasattr(self.native, 'out'):
            (Path(self.native.out) / 'integrated_trace.json').write_text(json.dumps(trace, indent=2, allow_nan=False) + '\n')

    def capture(self):
        return dict(current=copy.deepcopy(getattr(self.native, 'current', None)), cost=self.native.costs())

    def recover_tool_error(self, exc):
        if type(exc).__name__ != 'ToolUnavailable' or self.native.engine.status == 'paused':
            return False
        self.native.last_result = dict(status='tool_failed', error_class=type(exc).__name__)
        return True

    def is_control_exception(self, exc):
        return type(exc).__module__ == 'kira_shared_control.parent_lease'
