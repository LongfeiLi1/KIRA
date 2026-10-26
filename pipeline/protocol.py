"""Explain the unchanged strict decision contract; never repair a model answer."""
import copy
PROMPT_SUPPLEMENT = '\nDecision wire types are strict JSON: schema_version must be the integer 1 (not\nthe string "1", 1.0, true or null). state_id is the exact supplied string.\nReturn one object with exactly schema_version, state_id, diagnosis, action,\nknowledge_ids. The supplied decision_contract documents every required type,\nallowed action and argument. It describes the existing validator, not a tool\nexecution or an invitation to coerce values. No answer correction or retry occurs.\nCalibration is characterization to establish/check a local repair model. An\nabsent or unqualified repair model is not itself a prerequisite failure for\ncalibration. When calibration is offered, its current measurement, remaining\nbudget, device/parameter domains and native checks govern whether it can run;\nsuccessful characterization is not guaranteed. Never bypass a failed physical safety\ncheck. Offered characterize obtains current RC anchor/OTA/neighbor evidence and\nmust pass the original qualification. repair_passives produces only a proposal;\nremeasuring it can establish another current center, while finish performs the\nunchanged independent endpoint. Neither action is mandated by this explanation.\nSelect among currently offered actions using evidence and budget; these are\npurposes and preconditions, not a prescribed action sequence. Stop remains legal\nand means no success. Model qualification and predicted improvement never replace\nfresh independent endpoint receipts and the unchanged physical verifier.\n'

def decision_contract(state):
    """JSON Schema documentation plus the original stricter numeric-type note."""
    actions = []
    for name in state['allowed_actions']:
        properties = {key: dict(type=kind) for key, kind in state['tool_interface'][name]['schema'].items()}
        actions.append(dict(type='object', additionalProperties=False, required=['name', 'args'], properties=dict(name=dict(const=name), args=dict(type='object', additionalProperties=False, required=list(properties), properties=properties))))
    ids = lambda values: dict(type='array', minItems=1, uniqueItems=True, items=dict(type='string', enum=values))
    return dict(type='object', additionalProperties=False, required=['schema_version', 'state_id', 'diagnosis', 'action', 'knowledge_ids'], properties=dict(schema_version=dict(type='integer', const=1), state_id=dict(type='string', const=state['state_id']), diagnosis=dict(type='object', additionalProperties=False, required=['category', 'summary', 'evidence_ids'], properties=dict(category=dict(type='string', enum=['model_mismatch', 'realization_gap', 'planning_gap', 'constraints', 'success', 'unknown']), summary=dict(type='string', minLength=1, maxLength=1600), evidence_ids=ids(state['evidence_ids']))), action=dict(oneOf=actions), knowledge_ids=ids(state['knowledge_ids'])), python_validator_note='schema_version uses type(x) is int: even JSON 1.0 is refused; numeric args finite and never boolean')

def enrich_state(state):
    result = copy.deepcopy(state)
    result['decision_contract'] = decision_contract(state)
    return result

def explain_calibration(state, family):
    """Descriptions only: tool names, schemas, default args and availability stay."""
    result = copy.deepcopy(state)
    interface = result['tool_interface']
    if 'calibrate' in interface:
        purpose = 'Current anchor/OTA/neighbor characterization; establish original qualified local repair model (at most 3 SPICE)'
        interface['calibrate']['description'] = purpose + '. Prior model_trusted=false does not prohibit characterization. Current tool/native safety and budget checks apply; may fail; no endpoint acceptance.'
    result['calibration_semantics'] = dict(prior_model_qualification_required_for_characterization=False, qualified_model_required_for_model_guided_repair=True, safety_authority='unchanged native tool checks and current physical gates', autonomous_stop_unchanged=True)
    return result
