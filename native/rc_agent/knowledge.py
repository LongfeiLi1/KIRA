"""Read-only deterministic card retrieval and action planning; no LLM or SPICE.

Only the bundled allowlist is readable. Results/experiences are never ingested.
External native baselines cannot be consumers of framework-specific policy.
"""
from dataclasses import dataclass
from pathlib import Path
import copy, hashlib, json
_DIRECTORY = Path(__file__).resolve().parents[2] / 'resources/knowledge'
_ALLOWED = {'cards.json', 'sources.json'}
_DISABLED = {'K08', 'K09', 'K11'}
_RESIDUAL_TAGS = {'r_mod', 'r_real', 'r_plan', 'model_residual', 'realization_residual', 'planning_residual'}

def read_resource(name):
    if name not in _ALLOWED:
        raise PermissionError('Knowledge read is outside the fixed allowlist')
    path = (_DIRECTORY / name).resolve()
    if path.parent != _DIRECTORY.resolve():
        raise PermissionError('Knowledge resource escapes through a symlink')
    return path.read_bytes()

@dataclass(frozen=True)
class RetrievalContext:
    task_id: str
    candidate_id: str
    port_id: str
    model_id: str
    topology: str = 'sallen_key_lp_unity'
    split: str = 'development'
    passive_model: str = 'ideal'
    corner: str = 'tt'
    operating_mode: str = 'nominal'
    consumer: str = 'framework_no_llm'
    enabled_constraints: tuple = ('frequency', 'q', 'power', 'stability')
    evidence: tuple = ()
    stability_status: str = 'unknown'
    remaining_spice_processes: int = 0
    component_residual_ablation: bool = False

class KnowledgeBase:

    def __init__(self):
        raw_cards = read_resource('cards.json')
        raw_sources = read_resource('sources.json')
        self.cards = json.loads(raw_cards)
        self.sources = json.loads(raw_sources)
        if not isinstance(self.cards, list) or not self.cards or not isinstance(self.sources, dict):
            raise ValueError('Knowledge requires a nonempty card list and source mapping')
        if any(not isinstance(c, dict) or not isinstance(c.get('id'), str) for c in self.cards):
            raise ValueError('Each knowledge card requires a string identifier')
        if len({c['id'] for c in self.cards}) != len(self.cards):
            raise ValueError('Duplicate knowledge card identifier')
        self.manifest = {
            'knowledge_version': 'sallen_key_knowledge',
            'resources': {'cards.json': hashlib.sha256(raw_cards).hexdigest(),
                          'sources.json': hashlib.sha256(raw_sources).hexdigest()},
        }
        for c in self.cards:
            if c['version'] != '0.1' or c['data_partition'] != 'general_knowledge':
                raise ValueError('Unreviewed card version/partition')
            for claim in c['claims']:
                if claim['kind'] not in {'source_fact', 'derived', 'heuristic', 'policy_proposal'}:
                    raise ValueError('Unclassified claim')
                if claim['kind'] in {'source_fact', 'heuristic'} and (not claim.get('source_ids')):
                    raise ValueError('Source claims require provenance')
                if claim['kind'] == 'derived' and (not claim.get('assumptions')):
                    raise ValueError('Derived claims require assumptions')
            if any((s not in self.sources for s in c['source_ids'])):
                raise ValueError('Unknown source')

    def retrieve(self, context, tags=(), limit=3):
        if context.consumer not in {'framework_no_llm', 'framework_llm_interface'}:
            raise PermissionError('Native external baselines retain their own knowledge')
        if context.passive_model != 'ideal' or context.corner != 'tt' or context.operating_mode != 'nominal':
            raise ValueError('Current knowledge activation is limited to ideal passives and nominal tt; extensions require review')
        if context.topology != 'sallen_key_lp_unity':
            raise ValueError('Only reviewed nominal SK core is enabled')
        if context.split not in {'development', 'heldout'} or not all((context.task_id, context.candidate_id, context.port_id, context.model_id)):
            raise ValueError('Complete current-task provenance required')
        if limit != 3:
            raise ValueError('Frozen retrieval limit is three topical cards plus mandatory K01/K07')
        tags = tuple(tags)
        if any((not isinstance(x, str) for x in tags)):
            raise ValueError('Tags must be explicit strings')
        if context.stability_status not in {'pass', 'fail', 'unknown'}:
            raise ValueError('Explicit tri-state stability required')
        if context.component_residual_ablation and (set(tags) & _RESIDUAL_TAGS or any((origin == 'component_residual' or any((t in eid for t in _RESIDUAL_TAGS)) for _, eid, origin in context.evidence))):
            raise ValueError('Component residual ablation forbids residual query/evidence side channels')
        active = [c for c in self.cards if c['enabled_by_default'] and c['id'] not in _DISABLED and (context.topology in c['topology'])]
        mandatory = [c for c in active if c['id'] in {'K01', 'K07'}]
        ranked = sorted((c for c in active if c['id'] not in {'K01', 'K07'} and set(c['retrieval_tags']) & set(tags)), key=lambda c: (-len(set(c['retrieval_tags']) & set(tags)), c['id']))[:limit]
        selected = mandatory + ranked
        facts = {name for name, _, _ in context.evidence}
        cards = []
        for c in selected:
            result = copy.deepcopy(c)
            result['sources'] = [self.sources[k] for k in c['source_ids']]
            result['missing_evidence'] = sorted(set(c['evidence_required']) - facts)
            cards.append(result)
        return {'knowledge_version': self.manifest['knowledge_version'], 'knowledge_resource_hashes': self.manifest['resources'], 'task_id': context.task_id, 'candidate_id': context.candidate_id, 'port_id': context.port_id, 'model_id': context.model_id, 'split': context.split, 'cards': cards, 'stability_status_unchanged': context.stability_status, 'hard_constraints_unchanged': True, 'actions_are': 'policy_proposals_not_acceptance', 'retrieval_spice_calls': 0, 'llm_calls': 0, 'disabled_card_ids': sorted(_DISABLED)}
_ACTIONS = {'verify_mapping_units': ('map_sk_nodes', (), 0), 'compute_ideal_metrics': ('sk_constant_gain', ('node_mapping', 'ideal_reference_assumptions'), 0), 'propose_common_r': ('bounded_rc_coordinate', ('node_mapping', 'response_validity', 'rc_direction_evidence', 'legal_domain'), 0), 'propose_reciprocal_c': ('bounded_rc_coordinate', ('node_mapping', 'response_validity', 'rc_direction_evidence', 'legal_domain'), 0), 'propose_impedance_scale': ('bounded_rc_coordinate', ('node_mapping', 'port_identity', 'source_load_known', 'legal_domain'), 0), 'request_response': ('physical.measure_filter', (), 2), 'request_loaded_response': ('physical.measure_filter', ('port_identity', 'source_load_known'), 2), 'request_stability_validation': ('physical.measure_filter', ('node_mapping',), 2), 'request_amp_response': ('physical.characterize', ('port_identity',), 1), 'request_output_impedance': ('physical.characterize', ('port_identity',), 1), 'request_amp_local_measurement': ('physical.characterize', ('port_identity', 'actual_power'), 1), 'propose_amp_target': ('controller.propose', ('model_validity', 'amp_complex_response', 'port_identity', 'actual_power', 'legal_domain'), 0)}

def plan_action(context, retrieved, action_id, card_ids):
    if any((retrieved[k] != getattr(context, k) for k in ('task_id', 'candidate_id', 'port_id', 'model_id', 'split'))):
        raise ValueError('Stale or cross-candidate knowledge request')
    present = {c['id']: c for c in retrieved['cards']}
    facts = {x[0] for x in context.evidence}
    valid = action_id in _ACTIONS and bool(card_ids) and all((k in present and action_id in present[k]['action_ids'] for k in card_ids))
    missing = []
    tool = None
    cost = 0
    if valid:
        tool, required, cost = _ACTIONS[action_id]
        missing = sorted(set(required) - facts)
    if context.stability_status != 'pass' and action_id not in {'verify_mapping_units', 'compute_ideal_metrics', 'request_response', 'request_loaded_response', 'request_stability_validation'}:
        missing.append('qualified_stability_measurement')
    if cost > context.remaining_spice_processes:
        missing.append('remaining_spice_budget')
    accepted = valid and (not missing)
    return {'action_id': action_id, 'card_ids': list(card_ids), 'knowledge_version': retrieved['knowledge_version'], 'evidence_ids': [eid for _, eid, _ in context.evidence], 'preconditions_satisfied': accepted, 'missing_evidence': missing if valid else ['action_not_allowed_by_selected_cards'], 'tool_request': {'tool': tool, 'candidate_id': context.candidate_id, 'port_id': context.port_id, 'max_spice_processes': cost, 'execution_requires_common_budget': True} if accepted else None, 'expected_observation': 'Matched full-circuit measurements test the proposal; formula output remains predicted', 'rollback_condition': 'no improvement, model/port mismatch, domain boundary, missing hard evidence or budget exhaustion', 'fallback_action': 'request_stability_validation' if context.stability_status != 'pass' else 'request_response', 'hard_constraints_unchanged': True, 'success_decision': None, 'spice_calls': 0, 'llm_calls': 0}

def execute_math_tool(plan, *, passives=None, bounds=None, coordinate=None, log_step=None, k=1.0, f0_hz=None, q=None):
    """Execute only reviewed pure math. Physical calls require normal evaluator.

    Current passives and bounds must be supplied by the owning numerical runner,
    not extracted from retrieved prose. No returned value is a success claim.
    """
    if not plan.get('preconditions_satisfied') or not plan.get('tool_request'):
        raise ValueError('Unapproved or incomplete tool plan')
    from rc_agent.knowledge_tools import sk_constant_gain
    from rc_agent.knowledge_tools import bounded_rc_coordinate
    from rc_agent.knowledge_tools import f3db_relative_dc
    tool = plan['tool_request']['tool']
    if tool == 'sk_constant_gain':
        return sk_constant_gain(passives, k)
    if tool == 'bounded_rc_coordinate':
        expected = {'propose_common_r': 'common_r', 'propose_reciprocal_c': 'reciprocal_c', 'propose_impedance_scale': 'impedance_scale'}[plan['action_id']]
        if coordinate != expected:
            raise ValueError('Action/coordinate mapping mismatch')
        p = bounded_rc_coordinate(passives, coordinate, log_step, bounds, 0.08)
        return {'passives': p, 'predicted_ideal': sk_constant_gain(p), 'requires_full_circuit_validation': True, 'spice_calls': 0}
    raise NotImplementedError('This request requires the existing budgeted measurement/optimizer interface; it is not executed by knowledge retrieval')
