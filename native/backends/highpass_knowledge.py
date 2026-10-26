"""Audited static projection of the SK-card inventory; no outcome learning."""
from copy import deepcopy
from rc_agent.knowledge import KnowledgeBase
VERSION = 'sk_highpass_knowledge'
GENERAL = {'K03', 'K04', 'K06', 'K07', 'K10'}
DISABLED = {'K02', 'K05', 'K08', 'K09', 'K11'}

class HighpassKnowledge:
    version = VERSION

    def __init__(self):
        self.base = KnowledgeBase()

    def retrieve(self, case, candidate_id, binding, trusted):
        cards = []
        for original in self.base.cards:
            cid = original['id']
            if cid in GENERAL:
                c = {'id': cid, 'title': original['title'], 'claims': deepcopy([x for x in original['claims'] if x['kind'] == 'source_fact']), 'source_ids': original['source_ids'], 'source_inventory_id': cid, 'projection': 'source facts only; original topology-derived equations excluded'}
                cards.append(c)
        mapping = {'id': 'K01', 'title': 'Registered topology, SI coordinates, signed reference', 'claims': [{'kind': 'derived_protocol', 'text': 'Use the exact registered node graph and Case response profile. SK highpass has an s^2 numerator and a finite passband reference. Named parameter bounds and source/load conditions are immutable.'}], 'mapping': binding, 'source_ids': [], 'implementation_sources': ['numerics/topologies/registry.py', 'numerics/metrics/response.py', 'execution/device_model/model.py'], 'projection': 'explicit new typed protocol mapping, not the original SKLP formula'}
        by_id = {c['id']: c for c in cards}
        selected = [mapping] + [by_id[k] for k in ('K07', 'K03', 'K04', 'K10')]
        return {'knowledge_version': VERSION, 'original_inventory_version': self.base.manifest['knowledge_version'], 'original_resource_hashes': self.base.manifest['resources'], 'original_inventory_count': len(self.base.cards), 'type_id': case['type_id'], 'candidate_id': str(candidate_id), 'cards': selected, 'disabled_for_this_projection': sorted(DISABLED), 'available_general_cards': sorted(GENERAL), 'model_trusted': bool(trusted), 'no_original_topology_derived_formula_injected': True, 'experience_ingestion': False, 'hard_constraints_unchanged': True}
