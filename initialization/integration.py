"""Receipt wrapper preserves every original bootstrap/ablation wrapper."""
from types import MethodType
from .synthesis import synthesize

def install_receipt(backend):
    native = backend.initialize

    def initialize(self):
        from common_execution import save
        typed = self.case['type_id'] != 'T1'
        receipt = self.binding['initialization_receipt'] if typed else synthesize(dict(task_id=self.task['id'], topology='t1_sk_lp', frequency_hz=self.task['f0_hz'], q=self.task['q'], gain=1.0), self.bounds)
        self.initialization_receipt = receipt
        save(self.out / 'initialization_receipt.json', receipt)
        self.event('initialization_synthesized_before_first_measurement', receipt)
        result = native()
        if not self.rows or self.rows[0]['parameters'] != receipt['parameters']:
            raise ValueError('first measured candidate does not match initialization receipt')
        return result
    backend.initialize = MethodType(initialize, backend)
    return backend
