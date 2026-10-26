"""Independent unloaded OTA measurement at the Sallen-Key highpass DC point."""
from execution.device_model import characterization as common

def deck(devices,case,passives):
    if case.get('type_id')!='T2':raise ValueError('Only Sallen-Key highpass is available')
    return common.deck(devices)

def analyze(folder, devices, type_id):
    if type_id!='T2':raise ValueError('Only Sallen-Key highpass is available')
    eta, calibration, evidence = common.analyze(folder, devices, type_id)
    evidence.update(dc_condition='registered follower DC; identical shared T2 bench', source='independent DC-matched unloaded AC two-OTA A/Z; no full-filter fit')
    return (eta, calibration, evidence)
