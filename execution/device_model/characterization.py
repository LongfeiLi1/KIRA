"""Independent unloaded OTA measurement, identical to T1 M1's 0.001Hz sweep."""
from dataclasses import asdict
import numpy as np
from execution.highpass.adapter import physical
from rc_agent.physical import header, op_commands, scalar, table, first_crossing
from rc_agent.models import OTA
from .model import TopologyCalibration

def deck(devices):
    return header(devices) + '\nVdd vdd 0 1.8\nVin vp 0 DC 1.2 AC 1\nXB vdd 0 vbias bias\nXU vp vn out vdd 0 vbias ota\nLfb out vn 1e12\nCfb vn 0 1\nVddz vddz 0 1.8\nVinz vpz 0 DC 1.2 AC 0\nXBz vddz 0 vbiasz bias\nXUz vpz vnz outz vddz 0 vbiasz ota\nLfbz outz vnz 1e12\nCfbz vnz 0 1\nItest 0 outz DC 0 AC 1\n.control\nset wr_singlescale\nset wr_vecnames\nset numdgt=15\n' + op_commands() + '\nac dec 40 0.001 1e9\nlet ar=real(v(out))\nlet ai=imag(v(out))\nlet zr=real(v(outz))\nlet zi=imag(v(outz))\nwrdata descriptor_ac.dat ar ai zr zi\nquit\n.endc\n.end\n'

def analyze(folder, devices, type_id):
    log = (folder / 'ngspice.log').read_text()
    if physical.SIMULATOR_FAILURE.search(log):
        raise ValueError('Invalid independent OTA characterization')
    arr = table(folder, 'descriptor_ac.dat', 5)
    f = arr[:, 0]
    gain = arr[:, 1] + 1j * arr[:, 2]
    z = arr[:, 3] + 1j * arr[:, 4]
    a0 = float(abs(gain[0]))
    fp = first_crossing(f, 20 * np.log10(abs(gain)), 20 * np.log10(a0) - 3.01029995664)
    eta = OTA(a0, 2 * np.pi * fp, float(z[0].real))
    result = {'devices': asdict(devices), 'eta': asdict(eta), 'power_w': scalar(log, 'power_w'), 'dc_out': scalar(log, 'dc_out'), 'minimum_saturation_margin_v': min((scalar(log, 'margin' + str(i)) for i in range(1, 8))), 'unloaded': True, 'evidence': str(folder.relative_to(physical.ROOT)), 'source': 'independent unloaded two-OTA A/Z, never full-filter fitting'}
    calibration = TopologyCalibration(eta, tuple(f), tuple(gain), tuple(z), type_id)
    return (eta, calibration, result)
