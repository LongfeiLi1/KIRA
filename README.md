# KIRA

KIRA provides knowledge-based initialization, response diagnosis, constrained control and physical tool interfaces.

## Install

Python 3.10 or newer is required.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Examples

```sh
python kira.py --demo --topology lowpass
python kira.py --demo --topology highpass
python kira.py --spec my_spec.json
```

Both examples calculate valid initial component values at 50 kHz, Q=1/sqrt(2),
with explicit bounds. They do not call a model or run a simulator.
Initialization and analytical predictions are not physical acceptance.
`--arm no_initialization` returns a specification-blind domain-center start.

For Python use, run from the project directory:

```python
from api import demo_spec, initialize, analytical_response
row = demo_spec("highpass")
initial = initialize(row)
response = analytical_response(row, [1000, 50000, 1000000])
```

The JSON input has `family`, `spec` and `bounds`; `demo_spec()` shows the schema.
Only the two SK topology identifiers are accepted.

## Physical tools and orchestration

`python -m native_cli --help` exposes the retained native
measurement backend. Only T1/T2 tasks from an external task directory are
accepted. Set `KIRA_TASK_DATA` to the directory containing the supplied task
index and manifest. Original experiment inputs are not bundled.

Physical execution needs compatible ngspice, SKY130/open_pdks and a matching
external OTA renderer. Set `KIRA_OTA_RENDERER` to a Python file exposing
`ota_netlist(devices)` with the expected OTA/bias ports and XM1-XM8 measurement
identifiers. The device fields are `current_scale`, `output_scale`, `cc_pf` and
`rz_ohm`. The OTA netlist and PDK are not bundled. Set `KIRA_PDK_MANIFEST` to the model/simulator binding JSON
(`files` and `simulator` metadata) used by the physical runtime. Actual execution
requires explicit `--execute`; model requests require the caller's authorized client.

`pipeline.run` runs the shared decision loop around a caller-owned SK backend,
task and model client. It requires explicit independent receipt and endpoint
verifiers. Deterministic mode uses the same tool interface. The native backends
retain measurement, model qualification, bounded repair and fresh endpoint
checks. An initialization, model prediction or successful orchestration return
alone does not establish physical success.

See `LICENSE` for the project license.
