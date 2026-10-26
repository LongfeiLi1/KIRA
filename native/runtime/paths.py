"""Local paths configured without private files or credentials."""
import os
from pathlib import Path
CODE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = Path(os.environ.get('KIRA_BENCHMARK', str(CODE_ROOT.parent / 'KIRA-benchmark'))).expanduser().resolve()
BENCHMARK_DIR = Path(os.environ.get('KIRA_TASK_DATA', str(BENCHMARK_ROOT / 'data/tasks'))).expanduser().resolve()
PDK = Path(os.environ.get('KIRA_PDK_LIB', str(CODE_ROOT / 'external/sky130/libs.tech/ngspice/sky130.lib.spice'))).expanduser().resolve()
