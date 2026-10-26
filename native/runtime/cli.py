"""Explicit Sallen-Key task Full runs with fresh outputs."""
import argparse, json, os, random, shutil
from pathlib import Path

def ordered_cases(rows):
    groups = []
    rng = random.Random(0)
    for typ in ('T1', 'T2'):
        for level in ('L1', 'L2', 'L3'):
            group = [r for r in rows if r['type_id'] == typ and r['level'] == level]
            rng.shuffle(group)
            groups.append(group)
    ordered = []
    while any(groups):
        for group in groups:
            if group:
                ordered.append(group.pop(0))
    return ordered

def load_cases():
    from runtime.paths import BENCHMARK_DIR
    if not (BENCHMARK_DIR / 'runtime_index.json').is_file():
        raise ValueError('Task data are not distributed. Supply your separately obtained dataset with --benchmark PATH.')
    from runtime.integrity import verify
    from task_data import load_case
    from task_data import binding_report
    verify()
    rows = json.loads((BENCHMARK_DIR / 'runtime_index.json').read_text())['cases']
    rows=[r for r in rows if (r.get('type_id') or (r.get('case_id','').split('-')[1] if r.get('case_id','').startswith('V3-') else None)) in {'T1','T2'}]
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty unique Sallen-Key tasks')
    cases = []
    for row in rows:
        case = load_case(row['case_id'])
        binding_report(case)
        cases.append(case)
    return cases

def construct(case, out, ngspice, executor=None, arm='full'):
    if case.get('type_id') not in {'T1', 'T2'}:
        raise ValueError('Only Sallen-Key T1/T2 tasks are available')
    from runtime.paths import CODE_ROOT
    from backends.lowpass import T1Backend
    from backends.highpass import HighpassBackend
    from control.core import install
    from initialization.integration import install_receipt
    cls = HighpassBackend if case['type_id'] != 'T1' else T1Backend
    protocol = json.loads((CODE_ROOT / 'resources/protocols' / ('lowpass.json' if case['type_id'] == 'T1' else 'highpass.json')).read_text())
    kwargs = {'launch_authorized': True}
    if cls is HighpassBackend:
        kwargs['executor'] = executor
    backend = cls(case, out, protocol, ngspice, **kwargs)
    if cls is HighpassBackend:
        from execution.highpass.adapter import physical
        physical.SPICE = Path(ngspice).resolve()
    if cls is T1Backend and executor is not None:
        backend.engine.executor = executor
    if backend.engine.cache or backend.engine.candidates or backend.engine.api_calls or (backend.engine.endpoint is not None):
        raise ValueError('Fresh cold instance required')
    if arm not in ('full', 'rule', 'legacy_no_diagnostic_repair', 'legacy_fixed_allocation_targets'):
        raise ValueError('unknown controlled arm')
    factors = dict(no_diagnostic_repair=arm == 'legacy_no_diagnostic_repair', fixed_allocation_targets=arm == 'legacy_fixed_allocation_targets', fixed_rule_controller=arm == 'rule')
    install(backend, factors)
    install_receipt(backend)
    backend.experimental_arm = arm
    return backend

def execute(case, out, ngspice, arm='full'):
    from runtime.integrity import verify
    from control.loop import run_loop
    from control.controller import NoProvider
    from control.controller import run_loop as rule_loop
    from provider.telemetry import Trace
    from provider.telemetry import attach
    from provider.bootstrap import Client
    from rc_agent.qwen_adapter import QwenConnectivityAdapter
    from evidence import save
    source = verify()
    trace = Trace(out, enabled=True)
    try:
        with trace.span('case.worker'):
            with trace.span('backend.construction'):
                backend = construct(case, out, ngspice, arm=arm)
            attach(backend, trace)
            client = NoProvider() if arm == 'rule' else Client(backend.engine, trace=trace, adapter_factory=QwenConnectivityAdapter.from_existing_config, runtime_authority={'source_fingerprinted': True, 'execution_authorized': True, 'case_id': case['case_id'], 'source_sha256': source})
            with trace.span('method.loop'):
                result = (rule_loop if arm == 'rule' else run_loop)(backend, client, identity={'run_id': out.name, 'case_id': case['case_id']})
            if len(backend.engine.api_calls) > 32 or client.actual_retries > 1:
                raise ValueError('API/retry budget exceeded')
            result.update(release='kira-initialization-v2', experimental_arm=arm, initialization_receipt_sha256=backend.initialization_receipt['receipt_sha256'], source_sha256=source, numerical_local_seed=0, historical_answer_injected=False, automatic_case_restart=False, new_API_calls=len(backend.engine.api_calls), actual_transport_retries=client.actual_retries, adaptive_target_feedback=backend.adaptive_target_feedback, refinement_audit=backend.refinement_audit, scheduler_action_memory=backend.scheduler_memory, new_instance_endpoint_count=int(backend.engine.endpoint is not None))
            save(out / 'result.json', result)
        trace.summarize({'queue_wall_seconds': 0, 'execution': 'sequential local fresh case'})
        return result
    except BaseException as exc:
        out.mkdir(parents=True, exist_ok=True)
        save(out / 'entry_failure.json', {'method_success': False, 'error_class': type(exc).__name__, 'automatic_restart': False})
        raise

def main():
    parser = argparse.ArgumentParser(description='KIRA Sallen-Key circuit-sizing interface')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case')
    group.add_argument('--all', action='store_true')
    parser.add_argument('--benchmark', type=Path, help='Benchmark package root; default: sibling KIRA-benchmark')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--execute', action='store_true', help='Explicitly enable ngspice and paid model requests when needed')
    parser.add_argument('--out', type=Path, default=Path('output/sallen_key'), help='Fresh output directory within the code package output/')
    parser.add_argument('--ngspice', default='ngspice', help='ngspice executable with KLU and mdump support')
    args = parser.parse_args()
    if args.benchmark:
        os.environ['KIRA_BENCHMARK'] = str(args.benchmark.expanduser().resolve())
    from runtime.paths import CODE_ROOT
    from runtime.paths import PDK
    try:
        rows = load_cases()
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    selected = ordered_cases(rows) if args.all else [c for c in rows if c['case_id'] == args.case]
    if not selected:
        parser.error('Unknown case ID')
    if not args.execute:
        print(json.dumps({'status': 'PASS_readonly', 'cases': len(selected), 'unique_case_ids': [c['case_id'] for c in selected], 'API_started': 0, 'SPICE_started': 0, 'PDK_installed': PDK.is_file(), 'ngspice_found': shutil.which(args.ngspice) is not None}))
        return 0
    if not PDK.is_file():
        parser.error('Set KIRA_PDK_LIB to an installed open_pdks SKY130 sky130.lib.spice')
    if shutil.which(args.ngspice) is None:
        parser.error('ngspice executable not found')
    out = args.out.resolve() if args.out.is_absolute() else (CODE_ROOT / args.out).resolve()
    if not out.is_relative_to(CODE_ROOT / 'output'):
        parser.error('--out must be within code package output/')
    if out.exists():
        parser.error('Output already exists; use a new directory (no automatic resume/overwrite)')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for case in selected:
        folder = out / case['case_id']
        try:
            result = execute(case, folder, args.ngspice)
            record = {'case_id': case['case_id'], 'method_success': result.get('method_success', False), 'cost': result.get('cost', {}), 'token_accounting': result.get('token_accounting', {})}
        except Exception as exc:
            record = {'case_id': case['case_id'], 'method_success': False, 'error_class': type(exc).__name__}
        records.append(record)
        (out / 'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
    return 0 if all((r['method_success'] for r in records)) else 1
