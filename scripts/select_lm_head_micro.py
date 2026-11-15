"""Two independent RTX smoke lanes. Never dispatches formal training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from nanogpt_noise.artifacts import read_json, write_json, file_hash, digest, lock
from nanogpt_noise.lm_head.protocol import CENTERS, code_identity
from nanogpt_noise.lm_head.runner import input_identity

MICROS = (16, 32)
OBJECTIVES = ('G', 'T', 'K')


def gpu_inventory(devices):
    def query(arguments):
        value = subprocess.check_output(['nvidia-smi', *arguments,
                    '--format=csv,noheader,nounits'], text=True, timeout=15)
        return list(csv.reader(io.StringIO(value), skipinitialspace=True))
    rows = query(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version'])
    inventory = {r[0]: dict(index=r[0], uuid=r[1], name=r[2],
                 total_bytes=int(r[3])*1024**2, used_mib=int(r[4]),
                 utilization=int(r[5]), driver=r[6]) for r in rows}
    if len(devices) != 2 or len(set(devices)) != 2 or any(d not in inventory for d in devices):
        raise ValueError('exactly two distinct nvidia-smi GPU indices are required')
    selected = [inventory[d] for d in devices]
    apps = query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory'])
    busy = [r for r in apps if r and r[0] in {g['uuid'] for g in selected}]
    if busy or any(g['utilization'] or g['used_mib'] > 1024 for g in selected):
        raise RuntimeError(f'selected GPUs are occupied; no task started: {busy}, {selected}')
    return selected


def expected_proof(identity, micro, objective, statistics_hash):
    return dict(input_id=identity['input_id'], objective=objective, micro=micro,
                test=False, code=identity['code'],
                statistics_sha256=statistics_hash if objective == 'K' else None)


def inspect_success(attempt, identity, micro, objective, statistics_hash):
    """Revalidate evidence, including uninterrupted timing and all four optimizers."""
    directory = Path(attempt)/'smoke'/objective
    proof = read_json(directory/'passed.json')
    expected = expected_proof(identity, micro, objective, statistics_hash)
    if proof.get('passed') is not True or proof.get('identity') != expected:
        raise ValueError('smoke proof identity mismatch')
    rows = []
    for optimizer in CENTERS:
        candidate = directory/optimizer
        status = read_json(candidate/'status.json')
        metadata = read_json(candidate/'metadata.json')
        execution = metadata['execution']
        performance = read_json(candidate/'performance.json')
        run_id = metadata['run_id']
        if (status.get('status') != 'completed' or status.get('mode') != 'smoke'
                or status.get('step') != 32 or status.get('tokens') != 32*524288
                or status.get('run_id') != run_id or digest(execution) != run_id
                or execution.get('micro') != micro or execution.get('artifact_id') != identity['input_id']
                or execution.get('code') != identity['code'] or execution.get('target_steps') != 32
                or execution.get('statistics_sha256') != expected['statistics_sha256']
                or execution.get('config', {}).get('objective') != objective
                or execution.get('config', {}).get('optimizer') != optimizer
                or execution.get('device_type') != 'cuda' or execution.get('mode') != 'smoke'
                or not math.isfinite(status.get('loss', float('nan')))):
            raise ValueError(f'invalid completed smoke: {candidate}')
        seconds, peak = performance['wall_seconds'], performance['peak_allocated_bytes']
        if (performance.get('segment_start_step') != 0 or performance.get('segment_end_step') != 32
                or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
                or not isinstance(peak, int) or peak <= 0):
            raise ValueError('timing must cover a fresh complete 32-step GPU smoke')
        rows.append(dict(optimizer=optimizer, run_id=run_id, wall_seconds=seconds,
                         peak_allocated_bytes=peak, timings=performance.get('timings', {})))
    if proof.get('run_ids') != [r['run_id'] for r in rows]:
        raise ValueError('four-optimizer run IDs do not match proof')
    return rows


def group_result(root, identity, micro, objective, statistics_hash):
    base = Path(root)/f'micro-{micro}'/objective
    attempts = sorted(base.glob('attempt-*'))
    result = dict(micro=micro, objective=objective, status='pending')
    if not attempts:
        return result
    attempt = attempts[-1]
    result['attempt'] = str(attempt.relative_to(root))
    try:
        execution = read_json(attempt/'execution.json')
        result.update(command_seconds=execution.get('command_seconds'))
        if execution.get('exit_code') != 0:
            log = (attempt/'console.log').read_text(encoding='utf8', errors='replace')
            # Only explicit CUDA allocation failure is a terminal memory result.
            result['status'] = ('oom' if 'torch.OutOfMemoryError: CUDA out of memory' in log
                                or 'torch.cuda.OutOfMemoryError: CUDA out of memory' in log
                                else 'error')
            result['exit_code'] = execution.get('exit_code')
            return result
        rows = inspect_success(attempt, identity, micro, objective, statistics_hash)
        total = identity['gpus'][MICROS.index(micro)]['total_bytes']
        result.update(status='passed', rows=rows, wall_seconds=sum(r['wall_seconds'] for r in rows),
                      peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                      allocation_headroom_ok=all(r['peak_allocated_bytes'] <= .9*total for r in rows))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status='error', reason=str(exc))
    return result


def selection_report(root, identity, statistics_hash):
    groups = [group_result(root, identity, m, o, statistics_hash) for m in MICROS for o in OBJECTIVES]
    complete = all(g['status'] in ('passed', 'oom') for g in groups)
    eligible = [m for m in MICROS if all(g['status'] == 'passed' and g['allocation_headroom_ok']
                                       for g in groups if g['micro'] == m)]
    totals = {m: sum(g['wall_seconds'] for g in groups if g['micro'] == m) for m in eligible}
    selected = min(eligible, key=lambda m: (totals[m], m)) if complete and eligible else None
    return dict(schema=1, kind='RTX hardware selection, not optimizer results',
                identity=identity, statistics_sha256=statistics_hash, complete=complete,
                selected_micro=selected, eligible_micros=eligible, groups=groups,
                next_action='H100 fresh-root smoke at selected micro' if selected else 'resolve missing or failed stages',
                rule='All G/T/K pass, allocated peak <=90% RTX total; minimum sum of 12 full smoke wall times; ties choose 16')


class Supervisor:
    def __init__(self):
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.active = set()

    def interrupt(self, *_):
        self.stop.set()
        with self.guard:
            for process in self.active:
                if process.poll() is None:
                    process.terminate()

    def launch(self, command, attempt, gpu):
        started = time.monotonic()
        with (attempt/'console.log').open('w', encoding='utf8') as log:
            with self.guard:
                if self.stop.is_set():
                    return
                process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, 'CUDA_VISIBLE_DEVICES':gpu, 'PYTHONUNBUFFERED':'1',
                              'NANOGPT_SWANLAB':'0'})
                self.active.add(process)
            try:
                stop_started = None
                while process.poll() is None:
                    if self.stop.is_set():
                        stop_started = stop_started or time.monotonic()
                        if time.monotonic()-stop_started > 30:
                            process.kill()
                    time.sleep(.2)
                code = process.wait()
            finally:
                with self.guard:
                    self.active.discard(process)
        write_json(attempt/'execution.json', dict(command=command, exit_code=code,
                   command_seconds=time.monotonic()-started, stopped=self.stop.is_set()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root')
    parser.add_argument('--selection-root', required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--statistics')
    parser.add_argument('--stage', choices=('gt', 'all'), default='all')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--handoff-micro', action='store_true',
                        help='validate transferred selection.json against local code/input/K; print selected micro only')
    args = parser.parse_args(argv)
    root = Path(args.selection_root).resolve()
    if args.handoff_micro:
        if not args.input_root or not args.statistics or args.report_only:
            parser.error('--handoff-micro requires --input-root and --statistics and excludes --report-only')
        result = read_json(root/'selection.json')
        identity = result['identity']
        ident = input_identity(Path(args.input_root).resolve(), False)
        from nanogpt_noise.lm_head.measure import load_statistics
        from nanogpt_noise.model import Shape
        load_statistics(args.statistics, ident, Shape(**ident['shape']))
        if (result.get('complete') is not True or result.get('selected_micro') not in MICROS
                or identity['code'] != code_identity()['sha256']
                or identity['supervisor'] != file_hash(__file__)
                or identity['input_id'] != ident['id']
                or result['statistics_sha256'] != file_hash(args.statistics)):
            raise ValueError('completed RTX selection must match H100 code, input and approved K')
        print(result['selected_micro'])
        return 0
    if args.report_only:
        with lock(root/'supervisor'):
            identity = read_json(root/'selection_identity.json')
            if identity['code'] != code_identity()['sha256'] or identity['supervisor'] != file_hash(__file__):
                raise ValueError('report with the frozen selection code')
            stats = read_json(root/'statistics.json')['sha256'] if (root/'statistics.json').exists() else None
            result = selection_report(root, identity, stats)
            write_json(root/'selection.json', result)
            print(__import__('json').dumps(result, indent=2))
            return 0 if result['selected_micro'] else 3
    if not args.input_root:
        parser.error('--input-root is required to run')
    inputs = Path(args.input_root).resolve()
    if root == inputs or inputs in root.parents or root in inputs.parents:
        raise ValueError('selection outputs and immutable inputs must be separate trees')
    # Fail before creating outputs or loading GPUs when the input protocol is wrong.
    ident = input_identity(inputs, False)
    statistics_hash = None
    if args.statistics:
        from nanogpt_noise.lm_head.measure import load_statistics
        from nanogpt_noise.model import Shape
        load_statistics(args.statistics, ident, Shape(**ident['shape']))
        statistics_hash = file_hash(args.statistics)
    if args.stage == 'all' and not statistics_hash:
        parser.error('--stage all requires genuine --statistics K-approved.pt; use --stage gt for partial work')
    import torch
    devices = [s.strip() for s in args.devices.split(',')]
    with lock(root/'supervisor'):
        gpus = gpu_inventory(devices)
        identity = dict(schema=1, input_id=ident['id'], code=code_identity()['sha256'],
                        supervisor=file_hash(__file__), torch=str(torch.__version__), cuda=torch.version.cuda,
                        python=platform.python_version(),
                        gpus=[{k:g[k] for k in ('uuid','name','total_bytes','driver')} for g in gpus])
        identity_file = root/'selection_identity.json'
        if identity_file.exists() and read_json(identity_file) != identity:
            raise ValueError('selection identity changed; use a new selection root')
        write_json(root/'preflight.json', dict(time=datetime.now(timezone.utc).isoformat(), gpus=gpus))
        write_json(identity_file, identity)
        stats_file = root/'statistics.json'
        if stats_file.exists():
            bound = read_json(stats_file)['sha256']
            if statistics_hash and bound != statistics_hash:
                raise ValueError('K statistics cannot change within a selection root')
            statistics_hash = bound
        elif statistics_hash:
            write_json(stats_file, dict(sha256=statistics_hash))
        supervisor = Supervisor()
        old = {s:signal.signal(s, supervisor.interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
        objectives = OBJECTIVES if args.stage == 'all' else ('G', 'T')

        def lane(micro, gpu):
            for objective in objectives:
                if supervisor.stop.is_set():
                    return
                previous = group_result(root, identity, micro, objective, statistics_hash)
                if previous['status'] == 'passed' or (previous['status'] != 'pending' and not args.retry_failed):
                    print(f'micro={micro} {objective}: keep {previous["status"]}', flush=True)
                    continue
                base = root/f'micro-{micro}'/objective
                base.mkdir(parents=True, exist_ok=True)
                attempt = base/f'attempt-{len(list(base.glob("attempt-*"))):04d}'
                attempt.mkdir(exist_ok=False)
                command = [sys.executable, '-m', 'nanogpt_noise.lm_head', 'smoke',
                           '--input-root', str(inputs), '--run-root', str(attempt),
                           '--objective', objective, '--micro-batch', str(micro)]
                if objective == 'K':
                    command.extend(['--statistics', str(Path(args.statistics).resolve())])
                print(f'GPU {gpu}: micro={micro} {objective} {attempt.name}', flush=True)
                supervisor.launch(command, attempt, gpu)
                print(f'micro={micro} {objective}: '+group_result(root, identity, micro, objective, statistics_hash)['status'], flush=True)
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(lane, m, g['uuid']) for m, g in zip(MICROS, gpus)]
                for future in futures:
                    try:
                        future.result()
                    except BaseException:
                        supervisor.interrupt()
                        raise
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)
            result = selection_report(root, identity, statistics_hash)
            write_json(root/'selection.json', result)
        print(f'selected_micro={result["selected_micro"]}; report={root / "selection.json"}', flush=True)
        return 0 if result['selected_micro'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
