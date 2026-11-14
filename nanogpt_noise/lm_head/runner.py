"""Head-only searches, a dynamic single-GPU pool, and persistent pause."""
import argparse
import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import torch
from ..artifacts import lock, read_json, write_json, file_hash, save_checkpoint, digest
from ..data import verify
from ..input_bundle import verify_bundle
from ..pool import run_pool
from ..search import propose, summarize, winner
from ..storage import compact_steps, retain_winner_checkpoint
from .protocol import configuration, VERSION, CENTERS, code_identity, require_current_input
from .train import run, Runtime


def input_identity(root, test=False):
    root = Path(root)
    identity = verify(root/'openwebtext_subset', root/'frozen-artifacts')
    if identity['tiny'] != test:
        raise ValueError('input fixture mode mismatch')
    if not test:
        verify_bundle(root)
        require_current_input(identity)
    return identity


def bind_root(root, input_root, ident, test=False):
    root, inputs = Path(root).resolve(), Path(input_root).resolve()
    if root == inputs or inputs in root.parents or root in inputs.parents:
        raise ValueError('run output and read-only inputs must be separate trees')
    expected = dict(design=VERSION, scope='lm_head_only', input_id=ident['id'],
                    code=code_identity()['sha256'], test=test)
    path = root/'lm_head_run.json'
    # Workers only read an already committed immutable registration.
    if path.exists():
        if read_json(path) != expected:
            raise ValueError('run root identity mismatch; do not reuse legacy or changed-code outputs')
        return expected
    with lock(root):
        path = root/'lm_head_run.json'
        if path.exists():
            if read_json(path) != expected:
                raise ValueError('run root identity mismatch; do not reuse legacy or changed-code outputs')
        else:
            if any(p.name not in ('.lease', '.lock') for p in root.iterdir()):
                raise ValueError('refusing to adopt a nonempty unregistered run root')
            write_json(path, expected)
    return expected


def job_list(phase):
    nodes = [0] if phase == 'clean' else list(range(35))
    return [dict(node=n, optimizer=o, index=n*4+i)
            for n in nodes for i,o in enumerate(CENTERS)]


def run_node(job, input_root, run_root, objective, *, micro=4, statistics=None,
             test=False, runtime=None):
    root, inputs = Path(run_root), Path(input_root)
    output = root/objective/'jobs'/f"job_{job['index']:04d}"
    rt = runtime or Runtime(cpu=test)
    with lock(output):
        records = []
        for index in range(8):
            if (root/'pause.request').exists():
                return dict(status='paused')
            proposal = propose(records)
            config = configuration(objective, job['optimizer'], job['node'], proposal['x'])
            config['proposal'] = proposal
            if statistics:
                config['statistics'] = str(Path(statistics).resolve())
            candidate = output/f'candidate_{index:02d}'
            result = run(config, inputs/'openwebtext_subset', inputs/'frozen-artifacts',
                         candidate, micro=micro, runtime=rt, test=test,
                         pause_file=root/'pause.request')
            if result['status'] == 'infrastructure_failure':
                return dict(status='paused', candidate=index)
            objective_loss = (result['evaluations'][-1]['train_objective']
                              if result['status']=='completed' else None)
            records.append(dict(**proposal, candidate=index, status=result['status'],
                                loss=result.get('loss'), train_objective=objective_loss,
                                lr=config['peak_lr'], tokens=result['tokens'],
                                failure_field=result.get('field')))
            write_json(output/'search.json', dict(objective=objective, job=job, **summarize(records)))
            compact_steps(candidate)
        summary = summarize(records)
        primary = summary['winner']['candidate'] if summary['winner'] else None
        valid = [r for r in records if r['status']=='completed']
        auxiliary = winner([{**r,'loss':r['train_objective']} for r in valid])
        auxiliary_path = output/'training-objective-head.pt'
        if auxiliary and auxiliary['candidate'] != primary and not auxiliary_path.exists():
            state = torch.load(output/f"candidate_{auxiliary['candidate']:02d}"/'final.pt',
                               map_location='cpu', weights_only=True)
            save_checkpoint(auxiliary_path, dict(head=state['head'], candidate=auxiliary['candidate'],
                            input_id=state['input_id'], backbone_hash=state['backbone_hash']))
        retain_winner_checkpoint(output, primary, completed=range(8))
        if primary is not None and not (output/f'candidate_{primary:02d}'/'final.pt').exists():
            raise RuntimeError('winner checkpoint missing')
        summary.update(job=job, objective=objective, training_objective_candidate=
                       auxiliary['candidate'] if auxiliary else None)
        write_json(output/'search.json', summary)
        write_json(output/'retained.json', dict(
            complete=True, files={p.relative_to(output).as_posix():file_hash(p)
                for p in sorted(output.rglob('*.pt'))}))
        return summary


def smoke_identity(ident, objective, micro, statistics, test):
    return dict(input_id=ident['id'], objective=objective, micro=micro, test=test,
                code=code_identity()['sha256'],
                statistics_sha256=file_hash(statistics) if statistics else None)


def smoke(args, ident):
    rt = Runtime(cpu=args.test)
    records = []
    for optimizer in CENTERS:
        config = configuration(args.objective, optimizer, 0, 0.)
        if args.statistics:
            config['statistics'] = str(Path(args.statistics).resolve())
        result = run(config, Path(args.input_root)/'openwebtext_subset',
                     Path(args.input_root)/'frozen-artifacts',
                     Path(args.run_root)/'smoke'/args.objective/optimizer,
                     micro=args.micro_batch, runtime=rt,
                     smoke_steps=min(32, ident['budget']['steps']), test=args.test)
        records.append(result)
    if any(r['status'] != 'completed' for r in records):
        raise RuntimeError('smoke did not complete successfully for all optimizers')
    proof = dict(identity=smoke_identity(ident, args.objective, args.micro_batch,
                                        args.statistics, args.test),
                 passed=True, run_ids=[r['run_id'] for r in records])
    write_json(Path(args.run_root)/'smoke'/args.objective/'passed.json', proof)
    return proof


def collect(root, objective):
    rows = []
    for path in sorted((Path(root)/objective/'jobs').glob('job_*/search.json')):
        value = read_json(path)
        if value.get('complete'):
            rows.append(value)
    result = dict(design=VERSION, objective=objective, primary_metric='real_validation_ce',
                  completed_jobs=len(rows), rows=rows)
    write_json(Path(root)/objective/'summary.json', result)
    return result


def serve(args):
    root = Path(args.run_root)
    pause = root/'pause.request'
    devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('devices must be a nonempty list of unique CUDA tokens')
    with lock(root/'launcher'):
        # Persistent kernel leases, not PID guesses, guard orphan workers.
        for directory in (root/'workers').glob('job-*'):
            with lock(directory):
                pass
        if args.resume:
            pause.unlink(missing_ok=True)
        elif pause.exists():
            raise RuntimeError('run is paused; add --resume to continue the same run')
        stop = threading.Event()
        active, guard = set(), threading.Lock()
        def stopping():
            return stop.is_set() or pause.exists()
        def request_stop(*_):
            stop.set()
            write_json(pause, dict(reason='supervisor stop', time=datetime.now(timezone.utc).isoformat()))
        old = {s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        jobs = job_list(args.phase)
        def launch(gpu, index):
            job = jobs[index]
            command = [sys.executable, '-m', 'nanogpt_noise.lm_head', '_worker',
                       '--input-root',args.input_root,'--run-root',args.run_root,
                       '--objective',args.objective,'--micro-batch',str(args.micro_batch),
                       '--node',str(job['node']),'--optimizer',job['optimizer'],
                       '--parent-pid',str(os.getpid())]
            if args.test:command.append('--test')
            if args.statistics:command.extend(['--statistics',args.statistics])
            logs = root/'logs';logs.mkdir(exist_ok=True)
            with (logs/f"{args.objective}-job-{job['index']:04d}.log").open('a',encoding='utf-8') as log:
                with guard:
                    if stopping():return 75
                    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONUNBUFFERED':'1'})
                    active.add(process)
                print(f"GPU {gpu}: {args.objective} job {job['index']} pid={process.pid}",flush=True)
                try:return process.wait()
                finally:
                    with guard:active.discard(process)
        try:
            failures = run_pool(range(len(jobs)),devices,launch,
                                on_failure=request_stop,stop_requested=stopping)
            summary = collect(root,args.objective)
            if failures or stopping():
                return dict(status='paused',failures=failures,**summary)
            return dict(status='completed',**summary)
        finally:
            for s,handler in old.items():signal.signal(s,handler)


def worker(args):
    root = Path(args.run_root)
    stop = threading.Event()
    parent_handle = None
    if os.name == 'nt' and args.parent_pid:
        # Windows virtualenv redirectors introduce an intermediate parent.
        # Hold a kernel process handle instead of comparing the immediate PPID.
        import ctypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p,ctypes.c_uint32]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        parent_handle = kernel.OpenProcess(0x00100000,False,args.parent_pid)
        if not parent_handle:
            raise OSError('could not obtain a supervisor process handle')
    def parent_gone():
        if parent_handle:
            return kernel.WaitForSingleObject(parent_handle,0) != 0x102
        return args.parent_pid and os.getppid() != args.parent_pid
    def watch():
        while not stop.wait(1):
            if parent_gone():
                write_json(root/'pause.request',dict(reason='supervisor disappeared'))
                return
    index = args.node*4+list(CENTERS).index(args.optimizer)
    try:
        with lock(root/'workers'/f'job-{args.objective}-{index:04d}'):
            if parent_gone():
                return dict(status='paused')
            thread = threading.Thread(target=watch,daemon=True)
            thread.start()
            try:
                return run_node(dict(index=index,node=args.node,optimizer=args.optimizer),
                                args.input_root,root,args.objective,micro=args.micro_batch,
                                statistics=args.statistics,test=args.test)
            finally:
                stop.set();thread.join()
    finally:
        if parent_handle:
            kernel.CloseHandle(parent_handle)
