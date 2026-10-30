"""Run independent NanoGPT jobs on a dynamic one-job-per-GPU worker pool."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from .artifacts import lock
from .protocol import grid,SCOPES
from .progress import ScopeReporter


def row_batches(start,end):
    """Keep stable optimizer-major IDs; execute sigma1/sigma2 rows in order."""
    jobs=grid()
    groups={}
    for index in range(start,end+1):
        job=jobs[index]
        groups.setdefault((job['scope'],job['node']),[]).append(index)
    return [groups[key] for key in sorted(groups,key=lambda k:(SCOPES.index(k[0]),k[1]))]


def run_rows(start,end,gpus,runner,on_failure=None,stop_requested=None,on_row_complete=None):
    for batch in row_batches(start,end):
        if stop_requested is not None and stop_requested(): return []
        started=time.monotonic()
        failures=run_pool(batch,gpus,runner,on_failure,stop_requested)
        if failures or (stop_requested is not None and stop_requested()): return failures
        if on_row_complete: on_row_complete(batch,time.monotonic()-started)
    return []


def run_pool(jobs, gpus, runner, on_failure=None, stop_requested=None):
    """Run each job once; an idle GPU claims the next queued job."""
    jobs = list(jobs)
    gpus = list(gpus)
    if not gpus:
        raise ValueError('at least one GPU token is required')
    if len(set(gpus)) != len(gpus):
        raise ValueError('GPU tokens must be unique')
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    stop = threading.Event()
    failures = []
    failure_lock = threading.Lock()

    def stopping():
        return stop.is_set() or (stop_requested is not None and stop_requested())

    def worker(gpu):
        while not stopping():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            if stopping():
                pending.task_done()
                return
            try:
                code = runner(gpu, job)
            except Exception:
                code = 1
            finally:
                pending.task_done()
            if code:
                first = False
                with failure_lock:
                    if not failures:
                        failures.append((job, code))
                        stop.set()
                        first = True
                if first and on_failure is not None:
                    on_failure()
                return

    threads = [threading.Thread(target=worker, args=(gpu,), name=f'gpu-{gpu}')
               for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


class SubprocessRunner:
    def __init__(self, args, log_dir, stop_requested=None):
        self.args = args
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.active = set()
        self.lock = threading.RLock()
        self.stop_requested = stop_requested or (lambda: False)
        self.stopping = threading.Event()
        self.jobs = {}

    def snapshot(self):
        with self.lock:
            return dict(jobs={str(i):dict(v) for i,v in self.jobs.items()})

    def __call__(self, gpu, job):
        path = self.log_dir / f'job_{job:04d}.log'
        command = [sys.executable, '-m', 'nanogpt_noise.node',
                   '--data', self.args.data, '--artifacts', self.args.artifacts,
                   '--output', self.args.output, '--micro-batch', str(self.args.micro_batch),
                   '--job-index', str(job)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
        stamp = datetime.now(timezone.utc).isoformat()
        print(f'[pool] GPU {gpu}: start job {job}; log={path}', flush=True)
        with path.open('a', encoding='utf-8') as log:
            log.write(f'\n[{stamp}] GPU token {gpu}: {command!r}\n')
            log.flush()
            with self.lock:
                if self.stopping.is_set() or self.stop_requested():
                    return 75
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=env)
                self.active.add(process)
                started=time.monotonic()
                self.jobs[job]=dict(state='running',gpu=gpu,pid=process.pid)
            try:
                code = process.wait()
            finally:
                with self.lock:
                    self.active.discard(process)
                    self.jobs[job].update(state='exited',exit_code=process.returncode,
                                          elapsed_seconds=time.monotonic()-started)
        print(f'[pool] GPU {gpu}: job {job} exited {code}', flush=True)
        return code

    def terminate_all(self):
        self.stopping.set()
        with self.lock:
            processes = list(self.active)
        for process in processes:
            if process.poll() is None:
                process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    parser.add_argument('--gpus', required=True,
                        help='comma-separated CUDA_VISIBLE_DEVICES tokens')
    parser.add_argument('--data', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--resume-paused', action='store_true')
    args = parser.parse_args()
    if not (0 <= args.start <= args.end <= 839):
        parser.error('job interval must satisfy 0 <= start <= end <= 839')
    if grid()[args.start]['scope']!=grid()[args.end]['scope']:
        parser.error('one pool must stay inside one scope')
    gpus = [item.strip() for item in args.gpus.split(',') if item.strip()]
    if len(gpus) != 4 or len(set(gpus)) != 4:
        parser.error('--gpus must contain exactly four unique tokens')

    with lock(Path(args.output)/'launcher'):
        if (Path(args.output)/'scope_manifest.json').exists():
            raise RuntimeError('scope is already finalized; verify its manifest instead of training')
        serve(args, gpus)


def serve(args, gpus):
    """Called only while owning the scope's launcher lock."""
    pause_file = Path(args.output)/'pause.request'
    if args.resume_paused:
        pause_file.unlink(missing_ok=True)
    elif pause_file.exists():
        raise RuntimeError('scope is paused; use bash scripts/run_acp.sh resume SCOPE')
    interrupted = threading.Event()
    def stopping():
        return interrupted.is_set() or pause_file.exists()
    runner = SubprocessRunner(args, Path(args.output)/'launcher',stop_requested=stopping)
    execution=dict(policy='sigma1-major-row-barrier-v1',row_timings=[])
    def runtime_snapshot():
        return {**execution,**runner.snapshot()}
    reporter=ScopeReporter(args.output,grid()[args.start]['scope'],runtime_snapshot)

    def stop(signum, _frame):
        interrupted.set()
        print(f'[pool] received signal {signum}; stopping active jobs', flush=True)
        runner.terminate_all()

    previous = {number: signal.signal(number, stop)
                for number in (signal.SIGINT, signal.SIGTERM)}
    def stop_on_failure():
        if not pause_file.exists():
            runner.terminate_all()
    try:
        reporter.start()
        def row_complete(batch,elapsed):
            states=runner.snapshot()['jobs']
            busy=sum(states.get(str(i),{}).get('elapsed_seconds',0) for i in batch)
            execution['row_timings']=execution['row_timings']+[
                dict(node=grid()[batch[0]]['node'],wall_seconds=elapsed,
                     idle_gpu_seconds=max(0,len(gpus)*elapsed-busy))]
            reporter.publish('row_complete')
        failures = run_rows(args.start,args.end,gpus,runner,
                            on_failure=stop_on_failure,stop_requested=stopping,on_row_complete=row_complete)
        if pause_file.exists():
            print('SCOPE PAUSED: all worker processes exited; checkpoints retained',flush=True)
            raise SystemExit(75)
        if interrupted.is_set():
            raise SystemExit(130)
        if failures:
            runner.terminate_all()
            job, code = failures[0]
            print(f'[pool] stopped after job {job} failed with exit {code}; '
                  'rerun the same interval to resume', file=sys.stderr, flush=True)
            raise SystemExit(code)
    finally:
        reporter.close('interrupted' if stopping() else 'pool_exit')
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
