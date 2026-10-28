"""Short two-GPU timing experiment; does not certify convergence or formal runs."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OPS = ('adamw', 'muon', 'adam', 'sgd')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def estimate(seconds_per_step, event_seconds, setup_seconds, tokens):
    steps = math.ceil(tokens / 524288)
    # Initial evaluation, each 100M boundary, and final evaluation.
    events = 1 + (steps * 524288 // 100_000_000)
    if steps * 524288 // 100_000_000 == (steps - 1) * 524288 // 100_000_000:
        events += 1
    return steps * seconds_per_step + events * event_seconds + setup_seconds


def worker(a):
    sys.path.insert(0, str(ROOT))
    import torch
    from nanogpt_noise import train
    from nanogpt_noise.protocol import configuration
    times = {'eval': [], 'checkpoint': []}
    for name, key in [('evaluate', 'eval'), ('save_checkpoint', 'checkpoint')]:
        original = getattr(train, name)

        def timed(*args, _original=original, _key=key, **kwargs):
            torch.cuda.synchronize()
            start = time.monotonic()
            value = _original(*args, **kwargs)
            torch.cuda.synchronize()
            times[_key].append(time.monotonic() - start)
            return value

        setattr(train, name, timed)
    start = time.monotonic()
    result = train.run(configuration(a.scope, a.worker, 10, 0), a.data,
                       a.artifacts, a.output, micro=a.micro, smoke_steps=a.steps)
    torch.cuda.synchronize()
    wall = time.monotonic() - start
    if result['status'] != 'completed':
        raise RuntimeError(f"benchmark failed: {result['status']}")
    rows = [json.loads(p.read_text()) for p in sorted((a.output / 'steps').glob('*.json'))]
    # Exclude steps 1-3; include ordinary and diagnostic steps thereafter.
    deltas = [rows[i]['elapsed_seconds'] - rows[i-1]['elapsed_seconds']
              for i in range(3, len(rows))]
    mean = statistics.mean(deltas)
    # Initial checkpoint lacks initialized optimizer state; use the slower event
    # rather than averaging it with the representative final checkpoint.
    event = max(times['eval']) + max(times['checkpoint'])
    # Conservative startup allowance: includes warmup and any timing residual.
    setup = max(0, wall - sum(times['eval']) - sum(times['checkpoint'])
                - mean * a.steps)
    report = dict(scope=a.scope, optimizer=a.worker, gpu=torch.cuda.get_device_name(0),
                  mode='smoke_timing_only', steps=a.steps, micro=a.micro,
                  seconds_per_step=mean, median_seconds_per_step=statistics.median(deltas),
                  min_seconds_per_step=min(deltas), max_seconds_per_step=max(deltas),
                  training_tokens_per_second=524288 / mean,
                  peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  event_seconds=event, startup_seconds=setup, wall_seconds=wall,
                  measured_events=times,
                  candidate_2B_seconds=estimate(mean, event, setup, 2_000_000_000))
    write(a.output / 'timing.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--micro', type=int, default=4)
    p.add_argument('--minutes', type=float, default=30)
    p.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1'))
    p.add_argument('--worker', choices=OPS, help=argparse.SUPPRESS)
    p.add_argument('--scope', choices=('full','last1','last2','last3','last6','last12'), default='last1')
    a = p.parse_args()
    if a.worker:
        worker(a)
        return
    if not 12 <= a.steps <= 100 or a.micro < 1 or 512 % a.micro or a.minutes <= 0:
        p.error('steps must be 12..100, micro must divide 512, minutes must be positive')
    cards = a.gpus.split(',')
    if len(cards) != 2 or len(set(cards)) != 2 or not all(c.strip() for c in cards):
        p.error('exactly two distinct allocated GPUs are required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('run directly, not with torchrun or DDP')
    for path in [a.data/'train.bin', a.data/'val.bin', a.artifacts/'initial.pt',
                 a.artifacts/'windows.pt', a.artifacts/'identity.json']:
        if not path.is_file():
            p.error(f'missing input: {path}')
    a.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + a.minutes * 60
    outcomes = {}
    active = []
    try:
        for pair in (OPS[:2], OPS[2:]):
            if time.monotonic() >= deadline:
                break
            active = []
            for op, gpu in zip(pair, cards):
                folder = a.output/op
                folder.mkdir()
                log = (folder/'console.log').open('w', encoding='utf8')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.strip(),
                           OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
                    env.pop(key, None)
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', op,
                           '--data', str(a.data.resolve()), '--artifacts', str(a.artifacts.resolve()),
                           '--output', str(folder.resolve()), '--steps', str(a.steps),
                           '--micro', str(a.micro), '--scope', a.scope]
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
                active.append((op, proc, log))
                print(f'START {op}, GPU={gpu}, log={folder / "console.log"}', flush=True)
            while any(proc.poll() is None for _, proc, _ in active):
                if time.monotonic() >= deadline:
                    break
                time.sleep(1)
            for op, proc, log in active:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    outcomes[op] = 'time_limit'
                else:
                    outcomes[op] = f'exit_{proc.returncode}'
                log.close()
            active = []
    finally:
        for _, proc, log in active:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            log.close()
    reports = {}
    for op in OPS:
        path = a.output/op/'timing.json'
        if path.exists():
            reports[op] = json.loads(path.read_text(encoding='utf8'))
    summary = dict(mode='smoke_timing_only', scope=a.scope, outcomes=outcomes, measurements=reports,
                   complete=len(reports)==4,
                   assumptions='Two independent GPU workers, balanced queue, no retries; '
                   'Measured scope only. Cold-start short-run extrapolation, not convergence evidence.')
    if len(reports) == 4:
        sum2 = sum(r['candidate_2B_seconds'] for r in reports.values())
        estimates = {
            'four_candidates_2B': sum2 / 2 / 86400,
            'scope_35nodes_8LRs_2B': 35*8*sum2 / 2 / 86400,
        }
        summary['estimated_days_two_GPUs'] = estimates
        summary['planning_days_plus_20percent_not_confidence_interval'] = {
            k: v*1.2 for k,v in estimates.items()}
        for name, days in estimates.items():
            print(f'{name}: {days:.2f} days; planning allowance {days*1.2:.2f} days')
    else:
        print('INCOMPLETE: no aggregate estimate. Send summary.json and all console.log files.')
    write(a.output/'summary.json', summary)
    print(f'REPORT: {a.output / "summary.json"}', flush=True)
    if len(reports) != 4:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
