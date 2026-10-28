"""Fixed 24-candidate, two-GPU 300M-prefix screening pilot; never formal results."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY = 'joint-prefix-300m-2b-v2'
PREFIX_STEPS = 573
OPS = ('adamw', 'muon', 'adam', 'sgd')
TERMINAL = ('completed', 'numerical_failure')
LIMITATIONS = [
    '300M prefix of the v1.7 2B schedule; not a 300M cosine schedule.',
    'Five versus three INITIAL LR points only; adaptive additions are not tested.',
    'Small-noise comparison uses center LR only, not independently tuned optima.',
    'Does not validate 2B sufficiency, all noise nodes, or cross-seed robustness.',
    'Numerical failure is a scientific outcome; infrastructure failure needs recovery.',
]


def candidates():
    """Pair center clean/noisy first, then cover both omitted and outer LR points."""
    points = ((0, 0.), (1, 0.), (0, -1.), (0, -.5), (0, .5), (0, 1.))
    labels = {0.: '0', -1.: 'm1', -.5: 'm0p5', .5: 'p0p5', 1.: 'p1'}
    return [dict(name=f'{op}-n{node}-x{labels[x]}', optimizer=op,
                 node=node, exponent=x)
            for node, x in points for op in OPS]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_gpus(value):
    tokens = [item.strip() for item in value.split(',')]
    if len(tokens) != 2 or any(not t for t in tokens) or len(set(tokens)) != 2:
        raise ValueError('exactly two distinct allocated GPU tokens are required')
    return tokens


def manifest_payload(input_identity, micro, *, test=False, steps=PREFIX_STEPS):
    import torch
    from dataclasses import asdict
    from nanogpt_noise.artifacts import code_identity
    from nanogpt_noise.protocol import Budget
    if not isinstance(micro, int) or micro < 1:
        raise ValueError('micro-batch must be a positive integer')
    budget = Budget(**input_identity['budget'])
    budget.accumulations(1, micro)
    if input_identity['tiny'] != test:
        raise ValueError('real pilot requires real inputs; tiny inputs are test-only')
    if not test and (budget != Budget() or steps != PREFIX_STEPS):
        raise ValueError('pilot requires the v1.7 2B input budget and 573 steps')
    if not 1 <= steps <= budget.steps:
        raise ValueError('prefix exceeds frozen training windows')
    return dict(policy=POLICY, mode='test' if test else 'pilot_prefix',
                formal_result=False, target_steps=steps,
                target_tokens=steps * budget.step_tokens,
                schedule_budget=asdict(budget), input_id=input_identity['id'],
                micro_batch=micro, package_code=code_identity()['sha256'],
                runner_sha256=fingerprint(__file__),
                runtime=dict(python=sys.version, torch=str(torch.__version__),
                             cuda=torch.version.cuda),
                candidates=candidates(), limitations=LIMITATIONS)


def prepare(output, data, artifacts, micro, *, test=False, steps=PREFIX_STEPS):
    """Caller holds the root lock. Existing input files are verified, never rebuilt."""
    from nanogpt_noise.artifacts import digest, read_json, write_json
    from nanogpt_noise.data import verify
    output = Path(output)
    payload = manifest_payload(verify(data, artifacts), micro, test=test, steps=steps)
    identity = digest(payload)
    path = output / 'pilot_manifest.json'
    if path.exists():
        manifest = read_json(path)
        if manifest.get('identity') != identity or manifest.get('execution') != payload:
            raise ValueError('incompatible pilot resume: inputs, code, runtime, micro or plan changed')
    else:
        if any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError('nonempty output without pilot_manifest.json; use a new directory')
        manifest = dict(identity=identity, execution=payload,
                        created_utc=datetime.now(timezone.utc).isoformat())
        write_json(path, manifest)
    return manifest


def config_for(spec, manifest):
    from nanogpt_noise.protocol import configuration
    config = configuration('full', spec['optimizer'], spec['node'], spec['exponent'])
    config['pilot'] = dict(policy=POLICY, manifest_id=manifest['identity'],
                           candidate=spec['name'], formal_result=False,
                           prefix_steps=manifest['execution']['target_steps'])
    return config


def validate_manifest(manifest):
    from nanogpt_noise.artifacts import code_identity, digest
    payload = manifest['execution']
    if digest(payload) != manifest['identity'] or payload['policy'] != POLICY:
        raise ValueError('pilot manifest identity mismatch')
    if payload['candidates'] != candidates():
        raise ValueError('unexpected pilot candidate plan')
    if payload['package_code'] != code_identity()['sha256'] or payload['runner_sha256'] != fingerprint(__file__):
        raise ValueError('pilot code changed; restore the original code before resuming')


def expected_execution(spec, manifest):
    payload = manifest['execution']
    test = payload['mode'] == 'test'
    return dict(config=config_for(spec, manifest), mode='test' if test else 'smoke',
                artifact_id=payload['input_id'], code=payload['package_code'],
                world=1, micro=payload['micro_batch'], device_type='cpu' if test else 'cuda',
                torch=payload['runtime']['torch'], cuda=payload['runtime']['cuda'],
                target_steps=payload['target_steps'])


def read_result(spec, manifest, output):
    """Never trust a completed marker without matching candidate provenance."""
    from nanogpt_noise.artifacts import digest, read_json
    folder = Path(output) / spec['name']
    if not (folder / 'status.json').exists():
        return None
    result = read_json(folder / 'status.json')
    if result.get('status') not in TERMINAL:
        return result
    meta = read_json(folder / 'metadata.json')
    execution = expected_execution(spec, manifest)
    if (meta['execution'] != execution or meta['run_id'] != digest(execution)
            or result.get('run_id') != meta['run_id'] or result.get('mode') != execution['mode']):
        raise ValueError(f'{folder}: candidate provenance mismatch')
    payload = manifest['execution']
    if result['status'] == 'completed':
        if (result['step'] != payload['target_steps']
                or result['tokens'] != payload['target_tokens']
                or not isinstance(result.get('loss'), (int, float))
                or not math.isfinite(result['loss'])):
            raise ValueError(f'{folder}: invalid prefix completion')
        if not (folder / 'final.pt').is_file():
            raise ValueError(f'{folder}: completed prefix checkpoint is missing')
        evaluations = read_json(folder / 'evaluations.json')
        if (not evaluations or evaluations[-1]['step'] != payload['target_steps']
                or evaluations[-1]['val'] != result['loss']):
            raise ValueError(f'{folder}: terminal evaluation mismatch')
    return result


def execute_candidate(spec, manifest, data, artifacts, output, *, pause_after=None):
    import torch
    from nanogpt_noise.artifacts import read_json
    from nanogpt_noise.storage import compact_steps
    from nanogpt_noise.train import run
    validate_manifest(manifest)
    payload = manifest['execution']
    runtime = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda)
    if runtime != payload['runtime'] or read_json(Path(artifacts) / 'identity.json')['id'] != payload['input_id']:
        raise ValueError('worker runtime or input identity differs from the pilot manifest')
    print(f"[pilot] {spec['name']}: {payload['target_steps']} steps, original schedule", flush=True)
    result = run(config_for(spec, manifest), data, artifacts, Path(output) / spec['name'],
                 micro=payload['micro_batch'], smoke_steps=payload['target_steps'],
                 test=payload['mode'] == 'test', pause_after=pause_after)
    if result['status'] in TERMINAL:
        read_result(spec, manifest, output)
        compact_steps(Path(output) / spec['name'])
    return result


def diagnostic_summary(folder):
    """Read both active JSON and terminal compressed diagnostics without losing fields."""
    folder = Path(folder)
    archive = folder / 'steps.jsonl.gz'
    if archive.exists():
        from nanogpt_noise.artifacts import file_hash, read_json
        marker = folder / 'steps.compaction.json'
        if marker.exists() and read_json(marker)['sha256'] != file_hash(archive):
            raise ValueError(f'{folder}: diagnostic archive hash mismatch')
        with gzip.open(archive, 'rt', encoding='utf8') as stream:
            records = [json.loads(line) for line in stream]
    else:
        from nanogpt_noise.artifacts import read_json
        try:
            records = [read_json(p) for p in sorted((folder / 'steps').glob('*.json'))]
        except FileNotFoundError:
            if archive.exists():
                return diagnostic_summary(folder)
            raise
    summary = dict(logged_steps=len(records))
    for key in ('gradient_norm', 'noise_rms', 'noise_norm', 'actual_ratio',
                'update_norm', 'displacement'):
        values = [r[key] for r in records if isinstance(r.get(key), (int, float))
                  and math.isfinite(r[key])]
        summary[key] = (dict(min=min(values), median=statistics.median(values),
                             max=max(values), last=values[-1]) if values else None)
    return summary


def write_csv(path, rows, fields):
    from nanogpt_noise.artifacts import atomic
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    atomic(path, lambda stream: stream.write(buffer.getvalue().encode('utf8')))


def summarize(output, manifest):
    from nanogpt_noise.artifacts import atomic, read_json, write_json
    from nanogpt_noise.search import winner
    output = Path(output)
    rows, curve_rows = [], []
    for spec in candidates():
        result = read_result(spec, manifest, output)
        unknown = 'running_or_interrupted' if (output / spec['name'] / 'metadata.json').exists() else 'not_started'
        row = dict(**spec, status=result['status'] if result else unknown,
                   x=spec['exponent'], peak_lr=config_for(spec, manifest)['peak_lr'],
                   loss=result.get('loss') if result and result['status'] == 'completed' else None,
                   step=result.get('step') if result else None,
                   tokens=result.get('tokens') if result else None,
                   failure_field=result.get('field') if result else None,
                   failure_reason=result.get('reason') if result else None,
                   T_diverge=result.get('T_diverge') if result else None)
        folder = output / spec['name']
        if result and result['status'] in TERMINAL:
            row['diagnostics'] = diagnostic_summary(folder)
        if (folder / 'evaluations.json').exists():
            row['evaluations'] = read_json(folder / 'evaluations.json')
            curve_rows += [dict(candidate=spec['name'], optimizer=spec['optimizer'],
                                node=spec['node'], exponent=spec['exponent'], **e)
                           for e in row['evaluations']]
        rows.append(row)
    lr, pairs = {}, {}
    for op in OPS:
        clean = [r for r in rows if r['optimizer'] == op and r['node'] == 0]
        ready = all(r['status'] in TERMINAL for r in clean)
        fine = winner(clean) if ready else None
        coarse = winner([r for r in clean if r['x'] in (-1., 0., 1.)]) if ready else None
        lr[op] = dict(all_five_terminal=ready,
                      five_initial_winner=fine['name'] if fine else None,
                      three_initial_winner=coarse['name'] if coarse else None,
                      five_initial_loss=fine['loss'] if fine else None,
                      three_initial_loss=coarse['loss'] if coarse else None,
                      three_initial_minus_five_initial_loss=(coarse['loss'] - fine['loss'])
                      if fine and coarse else None,
                      coarse_failed_but_middle_valid=bool(ready and fine and not coarse),
                      five_initial_winner_at_outer_boundary=bool(fine and abs(fine['x']) == 1),
                      valid_five=sum(r['status'] == 'completed' for r in clean),
                      missing_candidates=[r['name'] for r in clean if r['status'] not in TERMINAL],
                      adaptive_search_tested=False)
        center = next(r for r in clean if r['x'] == 0)
        noisy = next(r for r in rows if r['optimizer'] == op and r['node'] == 1)
        both = center['status'] == noisy['status'] == 'completed'
        ce = {e['tokens']: e for e in center.get('evaluations', [])}
        ne = {e['tokens']: e for e in noisy.get('evaluations', [])}
        pairs[op] = dict(clean_status=center['status'], noisy_status=noisy['status'],
                         comparison='same center peak LR, paired inputs, early prefix only',
                         final_loss_noisy_minus_clean=noisy['loss'] - center['loss'] if both else None,
                         final_update_norm_noisy_minus_clean=None,
                         shared_evaluations=[dict(tokens=t, clean_val=ce[t]['val'],
                                                  noisy_val=ne[t]['val'],
                                                  noisy_minus_clean=ne[t]['val'] - ce[t]['val'])
                                             for t in sorted(ce.keys() & ne.keys())])
        if both:
            a = center['diagnostics']['update_norm']
            b = noisy['diagnostics']['update_norm']
            if a and b:
                pairs[op]['final_update_norm_noisy_minus_clean'] = b['last'] - a['last']
    counts = {state: sum(r['status'] == state for r in rows)
              for state in ('completed', 'numerical_failure', 'infrastructure_failure',
                            'not_started', 'running_or_interrupted')}
    report = dict(mode=manifest['execution']['mode'], policy=POLICY, formal_result=False,
                  manifest_id=manifest['identity'], complete=all(r['status'] in TERMINAL for r in rows),
                  counts=counts, prefix_steps=manifest['execution']['target_steps'],
                  prefix_tokens=manifest['execution']['target_tokens'],
                  updated_utc=datetime.now(timezone.utc).isoformat(),
                  limitations=LIMITATIONS, candidates=rows,
                  clean_initial_LR_comparison=lr, small_noise_center_pairs=pairs)
    write_json(output / 'summary.json', report)
    write_csv(output / 'candidates.csv', rows,
              ['name', 'optimizer', 'node', 'exponent', 'peak_lr', 'status',
               'step', 'tokens', 'loss', 'failure_field', 'failure_reason', 'T_diverge'])
    write_csv(output / 'evaluations.csv', curve_rows,
              ['candidate', 'optimizer', 'node', 'exponent', 'step', 'tokens', 'train', 'val'])
    lines = [f'JOINT PREFIX PILOT: complete={report["complete"]}, counts={counts}',
             'NOT formal results. See summary.json for curves, failures and comparisons.', *LIMITATIONS]
    for op in OPS:
        a, b = lr[op], pairs[op]
        lines += [f'{op}: initial-grid loss gap={a["three_initial_minus_five_initial_loss"]}; '
                  f'coarse_failed_middle_valid={a["coarse_failed_but_middle_valid"]}; '
                  f'outer_boundary={a["five_initial_winner_at_outer_boundary"]}',
                  f'{op}: small-noise minus clean center loss={b["final_loss_noisy_minus_clean"]}']
    atomic(output / 'report.txt', lambda stream: stream.write(('\n'.join(lines) + '\n').encode('utf8')))
    return report


def analysis_bundle(output):
    """Collect small analysis inputs only; checkpoints remain in their original folders."""
    from nanogpt_noise.artifacts import atomic
    output = Path(output)
    paths = [output / name for name in ('pilot_manifest.json', 'pilot_status.json',
             'summary.json', 'candidates.csv', 'evaluations.csv', 'report.txt')]
    paths += list((output / 'launcher').glob('*.log'))
    for spec in candidates():
        folder = output / spec['name']
        paths += [folder / name for name in ('metadata.json', 'status.json', 'evaluations.json',
                  'steps.compaction.json', 'steps.jsonl.gz')]
        paths += list((folder / 'steps').glob('*.json'))
    def write(stream):
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                if path.is_file():
                    archive.write(path, path.relative_to(output).as_posix())
    atomic(output / 'analysis_bundle.zip', write)


def supervise(jobs, gpus, launch, *, stop_requested=lambda: False, grace=120.):
    """Single-threaded queue: no new work after a signal or infrastructure failure."""
    pending = list(jobs)
    active = {}
    stopping = False
    deadline = None
    failure = 0
    while pending or active:
        if stop_requested() and not stopping:
            failure = failure or 130
            stopping = True
        for gpu, child in list(active.items()):
            code = child.poll()
            if code is not None:
                del active[gpu]
                if code:
                    failure = failure or code
                    stopping = True
        if stopping:
            if deadline is None:
                deadline = time.monotonic() + grace
                for child in active.values():
                    child.terminate()
            if time.monotonic() >= deadline:
                for child in active.values():
                    if child.poll() is None:
                        child.kill()
            if not active:
                break
        else:
            for gpu in gpus:
                if stop_requested():
                    break
                if pending and gpu not in active:
                    try:
                        active[gpu] = launch(gpu, pending.pop(0))
                    except Exception:
                        for child in active.values():
                            child.terminate()
                            child.wait()
                        raise
        if pending or active:
            time.sleep(.5)
    return failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('run directly, not with torchrun or DDP')
    if args.micro_batch < 1 or 512 % args.micro_batch:
        parser.error('micro-batch must divide 512')
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(dict(policy=POLICY, mode='pilot_prefix', steps=PREFIX_STEPS,
                              tokens_per_candidate=PREFIX_STEPS * 524288,
                              total_tokens=24 * PREFIX_STEPS * 524288,
                              candidates=candidates(), limitations=LIMITATIONS), indent=2))
        return 0
    from nanogpt_noise.artifacts import lock, read_json, write_json
    output = args.output.resolve()
    if args.worker:
        manifest = read_json(output / 'pilot_manifest.json')
        spec = next((s for s in candidates() if s['name'] == args.worker), None)
        if spec is None:
            parser.error('unknown candidate')
        result = execute_candidate(spec, manifest, args.data, args.artifacts, output)
        return 0 if result['status'] in TERMINAL else 75
    if args.summarize:
        manifest = read_json(output / 'pilot_manifest.json')
        validate_manifest(manifest)
        report = summarize(output, manifest)
        print(json.dumps(dict(complete=report['complete'], counts=report['counts']), indent=2))
        print(f'REPORT: {output / "summary.json"}')
        return 0
    with lock(output):
        manifest = prepare(output, args.data, args.artifacts, args.micro_batch)
        print(f'PILOT OUTPUT: {output}', flush=True)
        print('Resume with the SAME PILOT_OUT and unchanged code/input/micro.', flush=True)
        pending = []
        from nanogpt_noise.storage import compact_steps
        for spec in candidates():
            result = read_result(spec, manifest, output)
            if result and result['status'] in TERMINAL:
                compact_steps(output / spec['name'])
            else:
                pending.append(spec)
        summarize(output, manifest)
        (output / 'launcher').mkdir(exist_ok=True)
        interrupted = [False]
        logs = []

        def stop(_signum, _frame):
            interrupted[0] = True

        def launch(gpu, spec):
            current = summarize(output, manifest)
            print(f'PROGRESS: {current["counts"]}', flush=True)
            log = (output / 'launcher' / f'{spec["name"]}.log').open('a', encoding='utf8')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', spec['name'],
                       '--data', str(args.data.resolve()), '--artifacts', str(args.artifacts.resolve()),
                       '--output', str(output), '--micro-batch', str(args.micro_batch),
                       '--gpus', args.gpus]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(key, None)
            log.write(f'\n{datetime.now(timezone.utc).isoformat()} GPU={gpu} {command!r}\n')
            log.flush()
            print(f'START {spec["name"]} GPU={gpu} log={log.name}', flush=True)
            return subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            code = supervise(pending, gpus, launch, stop_requested=lambda: interrupted[0])
        finally:
            for log in logs:
                log.close()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            report = summarize(output, manifest)
            write_json(output / 'pilot_status.json', dict(complete=report['complete'],
                       interrupted=interrupted[0], counts=report['counts']))
            analysis_bundle(output)
        print(f'REPORT: {output / "summary.json"}', flush=True)
        print(f'SEND FOR ANALYSIS: {output / "analysis_bundle.zip"}', flush=True)
        if code or not report['complete']:
            print('PILOT INCOMPLETE: read launcher logs; rerun the same output to resume.', flush=True)
            return 75 if code != 130 else 130
        print(f'PILOT FINISHED: {report["counts"]}. Numerical failures remain explicitly labeled.', flush=True)
        return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'PILOT ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
