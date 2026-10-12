"""Prepare, estimate, close and verify one independently transferred scope."""
import argparse
from pathlib import Path
import shutil

from .artifacts import file_hash, read_json, write_json
from .protocol import SCOPES


JOBS_PER_SCOPE = 140
MODEL_PARAMETERS = 124_373_760
BLOCK_PARAMETERS = 7_079_424
BLOCK_MATRIX_PARAMETERS = 7_077_888
ACTIVE_WORKERS = 4
SHARED_AND_LOG_OVERHEAD_BYTES = 4_000_000_000


def scope_bounds(scope):
    if scope not in SCOPES:
        raise ValueError(f'unknown scope: {scope}')
    start = SCOPES.index(scope)*JOBS_PER_SCOPE
    return start, start+JOBS_PER_SCOPE-1


def scope_indices(scope):
    start, end = scope_bounds(scope)
    return range(start, end+1)


def _checkpoint_bytes(scope, optimizer):
    model = 4*MODEL_PARAMETERS
    trainable = MODEL_PARAMETERS if scope == 'full' else BLOCK_PARAMETERS*int(scope[4:])
    if optimizer in ('adamw', 'adam'):
        state = 8*trainable
    elif optimizer == 'sgd':
        state = 4*trainable
    elif optimizer == 'muon':
        if scope == 'full':
            main = 12*BLOCK_MATRIX_PARAMETERS
            auxiliary = trainable-main
        else:
            layers = int(scope[4:])
            main = layers*BLOCK_MATRIX_PARAMETERS
            auxiliary = trainable-main
        state = 4*main+8*auxiliary
    else:
        raise ValueError(optimizer)
    return model+state


def estimate_scope(scope):
    checkpoints = {name:_checkpoint_bytes(scope, name)
                   for name in ('adamw', 'adam', 'sgd', 'muon')}
    # 35 noise nodes per optimizer. During an atomic evaluation write, each of
    # four active jobs can temporarily hold winner final + old latest + new temp.
    retained = 35*sum(checkpoints.values())
    transient = ACTIVE_WORKERS*2*max(checkpoints.values())
    peak = retained+transient+SHARED_AND_LOG_OVERHEAD_BYTES
    return dict(scope=scope, retained_checkpoint_bytes=retained,
                active_write_headroom_bytes=transient,
                shared_and_log_headroom_bytes=SHARED_AND_LOG_OVERHEAD_BYTES,
                peak_bytes=peak, checkpoint_bytes=checkpoints,
                assumptions=dict(active_workers=ACTIVE_WORKERS,
                                 jobs=JOBS_PER_SCOPE,
                                 one_winner_checkpoint_per_job=True))


def _job_directories(root):
    result = []
    for path in Path(root).glob('job_[0-9][0-9][0-9][0-9]'):
        if path.is_dir():
            result.append((int(path.name[4:]), path))
    return result


def prepare_scope(root, scope):
    root = Path(root)
    if root.resolve().name != scope:
        raise ValueError(f'scope output directory must be named exactly {scope}')
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root/'scope_run.json'
    expected = dict(format='nanogpt-scope-run-v1', scope=scope,
                    first_job=scope_bounds(scope)[0], last_job=scope_bounds(scope)[1])
    if marker_path.exists():
        actual = read_json(marker_path)
        if actual != expected:
            raise ValueError(f'{root} belongs to scope {actual.get("scope")}, not {scope}')
    else:
        write_json(marker_path, expected)
    allowed = set(scope_indices(scope))
    outside = [index for index, _ in _job_directories(root) if index not in allowed]
    if outside:
        raise ValueError(f'{root}: job {outside[0]:04d} is outside scope {scope}')
    if (root/'scope_manifest.json').exists():
        raise ValueError(f'{root}: scope is already finalized; use a new directory')
    return expected


def _manifest_files(root):
    excluded = {'scope_manifest.json', '.lock'}
    files = []
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size,
                          sha256=file_hash(path)))
    return files


def finalize_scope(root, scope):
    root = Path(root)
    marker = read_json(root/'scope_run.json')
    if marker.get('scope') != scope:
        raise ValueError(f'{root} belongs to scope {marker.get("scope")}, not {scope}')
    leftovers = [path for path in root.rglob('*')
                 if path.is_file() and (path.name == '.lock' or path.name.endswith('.tmp'))]
    if leftovers:
        raise ValueError(f'{leftovers[0]}: active/stale lock or atomic temporary file remains')
    expected_jobs = set(scope_indices(scope))
    present_jobs = {index for index, _ in _job_directories(root)}
    if present_jobs != expected_jobs:
        missing = sorted(expected_jobs-present_jobs)
        extra = sorted(present_jobs-expected_jobs)
        raise ValueError(f'scope jobs incomplete: missing={missing[:5]} extra={extra[:5]}')
    for index in sorted(expected_jobs):
        job = root/f'job_{index:04d}'
        search = read_json(job/'search.json')
        if not search.get('complete'):
            raise ValueError(f'{job}: LR search is incomplete')
        candidate_ids = [item['candidate'] for item in search.get('candidates', [])]
        if not candidate_ids:
            raise ValueError(f'{job}: no candidate records')
        for candidate_id in candidate_ids:
            candidate = job/f'candidate_{candidate_id:02d}'
            if not (candidate/'steps.jsonl.gz').exists() or not (candidate/'steps.compaction.json').exists():
                raise ValueError(f'{candidate}: diagnostics are not compacted')
            if (candidate/'steps').exists():
                raise ValueError(f'{candidate}: individual step files remain')
        winner = search.get('winner')
        expected_checkpoint = (job/f"candidate_{winner['candidate']:02d}"/'final.pt'
                               if winner else None)
        checkpoints = list(job.rglob('*.pt'))
        if expected_checkpoint is None and checkpoints:
            raise ValueError(f'{job}: no-valid-LR job must not retain checkpoints')
        if expected_checkpoint is not None and checkpoints != [expected_checkpoint]:
            raise ValueError(f'{job}: checkpoint retention does not match winner')
    files = _manifest_files(root)
    manifest = dict(format='nanogpt-scope-manifest-v1', scope=scope,
                    first_job=marker['first_job'], last_job=marker['last_job'],
                    estimate=estimate_scope(scope), file_count=len(files), files=files)
    write_json(root/'scope_manifest.json', manifest)
    return manifest


def verify_scope(root):
    root = Path(root)
    manifest = read_json(root/'scope_manifest.json')
    expected = {item['path']:item for item in manifest['files']}
    actual = {item['path']:item for item in _manifest_files(root)}
    if set(actual) != set(expected):
        missing = sorted(set(expected)-set(actual))
        extra = sorted(set(actual)-set(expected))
        raise ValueError(f'manifest file set mismatch: missing={missing[:5]} extra={extra[:5]}')
    for relative, item in expected.items():
        found = actual[relative]
        if found['size'] != item['size']:
            raise ValueError(f'{relative}: size mismatch')
        if found['sha256'] != item['sha256']:
            raise ValueError(f'{relative}: hash mismatch')
    return manifest


def cleanup_scope(root, scope, confirmation):
    """Remove one finalized scope only after an explicit scope-specific token."""
    root = Path(root)
    resolved = root.resolve()
    if confirmation != f'DELETE-{scope}':
        raise ValueError(f'confirmation must be DELETE-{scope}')
    if root.is_symlink() or resolved.name != scope:
        raise ValueError('cleanup path must be a real directory named exactly as its scope')
    protected = {Path('/').resolve(), Path.home().resolve(), Path.cwd().resolve(),
                 Path(__file__).resolve().parents[1]}
    if resolved in protected or resolved.parent in protected:
        raise ValueError('refusing to remove a protected or overly broad directory')
    manifest = verify_scope(resolved)
    if manifest.get('scope') != scope:
        raise ValueError(f'{resolved} belongs to scope {manifest.get("scope")}, not {scope}')
    shutil.rmtree(resolved)
    return dict(scope=scope, removed=str(resolved))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'finalize'):
        child = sub.add_parser(name)
        child.add_argument('--runs', required=True)
        child.add_argument('--scope', required=True, choices=SCOPES)
    child = sub.add_parser('verify')
    child.add_argument('--runs', required=True)
    child = sub.add_parser('estimate')
    child.add_argument('--scope', required=True, choices=SCOPES)
    child = sub.add_parser('cleanup')
    child.add_argument('--runs', required=True)
    child.add_argument('--scope', required=True, choices=SCOPES)
    child.add_argument('--confirm', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_scope(args.runs, args.scope)
        result['estimate'] = estimate_scope(args.scope)
    elif args.command == 'finalize':
        result = finalize_scope(args.runs, args.scope)
    elif args.command == 'verify':
        result = verify_scope(args.runs)
    elif args.command == 'cleanup':
        result = cleanup_scope(args.runs, args.scope, args.confirm)
    else:
        result = estimate_scope(args.scope)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
