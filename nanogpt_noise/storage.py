"""Lossless diagnostic compaction and winner-only checkpoint retention."""
import gzip
import json
import os
from pathlib import Path

from .artifacts import atomic, file_hash, read_json, write_json


CHECKPOINTS = ('latest.pt', 'best.pt', 'final.pt', 'failure.pt')


def _read_archive(path):
    records = []
    with gzip.open(path, 'rt', encoding='utf8') as source:
        for line_number, line in enumerate(source, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}: invalid JSONL line {line_number}') from exc
    return records


def _validate_archive(candidate, manifest):
    archive = candidate/'steps.jsonl.gz'
    if not archive.exists():
        raise ValueError(f'{archive}: missing compacted diagnostics')
    if file_hash(archive) != manifest['sha256']:
        raise ValueError(f'{archive}: compacted diagnostics hash mismatch')
    records = _read_archive(archive)
    if len(records) != manifest['count']:
        raise ValueError(f'{archive}: compacted diagnostic count mismatch')
    steps = [record.get('step') for record in records]
    if steps != list(range(1, len(records)+1)):
        raise ValueError(f'{archive}: diagnostic steps are not contiguous from one')
    return records


def compact_steps(candidate):
    """Atomically replace per-step JSON files with verified gzip JSONL."""
    candidate = Path(candidate)
    archive = candidate/'steps.jsonl.gz'
    manifest_path = candidate/'steps.compaction.json'
    steps_dir = candidate/'steps'
    sources = sorted(steps_dir.glob('*.json')) if steps_dir.exists() else []

    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _validate_archive(candidate, manifest)
    else:
        records = []
        for expected, path in enumerate(sources, 1):
            if path.name != f'{expected:06d}.json':
                raise ValueError(f'{path}: diagnostic filenames are not contiguous from one')
            record = read_json(path)
            if record.get('step') != expected:
                raise ValueError(f'{path}: diagnostic step does not match filename')
            records.append(record)

        def write_archive(raw):
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as compressed:
                for record in records:
                    line = json.dumps(record, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, separators=(',', ':'))
                    compressed.write(line.encode('utf8')+b'\n')

        atomic(archive, write_archive)
        manifest = dict(format='gzip-jsonl-v1', count=len(records),
                        first_step=records[0]['step'] if records else None,
                        last_step=records[-1]['step'] if records else None,
                        sha256=file_hash(archive))
        _validate_archive(candidate, manifest)
        write_json(manifest_path, manifest)

    # Source deletion happens only after both archive and manifest verify.
    for path in sources:
        path.unlink()
    if steps_dir.exists():
        try:
            steps_dir.rmdir()
        except OSError as exc:
            raise ValueError(f'{steps_dir}: unexpected files prevent compaction cleanup') from exc
    return manifest


def retain_winner_checkpoint(job_directory, winner):
    """Keep only the winner's final checkpoint; remove known transient files."""
    job_directory = Path(job_directory)
    for candidate in sorted(job_directory.glob('candidate_[0-9][0-9]')):
        try:
            index = int(candidate.name.split('_')[1])
        except (IndexError, ValueError):
            continue
        for name in CHECKPOINTS:
            if index == winner and name == 'final.pt':
                continue
            (candidate/name).unlink(missing_ok=True)
