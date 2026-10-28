"""Derive v1.7 inputs without resampling data or changing initial weights.

The destination must not exist. Interrupted partial outputs are preserved for
diagnosis; use a fresh destination to retry. Source files are never modified.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import torch
from .artifacts import digest, file_hash, save_checkpoint, write_json
from .data import verify
from .input_bundle import register_bundle, verify_bundle
from .model import Shape
from .protocol import Budget, VERSION


def migrate(source_root, output_root, *, _test_budget=None):
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('source and destination must be separate directories')
    data, artifacts = source/'openwebtext_subset', source/'frozen-artifacts'
    identity = verify(data, artifacts)
    if (source/'registration.json').exists():
        verify_bundle(source)
    target = _test_budget or Budget()
    old = Budget(**identity['budget'])
    if _test_budget is None:
        if (identity['tiny'] or identity['design'] not in ('v1.5', 'v1.6')
                or identity['shape'] != asdict(Shape()) or old != Budget(steps=4960)):
            raise ValueError('migration requires verified v1.5/v1.6 4960-step real inputs')
    elif not identity['tiny']:
        raise ValueError('test migration requires tiny fixtures')
    if old.steps < target.steps or (asdict(old) | {'steps': target.steps}) != asdict(target):
        raise ValueError('only shortening the training horizon is supported')
    indices = torch.load(artifacts/'windows.pt', map_location='cpu', weights_only=True)
    if (set(indices) != {'train', 'eval_train', 'eval_val'}
            or tuple(indices['train'].shape) != (old.steps, old.batch)
            or any(tuple(indices[key].shape) != (old.eval_sequences,)
                   for key in ('eval_train', 'eval_val'))):
        raise ValueError('source window shapes disagree with its budget')
    # mkdir(exist_ok=False) is the no-overwrite boundary, including concurrent calls.
    output.mkdir(parents=True, exist_ok=False)
    dest_data, dest_artifacts = output/'openwebtext_subset', output/'frozen-artifacts'
    dest_data.mkdir()
    dest_artifacts.mkdir()
    for name in ('train.bin', 'val.bin', 'meta.json'):
        shutil.copyfile(data/name, dest_data/name)
    shutil.copyfile(artifacts/'initial.pt', dest_artifacts/'initial.pt')
    cropped = {**indices, 'train': indices['train'][:target.steps].clone()}
    save_checkpoint(dest_artifacts/'windows.pt', cropped)
    # Validate byte preservation before publishing a completion identity.
    for name, expected in identity['data'].items():
        if file_hash(dest_data/name) != expected:
            raise ValueError('copied data hash mismatch')
    if file_hash(dest_artifacts/'initial.pt') != identity['artifacts']['initial.pt']:
        raise ValueError('copied initialization hash mismatch')
    derived = {key: value for key, value in identity.items() if key != 'id'}
    derived.update(design=VERSION, budget=asdict(target),
                   artifacts={name: file_hash(dest_artifacts/name)
                              for name in ('initial.pt', 'windows.pt')},
                   migration=dict(source_input_id=identity['id'], source_design=identity['design'],
                                  source_budget=identity['budget'],
                                  source_windows_sha256=identity['artifacts']['windows.pt'],
                                  operation='training-window prefix; unchanged corpus, initial weights and evaluation windows',
                                  torch=str(torch.__version__)))
    derived['id'] = digest(derived)
    # Detect source changes while copying; retain any partial destination on failure.
    if verify(data, artifacts)['id'] != identity['id']:
        raise ValueError('source identity changed during migration')
    write_json(dest_artifacts/'identity.json', derived)
    return register_bundle(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    print(migrate(args.source_root, args.output_root))


if __name__ == '__main__':
    main()
