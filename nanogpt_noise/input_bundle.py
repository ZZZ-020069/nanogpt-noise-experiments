"""Register and verify one immutable input realization for every formal scope."""
import argparse
from pathlib import Path
from .artifacts import atomic, file_hash, read_json, write_json
from .data import verify

FILES = (
    'openwebtext_subset/train.bin',
    'openwebtext_subset/val.bin',
    'openwebtext_subset/meta.json',
    'frozen-artifacts/initial.pt',
    'frozen-artifacts/windows.pt',
    'frozen-artifacts/identity.json',
)


def _paths(input_root, data=None, artifacts=None):
    root = Path(input_root)
    return root, Path(data or root/'openwebtext_subset'), Path(artifacts or root/'frozen-artifacts')


def _registration(identity):
    meta = identity['meta']
    provenance_keys = ('source_dataset','source_revision','source_split','tokenizer',
                       'shuffle_seed','shuffle_buffer','dependencies')
    return dict(schema=1, input_id=identity['id'], design=identity['design'],
                provenance={key:meta.get(key) for key in provenance_keys},
                lengths=identity['lengths'], data=identity['data'],
                artifacts=identity['artifacts'])


def register_bundle(input_root, data=None, artifacts=None):
    root, data, artifacts = _paths(input_root, data, artifacts)
    identity = verify(data, artifacts)
    value = _registration(identity)
    lines = ''.join(f'{file_hash(root/path)}  {path}\n' for path in FILES)
    sums = root/'SHA256SUMS'
    if sums.exists() and sums.read_text(encoding='utf8') != lines:
        raise ValueError('SHA256SUMS already describes different input files')
    if not sums.exists():
        atomic(sums, lambda stream: stream.write(lines.encode('utf8')))
    # registration.json is the completion marker and is therefore written last.
    registration = root/'registration.json'
    if registration.exists():
        if read_json(registration) != value:
            raise ValueError('registration.json already binds a different input realization')
    else:
        write_json(registration, value)
    return verify_bundle(root, data, artifacts)


def verify_bundle(input_root, data=None, artifacts=None):
    root, data, artifacts = _paths(input_root, data, artifacts)
    registration = read_json(root/'registration.json')
    identity = verify(data, artifacts)
    if registration != _registration(identity):
        raise ValueError('registered input identity does not match the current files')
    for relative in FILES:
        if not (root/relative).is_file():
            raise FileNotFoundError(root/relative)
    expected_lines = ''.join(f'{file_hash(root/path)}  {path}\n' for path in FILES)
    if (root/'SHA256SUMS').read_text(encoding='utf8') != expected_lines:
        raise ValueError('SHA256SUMS does not match the registered files')
    return registration['input_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'verify'))
    parser.add_argument('--input-root', required=True)
    args = parser.parse_args()
    action = register_bundle if args.command == 'register' else verify_bundle
    print(action(args.input_root))


if __name__ == '__main__':
    main()
