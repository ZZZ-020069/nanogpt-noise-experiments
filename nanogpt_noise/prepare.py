"""Prepare and freeze one deterministic OpenWebText input realization."""
import argparse
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import numpy as np
from .artifacts import write_json,read_json,lock
from .data import REVISION, freeze
from .input_bundle import register_bundle, verify_bundle
from .protocol import Budget, VERSION, INPUT_POLICY, require_current_input


def fill(stream,encoder,path,count):
    consumed=0
    truncated=False
    written=0
    with open(path,'xb') as f:
        while written<count:
            ids=encoder.encode_ordinary(next(stream)['text'])+[encoder.eot_token]
            consumed+=1
            take=min(count-written,len(ids))
            f.write(np.asarray(ids[:take],dtype='<u2').tobytes())
            truncated=take<len(ids)
            written+=take
            if consumed%10000==0:
                print(f'{path.name}: {written}/{count} tokens',flush=True)
        f.flush()
        os.fsync(f.fileno())
    return dict(tokens=written,documents_consumed=consumed,boundary_document_truncated=truncated)


def prepare_corpus(output, restart_incomplete=False):
    from datasets import load_dataset
    import tiktoken
    output=Path(output)
    complete=tuple(output/name for name in ('train.bin','val.bin','meta.json'))
    if all(path.is_file() for path in complete):
        meta = read_json(output/'meta.json')
        if meta.get('input_policy') != INPUT_POLICY or meta.get('preparation_design') != VERSION:
            raise ValueError('existing corpus is not from this fresh preparation; use a new INPUT_ROOT')
        return
    if any(path.exists() for path in complete):
        raise RuntimeError('partial final corpus found; preserve it for diagnosis and use a fresh input root')
    with lock(output):
        incomplete=tuple(output/f'{split}.bin.incomplete' for split in ('val','train'))
        if any(path.exists() for path in incomplete):
            if not restart_incomplete:
                raise RuntimeError('incomplete corpus found; rerun with --restart-incomplete after confirming no preparation job remains')
            for path in incomplete:
                path.unlink(missing_ok=True)
        if any(path.name not in ('.lock','.lease') for path in output.iterdir()):
            raise ValueError('use a fresh directory; corpus preparation never overwrites unknown files')
        encoder=tiktoken.get_encoding('gpt2')
        stream=iter(load_dataset('Skylion007/openwebtext',split='train',streaming=True,
                                revision=REVISION).shuffle(seed=2357,buffer_size=10000))
        val=fill(stream,encoder,output/'val.bin.incomplete',10_000_000)
        train=fill(stream,encoder,output/'train.bin.incomplete',1_000_000_000)
        for split in ('val','train'):
            os.replace(output/f'{split}.bin.incomplete',output/f'{split}.bin')
        write_json(output/'meta.json',dict(input_policy=INPUT_POLICY, preparation_design=VERSION,
                   source_dataset='Skylion007/openwebtext',source_revision=REVISION,
                   source_split='train',streaming=True,shuffle_seed=2357,shuffle_buffer=10000,
                   tokenizer='tiktoken:gpt2',vocab_size=encoder.n_vocab,eot_token=encoder.eot_token,
                   dtype='uint16',validation=val,train=train,
                   split_policy='validation first; unused boundary tail discarded; no document crosses splits',
                   dependencies={name:importlib.metadata.version(name) for name in ('datasets','tiktoken','numpy','huggingface-hub')}))


def prepare_bundle(input_root, restart_incomplete=False):
    root=Path(input_root)
    # Own the entire preparation, including the gap between corpus and artifacts.
    with lock(root):
        marker = root/'preparation.json'
        expected = dict(schema=1, policy=INPUT_POLICY, design=VERSION,
                        source_revision=REVISION, budget=asdict(Budget()))
        if marker.exists():
            if read_json(marker) != expected:
                raise ValueError('preparation protocol changed; use a fresh INPUT_ROOT')
        else:
            if any(path.name not in ('.lock','.lease') for path in root.iterdir()):
                raise ValueError('fresh preparation requires an empty INPUT_ROOT; existing files are preserved')
            write_json(marker, expected)
        if (root/'registration.json').exists():
            input_id = verify_bundle(root)
            require_current_input(read_json(root/'frozen-artifacts'/'identity.json'))
            return input_id
        data=root/'openwebtext_subset'
        artifacts=root/'frozen-artifacts'
        prepare_corpus(data,restart_incomplete=restart_incomplete)
        freeze(data,artifacts)
        return register_bundle(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    target=p.add_mutually_exclusive_group(required=True)
    target.add_argument('--output',help='prepare only the token corpus (advanced use)')
    target.add_argument('--input-root',help='prepare, freeze and register the complete bundle')
    p.add_argument('--restart-incomplete',action='store_true',
                   help='discard only named .incomplete corpus files and restart from the source')
    a=p.parse_args()
    if a.input_root:
        print(prepare_bundle(a.input_root,restart_incomplete=a.restart_incomplete))
    else:
        prepare_corpus(a.output,restart_incomplete=a.restart_incomplete)


if __name__=='__main__': main()
