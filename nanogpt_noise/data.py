"""Freeze existing token files and all window indices once before training."""
import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .artifacts import file_hash,write_json,read_json,save_checkpoint,tensor_hash,digest,lock
from .model import GPT,Shape
from .protocol import Budget,VERSION,INPUT_POLICY,require_current_input

REVISION = '79d93d786212f7344586290adb811d4ae6a1762c'


def starts(length, count, sequence, seed):
    gen = torch.Generator(device='cpu').manual_seed(seed % (2**63-1))
    return torch.randint(length-sequence,(count,),generator=gen)


def windows(lengths, budget):
    return dict(train=torch.stack([starts(lengths['train'],budget.batch,budget.sequence,1337+97409*s)
                                  for s in range(budget.steps)]),
                eval_train=starts(lengths['train'],budget.eval_sequences,budget.sequence,20260829),
                eval_val=starts(lengths['val'],budget.eval_sequences,budget.sequence,20260830))


def freeze(data, output, tiny=False):
    data, output = Path(data),Path(output)
    with lock(output):
        if (output/'identity.json').exists():
            verify(data,output)
            existing = read_json(output/'identity.json')
            if existing['tiny'] != tiny:
                raise ValueError('artifact mode mismatch')
            if not tiny:
                require_current_input(existing)
            return existing
        if any((output/p).exists() for p in ('initial.pt','windows.pt')):
            raise RuntimeError('partial artifact preparation: use a fresh output directory')
        budget = Budget(sequence=8,batch=4,steps=4,warmup=32,eval_every=64,eval_sequences=4) if tiny else Budget()
        shape = Shape(n_layer=2,n_head=2,n_embd=16,block_size=8,vocab_size=64) if tiny else Shape()
        lengths = {s:(data/f'{s}.bin').stat().st_size//2 for s in ('train','val')}
        meta = read_json(data/'meta.json')
        if not tiny:
            required = dict(input_policy=INPUT_POLICY, preparation_design=VERSION,
                            source_dataset='Skylion007/openwebtext',source_revision=REVISION,
                            source_split='train',shuffle_seed=2357,shuffle_buffer=10000,
                            tokenizer='tiktoken:gpt2',dtype='uint16',eot_token=50256,vocab_size=50257)
            if any(meta.get(k)!=v for k,v in required.items()) or lengths != dict(train=1_000_000_000,val=10_000_000):
                raise ValueError('data metadata or sizes disagree with protocol')
        for split in ('train','val'):
            arr = np.memmap(data/f'{split}.bin',dtype='<u2',mode='r')
            if len(arr)<=budget.sequence or int(arr.max())>= (64 if tiny else 50257):
                raise ValueError('invalid token range or length')
        hashes = {p:file_hash(data/p) for p in ('train.bin','val.bin','meta.json')}
        torch.manual_seed(1337)
        model = GPT(shape)
        save_checkpoint(output/'initial.pt',model.state_dict())
        save_checkpoint(output/'windows.pt',windows(lengths,budget))
        identity = dict(design=VERSION,tiny=tiny,shape=asdict(shape),budget=asdict(budget),
                        data=hashes,meta=meta,lengths=lengths,
                        initialization_tensor_hash=tensor_hash(model.named_parameters()),
                        artifacts={p:file_hash(output/p) for p in ('initial.pt','windows.pt')},
                        preparation_environment=dict(torch=str(torch.__version__),numpy=np.__version__))
        identity['id'] = digest(identity)
        write_json(output/'identity.json',identity)
        return identity


def verify(data, artifacts):
    data,artifacts = Path(data),Path(artifacts)
    ident = read_json(artifacts/'identity.json')
    check = dict(ident)
    expected_id = check.pop('id')
    if digest(check)!=expected_id:
        raise ValueError('identity manifest edited')
    for base,kind in ((data,'data'),(artifacts,'artifacts')):
        for name,expected in ident[kind].items():
            if file_hash(base/name)!=expected:
                raise ValueError(f'hash mismatch: {base/name}')
    return ident


def batch(array, positions, sequence, device):
    values = np.stack([array[int(i):int(i)+sequence+1] for i in positions]).astype(np.int64)
    t = torch.from_numpy(values).to(device)
    return t[:,:-1].contiguous(), t[:,1:].contiguous()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--artifacts',required=True)
    p.add_argument('--verify',action='store_true')
    a = p.parse_args()
    result = verify(a.data,a.artifacts) if a.verify else freeze(a.data,a.artifacts)
    print(result['id'])


if __name__=='__main__':
    main()
