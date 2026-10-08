"""Optional fresh data preparation; prefer transferring the fixed hashed corpus."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import numpy as np
from .artifacts import write_json,lock
from .data import REVISION


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


def main():
    from datasets import load_dataset
    import tiktoken
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    output=Path(a.output)
    with lock(output):
        if any(p.name!='.lock' for p in output.iterdir()):
            raise ValueError('use a fresh empty directory; never overwrite a frozen corpus')
        encoder=tiktoken.get_encoding('gpt2')
        stream=iter(load_dataset('Skylion007/openwebtext',split='train',streaming=True,
                                revision=REVISION).shuffle(seed=2357,buffer_size=10000))
        val=fill(stream,encoder,output/'val.bin.incomplete',10_000_000)
        train=fill(stream,encoder,output/'train.bin.incomplete',1_000_000_000)
        for split in ('val','train'):
            os.replace(output/f'{split}.bin.incomplete',output/f'{split}.bin')
        write_json(output/'meta.json',dict(source_dataset='Skylion007/openwebtext',source_revision=REVISION,
                   source_split='train',streaming=True,shuffle_seed=2357,shuffle_buffer=10000,
                   tokenizer='tiktoken:gpt2',vocab_size=encoder.n_vocab,eot_token=encoder.eot_token,
                   dtype='uint16',validation=val,train=train,
                   split_policy='validation first; unused boundary tail discarded; no document crosses splits',
                   dependencies={name:importlib.metadata.version(name) for name in ('datasets','tiktoken','numpy','huggingface-hub')}))


if __name__=='__main__': main()
