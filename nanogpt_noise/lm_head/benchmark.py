"""Empirical inputs for hardware timing only; never formal K approval."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from ..artifacts import save_checkpoint, tensor_hash, lock
from ..data import verify, batch
from ..model import Shape
from .protocol import Budget, VERSION, code_identity


def require_benchmark_input(identity):
    budget = identity.get('budget', {})
    expected = asdict(Budget())
    if (identity.get('tiny') is not False or identity.get('shape') != asdict(Shape())
            or any(budget.get(k) != expected[k] for k in ('sequence','batch','eval_sequences'))
            or budget.get('steps', 0) < 32):
        raise ValueError('hardware benchmark requires full GPT shape and original batch/sequence/eval sizes')


def load_benchmark_statistics(path, identity, shape):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value.get('schema') != 'lm-head-K-benchmark-v1' or value.get('passed') is not False
            or value.get('benchmark_only') is not True or value.get('design') != VERSION
            or value.get('input_id') != identity['id'] or value.get('code') != code_identity()['sha256']
            or value.get('initialization_hash') != identity['initialization_tensor_hash']
            or value.get('shape') != asdict(shape)):
        raise ValueError('requires matching empirical benchmark K, not a formal approval')
    stats = value['statistics']
    if (stats['B'].shape != (shape.n_embd,shape.n_embd) or stats['pbar'].shape != (shape.vocab_size,)
            or stats['alpha'].numel() != 1 or stats['alpha'] <= 0
            or not all(torch.isfinite(stats[k]).all() for k in ('B','pbar','alpha'))):
        raise ValueError('invalid benchmark statistics')
    return stats


def empirical_statistics(input_root, output, micro=16):
    from .train import Runtime
    from .model import HeadGPT
    from .curvature import probabilities, trace
    root, output = Path(input_root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('benchmark statistics and source input must be separate trees')
    identity = verify(root/'openwebtext_subset', root/'frozen-artifacts')
    require_benchmark_input(identity)
    Budget().accumulations(1, micro)
    shape=Shape(**identity['shape'])
    with lock(output):
        target=output/'K-benchmark.pt'
        if target.exists():
            load_benchmark_statistics(target,identity,shape)
            return target
        rt=Runtime()
        if rt.world != 1: raise ValueError('benchmark statistics is single GPU')
        model=HeadGPT(shape)
        model.load_state_dict(torch.load(root/'frozen-artifacts/initial.pt',weights_only=True,map_location='cpu'))
        if tensor_hash(model.named_parameters()) != identity['initialization_tensor_hash']:
            raise ValueError('initialization mismatch')
        model.activate_head().to(rt.device).eval()
        indices=torch.load(root/'frozen-artifacts/windows.pt',weights_only=True,map_location='cpu')['eval_train'][:32]
        array=np.memmap(root/'openwebtext_subset/train.bin',dtype='<u2',mode='r')
        b=torch.zeros(shape.n_embd,shape.n_embd,device=rt.device,dtype=torch.float64)
        pbar=torch.zeros(shape.vocab_size,device=rt.device,dtype=torch.float64)
        curvature=torch.zeros((),device=rt.device,dtype=torch.float64)
        n=0
        try:
            with torch.no_grad():
                w=model.reference_head.double()
                for positions in indices.split(micro):
                    x,_=batch(array,positions,1024,rt.device)
                    with rt.autocast(): h=model.features(x).flatten(0,1)
                    for chunk in h.split(64):
                        chunk=chunk.double()
                        p=probabilities(chunk,w)
                        b+=chunk.T@chunk
                        pbar+=p.sum(0)
                        curvature+=trace(chunk,p)*len(chunk)
                        n+=len(chunk)
                    print(f'empirical K: {n}/32768 tokens',flush=True)
        finally:
            array._mmap.close()
        b,pbar,curvature=b/n,pbar/n,curvature/n
        alpha=curvature/((1-pbar.square().sum())*b.trace())
        save_checkpoint(target,dict(schema='lm-head-K-benchmark-v1',design=VERSION,benchmark_only=True,
                        passed=False,input_id=identity['id'],code=code_identity()['sha256'],shape=asdict(shape),
                        initialization_hash=identity['initialization_tensor_hash'],tokens=n,windows=indices,
                        statistics=dict(B=b.cpu(),pbar=pbar.cpu(),alpha=alpha.cpu())))
        load_benchmark_statistics(target,identity,shape)
        return target
