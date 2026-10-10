"""Run with torchrun --standalone --nproc-per-node=2 -m tests.ddp_smoke."""
import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from unittest.mock import patch
from nanogpt_noise.train import Runtime,run
from nanogpt_noise.protocol import configuration
from nanogpt_noise.artifacts import tensor_hash,read_json
from tests.test_integration import fixture
from nanogpt_noise.model import GPT


def worker(root):
    root=Path(root)
    rt=Runtime(cpu=True)
    if rt.rank==0:
        root.mkdir(parents=True,exist_ok=True)
        fixture(root, batch_size=8)
    rt.barrier()
    for scope in ('full', 'last1'):
        for opt in ('adamw','adam','sgd','muon'):
            name=f'{scope}_{opt}'
            cfg=configuration(scope,opt,10,0)
            run(cfg,root/'data',root/'artifacts',root/f'{name}_whole',micro=1,runtime=rt,test=True)
            paused=run(cfg,root/'data',root/'artifacts',root/f'{name}_resume',micro=1,runtime=rt,test=True,pause_after=1)
            assert paused['status']=='infrastructure_failure'
            run(cfg,root/'data',root/'artifacts',root/f'{name}_resume',micro=1,runtime=rt,test=True)
            whole=torch.load(root/f'{name}_whole'/'final.pt',weights_only=True)
            resumed=torch.load(root/f'{name}_resume'/'final.pt',weights_only=True)
            for parameter, value in whole['model'].items():
                restored=resumed['model'][parameter]
                assert torch.equal(value,restored), (
                    name,parameter,'max_abs_error',float((value-restored).abs().max()))
            assert tensor_hash(whole['model'].items())==tensor_hash(resumed['model'].items())
            status=read_json(root/f'{name}_whole'/'status.json')
            assert status['tokens']==256 and status['status']=='completed'
            if rt.rank==0: print(f'{rt.world}-rank {name}: bitwise resume passed',flush=True)
    # A nonfinite loss on only one rank must terminate all ranks without hanging.
    original_forward=GPT.forward
    def broken(model,x,y):
        value=original_forward(model,x,y)
        return value*float('nan') if model.training and rt.rank==1 else value
    with patch.object(GPT,'forward',broken):
        failure=run(configuration('full','adamw',0,0),root/'data',root/'artifacts',
                    root/'rank_failure',micro=1,runtime=rt,test=True)
    assert failure['status']=='numerical_failure' and failure['tokens']==rt.world*8
    if rt.rank==0: print(f'{rt.world}-rank DDP Gloo training/resume/failure passed (not H100 validation)')
    dist.destroy_process_group()


def spawn_worker(rank,root):
    os.environ.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',
                      NANOGPT_TEST_RENDEZVOUS=str(Path(root)/'gloo.store'))
    worker(root)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--spawn',action='store_true',help='FileStore fallback for Windows CPU builds without libuv')
    a=p.parse_args()
    if a.spawn:
        Path(a.output).mkdir(parents=True,exist_ok=True)
        torch.multiprocessing.spawn(spawn_worker,args=(a.output,),nprocs=2,join=True)
    else:
        worker(a.output)


if __name__=='__main__': main()
