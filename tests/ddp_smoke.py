"""Run with torchrun --standalone --nproc-per-node=2 -m tests.ddp_smoke."""
import argparse
import os
from pathlib import Path
import torch
import torch.distributed as dist
from nanogpt_noise.train import Runtime,run
from nanogpt_noise.protocol import configuration
from nanogpt_noise.artifacts import tensor_hash,read_json
from tests.test_integration import fixture


def worker(root):
    root=Path(root)
    rt=Runtime(cpu=True)
    if rt.rank==0:
        root.mkdir(parents=True,exist_ok=True)
        fixture(root)
    rt.barrier()
    for opt in ('adamw','muon'):
        cfg=configuration('last1',opt,10,0)
        run(cfg,root/'data',root/'artifacts',root/f'{opt}_whole',micro=1,runtime=rt,test=True)
        run(cfg,root/'data',root/'artifacts',root/f'{opt}_resume',micro=1,runtime=rt,test=True,pause_after=1)
        run(cfg,root/'data',root/'artifacts',root/f'{opt}_resume',micro=1,runtime=rt,test=True)
        whole=torch.load(root/f'{opt}_whole'/'final.pt',weights_only=True)
        resumed=torch.load(root/f'{opt}_resume'/'final.pt',weights_only=True)
        assert tensor_hash(whole['model'].items())==tensor_hash(resumed['model'].items())
        assert read_json(root/f'{opt}_whole'/'status.json')['tokens']==128
    if rt.rank==0: print('DDP Gloo resume smoke passed (not H100 validation)')
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
