"""One independent candidate, with transactional checkpoints and DDP."""
import argparse
import contextlib
from dataclasses import asdict
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from . import optim
from .artifacts import (read_json,write_json,save_checkpoint,tensor_hash,digest,
                        environment,code_identity,lock)
from .data import batch,verify
from .model import GPT,Shape,set_scope
from .noise import inject,norm,ratio
from .protocol import Budget,configuration,SCOPES,CENTERS,require_current_input


class NumericalFailure(Exception):
    pass


class Runtime:
    def __init__(self, cpu=False):
        self.world = int(os.environ.get('WORLD_SIZE','1'))
        self.rank = int(os.environ.get('RANK','0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK','0'))
        if cpu:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)
        else:
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA with BF16 required; CPU is only allowed for explicit tests')
            self.device = torch.device('cuda',self.local_rank)
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if self.world>1 and not dist.is_initialized():
            test_store = os.environ.get('NANOGPT_TEST_RENDEZVOUS') if cpu else None
            kwargs = dict(store=dist.FileStore(test_store,self.world)) if test_store else {}
            dist.init_process_group('gloo' if cpu else 'nccl',
                                    rank=self.rank,world_size=self.world,**kwargs)

    def barrier(self):
        if self.world>1:
            dist.barrier()

    def broadcast(self, value):
        values = [value if self.rank==0 else None]
        if self.world>1:
            dist.broadcast_object_list(values,src=0)
        return values[0]

    def any(self, flag):
        t = torch.tensor(int(flag),device=self.device)
        if self.world>1:
            dist.all_reduce(t,op=dist.ReduceOp.MAX)
        return bool(t.item())

    def check(self, named):
        named=list(named)
        flags=[torch.isfinite(t).all().to(self.device) for _,t in named]
        failed=bool(flags) and not bool(torch.stack(flags).all())
        bad = next((n for n,t in named if not bool(torch.isfinite(t).all())),None) if failed else None
        if self.any(bad is not None):
            fields = [None]*self.world
            if self.world>1:
                dist.all_gather_object(fields,bad)
            else:
                fields=[bad]
            raise NumericalFailure(next(v for v in fields if v is not None))

    def autocast(self):
        return torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, arrays, indices, budget, micro, rt):
    model.eval()
    losses = {}
    for split in ('train','val'):
        positions = indices['eval_'+split][rt.rank::rt.world]
        total = torch.zeros(2,dtype=torch.float64,device=rt.device)
        for chunk in positions.split(micro):
            if not len(chunk):
                continue
            x,y = batch(arrays[split],chunk,budget.sequence,rt.device)
            with rt.autocast():
                loss = model(x,y)
            total[0] += loss.double()*len(chunk)
            total[1] += len(chunk)
        if rt.world>1:
            dist.all_reduce(total)
        rt.check([(f'eval_{split}_loss',total)])
        losses[split] = float(total[0]/total[1])
    model.train()
    return losses


def run(config, data, artifacts, output, *, micro=4, runtime=None,
        smoke_steps=None, test=False, pause_after=None):
    rt = runtime or Runtime(cpu=test)
    output, data, artifacts = Path(output),Path(data),Path(artifacts)
    mode = 'test' if test else ('smoke' if smoke_steps is not None else 'formal')
    if rt.device.type=='cpu' and not test:
        raise ValueError('CPU outputs cannot be formal or H100 smoke results')
    with lock(output) if rt.rank==0 else contextlib.nullcontext():
        rt.barrier()
        return _run(config,data,artifacts,output,micro,rt,mode,smoke_steps,pause_after)


def _run(config,data,artifacts,output,micro,rt,mode,smoke_steps,pause_after):
    # Every rank checks identity: mismatched node-local data must not silently train.
    ident = verify(data,artifacts)
    if ident['tiny'] != (mode=='test'):
        raise ValueError('tiny fixtures cannot be used for real-model runs')
    if mode != 'test':
        require_current_input(ident)
        expected = configuration(config['scope'], config['optimizer'], config['node'], config['exponent'])
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('candidate config does not match the current protocol')
    budget, shape = Budget(**ident['budget']),Shape(**ident['shape'])
    accumulation = budget.accumulations(rt.world,micro)
    target = budget.steps if smoke_steps is None else smoke_steps
    if not 1<=target<=budget.steps:
        raise ValueError('invalid smoke step count')
    execution = dict(config=config,mode=mode,artifact_id=ident['id'],
                     code=code_identity()['sha256'],world=rt.world,micro=micro,
                     device_type=rt.device.type,torch=str(torch.__version__),
                     cuda=torch.version.cuda,target_steps=target)
    identity = digest(execution)
    metadata_path = output/'metadata.json'
    if metadata_path.exists() and read_json(metadata_path)['run_id'] != identity:
        raise ValueError('refusing incompatible resume/output reuse: config, data, code or layout changed')
    if (output/'status.json').exists():
        previous = read_json(output/'status.json')
        if previous['status'] in ('completed','numerical_failure'):
            return previous
    torch.manual_seed(1337)
    model = GPT(shape)
    model.load_state_dict(torch.load(artifacts/'initial.pt',map_location='cpu',weights_only=True))
    if tensor_hash(model.named_parameters()) != ident['initialization_tensor_hash']:
        raise ValueError('initial tensor identity mismatch')
    named = set_scope(model,config['scope'])
    dimension = sum(p.numel() for _,p in named)
    if mode!='test':
        expected = 124373760 if config['scope']=='full' else 7079424*int(config['scope'][4:])
        if dimension != expected:
            raise ValueError('unexpected trainable parameter count')
    frozen = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    frozen_hash = tensor_hash(frozen)
    initial = {n:p.detach().clone().to(rt.device) for n,p in named}
    model.to(rt.device)
    # References collected before .to() remain valid for ordinary CPU/CUDA moves.
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen = [(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    bundle = optim.build(named,config['optimizer'],config['peak_lr'])
    metadata = dict(run_id=identity,execution=execution,environment=environment(rt.device),
                    artifact_identity=ident,D=dimension,frozen_hash=frozen_hash,
                    parameters=[dict(name=n,shape=list(p.shape),count=p.numel(),trainable=p.requires_grad)
                                for n,p in model.named_parameters()],optimizer_groups=bundle.groups)
    if rt.rank==0:
        write_json(metadata_path,metadata)
    indices = torch.load(artifacts/'windows.pt',map_location='cpu',weights_only=True)
    step, processed, best, evaluations = 0,0,float('inf'),[]
    latest = output/'latest.pt'
    final = output/'final.pt'
    resume_checkpoint = latest if latest.exists() else (final if final.exists() else None)
    if resume_checkpoint is not None:
        state = torch.load(resume_checkpoint,map_location=rt.device,weights_only=True)
        if state['run_id']!=identity:
            raise ValueError('checkpoint belongs to a different candidate')
        model.load_state_dict(state['model'])
        bundle.load_state_dict(state['optimizer'])
        step,processed,best,evaluations = state['step'],state['tokens'],state['best'],state['evaluations']
    # Keep bucket layout fixed across construction/resume. Default DDP rebuilds
    # buckets after its first backward; a fresh resumed reducer would otherwise
    # use a different reduction order than an uninterrupted warmed-up reducer.
    # In PyTorch 2.x find_unused_parameters=True with static_graph=False disables
    # this rebuild. Missing trainable gradients remain an explicit error below.
    wrapped = DDP(model,device_ids=[rt.local_rank] if rt.device.type=='cuda' else None,
                  broadcast_buffers=False,find_unused_parameters=True,
                  static_graph=False) if rt.world>1 else model
    if rt.rank==0 and mode!='test':
        print(f"[{mode}] {config['scope']}/{config['optimizer']} node={config['node']} "
              f"peak_lr={config['peak_lr']:.8g} world={rt.world} micro={micro} "
              f"accumulation={accumulation} step={step}/{target} output={output}", flush=True)
    interrupted = [False]
    old_handlers = {}
    for sig in (signal.SIGTERM,signal.SIGINT):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))

    def snapshot():
        return dict(run_id=identity,model=model.state_dict(),optimizer=bundle.state_dict(),
                    step=step,tokens=processed,best=best,evaluations=evaluations,
                    execution=execution,artifact_identity=ident,
                    rng_reconstruction=dict(next_step=step,noise_seed=config['seeds']['noise'],
                                            windows_hash=ident['artifacts']['windows.pt']))

    def status(kind, **extra):
        value = dict(status=kind,mode=mode,step=step,tokens=processed,
                     target_steps=target,run_id=identity,**extra)
        if rt.rank==0:
            write_json(output/'status.json',value)
        rt.barrier()
        return value

    def checkpoint_evaluation():
        nonlocal best
        values = evaluate(model,arrays,indices,budget,micro,rt)
        if tensor_hash(frozen)!=frozen_hash:
            raise RuntimeError('frozen parameter checksum changed')
        if rt.world>1:
            replica_hashes=[None]*rt.world
            dist.all_gather_object(replica_hashes,tensor_hash(model.named_parameters()))
            if len(set(replica_hashes))!=1:
                raise RuntimeError('DDP replicas differ; check synchronization and noise RNG')
        item = dict(step=step,tokens=step*budget.step_tokens,**values)
        evaluations.append(item)
        best = min(best,values['val'])
        if rt.rank==0:
            write_json(output/'evaluations.json',evaluations)
            if mode!='test':
                print(f"eval step={step}/{target} train={values['train']:.6f} "
                      f"val={values['val']:.6f}", flush=True)
            save_checkpoint(latest,snapshot())
        rt.barrier()

    arrays = {s:np.memmap(data/f'{s}.bin',dtype='<u2',mode='r') for s in ('train','val')}
    started = time.monotonic()
    try:
        rt.check([('parameters:'+n,p) for n,p in model.named_parameters()])
        rt.check([('optimizer_state',t) for t in bundle.state_tensors()])
        if not evaluations:
            checkpoint_evaluation()
        while step<target:
            bundle.zero_grad()
            lr = config['peak_lr']*budget.multiplier(step*budget.step_tokens)
            bundle.set_lr(lr)
            positions = indices['train'][step].reshape(rt.world,-1)[rt.rank]
            train_loss = torch.zeros((),device=rt.device,dtype=torch.float64)
            diagnostic = step==0 or (step+1)%10==0 or step+1==target
            for i,chunk in enumerate(positions.split(micro)):
                sync = wrapped.no_sync() if rt.world>1 and i<accumulation-1 else contextlib.nullcontext()
                with sync:
                    x,y = batch(arrays['train'],chunk,budget.sequence,rt.device)
                    with rt.autocast():
                        loss = wrapped(x,y)
                    processed += micro*rt.world*budget.sequence
                    rt.check([('training_loss',loss)])
                    train_loss += loss.detach().double()/accumulation
                    (loss/accumulation).backward()
            if rt.world>1:
                dist.all_reduce(train_loss)
                train_loss /= rt.world
            if any(p.grad is None for _,p in named):
                raise RuntimeError('missing trainable gradient')
            rt.check([('gradient:'+n,p.grad) for n,p in named])
            rt.check([('parameters:'+n,p) for n,p in model.named_parameters()])
            rt.check([('optimizer_state',t) for t in bundle.state_tensors()])
            metrics = inject(named,config['sigma1'],config['sigma2'],step,diagnostic)
            rt.check([('noisy_gradient:'+n,p.grad) for n,p in named])
            before = [p.detach().clone() for _,p in named] if diagnostic else None
            bundle.step()
            rt.check([('updated_parameters:'+n,p) for n,p in model.named_parameters()])
            rt.check([('updated_optimizer_state',t) for t in bundle.state_tensors()])
            step += 1
            if diagnostic:
                update = norm(p.detach()-old for (_,p),old in zip(named,before))
                metrics.update(parameter_norm=norm(p for _,p in named),update_norm=update,
                               displacement=norm(p.detach()-initial[n] for n,p in named),
                               update_to_gradient=ratio(update,metrics['gradient_norm']))
                del before
            metrics.update(step=step,tokens=processed,train_loss=float(train_loss),lr=lr,
                           auxiliary_lr=.03*lr if config['optimizer']=='muon' else None,
                           finite=True,elapsed_seconds=time.monotonic()-started)
            if rt.rank==0:
                write_json(output/'steps'/f'{step:06d}.json',metrics)
                if diagnostic and mode!='test':
                    print(f"step={step}/{target} tokens={processed} "
                          f"loss={float(train_loss):.6f} lr={lr:.8g} "
                          f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            if budget.evaluate_at(step) or step==target:
                checkpoint_evaluation()
            stop = rt.any(interrupted[0] or (pause_after is not None and step>=pause_after))
            if stop and step<target:
                if rt.rank==0:
                    save_checkpoint(latest,snapshot())
                return status('infrastructure_failure',reason='interrupted; resume same candidate')
        if rt.rank==0:
            # The final evaluation has already written the complete latest
            # snapshot. A rename avoids a second full-checkpoint write.
            if latest.exists():
                os.replace(latest,final)
            elif not final.exists():
                raise RuntimeError('terminal checkpoint is missing')
        return status('completed',loss=evaluations[-1]['val'],
                      frozen_hash_final=tensor_hash(frozen),evaluations=evaluations)
    except NumericalFailure as exc:
        if rt.rank==0:
            save_checkpoint(output/'failure.pt',snapshot())
        return status('numerical_failure',field=str(exc),failure_step=step,
                      T_diverge=processed,loss=None)
    except Exception as exc:
        # Do not overwrite the last known-good checkpoint with partial state.
        if rt.rank==0:
            write_json(output/'status.json',dict(status='infrastructure_failure',mode=mode,
                       step=step,tokens=processed,run_id=identity,reason=repr(exc)))
        raise
    finally:
        for array in arrays.values():
            array._mmap.close()
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--artifacts',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--scope',choices=SCOPES,default='last1')
    p.add_argument('--optimizer',choices=CENTERS,default='adamw')
    p.add_argument('--node',type=int,default=0)
    p.add_argument('--exponent',type=float,default=0)
    p.add_argument('--micro-batch',type=int,default=4)
    p.add_argument('--smoke-steps',type=int)
    a=p.parse_args()
    rt=Runtime()
    result=run(configuration(a.scope,a.optimizer,a.node,a.exponent),a.data,a.artifacts,a.output,
               micro=a.micro_batch,runtime=rt,smoke_steps=a.smoke_steps)
    if rt.rank==0:
        print(result['status'])
    if dist.is_initialized():
        dist.destroy_process_group()
    if result['status']=='infrastructure_failure':
        raise SystemExit(75)
    if a.smoke_steps is not None and result['status']!='completed':
        # A failed smoke must not look like successful hardware acceptance.
        raise SystemExit(2)


if __name__=='__main__':
    main()
