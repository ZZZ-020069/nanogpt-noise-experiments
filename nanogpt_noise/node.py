"""Run one scope/optimizer/noise configuration's eight sequential candidates."""
import argparse
import contextlib
from pathlib import Path
import torch.distributed as dist
from .artifacts import read_json,write_json,lock
from .protocol import grid,configuration
from .search import propose,summarize
from .storage import compact_steps,retain_winner_checkpoint
from .train import run,Runtime


def run_node(job,data,artifacts,output,micro=4,runtime=None,test=False):
    rt=runtime or Runtime(cpu=test)
    output=Path(output)
    with lock(output) if rt.rank==0 else contextlib.nullcontext():
        rt.barrier()
        records=[]
        for index in range(8):
            if (output.parent/'pause.request').exists():
                return dict(status='infrastructure_failure',reason='scope pause requested')
            candidate=propose(records)
            config=configuration(job['scope'],job['optimizer'],job['node'],candidate['x'])
            config['proposal']=candidate
            result=run(config,data,artifacts,output/f'candidate_{index:02d}',micro=micro,
                       runtime=rt,test=test,pause_file=output.parent/'pause.request')
            if result['status']=='infrastructure_failure':
                return result
            records.append(dict(**candidate,status=result['status'],loss=result.get('loss'),
                                lr=config['peak_lr'],candidate=index,tokens=result['tokens'],
                                failure_field=result.get('field'),T_diverge=result.get('T_diverge')))
            summary=summarize(records)
            if rt.rank==0:
                write_json(output/'search.json',dict(job=job,mode='test' if test else 'formal',**summary))
                compact_steps(output/f'candidate_{index:02d}')
                winner=summary['winner']['candidate'] if summary['winner'] else None
                retain_winner_checkpoint(output,winner,completed=[r['candidate'] for r in records])
            rt.barrier()
        winner=summary['winner']['candidate'] if summary['winner'] else None
        if winner is not None and not (output/f'candidate_{winner:02d}'/'final.pt').is_file():
            raise RuntimeError('final winner checkpoint is missing; preserve outputs for diagnosis')
        return summarize(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-index',type=int,required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--artifacts',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--micro-batch',type=int,default=4)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not 0<=a.job_index<len(grid()):
        p.error('job-index must be 0..839')
    job=grid()[a.job_index]
    if a.dry_run:
        print(job)
        return
    rt=Runtime()
    result=run_node(job,a.data,a.artifacts,Path(a.output)/f'job_{a.job_index:04d}',a.micro_batch,rt)
    if dist.is_initialized():
        dist.destroy_process_group()
    if result.get('status')=='infrastructure_failure':
        raise SystemExit(75)


if __name__=='__main__':
    main()
