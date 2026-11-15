"""Explicit legacy-input hardware timing; writes no formal smoke proof."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nanogpt_noise.artifacts import write_json, file_hash
from nanogpt_noise.data import verify
from nanogpt_noise.lm_head.benchmark import require_benchmark_input, empirical_statistics
from nanogpt_noise.lm_head.protocol import CENTERS, configuration, code_identity
from nanogpt_noise.lm_head.train import run, Runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',required=True)
    p.add_argument('--run-root',required=True)
    p.add_argument('--objective',choices=('G','T','K'),default='G')
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--statistics')
    p.add_argument('--prepare-statistics',action='store_true')
    a=p.parse_args()
    root,output=Path(a.input_root).resolve(),Path(a.run_root).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('benchmark output and inputs must be separate trees')
    if a.prepare_statistics:
        print(empirical_statistics(root,output,a.micro_batch));return
    ident=verify(root/'openwebtext_subset',root/'frozen-artifacts')
    require_benchmark_input(ident)
    if (a.objective=='K') != bool(a.statistics): p.error('only K requires --statistics')
    rt=Runtime(); records=[]
    for optimizer in CENTERS:
        config=configuration(a.objective,optimizer,0,0.)
        if a.statistics: config['statistics']=str(Path(a.statistics).resolve())
        result=run(config,root/'openwebtext_subset',root/'frozen-artifacts',
                   output/'smoke'/a.objective/optimizer,micro=a.micro_batch,runtime=rt,
                   smoke_steps=32,benchmark=True)
        records.append(result)
        if result['status']!='completed':
            raise RuntimeError(f'benchmark did not complete: {result["status"]}')
    identity=dict(input_id=ident['id'],objective=a.objective,micro=a.micro_batch,test=False,
                  code=code_identity()['sha256'],benchmark_only=True,
                  statistics_sha256=file_hash(a.statistics) if a.statistics else None)
    write_json(output/'smoke'/a.objective/'benchmark.json',dict(identity=identity,passed=True,
               run_ids=[r['run_id'] for r in records],purpose='hardware timing only, no formal approval'))


if __name__=='__main__': main()
