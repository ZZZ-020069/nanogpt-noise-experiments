"""Explicit entry points; no command automatically chains training stages."""
import argparse
import json
from pathlib import Path
from ..artifacts import write_json, read_json
from .protocol import CENTERS
from .runner import (input_identity,bind_root,job_list,smoke,serve,worker,
                     smoke_identity,collect)
from .measure import measure
from .measure import load_statistics
from ..model import Shape
from .protocol import Budget
from .analyze import report as comparison_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','smoke','run','_worker'):
        p=sub.add_parser(name)
        p.add_argument('--input-root',required=True)
        p.add_argument('--run-root',required=True)
        p.add_argument('--objective',choices=('G','T','K'),default='G')
        p.add_argument('--statistics')
        p.add_argument('--micro-batch',type=int,default=4)
        p.add_argument('--test',action='store_true',help='CPU synthetic fixtures only, never formal output')
        if name=='run':
            p.add_argument('--phase',choices=('clean','grid'),default='clean')
            p.add_argument('--devices',default='0')
            p.add_argument('--resume',action='store_true')
            p.add_argument('--dry-run',action='store_true')
        if name=='_worker':
            p.add_argument('--node',type=int,required=True)
            p.add_argument('--optimizer',choices=CENTERS,required=True)
            p.add_argument('--parent-pid',type=int,default=0)
    for name in ('pause','status','report'):
        p=sub.add_parser(name)
        p.add_argument('--run-root',required=True)
    p=sub.add_parser('curvature')
    p.add_argument('--input-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('fit','check'),default='fit')
    p.add_argument('--fit-dir')
    p.add_argument('--checkpoint')
    p.add_argument('--tokens',type=int,default=32768)
    p.add_argument('--blocks',type=int)
    p.add_argument('--chunk-tokens',type=int,default=64)
    p.add_argument('--micro-batch',type=int,default=4)
    p.add_argument('--test',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='run' and args.dry_run:
        print(json.dumps(dict(objective=args.objective,phase=args.phase,
                              candidates_per_job=8,jobs=job_list(args.phase)),indent=2))
        return 0
    if args.command in ('pause','status','report'):
        root=Path(args.run_root)
        identity=read_json(root/'lm_head_run.json')
        if identity.get('scope')!='lm_head_only':
            raise ValueError('not a registered Head-only run')
        if args.command=='report':
            result=comparison_report(root)
            print(json.dumps(dict(rows=len(result['rows']),totals=result['totals']),indent=2))
        elif args.command=='pause':
            write_json(root/'pause.request',dict(reason='user pause'))
            print('Pause requested. Workers save at step boundaries; verify status and GPU release.')
        else:
            print(json.dumps(dict(identity=identity,pause_requested=(root/'pause.request').exists(),
                  candidates=[dict(path=str(p.relative_to(root)),**read_json(p))
                    for p in root.glob('*/jobs/job_*/candidate_*/status.json')]),ensure_ascii=False,indent=2))
        return 0
    if args.command=='curvature':
        report=measure(args.input_root,args.output,args.mode,tokens=args.tokens,count=args.blocks,
                       fit_dir=args.fit_dir,checkpoint=args.checkpoint,
                       chunk_tokens=args.chunk_tokens,micro=args.micro_batch,test=args.test)
        print(json.dumps({k:v for k,v in report.items() if k!='blocks'},indent=2))
        return 0 if report['passed'] else 3
    if args.objective=='K' and not args.statistics:
        parser.error('K requires --statistics from a passing held-out curvature check')
    if args.objective!='K' and args.statistics:
        parser.error('--statistics is only valid for K')
    ident=input_identity(args.input_root,args.test)
    Budget(**ident['budget']).accumulations(1,args.micro_batch)
    if args.statistics:
        load_statistics(args.statistics,ident,Shape(**ident['shape']),allow_test=args.test)
    bind_root(args.run_root,args.input_root,ident,args.test)
    if args.command=='check':
        print(json.dumps(dict(input_id=ident['id'],design='LH-QP-20261114-draft2',
                              status='inputs_verified'),indent=2));return 0
    if args.command=='smoke':
        print(json.dumps(smoke(args,ident),indent=2));return 0
    if args.command=='_worker':
        result=worker(args)
    else:
        proof=read_json(Path(args.run_root)/'smoke'/args.objective/'passed.json')
        if len(proof.get('run_ids',[])) != 4 or proof != dict(identity=smoke_identity(ident,args.objective,args.micro_batch,
                                                args.statistics,args.test),
                         passed=True,run_ids=proof.get('run_ids',[])):
            raise ValueError('matching successful smoke required before formal search')
        result=serve(args)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 75 if result.get('status')=='paused' else 0


if __name__=='__main__':
    raise SystemExit(main())
