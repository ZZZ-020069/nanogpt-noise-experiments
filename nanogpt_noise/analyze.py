"""Summarize completed searches; missing results remain missing, never zero."""
import argparse
import csv
import io
from pathlib import Path
from .artifacts import read_json,write_json,atomic
from .protocol import grid,CENTERS,SCOPES,SIGMA1,SIGMA2


def auc(curve):
    if len(curve)<2 or curve[-1]['tokens']<=0:
        return None
    return sum((b['tokens']-a['tokens'])*(a['val']+b['val'])/2
               for a,b in zip(curve,curve[1:]))/curve[-1]['tokens']


def collect(roots):
    if isinstance(roots,(str,Path)):
        roots=[roots]
    roots=[Path(root) for root in roots]
    rows=[]
    cohort=None
    for index,job in enumerate(grid()):
        matches=[root/f'job_{index:04d}'/'search.json' for root in roots
                 if (root/f'job_{index:04d}'/'search.json').exists()]
        if len(matches)>1:
            raise ValueError(f'job {index:04d} appears in multiple scope roots')
        path=matches[0] if matches else roots[0]/f'job_{index:04d}'/'search.json'
        row=dict(job_index=index,**job,search_complete=False,status='missing',loss=None,
                 best_lr=None,valid_count=0,boundary_unclosed=None,delta_loss=None,auc=None)
        if path.exists():
            result=read_json(path)
            if result['mode']!='formal':
                row['status']='excluded_nonformal'
                rows.append(row)
                continue
            if result['job']!=job:
                raise ValueError(f'job identity mismatch: {path}')
            for candidate in result['candidates']:
                meta=read_json(path.parent/f"candidate_{candidate['candidate']:02d}"/'metadata.json')
                e=meta['execution']
                signature=(e['artifact_id'],e['code'],e['torch'],e['cuda'],e['mode'])
                if cohort is not None and signature!=cohort:
                    raise ValueError('mixed code/data/environment cohorts; summarize separately')
                cohort=signature
            row.update(search_complete=result['complete'],valid_count=result['valid_count'],
                       numerical_failure_count=sum(r['status']=='numerical_failure' for r in result['candidates']),
                       boundary_unclosed=result['boundary_unclosed'],
                       adjacent_failed_exponents=result['adjacent_failed_exponents'])
            w=result['winner']
            row['status']='incomplete_search'
            if result['complete']:
                row['status']='completed' if w else 'no_valid_lr'
                if w:
                    status=read_json(path.parent/f"candidate_{w['candidate']:02d}"/'status.json')
                    row.update(loss=w['loss'],best_lr=w['lr'],auc=auc(status['evaluations']))
        rows.append(row)
    clean={(r['scope'],r['optimizer']):r['loss'] for r in rows if r['node']==0}
    for row in rows:
        baseline=clean.get((row['scope'],row['optimizer']))
        if row['loss'] is not None and baseline is not None:
            row['delta_loss']=row['loss']-baseline
        peers=[r for r in rows if r['scope']==row['scope'] and r['node']==row['node'] and r['loss'] is not None]
        row['rank']=None if row['loss'] is None else 1+sum(r['loss']<row['loss']-1e-6 for r in peers)
        row['ranking_complete']=sum(r['scope']==row['scope'] and r['node']==row['node'] and r['search_complete'] for r in rows)==4
    return rows


def plot(rows,output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for scope in SCOPES:
        for metric in ('loss','delta_loss','best_lr','rank','numerical_failure_count'):
            fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
            for ax,opt in zip(axes,CENTERS):
                a=np.full((5,7),np.nan)
                for r in rows:
                    if r['scope']==scope and r['optimizer']==opt and r.get(metric) is not None:
                        a[r['node']//7,r['node']%7]=r[metric]
                im=ax.imshow(a,aspect='auto',origin='lower')
                ax.set_title(opt)
                ax.set_xticks(range(7),[f'{x:g}' for x in SIGMA2],rotation=45)
                ax.set_yticks(range(5),[f'{x:g}' for x in SIGMA1])
                ax.set_xlabel('sigma2 (absolute)')
                ax.set_ylabel('sigma1')
                fig.colorbar(im,ax=ax)
            fig.suptitle(f'{scope}: {metric}; blank = missing/no valid winner; single seed')
            fig.savefig(Path(output)/f'{scope}_{metric}.png',dpi=160)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True,nargs='+',
                   help='one or more downloaded scope directories')
    p.add_argument('--output',required=True)
    p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    rows=collect(a.runs)
    write_json(Path(a.output)/'surfaces.json',rows)
    stream=io.StringIO()
    fields=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic(Path(a.output)/'surfaces.csv',lambda f:f.write(stream.getvalue().encode('utf8')))
    if a.plots:
        plot(rows,a.output)


if __name__=='__main__':
    main()
