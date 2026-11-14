"""Paired rankings on actual validation CE; missing and failed jobs stay explicit."""
import csv
import io
import itertools
from pathlib import Path
from ..artifacts import read_json,write_json,atomic,file_hash
from ..search import TOL,winner
from ..protocol import SIGMA1,SIGMA2
from .protocol import CENTERS,VERSION


def ranking(scores):
    valid = {k:v for k,v in scores.items() if isinstance(v,(float,int))}
    best = min(valid.values()) if valid else None
    return dict(
        first=sorted(k for k,v in valid.items() if v<=best+TOL) if valid else [],
        complete=len(valid)==len(CENTERS),
        pairs={a+'|'+b:(None if a not in valid or b not in valid else
               0 if abs(valid[a]-valid[b])<=TOL else -1 if valid[a]<valid[b] else 1)
               for a,b in itertools.combinations(CENTERS,2)})


def compare(left,right):
    complete=left['complete'] and right['complete']
    return dict(comparable=complete,
                first_matches=left['first']==right['first'] if complete else None,
                all_pairs_match=left['pairs']==right['pairs'] if complete else None,
                matched_pairs=sum(a is not None and a==right['pairs'][k]
                                  for k,a in left['pairs'].items()),
                pairs=6)


def report(root):
    root=Path(root)
    identity=read_json(root/'lm_head_run.json')
    if identity['scope']!='lm_head_only':
        raise ValueError('not a Head-only result tree')
    jobs={}
    sources={}
    rows=[]
    for objective in ('G','T','K'):
        for path in (root/objective/'jobs').glob('job_*/search.json'):
            value=read_json(path)
            if not value.get('complete'):
                continue
            job=value['job']
            if value['objective']!=objective:
                raise ValueError('objective metadata mismatch')
            scores=[]
            for item in value['candidates']:
                status_path=path.parent/f"candidate_{item['candidate']:02d}"/'status.json'
                status=read_json(status_path)
                metadata=read_json(status_path.parent/'metadata.json')
                execution=metadata['execution']
                if (execution['artifact_id']!=identity['input_id']
                        or execution['code']!=identity['code']
                        or execution['config']['objective']!=objective
                        or execution['config']['node']!=job['node']
                        or execution['config']['optimizer']!=job['optimizer']
                        or execution['mode']!=('test' if identity['test'] else 'formal')
                        or status['status']!=item['status']
                        or status.get('loss')!=item.get('loss')):
                    raise ValueError('candidate identity/result mismatch')
                scores.append(item)
            if winner(scores)!=value['winner']:
                raise ValueError('stored CE winner disagrees with candidate scores')
            auxiliary=winner([{**s,'loss':s.get('train_objective')} for s in scores])
            result=value['winner']
            item=dict(objective=objective,node=job['node'],optimizer=job['optimizer'],
                      sigma1=SIGMA1[job['node']//7],sigma2=SIGMA2[job['node']%7],
                      status='completed' if result else 'all_numerical_failure',
                      ce=result['loss'] if result else None,lr=result['lr'] if result else None,
                      candidate=result['candidate'] if result else None,
                      boundary_unclosed=value['boundary_unclosed'],
                      valid_candidates=value['valid_count'],
                      training_objective=auxiliary['loss'] if auxiliary else None,
                      training_candidate=auxiliary['candidate'] if auxiliary else None)
            jobs[objective,job['node'],job['optimizer']]=item
            sources[path.relative_to(root).as_posix()]=file_hash(path)
            rows.append(item)
    nodes={}
    for node in range(35):
        node_result={}
        for objective in ('G','T','K'):
            scores={o:jobs.get((objective,node,o),{}).get('ce') for o in CENTERS}
            node_result[objective]=ranking(scores)
        node_result['G_to_T']=compare(node_result['G'],node_result['T'])
        node_result['T_to_K']=compare(node_result['T'],node_result['K'])
        nodes[node]=node_result
    totals={}
    for key in ('G_to_T','T_to_K'):
        comparable=[n[key] for n in nodes.values() if n[key]['comparable']]
        totals[key]=dict(expected_nodes=35,comparable_nodes=len(comparable),
                        first_matches=sum(n['first_matches'] for n in comparable),
                        all_pairs_matches=sum(n['all_pairs_match'] for n in comparable))
    for item in rows:
        clean=jobs.get((item['objective'],0,item['optimizer']),{})
        item['ce_change_from_clean']=(item['ce']-clean['ce']
            if item['ce'] is not None and clean.get('ce') is not None else None)
        item['lr_ratio_to_clean']=(item['lr']/clean['lr']
            if item['lr'] is not None and clean.get('lr') is not None else None)
    result=dict(design=VERSION,identity=identity,kind='CPU synthetic tests' if identity['test']
                else 'formal completed candidates',metric='real_validation_ce',sources=sources,
                nodes=nodes,totals=totals,rows=rows,
                training_scores='Diagnostic best within the CE-driven candidate set; no separate search.')
    write_json(root/'comparison.json',result)
    if rows:
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
        atomic(root/'comparison.csv',lambda f:f.write(stream.getvalue().encode('utf-8-sig')))
    return result
