"""Automatic, read-only stage reports. No checkpoints are loaded or copied."""
import csv
from datetime import datetime,timezone
import html
import io
import json
from pathlib import Path
import threading
import uuid
import zipfile
from .artifacts import atomic,digest,read_json,write_json
from .analyze import collect
from .protocol import Budget,CENTERS,SCOPES,SIGMA1,SIGMA2,configuration,grid


def read_optional(path,default=None):
    return read_json(path) if path.is_file() else default


def snapshot(root,scope,execution=None,event='periodic'):
    root=Path(root)
    jobs=grid()
    finals={r['job_index']:r for r in collect(root) if r['scope']==scope}
    candidates=[]
    cohort=None
    runtime=(execution or {}).get('jobs',{})
    for index,job in enumerate(jobs):
        if job['scope']!=scope: continue
        directory=root/f'job_{index:04d}'
        entries=[]
        for number in range(8):
            path=directory/f'candidate_{number:02d}'
            meta=read_optional(path/'metadata.json')
            if meta is None: continue
            e=meta['execution']; config=e['config']
            entry=dict(job_index=index,candidate=number,optimizer=job['optimizer'],node=job['node'],
                       sigma1=job['sigma1'],sigma2=job['sigma2'],state='excluded_nonformal')
            if e['mode']!='formal':
                candidates.append(entry);continue
            expected=configuration(scope,job['optimizer'],job['node'],config['exponent'])
            if (meta['run_id']!=digest(e) or any(config.get(k)!=v for k,v in expected.items())
                    or e['target_steps']!=Budget().steps or meta['artifact_identity']['tiny']):
                raise ValueError(f'invalid formal candidate identity: {path}')
            signature=(e['artifact_id'],e['code'],e['torch'],e['cuda'],e['mode'])
            if cohort is not None and signature!=cohort:
                raise ValueError('mixed code/data/environment cohorts in live report')
            cohort=signature
            status=read_optional(path/'status.json',{})
            progress=read_optional(path/'progress.json',{})
            for value in (status,progress):
                if value and value.get('run_id')!=meta['run_id']:
                    raise ValueError(f'progress/status identity mismatch: {path}')
            state=status.get('status',progress.get('state','starting'))
            if state not in ('completed','numerical_failure'):
                state='running' if runtime.get(str(index),{}).get('state')=='running' else 'pending_resume'
            if state=='completed' and (status.get('step')!=Budget().steps or status.get('tokens')!=Budget().tokens):
                raise ValueError(f'incomplete budget marked completed: {path}')
            entry.update(state=state,peak_lr=config['peak_lr'],exponent=config['exponent'],
                         run_id=meta['run_id'],D=meta['D'],frozen_hash=meta['frozen_hash'],
                         progress=progress,evaluations=read_optional(path/'evaluations.json',[]),
                         loss=status.get('loss') if state=='completed' else None,
                         failure_field=status.get('field'),T_diverge=status.get('T_diverge'))
            entries.append(entry);candidates.append(entry)
        final=finals[index]
        if not final['ranking_complete']: final['rank']=None
        if final['search_complete'] and final['status']!='excluded_nonformal':
            if len(entries)!=8 or any(e['state'] not in ('completed','numerical_failure') for e in entries):
                raise ValueError(f'search summary is ahead of valid candidate evidence: {directory}')
        final['state']=final['status'] if final['search_complete'] or final['status']=='excluded_nonformal' else (
            'running' if runtime.get(str(index),{}).get('state')=='running' else
            'pending_resume' if entries else 'not_started')
        final['candidates_finished']=sum(e['state'] in ('completed','numerical_failure') for e in entries)
    rows=[]
    for node in range(35):
        cells=[finals[i] for i,j in enumerate(jobs) if j['scope']==scope and j['node']==node]
        complete=all(c['search_complete'] and c['status'] in ('completed','no_valid_lr') for c in cells)
        rows.append(dict(node=node,sigma1=SIGMA1[node//7],sigma2=SIGMA2[node%7],
                         complete=complete,cells=cells))
    return dict(schema=1,generation=uuid.uuid4().hex,updated_at=datetime.now(timezone.utc).isoformat(),
                scope=scope,event=event,kind='live_stage_report',
                snapshot_note='Read-only observation of atomic files; not a transactional training checkpoint. '
                              'Incomplete candidates/searches are not final optimizer rankings.',
                order='sigma1-major, sigma2-minor; row-priority queue; idle GPU claims next search',
                cohort=cohort,rows=rows,candidates=candidates,execution=execution or {},
                counts=dict(rows_complete=sum(r['complete'] for r in rows),
                            cells_complete=sum(c['search_complete'] for c in finals.values()),
                            candidates_finished=sum(c['state'] in ('completed','numerical_failure') for c in candidates)))


def csv_text(records,fields):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
    writer.writeheader();writer.writerows(records);return stream.getvalue()


def svg_curves(candidates):
    curves=[c for c in candidates if c.get('evaluations')]
    values=[p['val'] for c in curves for p in c['evaluations']]
    if not values: return '<p>No evaluations yet.</p>'
    lo,hi=min(values),max(values);span=max(hi-lo,1e-9)
    colors=dict(zip(CENTERS,('#2563eb','#db2777','#059669','#d97706')))
    lines=[]
    for c in curves:
        points=' '.join(f"{40+640*p['tokens']/Budget().tokens:.1f},{260-220*(p['val']-lo)/span:.1f}"
                        for p in c['evaluations'])
        title=f"{c['optimizer']} node={c['node']} candidate={c['candidate']} peak_lr={c['peak_lr']}"
        lines.append(f'<polyline fill="none" stroke="{colors[c["optimizer"]]}" points="{points}"><title>{html.escape(title)}</title></polyline>')
    return f'<p>Validation loss {lo:.5g}–{hi:.5g}; x-axis: 0–2B tokens. Curves are individual LR candidates.</p>'+\
           '<svg viewBox="0 0 720 290" role="img" aria-label="Candidate validation curves">'+''.join(lines)+'</svg>'


def render(report):
    body=[]
    for row in report['rows']:
        cells=[]
        for c in row['cells']:
            value=f"loss={c['loss']:.6g}; LR={c['best_lr']:.6g}" if c['loss'] is not None else '—'
            cells.append('<td>'+html.escape(f"{c['state']} ({c['candidates_finished']}/8); {value}")+'</td>')
        body.append(f'<tr><td>{row["node"]}</td><td>{row["sigma1"]:g}</td><td>{row["sigma2"]:g}</td>'+''.join(cells)+'</tr>')
    candidate_rows=[]
    for c in report['candidates']:
        p=c.get('progress',{});m=p.get('metrics',{})
        values=[c['node'],c['optimizer'],c['candidate'],c['state'],c.get('peak_lr'),p.get('step'),
                p.get('tokens'),p.get('checkpoint_step'),p.get('last_verified_eval_step'),c.get('D'),
                m.get('step'),m.get('lr'),m.get('auxiliary_lr'),m.get('train_loss'),
                c.get('evaluations',[])[-1]['val'] if c.get('evaluations') else None,
                m.get('gradient_norm'),m.get('noise_rms'),m.get('noise_norm'),
                m.get('actual_ratio'),m.get('update_norm'),c.get('failure_field')]
        candidate_rows.append(f'<tr data-node="{c["node"]}">'+''.join('<td>'+html.escape(str(v) if v is not None else '—')+'</td>' for v in values)+'</tr>')
    headers=('node','optimizer','candidate','state','peak LR','step','tokens','saved step','verified eval step',
             'D','diagnostic step','diagnostic LR','auxiliary LR','training loss','latest validation loss',
             'gradient norm','noise RMS','noise norm','noise/gradient','update norm','failure')
    curves=''.join(f'<div data-node="{node}"><h3>Node {node}: sigma1={SIGMA1[node//7]:g}, sigma2={SIGMA2[node%7]:g}</h3>'+\
                   svg_curves([c for c in report['candidates'] if c['node']==node])+'</div>' for node in range(35))
    options=''.join(f'<option value="{n}">Node {n}: ({SIGMA1[n//7]:g}, {SIGMA2[n%7]:g})</option>' for n in range(35))
    return '<!doctype html><meta charset="utf-8"><title>Stage report</title><style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse;margin:16px 0}td,th{border:1px solid #ccc;padding:7px}svg{max-width:900px;width:100%}</style>'+\
        f'<h1>{html.escape(report["scope"])} stage report</h1><p>{report["updated_at"]} · {report["generation"]} · {report["event"]}</p>'+\
        '<p>'+html.escape(report['snapshot_note'])+'</p><p>'+html.escape(str(report['counts']))+'</p>'+\
        '<table><tr><th>node</th><th>sigma1</th><th>sigma2</th>'+''.join(f'<th>{o}</th>' for o in CENTERS)+'</tr>'+''.join(body)+'</table>'+\
        '<h2>Individual candidate curves (provisional until searches finish)</h2>'+\
        '<p>Colors: AdamW blue; Adam pink; SGD green; Muon orange. Hover a curve for its LR.</p>'+\
        '<label>Detail node: <select id="node"><option value="all">All nodes</option>'+options+'</select></label>'+curves+\
        '<h2>Candidate progress and diagnostics</h2><table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join(candidate_rows)+'</table>'+\
        '<script>const select=document.getElementById("node");function filter(){document.querySelectorAll("[data-node]").forEach(e=>e.style.display=(select.value==="all"||e.dataset.node===select.value)?"":"none");}select.value="0";select.onchange=filter;filter();</script>'


class ScopeReporter:
    def __init__(self,root,scope,runtime,interval=300,output=None):
        self.root=Path(root);self.scope=scope;self.runtime=runtime;self.interval=interval
        self.output=Path(output) if output else self.root.parent/'reports'/scope
        self.stop_event=threading.Event();self.writer=threading.Lock();self.thread=None
        self.announced=set()
        self.wake=threading.Event()

    def publish(self,event='periodic'):
        with self.writer:
            try:
                report=snapshot(self.root,self.scope,self.runtime(),event)
                page=render(report)
                cells=[c for r in report['rows'] for c in r['cells']]
                wide=[];details=[];evaluations=[]
                for row in report['rows']:
                    value={k:row[k] for k in ('node','sigma1','sigma2','complete')}
                    for c in row['cells']:
                        for key in ('state','loss','best_lr','candidates_finished','rank','boundary_unclosed'):
                            value[c['optimizer']+'_'+key]=c[key]
                    wide.append(value)
                for c in report['candidates']:
                    p=c.get('progress',{});m=p.get('metrics',{})
                    details.append({**c,**{k:p.get(k) for k in ('step','tokens','checkpoint_step','last_verified_eval_step')},
                                    'diagnostic_step':m.get('step'),'diagnostic_lr':m.get('lr'),
                                    'gradient_norm':m.get('gradient_norm'),'noise_norm':m.get('noise_norm'),
                                    'actual_ratio':m.get('actual_ratio'),'update_norm':m.get('update_norm')})
                    evaluations.extend(dict(node=c['node'],optimizer=c['optimizer'],candidate=c['candidate'],**e)
                                       for e in c.get('evaluations',[]))
                files={'report.json':json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2),
                       'report.html':page,
                       'big_table.csv':csv_text(wide,list(wide[0])),
                       'evaluations.csv':csv_text(evaluations,['node','optimizer','candidate','step','tokens','train','val']),
                       'cells.csv':csv_text(cells,['node','sigma1','sigma2','optimizer','state','candidates_finished',
                                                   'loss','best_lr','valid_count','boundary_unclosed','delta_loss','rank','ranking_complete']),
                       'candidates.csv':csv_text(details,['node','optimizer','candidate','state','peak_lr','loss','step','tokens',
                                                         'checkpoint_step','last_verified_eval_step','diagnostic_step','diagnostic_lr',
                                                         'gradient_norm','noise_norm','actual_ratio','update_norm','failure_field','T_diverge'])}
                def archive(stream):
                    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
                        for name,value in files.items(): z.writestr(name,value)
                atomic(self.output/'latest.zip',archive)
                atomic(self.output/'latest.html',lambda f:f.write(page.encode('utf-8')))
                write_json(self.output/'latest.json',report)
                for row in report['rows']:
                    path=self.output/f'row_{row["node"]:02d}.zip'
                    if row['complete'] and not path.exists(): atomic(path,archive)
                (self.output/'report_error.json').unlink(missing_ok=True)
                print(f'[report] {event}: {report["counts"]}; {self.output}/latest.zip',flush=True)
                for c in report['candidates']:
                    if c['state']=='running':
                        p=c.get('progress',{});m=p.get('metrics',{});curve=c.get('evaluations',[])
                        print(f'[report] live node={c["node"]} {c["optimizer"]} candidate={c["candidate"]} '
                              f'step={p.get("step")} tokens={p.get("tokens")} saved={p.get("checkpoint_step")} '
                              f'val={curve[-1]["val"] if curve else None} diagnostic_step={m.get("step")} '
                              f'lr={m.get("lr")} gradient={m.get("gradient_norm")} noise_ratio={m.get("actual_ratio")}',flush=True)
                for row in report['rows']:
                    if row['complete'] and row['node'] not in self.announced:
                        print(f'[report] completed row={row["node"]} sigma1={row["sigma1"]} sigma2={row["sigma2"]}: '+
                              '; '.join(f'{c["optimizer"]}: {c["state"]}, loss={c["loss"]}, LR={c["best_lr"]}'
                                        for c in row['cells']),flush=True)
                        self.announced.add(row['node'])
            except Exception as exc:
                print(f'[report] ERROR: {exc!r}; training decisions unchanged; retrying next update',flush=True)
                try: write_json(self.output/'report_error.json',dict(error=repr(exc),updated_at=datetime.now(timezone.utc).isoformat()))
                except OSError: pass

    def request_update(self):
        """Wake the reporter without blocking GPU workers on report I/O."""
        self.wake.set()

    def start(self):
        self.publish('startup')
        def loop():
            while not self.stop_event.is_set():
                notified=self.wake.wait(self.interval)
                self.wake.clear()
                if self.stop_event.is_set(): break
                self.publish('row_complete' if notified else 'periodic')
        self.thread=threading.Thread(target=loop,name='stage-reporter',daemon=True);self.thread.start()

    def close(self,event):
        self.stop_event.set()
        self.wake.set()
        if self.thread: self.thread.join()
        self.publish(event)
