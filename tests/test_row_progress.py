"""Synthetic report evidence and scheduling tests; these are not research results."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile
from nanogpt_noise.artifacts import digest,read_json,write_json
from nanogpt_noise.pool import row_batches,run_rows
from nanogpt_noise.progress import ScopeReporter,snapshot
from nanogpt_noise.protocol import Budget,configuration,grid
from nanogpt_noise.search import propose,summarize


class RowSchedulingTests(unittest.TestCase):
    def test_partial_expert_interval_still_orders_nodes(self):
        nodes=[grid()[batch[0]]['node'] for batch in row_batches(160,180)]
        self.assertEqual(nodes,sorted(nodes))
    def test_exact_stable_ids_and_sigma1_major_order(self):
        for first in (0,140,280,420,560,700):
            batches=row_batches(first,first+139)
            self.assertEqual(batches[0],[first,first+35,first+70,first+105])
            self.assertEqual(sorted(i for batch in batches for i in batch),list(range(first,first+140)))
            self.assertEqual([grid()[batch[0]]['node'] for batch in batches],list(range(35)))

    def test_next_row_waits_for_all_four_and_stop_blocks_following_row(self):
        completed=set();seen=[];guard=threading.Lock();stop=threading.Event()
        first=set(row_batches(140,279)[0])
        def runner(gpu,job):
            with guard:
                if grid()[job]['node']>0: self.assertTrue(first<=completed)
                seen.append(job)
            time.sleep(.02 if job==140 else .001)
            with guard: completed.add(job)
            return 0
        def finished(batch,elapsed):
            if grid()[batch[0]]['node']==1: stop.set()
        result=run_rows(140,279,['0','1','2','3'],runner,stop_requested=stop.is_set,on_row_complete=finished)
        self.assertEqual(result,[])
        self.assertEqual(len(seen),8)
        self.assertEqual({grid()[i]['node'] for i in seen},{0,1})

    def test_failure_never_advances_to_next_row(self):
        seen=[]
        def runner(gpu,job): seen.append(job);return 75
        result=run_rows(140,279,['0','1','2','3'],runner)
        self.assertTrue(result)
        self.assertTrue(all(grid()[i]['node']==0 for i in seen))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'last1';self.root.mkdir()

    def candidate(self,job,index,proposal,state='completed',code='test-code'):
        j=grid()[job];cfg=configuration(j['scope'],j['optimizer'],j['node'],proposal['x'])
        e=dict(config=cfg,artifact_id='synthetic',code=code,torch='test',cuda='test',
               mode='formal',target_steps=Budget().steps)
        identity=digest(e);path=self.root/f'job_{job:04d}'/f'candidate_{index:02d}'
        write_json(path/'metadata.json',dict(run_id=identity,execution=e,D=7079424,
                                            frozen_hash='synthetic',artifact_identity=dict(tiny=False)))
        step=Budget().steps if state=='completed' else 191
        evaluations=[dict(step=0,tokens=0,train=10.,val=10.),dict(step=step,tokens=step*Budget().step_tokens,train=5.,val=5.)]
        write_json(path/'evaluations.json',evaluations)
        write_json(path/'progress.json',dict(run_id=identity,step=step,tokens=step*Budget().step_tokens,
                                            checkpoint_step=step,state=state,metrics=dict(gradient_norm=1.)))
        write_json(path/'status.json',dict(run_id=identity,status=state,step=step,
                                          tokens=step*Budget().step_tokens,loss=5.,evaluations=evaluations))
        return dict(**proposal,candidate=index,status=state,loss=5.,lr=cfg['peak_lr'],tokens=step*Budget().step_tokens)

    def search(self,job):
        records=[]
        for i in range(8): records.append(self.candidate(job,i,propose(records)))
        write_json(self.root/f'job_{job:04d}'/'search.json',dict(job=grid()[job],mode='formal',**summarize(records)))

    def test_live_candidate_does_not_fill_final_cell(self):
        self.candidate(140,0,dict(x=-1.,reason='initial'),state='infrastructure_failure')
        report=snapshot(self.root,'last1',dict(jobs={'140':dict(state='running')}))
        self.assertEqual(report['counts']['rows_complete'],0)
        cell=report['rows'][0]['cells'][0]
        self.assertEqual(cell['state'],'running')
        self.assertIsNone(cell['loss']);self.assertIsNone(cell['rank'])
        self.assertEqual(report['candidates'][0]['progress']['step'],191)
        interrupted=snapshot(self.root,'last1')
        self.assertEqual(interrupted['candidates'][0]['state'],'pending_resume')

    def test_only_full_row_has_final_peer_ranks(self):
        self.search(140)
        report=snapshot(self.root,'last1')
        self.assertEqual(report['counts']['cells_complete'],1)
        self.assertIsNone(report['rows'][0]['cells'][0]['rank'])
        for index in (175,210,245): self.search(index)
        report=snapshot(self.root,'last1')
        self.assertEqual(report['counts']['rows_complete'],1)
        self.assertTrue(all(c['ranking_complete'] for c in report['rows'][0]['cells']))

    def test_zip_is_one_generation_has_no_weights_and_row_archive_is_immutable(self):
        for index in (140,175,210,245): self.search(index)
        output=Path(self.tmp.name)/'reports'
        reporter=ScopeReporter(self.root,'last1',lambda:dict(jobs={}),interval=.01,output=output)
        reporter.start();time.sleep(.03);reporter.close('interrupted')
        before=(output/'row_00.zip').read_bytes()
        reporter.publish()
        self.assertEqual((output/'row_00.zip').read_bytes(),before)
        with zipfile.ZipFile(output/'latest.zip') as z:
            self.assertEqual(set(z.namelist()),{'report.json','report.html','big_table.csv','evaluations.csv','cells.csv','candidates.csv'})
            self.assertEqual(len(z.read('big_table.csv').decode().splitlines()),36)
            report=json.loads(z.read('report.json'))
            self.assertIn(report['generation'],z.read('report.html').decode())
        self.assertFalse(reporter.thread.is_alive())

    def test_mixed_or_tampered_evidence_keeps_previous_report_and_exposes_error(self):
        self.candidate(140,0,dict(x=-1.,reason='initial'))
        output=Path(self.tmp.name)/'reports'
        reporter=ScopeReporter(self.root,'last1',lambda:{},output=output)
        reporter.publish();before=(output/'latest.zip').read_bytes()
        self.candidate(175,0,dict(x=-1.,reason='initial'),code='different-code')
        reporter.publish()
        self.assertEqual((output/'latest.zip').read_bytes(),before)
        self.assertIn('mixed',read_json(output/'report_error.json')['error'])

    def test_completed_status_with_short_budget_is_rejected(self):
        self.candidate(140,0,dict(x=-1.,reason='initial'))
        p=self.root/'job_0140/candidate_00/status.json';value=read_json(p);value['step']=1;write_json(p,value)
        with self.assertRaisesRegex(ValueError,'incomplete budget'): snapshot(self.root,'last1')
