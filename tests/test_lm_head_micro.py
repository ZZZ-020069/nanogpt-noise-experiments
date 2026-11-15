"""Micro layout invariants; synthetic CPU values are not H100 benchmarks."""
import contextlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from nanogpt_noise.artifacts import read_json,write_json,file_hash
from nanogpt_noise.protocol import Budget
from nanogpt_noise.lm_head.protocol import code_identity,configuration
from nanogpt_noise.lm_head.runner import bind_root,run_node,smoke,serve
from nanogpt_noise.lm_head.train import run,Runtime,evaluate
from nanogpt_noise.lm_head.measure import measure
from nanogpt_noise.lm_head.__main__ import main
from nanogpt_noise.lm_head.analyze import report
from nanogpt_noise.search import propose,summarize
from tests.test_integration import fixture


class MicroLayout(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.inputs=self.base/'inputs'
        self.ident=dict(id='synthetic-layout',budget=vars(Budget()))

    def test_default_and_global_budget_all_entrypoints(self):
        for function in (run,run_node,measure,bind_root):
            self.assertEqual(inspect.signature(function).parameters['micro'].default,16)
        micro,accumulation=16,32
        self.assertEqual(Budget().accumulations(1,micro),accumulation)
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(['run','--input-root','missing','--run-root',str(self.base/'absent'),
                  '--dry-run','--micro-batch',str(micro)])
        plan=json.loads(stream.getvalue())
        self.assertEqual(plan['micro'],micro)
        self.assertEqual(plan['accumulations'],accumulation)
        self.assertEqual(plan['step_tokens'],524288)
        self.assertEqual(plan['steps'],3815)
        self.assertFalse((self.base/'absent').exists())
        with patch('nanogpt_noise.lm_head.__main__.input_identity',return_value=self.ident), \
                contextlib.redirect_stdout(io.StringIO()):
            main(['check','--input-root',str(self.inputs),'--run-root',str(self.base/'run')])
        self.assertEqual(read_json(self.base/'run/lm_head_run.json')['micro'],16)
        with patch('nanogpt_noise.lm_head.__main__.measure',return_value={'passed':True}) as call, \
                contextlib.redirect_stdout(io.StringIO()):
            main(['curvature','--input-root','unused','--output','unused'])
        self.assertEqual(call.call_args.kwargs['micro'],16)

    def test_root_lock_cross_objective_and_direct_dispatch(self):
        root=self.base/'run'
        bound=bind_root(root,self.inputs,self.ident,micro=16)
        before=(root/'lm_head_run.json').read_bytes()
        self.assertEqual(bind_root(root,self.inputs,self.ident,micro=16),bound)
        with self.assertRaisesRegex(ValueError,'micro'):
            bind_root(root,self.inputs,self.ident,micro=32)
        with patch('nanogpt_noise.lm_head.__main__.input_identity',return_value=self.ident), \
                self.assertRaisesRegex(ValueError,'micro'):
            main(['check','--input-root',str(self.inputs),'--run-root',str(root),
                  '--objective','T','--micro-batch','32'])
        args=SimpleNamespace(run_root=str(root),input_root=str(self.inputs),micro_batch=32,
                             objective='T',test=False)
        with patch('nanogpt_noise.lm_head.runner.Runtime') as runtime:
            for call in (lambda:smoke(args,self.ident),lambda:serve(args),
                         lambda:run_node(dict(index=0,node=0,optimizer='sgd'),
                                        self.inputs,root,'K',micro=32)):
                with self.assertRaisesRegex(ValueError,'micro'):call()
            runtime.assert_not_called()
        self.assertEqual((root/'lm_head_run.json').read_bytes(),before)

    def test_formal_micro_guard_precedes_gpu_and_input_work(self):
        with patch('nanogpt_noise.lm_head.train.Runtime') as runtime, \
             patch('nanogpt_noise.lm_head.__main__.input_identity') as inputs:
            for call in (
                lambda: run({},'missing','missing',self.base/'run',micro=8),
                lambda: measure('missing',self.base/'curve',micro=8),
                lambda: main(['run','--input-root','missing','--run-root','missing',
                              '--micro-batch','8','--dry-run']),
                lambda: main(['curvature','--input-root','missing','--output','missing','--micro-batch','8']),
            ):
                with self.assertRaisesRegex(ValueError,'micro=16'):call()
            runtime.assert_not_called()
            inputs.assert_not_called()

    def test_old_root_and_invalid_micro_rejected_without_migration(self):
        root=self.base/'run'
        bound=bind_root(root,self.inputs,self.ident,micro=16)
        del bound['micro']
        write_json(root/'lm_head_run.json',bound)
        before=(root/'lm_head_run.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'micro'):
            bind_root(root,self.inputs,self.ident,micro=16)
        with self.assertRaisesRegex(ValueError,'micro'):report(root)
        self.assertEqual((root/'lm_head_run.json').read_bytes(),before)
        for micro in (0,-1,3,1024):
            fresh=self.base/f'invalid-{micro}'
            with self.assertRaises(ValueError):bind_root(fresh,self.inputs,self.ident,micro=micro)
            self.assertFalse(fresh.exists())

    def make_report_fixture(self,root):
        bound=bind_root(root,self.inputs,self.ident,test=True,micro=16)
        for objective in ('G','T','K'):
            job=dict(index=0,node=0,optimizer='adamw')
            path=root/objective/'jobs/job_0000'
            rows=[]
            for index in range(8):
                proposal=propose(rows)
                item=dict(**proposal,candidate=index,status='completed',loss=1.+index/10,
                          train_objective=2.+index/10,lr=.001*10**proposal['x'],tokens=128)
                rows.append(item)
                candidate=path/f'candidate_{index:02d}'
                write_json(candidate/'status.json',dict(status='completed',loss=item['loss']))
                write_json(candidate/'metadata.json',dict(execution=dict(
                    artifact_id=bound['input_id'],code=bound['code'],micro=16,mode='test',
                    config=dict(objective=objective,node=0,optimizer='adamw'))))
            write_json(path/'search.json',dict(job=job,objective=objective,**summarize(rows)))

    def test_report_checks_every_candidate_before_writing(self):
        for value in (32,None):
            with self.subTest(value=value):
                root=self.base/str(value)
                self.make_report_fixture(root)
                path=root/'T/jobs/job_0000/candidate_07/metadata.json'
                metadata=read_json(path)
                if value is None:del metadata['execution']['micro']
                else:metadata['execution']['micro']=value
                write_json(path,metadata)
                with self.assertRaisesRegex(ValueError,'micro'):report(root)
                self.assertFalse((root/'comparison.json').exists())
                self.assertFalse((root/'comparison.csv').exists())
        root=self.base/'consistent'
        self.make_report_fixture(root)
        result=report(root)
        self.assertEqual(len(result['rows']),3)
        self.assertTrue(all(row['micro']==16 for row in result['rows']))

    def test_checkpoint_rejects_changed_micro(self):
        data,artifacts=fixture(self.base)
        output=self.base/'candidate'
        cfg=configuration('G','sgd',10,0.)
        run(cfg,data,artifacts,output,micro=2,test=True,pause_after=1)
        before=file_hash(output/'latest.pt')
        with self.assertRaisesRegex(ValueError,'incompatible'):
            run(cfg,data,artifacts,output,micro=4,test=True)
        self.assertEqual(file_hash(output/'latest.pt'),before)
        with self.assertRaises(ValueError):
            run(cfg,data,artifacts,self.base/'tiny-default',test=True)

    def test_200_sequence_evaluation_weights_actual_tail(self):
        class DeterministicLoss(torch.nn.Module):
            def features(self,ids):return ids.double()
            def loss_from_features(self,h,y,objective=None):
                return h.square().mean()+(0 if (objective or self.objective)=='G' else 7.)
        model=DeterministicLoss()
        arrays={s:np.arange(300,dtype=np.uint16) for s in ('train','val')}
        indices={f'eval_{s}':torch.arange(200) for s in arrays}
        budget=Budget(sequence=2)
        expected=np.mean([np.mean(np.arange(i,i+2,dtype=float)**2) for i in range(200)])
        for objective in ('G','T','K'):
            model.objective=objective
            for micro in (16,8):
                values=evaluate(model,arrays,indices,budget,micro,Runtime(cpu=True))
                self.assertAlmostEqual(values['val'],expected,places=10)
                self.assertAlmostEqual(values['train_objective'],
                    expected+(0 if objective=='G' else 7.),places=10)

    def test_fixed_16_full_batches_and_eval_tails_with_reference_chunks(self):
        from nanogpt_noise.lm_head.diagnostics import fixed_gradient_probe
        class FixedFeatures(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head=torch.nn.Linear(3,4,bias=False).double()
            def features(self,ids):
                x=ids.double()
                return torch.stack((x,x*x,torch.ones_like(x)),dim=-1)/5
        torch.manual_seed(12)
        model=FixedFeatures()
        for split in ('train','val'):
            (np.arange(4096)%4).astype('<u2').tofile(self.base/f'{split}.bin')
        budget=Budget(sequence=2,batch=512,steps=16,eval_sequences=200)
        indices=dict(train=torch.stack([(torch.arange(512)+i*17)%4000 for i in range(16)]),
                     eval_train=torch.arange(200),eval_val=torch.arange(1,201))
        stats=dict(B=torch.eye(3,dtype=torch.float64))
        for mode in ('fit','check'):
            reports=[]
            for micro in (16,8):
                with contextlib.redirect_stdout(io.StringIO()):
                    reports.append(fixed_gradient_probe(model,self.base,indices,budget,
                        Runtime(cpu=True),micro,stats,mode,chunk_tokens=7))
            self.assertEqual(reports[0]['count'],16 if mode=='fit' else 2)
            for first,second in zip(reports[0]['batches'],reports[1]['batches']):
                self.assertEqual(first['sequences'],512 if mode=='fit' else 200)
                self.assertAlmostEqual(first['gradient_norm'],second['gradient_norm'],places=12)
                self.assertAlmostEqual(first['random_direction_projection'],
                                       second['random_direction_projection'],places=12)

    def test_curvature_features_and_blocks_with_reference_chunks_with_tail(self):
        self.inputs.mkdir()
        data,artifacts=fixture(self.inputs,batch_size=512)
        data.rename(self.inputs/'openwebtext_subset')
        artifacts.rename(self.inputs/'frozen-artifacts')
        results=[]
        for micro in (16,8):
            output=self.base/f'curvature-{micro}'
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(measure(self.inputs,output,tokens=192,chunk_tokens=7,
                                       micro=micro,test=True))
            identity=read_json(output/'identity.json')
            self.assertEqual(identity['micro'],micro)
            self.assertEqual(len(identity['windows']),24)
        self.assertAlmostEqual(results[0]['trace'],results[1]['trace'],places=6)
        a=torch.load(self.base/'curvature-16/features.pt',weights_only=True)
        b=torch.load(self.base/'curvature-8/features.pt',weights_only=True)
        self.assertTrue(torch.equal(a['y'],b['y']))
        self.assertTrue(torch.allclose(a['h'],b['h'],atol=1e-6,rtol=1e-6))
        for index in range(10):
            a=torch.load(self.base/f'curvature-16/block-{index:02d}.pt',weights_only=True)
            b=torch.load(self.base/f'curvature-8/block-{index:02d}.pt',weights_only=True)
            self.assertEqual(a['definition'],b['definition'])
            self.assertTrue(torch.allclose(a['hessian'],b['hessian'],atol=1e-8,rtol=1e-6))
