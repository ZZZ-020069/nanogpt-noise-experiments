import hashlib
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import contextlib
import io

import torch
from torch.nn import functional as F

from nanogpt_noise.artifacts import read_json, tensor_hash, save_checkpoint
from nanogpt_noise.model import GPT, Shape
from nanogpt_noise.lm_head.model import HeadGPT, build_optimizer
from nanogpt_noise.lm_head.protocol import configuration, code_identity, VERSION
from nanogpt_noise.lm_head.curvature import (
    probabilities, hessian_block, trace, inclusion_moments, fitted_statistics,
    sample_blocks,statistics_block)
from nanogpt_noise.lm_head.train import run, Runtime
from tests.test_integration import fixture


class Mathematics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_untie_preserves_function_and_all_only_head_updates(self):
        shape = Shape(n_layer=2,n_head=2,n_embd=8,block_size=4,vocab_size=16)
        old, new = GPT(shape), HeadGPT(shape)
        new.load_state_dict(old.state_dict())
        new.activate_head()
        x = torch.randint(0,16,(2,4))
        y = torch.randint(0,16,(2,4))
        self.assertTrue(torch.equal(old(x,y),new(x,y)))
        named = [(n,p) for n,p in new.named_parameters() if p.requires_grad]
        self.assertEqual([n for n,_ in named],['lm_head.weight'])
        self.assertNotEqual(new.lm_head.weight.data_ptr(),new.transformer.wte.weight.data_ptr())
        frozen=[(n,p) for n,p in new.named_parameters() if not p.requires_grad]
        before=tensor_hash(frozen)
        new(x,y).backward()
        self.assertTrue((new.lm_head.weight.grad.abs().sum(1)>0).all())
        bundle=build_optimizer(named,'muon',.01)
        self.assertEqual(len(bundle.optimizers),1)
        bundle.step()
        self.assertEqual(tensor_hash(frozen),before)

    def test_sampled_hessian_matches_true_complete_softmax_autograd(self):
        h=torch.randn(7,3,dtype=torch.float64)
        w=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,1,2,3,1,0,2])
        full=torch.autograd.functional.hessian(
            lambda v:F.cross_entropy(h@v.reshape(4,3).T,y),w.flatten())
        rows,cols=[0,2],[0,1]
        indices=torch.tensor([r*3+c for r in rows for c in cols])
        p=probabilities(h,w.detach())
        block=hessian_block(h,p,rows,cols)
        self.assertTrue(torch.allclose(block,full[indices][:,indices],atol=1e-12))
        self.assertAlmostEqual(float(trace(h,p)),float(full.trace()),places=12)
        moments=[]
        for rs in itertools.combinations(range(4),2):
            for cs in itertools.combinations(range(3),2):
                b=hessian_block(h,p,list(rs),list(cs))
                moments.append(inclusion_moments(b,rs,cs,4,3))
        self.assertAlmostEqual(sum(m['trace'] for m in moments)/len(moments),float(full.trace()),places=6)
        self.assertAlmostEqual(sum(m['second_moment'] for m in moments)/len(moments),float(full.square().sum()),places=6)

    def test_taylor_matches_gradient_and_hessian_at_reference(self):
        shape=Shape(n_layer=1,n_head=1,n_embd=3,block_size=2,vocab_size=4)
        model=HeadGPT(shape).activate_head('T').double()
        h=torch.randn(2,2,3,dtype=torch.float64)
        y=torch.tensor([[0,1],[2,3]])
        # Production losses deliberately reduce logits in FP32; compare at that tolerance.
        def gradient_of(kind):
            model.zero_grad()
            model.loss_from_features(h,y,kind).backward()
            return model.lm_head.weight.grad.clone()
        self.assertTrue(torch.allclose(gradient_of('G'),gradient_of('T'),atol=2e-7))
        ids=torch.tensor([[0,1],[2,3]])
        def full_hessian(kind):
            model.objective=kind
            return torch.autograd.functional.hessian(lambda v:torch.func.functional_call(
                model,{'lm_head.weight':v},(ids,y)),model.lm_head.weight).reshape(12,12)
        self.assertTrue(torch.allclose(full_hessian('G'),full_hessian('T'),atol=2e-7))
        h32=h.float()
        w=model.reference_head.float()
        p=probabilities(h32.flatten(0,1),w)
        stats=fitted_statistics(h32.flatten(0,1),p)
        k=HeadGPT(shape)
        k.load_state_dict(GPT(shape).state_dict())
        k.activate_head('K',stats)
        g0=torch.autograd.grad(k.loss_from_features(h32,y,'G'),k.lm_head.weight)[0]
        gk=torch.autograd.grad(k.loss_from_features(h32,y,'K'),k.lm_head.weight)[0]
        self.assertTrue(torch.allclose(g0,gk,atol=2e-7))
        kh=torch.autograd.functional.hessian(lambda v:torch.func.functional_call(
            k,{'lm_head.weight':v},(ids,y)),k.lm_head.weight).reshape(12,12)
        self.assertTrue(torch.allclose(kh.double(),statistics_block(stats,list(range(4)),
            list(range(3))),atol=2e-7))

    def test_nested_sampling_and_rng_isolation(self):
        from nanogpt_noise.lm_head.measure import probe_windows
        state=torch.random.get_rng_state()
        small=sample_blocks(64,16,tiny=True,count=5)
        large=sample_blocks(64,16,tiny=True,count=10)
        for b in small:
            self.assertIn(b,large)
        p=probe_windows(4096,8,8,True)
        q=probe_windows(4096,8,16,True)
        for key in p:self.assertEqual(p[key],q[key][:8])
        self.assertTrue(all(abs(a-b)>8 for a in q['fit'] for b in q['check']))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_taylor_diagnostic_and_nullspace_failure(self):
        from nanogpt_noise.lm_head.diagnostics import taylor_chunk,compatible
        h=torch.randn(5,3,dtype=torch.float64)
        w=torch.randn(4,3,dtype=torch.float64)
        y=torch.tensor([0,1,2,3,0])
        d=taylor_chunk(h,y,w,w)
        self.assertEqual(float(d['real']),float(d['taylor']))
        self.assertTrue(torch.equal(d['gradient'],d['taylor_gradient']))
        b=torch.diag(torch.tensor([1.,1.,0.],dtype=torch.float64))
        g=torch.zeros_like(w);g[0,2]=1;g[1,2]=-1
        self.assertFalse(compatible(dict(B=b),g)['passed'])

    def test_rank_ties_missing_and_failure(self):
        from nanogpt_noise.lm_head.analyze import ranking,compare
        a=ranking(dict(adamw=1.,adam=1.+1e-8,sgd=2.,muon=3.))
        b=ranking(dict(adamw=1.,adam=1.1,sgd=2.,muon=3.))
        self.assertEqual(a['first'],['adam','adamw'])
        self.assertFalse(compare(a,b)['all_pairs_match'])
        c=ranking(dict(adamw=None,adam=1.,sgd=2.,muon=3.))
        self.assertFalse(compare(c,c)['comparable'])

    def test_code_identity_includes_new_implementation(self):
        names=code_identity()['files']
        self.assertIn('nanogpt_noise/lm_head/train.py',names)
        self.assertIn('nanogpt_noise/lm_head/model.py',names)


class Recovery(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_head_only_resume_all_optimizers_and_objectives(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data,artifacts=fixture(root)
            ident=read_json(artifacts/'identity.json')
            shape=Shape(**ident['shape'])
            statpath=root/'test-statistics.pt'
            stats=dict(B=torch.eye(shape.n_embd,dtype=torch.float64),
                       pbar=torch.ones(shape.vocab_size,dtype=torch.float64)/shape.vocab_size,
                       alpha=torch.tensor(1.,dtype=torch.float64))
            save_checkpoint(statpath,dict(schema='lm-head-K-v1',design=VERSION,
                input_id=ident['id'],initialization_hash=ident['initialization_tensor_hash'],
                shape=ident['shape'],code=code_identity()['sha256'],
                passed=True,test=True,statistics=stats))
            for objective,optimizer in itertools.product(('G','T','K'),('sgd','adam','adamw','muon')):
                with self.subTest(objective=objective,optimizer=optimizer):
                    cfg=configuration(objective,optimizer,10,0.)
                    if objective=='K':cfg['statistics']=str(statpath)
                    prefix=root/(objective+'-'+optimizer)
                    paused=Path(str(prefix)+'-pause')
                    whole=Path(str(prefix)+'-whole')
                    rt=Runtime(cpu=True)
                    first=run(cfg,data,artifacts,paused,micro=2,runtime=rt,test=True,pause_after=1)
                    self.assertEqual(first['step'],1)
                    actual=run(cfg,data,artifacts,paused,micro=2,runtime=rt,test=True)
                    expected=run(cfg,data,artifacts,whole,micro=2,runtime=rt,test=True)
                    self.assertEqual(actual['loss'],expected['loss'])
                    a=torch.load(paused/'final.pt',weights_only=True)
                    b=torch.load(whole/'final.pt',weights_only=True)
                    self.assertNotIn('model',a)
                    self.assertTrue(torch.equal(a['head'],b['head']))
                    self.assertEqual(a['backbone_hash'],b['backbone_hash'])
                    for sa,sb in zip(a['optimizer'],b['optimizer']):
                        for key,values in sa['state'].items():
                            for name,value in values.items():
                                other=sb['state'][key][name]
                                if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,other))
                                else:self.assertEqual(value,other)


class Workflow(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.inputs=self.root/'inputs';self.inputs.mkdir()
        data,artifacts=fixture(self.inputs)
        data.rename(self.inputs/'openwebtext_subset')
        artifacts.rename(self.inputs/'frozen-artifacts')

    def tearDown(self):
        self.tmp.cleanup()

    def test_measurements_and_failed_k_gate(self):
        from nanogpt_noise.lm_head.measure import measure,load_statistics
        output=self.root/'curvature'
        with contextlib.redirect_stdout(io.StringIO()):
            result=measure(self.inputs,output,tokens=256,chunk_tokens=32,test=True)
        self.assertEqual(len(result['blocks']),10)
        self.assertEqual(result['taylor']['displacement'],0.)
        self.assertLess(result['taylor']['absolute_gradient_error'],1e-12)
        self.assertTrue(result['numeric_passed'])
        ident=read_json(self.inputs/'frozen-artifacts'/'identity.json')
        with self.assertRaises(ValueError):
            load_statistics(output/'fit.pt',ident,Shape(**ident['shape']),allow_test=True)
        with self.assertRaises(ValueError):
            measure(self.inputs,self.inputs/'bad',tokens=256,test=True)
        with self.assertRaises(ValueError):
            measure(self.inputs,output,tokens=128,test=True)
        if not result['passed']:
            with self.assertRaises(ValueError),contextlib.redirect_stdout(io.StringIO()):
                measure(self.inputs,self.root/'check',mode='check',fit_dir=output,tokens=256,test=True)
            self.assertFalse((self.root/'check'/'K-approved.pt').exists())

    def test_input_and_output_gates_and_readonly_plan(self):
        from nanogpt_noise.lm_head.runner import bind_root,input_identity,job_list
        from nanogpt_noise.lm_head.__main__ import main
        ident=input_identity(self.inputs,True)
        runroot=self.root/'runs'
        self.assertEqual(len(job_list('clean')),4)
        self.assertEqual(len(job_list('grid')),140)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['run','--input-root',str(self.root/'absent'),
                '--run-root',str(runroot),'--dry-run']),0)
        self.assertFalse(runroot.exists())
        expected=bind_root(runroot,self.inputs,ident,True)
        self.assertEqual(bind_root(runroot,self.inputs,ident,True),expected)
        with self.assertRaises(ValueError):
            bind_root(self.inputs/'bad',self.inputs,ident,True)
        legacy=self.root/'legacy';legacy.mkdir();(legacy/'old.txt').write_text('preserve')
        with self.assertRaises(ValueError):bind_root(legacy,self.inputs,ident,True)
        with self.assertRaises(ValueError):input_identity(self.inputs,False)
        from nanogpt_noise.artifacts import write_json
        write_json(runroot/'lm_head_run.json',{**expected,'code':'changed'})
        with self.assertRaises(ValueError):bind_root(runroot,self.inputs,ident,True)

    def test_checkpoint_drift_uses_initial_block_prefix_and_never_approves_k(self):
        from nanogpt_noise.lm_head.measure import measure
        candidate=self.root/'candidate'
        run(configuration('G','sgd',0,0.),self.inputs/'openwebtext_subset',
            self.inputs/'frozen-artifacts',candidate,micro=2,test=True)
        with contextlib.redirect_stdout(io.StringIO()):
            fitted=measure(self.inputs,self.root/'drift-fit',checkpoint=candidate/'final.pt',
                           tokens=64,test=True)
            checked=measure(self.inputs,self.root/'drift-check','check',fit_dir=self.root/'drift-fit',
                            checkpoint=candidate/'final.pt',tokens=64,test=True)
        self.assertEqual(len(fitted['blocks']),4)
        self.assertEqual(len(checked['blocks']),4)
        self.assertEqual(checked['purpose'],'trajectory_drift')
        self.assertTrue(checked['numeric_passed'])
        self.assertFalse((self.root/'drift-check'/'K-approved.pt').exists())
        initial=sample_blocks(64,16,tiny=True)
        for b in fitted['blocks']:
            expected=next(s for s in initial if s['pool']==b['pool']
                          and s['layout']==b['layout'] and s['index']==b['index'])
            self.assertEqual(b['rows'],expected['rows'])
            self.assertEqual(b['cols'],expected['cols'])

    def test_b8_pool_pause_replay_retention_and_report(self):
        from nanogpt_noise.lm_head.runner import input_identity,bind_root,smoke,serve
        from nanogpt_noise.lm_head.analyze import report
        from nanogpt_noise.artifacts import write_json
        output=self.root/'runs'
        args=SimpleNamespace(input_root=str(self.inputs),run_root=str(output),objective='G',
            statistics=None,test=True,micro_batch=2,devices='0,1',phase='clean',resume=False)
        ident=input_identity(self.inputs,True);bind_root(output,self.inputs,ident,True)
        smoke(args,ident)
        write_json(output/'pause.request',dict(reason='test'))
        with self.assertRaises(RuntimeError):serve(args)
        args.resume=True
        with contextlib.redirect_stdout(io.StringIO()):
            result=serve(args)
        self.assertEqual(result['status'],'completed','\n'.join(
            p.read_text(encoding='utf-8',errors='replace')[-4000:] for p in (output/'logs').glob('*.log')))
        self.assertEqual(result['completed_jobs'],4)
        from nanogpt_noise.artifacts import file_hash
        before={p.relative_to(output).as_posix():file_hash(p) for p in output.rglob('final.pt')}
        self.assertEqual(len(before),8)  # four smoke plus four retained winners
        with contextlib.redirect_stdout(io.StringIO()):
            replay=serve(args)
        self.assertEqual(replay['completed_jobs'],4)
        self.assertEqual(before,{p.relative_to(output).as_posix():file_hash(p)
                                 for p in output.rglob('final.pt')})
        comparison=report(output)
        self.assertEqual(len(comparison['rows']),4)
        self.assertEqual(comparison['totals']['G_to_T']['comparable_nodes'],0)
        self.assertTrue((output/'comparison.csv').exists())


if __name__=='__main__':
    unittest.main()
