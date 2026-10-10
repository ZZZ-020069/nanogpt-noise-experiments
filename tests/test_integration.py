import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from nanogpt_noise.artifacts import write_json,read_json,tensor_hash
from nanogpt_noise.data import freeze,verify
from nanogpt_noise.protocol import configuration,grid
from nanogpt_noise.train import run,Runtime
from nanogpt_noise.node import run_node
from nanogpt_noise.analyze import auc,collect


def fixture(root, batch_size=4):
    data=root/'data'
    data.mkdir()
    for split,size in (('train',1024),('val',256)):
        (np.arange(size)%64).astype('<u2').tofile(data/f'{split}.bin')
    write_json(data/'meta.json',dict(kind='synthetic_test_fixture'))
    artifacts=root/'artifacts'
    from nanogpt_noise.protocol import Budget
    with patch('nanogpt_noise.data.Budget',
               side_effect=lambda **kw: Budget(**{**kw, 'batch': batch_size})):
        freeze(data,artifacts,tiny=True)
    return data,artifacts


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.data,self.artifacts=fixture(self.root)
        self.rt=Runtime(cpu=True)

    def tearDown(self): self.tmp.cleanup()

    def test_all_optimizers_resume_equivalence(self):
        for kind in ('adamw','adam','sgd','muon'):
            cfg=configuration('last1',kind,10,0)
            uninterrupted=self.root/f'{kind}_whole'
            resumed=self.root/f'{kind}_resume'
            a=run(cfg,self.data,self.artifacts,uninterrupted,micro=2,runtime=self.rt,test=True)
            paused=run(cfg,self.data,self.artifacts,resumed,micro=2,runtime=self.rt,test=True,pause_after=1)
            self.assertEqual(paused['status'],'infrastructure_failure')
            b=run(cfg,self.data,self.artifacts,resumed,micro=2,runtime=self.rt,test=True)
            self.assertEqual(a['loss'],b['loss'])
            self.assertEqual(a['tokens'],128)
            self.assertEqual(a['status'],'completed')
            self.assertEqual(a['mode'],'test')
            sa=torch.load(uninterrupted/'final.pt',weights_only=True)
            sb=torch.load(resumed/'final.pt',weights_only=True)
            for name,value in sa['model'].items():
                self.assertTrue(torch.equal(value,sb['model'][name]),name)
            meta=read_json(resumed/'metadata.json')
            self.assertEqual(meta['frozen_hash'],b['frozen_hash_final'])
            again=run(cfg,self.data,self.artifacts,resumed,micro=2,runtime=self.rt,test=True)
            self.assertEqual(again,b)
            with self.assertRaises(ValueError):
                run(configuration('last1',kind,10,.5),self.data,self.artifacts,resumed,micro=2,runtime=self.rt,test=True)

    def test_failure_is_terminal_not_infrastructure(self):
        cfg=configuration('full','adamw',0,0)
        from nanogpt_noise.model import GPT
        forward=GPT.forward
        def broken(m,x,y): return forward(m,x,y)*float('nan')
        with patch.object(GPT,'forward',broken):
            result=run(cfg,self.data,self.artifacts,self.root/'failed',micro=2,runtime=self.rt,test=True)
        self.assertEqual(result['status'],'numerical_failure')
        self.assertEqual(result['field'],'eval_train_loss')
        self.assertEqual(result['tokens'],0)
        self.assertTrue((self.root/'failed'/'failure.pt').exists())

    def test_eight_candidate_node(self):
        result=run_node(grid()[0],self.data,self.artifacts,self.root/'node',micro=2,runtime=self.rt,test=True)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['candidates']),8)
        self.assertEqual(result['valid_count'],8)
        again=run_node(grid()[0],self.data,self.artifacts,self.root/'node',micro=2,runtime=self.rt,test=True)
        self.assertEqual(again,result)
        self.assertIsNotNone(auc(read_json(self.root/'node'/'candidate_00'/'evaluations.json')))

    def test_identity_tamper_and_missing_surface(self):
        verify(self.data,self.artifacts)
        with open(self.data/'val.bin','ab') as f: f.write(b'00')
        with self.assertRaises(ValueError): verify(self.data,self.artifacts)
        rows=collect(self.root/'empty')
        self.assertEqual(len(rows),840)
        self.assertTrue(all(r['loss'] is None for r in rows))

    def test_finite_large_loss_is_not_early_stopped(self):
        cfg=configuration('full','sgd',0,0)
        from nanogpt_noise.model import GPT
        forward=GPT.forward
        def finite(m,x,y): return forward(m,x,y)*0+1e12
        with patch.object(GPT,'forward',finite):
            result=run(cfg,self.data,self.artifacts,self.root/'large',micro=2,runtime=self.rt,test=True)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['step'],4)

    def test_training_nonfinite_counts_processed_tokens(self):
        cfg=configuration('full','adamw',0,0)
        from nanogpt_noise.model import GPT
        forward=GPT.forward
        def broken(m,x,y):
            value=forward(m,x,y)
            return value*float('nan') if m.training else value
        with patch.object(GPT,'forward',broken):
            result=run(cfg,self.data,self.artifacts,self.root/'train-failed',micro=2,runtime=self.rt,test=True)
        self.assertEqual(result['status'],'numerical_failure')
        self.assertEqual(result['field'],'training_loss')
        self.assertEqual(result['step'],0)
        self.assertEqual(result['tokens'],16)


if __name__=='__main__': unittest.main()
