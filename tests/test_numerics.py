import unittest
import torch
from nanogpt_noise.model import GPT,Shape,set_scope
from nanogpt_noise.optim import build,Muon,orthogonalize
from nanogpt_noise.noise import inject
from nanogpt_noise.data import starts,windows
from nanogpt_noise.protocol import Budget,SCOPES


class NumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_exact_counts_and_tying(self):
        with torch.device('meta'): m=GPT()
        self.assertIs(m.lm_head.weight,m.transformer.wte.weight)
        for scope in SCOPES:
            named=set_scope(m,scope)
            expected=124373760 if scope=='full' else 7079424*int(scope[4:])
            self.assertEqual(sum(p.numel() for _,p in named),expected)
            self.assertEqual(len({id(p) for _,p in named}),len(named))
            if scope!='full': self.assertFalse(m.transformer.wte.weight.requires_grad)

    def test_freeze_and_groups(self):
        m=GPT(Shape(2,2,16,8,64))
        named=set_scope(m,'last1')
        before={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
        opt=build(named,'muon',.02)
        m(torch.ones(2,8,dtype=torch.long),torch.zeros(2,8,dtype=torch.long)).backward()
        self.assertTrue(all(p.grad is not None for _,p in named))
        opt.set_lr(.1)
        self.assertEqual(opt.optimizers[1].param_groups[0]['lr'],.003)
        inject(named,.3,1.,0)
        opt.step()
        for n,p in m.named_parameters():
            if n in before: self.assertTrue(torch.equal(p,before[n]))

    def test_muon_recurrence(self):
        p=torch.nn.Parameter(torch.arange(8,dtype=torch.float32).reshape(4,2)/10)
        p.grad=torch.ones_like(p)*.2
        old=p.detach().clone()
        opt=Muon([p],.02)
        expected=(1-.02*.01)*old-.02*(2**.5)*orthogonalize(.05*p.grad+.95*.05*p.grad).float()
        opt.step()
        torch.testing.assert_close(p,expected)
        torch.testing.assert_close(opt.state[p]['momentum'],.05*p.grad)

    def test_noise_moments_and_independent_rng(self):
        p=torch.nn.Parameter(torch.zeros(100000))
        p.grad=torch.ones_like(p)*.01
        original=p.grad.clone()
        state=torch.random.get_rng_state()
        stats=inject([('p',p)],.3,1.,7)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        xi=p.grad-original
        self.assertAlmostEqual(stats['noise_rms']**2,.09*10+1,places=5)
        self.assertLess(abs(stats['noise_norm']/stats['noise_rms']-1),.02)
        self.assertLess(abs(float(xi.mean())),stats['noise_std']*.02)
        first=xi.clone()
        p.grad=original.clone()
        inject([('p',p)],.3,1.,7)
        self.assertTrue(torch.equal(first,p.grad-original))
        p.grad=original.clone()
        inject([('p',p)],0,0,7)
        self.assertTrue(torch.equal(p.grad,original))

    def test_windows_layout_independent(self):
        a=starts(10000,512,1024,1337)
        for world in (1,2,4,8):
            self.assertTrue(torch.equal(a.reshape(world,-1).flatten(),a))
        b=Budget(sequence=8,batch=4,steps=4,warmup=32,eval_every=64,eval_sequences=4)
        self.assertTrue(torch.equal(windows(dict(train=1000,val=100),b)['train'][0],starts(1000,4,8,1337)))

    def test_noise_zero_and_missing_gradient(self):
        p=torch.nn.Parameter(torch.zeros(10))
        with self.assertRaises(RuntimeError): inject([('p',p)],0,1,0)
        p.grad=torch.zeros_like(p)
        result=inject([('p',p)],0,1,0)
        self.assertIsNone(result['actual_ratio'])
        self.assertEqual(result['undefined_reason'],'zero denominator')

    def test_adam_and_sgd_decay_recipes(self):
        for kind in ('adam','adamw','sgd'):
            p=torch.nn.Parameter(torch.ones(2,2))
            p.grad=torch.full_like(p,.5)
            opt=build([('matrix',p)],kind,.01)
            opt.step()
            if kind=='sgd': expected=1-.01*(.5+.0001)
            elif kind=='adam': expected=1-.01*.6/(.6+1e-8)
            else: expected=(1-.01*.1)-.01*.5/(.5+1e-8)
            torch.testing.assert_close(p,torch.full_like(p,expected))

    def test_zero_lr_updates_state_not_parameters(self):
        for kind in ('adam','adamw','sgd','muon'):
            p=torch.nn.Parameter(torch.ones(2,2))
            v=torch.nn.Parameter(torch.ones(2))
            p.grad=torch.ones_like(p)
            v.grad=torch.ones_like(v)
            opt=build([('transformer.h.0.matrix',p),('transformer.h.0.norm',v)],kind,0)
            opt.step()
            self.assertTrue(torch.equal(p,torch.ones_like(p)))
            self.assertTrue(any(float(t.abs().sum())>0 for t in opt.state_tensors()))
