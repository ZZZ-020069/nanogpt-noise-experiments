import math
import unittest
from nanogpt_noise.protocol import Budget,grid,configuration,INITIAL
from nanogpt_noise.search import propose,winner,summarize


def initial(losses):
    return [dict(x=x,loss=y,status='numerical_failure' if y is None else 'completed',reason='initial')
            for x,y in zip(INITIAL,losses)]


class ProtocolTests(unittest.TestCase):
    def test_budget_grid(self):
        b=Budget()
        self.assertEqual(b.tokens,2000158720)
        self.assertEqual(b.accumulations(4,4),32)
        self.assertEqual(len(grid()),840)
        self.assertEqual(sum(b.evaluate_at(step) for step in range(b.steps+1)),21)
        self.assertEqual(b.steps,3815)
        for job in grid():
            c=configuration(job['scope'],job['optimizer'],job['node'],0)
            self.assertEqual(c['budget']['steps'],3815)
            self.assertEqual(c['design'],'v1.8')
        self.assertEqual(len({(j['scope'],j['optimizer'],j['node']) for j in grid()}),840)
        with self.assertRaises(ValueError): b.accumulations(3,4)
        self.assertEqual(b.multiplier(0),0)
        self.assertEqual(b.multiplier(100000000),1)
        self.assertEqual(b.multiplier(b.tokens),.1)
        self.assertFalse(b.evaluate_at(190))
        self.assertTrue(b.evaluate_at(191))
        self.assertTrue(b.evaluate_at(3815))
        self.assertAlmostEqual(configuration('full','muon',0,-.5)['peak_lr'],.02/math.sqrt(10))

    def test_expansion_and_feedback(self):
        r=initial([5,4,3,2,1])
        q=propose(r)
        self.assertEqual(q['x'],1.5)
        for loss in (2,None,1,1-5e-7):
            s=r+[dict(**q,loss=loss,status='numerical_failure' if loss is None else 'completed')]
            self.assertEqual(propose(s)['x'],1.25)
        r.append(dict(**q,loss=.5,status='completed'))
        self.assertEqual(propose(r)['x'],2)

    def test_lower_tie_feedback(self):
        r=initial([1,2,3,4,5])
        q=propose(r)
        r.append(dict(**q,loss=1,status='completed'))
        self.assertEqual(winner(r)['x'],-1.5)
        self.assertEqual(propose(r)['x'],-1.25)

    def test_internal_failures(self):
        r=initial([5,4,1,4,5])
        self.assertEqual(propose(r)['x'],-.25)
        r.append(dict(**propose(r),loss=2,status='completed'))
        self.assertEqual(propose(r)['x'],.25)
        failures=initial([None]*5)
        for _ in range(3):
            failures.append(dict(**propose(failures),loss=None,status='numerical_failure'))
        self.assertEqual(failures[-1]['x'],-2.5)
        self.assertIsNone(propose(failures))
        self.assertIsNone(winner(failures))
        self.assertTrue(summarize(failures)['complete'])
        with self.assertRaises(ValueError): propose([dict(status='infrastructure_failure')])

    def test_eight_unique_many_surfaces(self):
        for center in (-5,-1,-.3,0,.9,5):
            r=[]
            for _ in range(8):
                q=propose(r)
                r.append(dict(**q,status='completed',loss=(q['x']-center)**2))
            self.assertEqual(len({a['x'] for a in r}),8)
            self.assertIsNone(propose(r))
