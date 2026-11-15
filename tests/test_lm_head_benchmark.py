from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
import torch
from nanogpt_noise.artifacts import save_checkpoint
from nanogpt_noise.model import Shape
from nanogpt_noise.lm_head.protocol import Budget, VERSION, code_identity
from nanogpt_noise.lm_head.benchmark import require_benchmark_input,load_benchmark_statistics
from nanogpt_noise.lm_head.measure import load_statistics
from nanogpt_noise.lm_head.train import run


class BenchmarkTests(unittest.TestCase):
    def test_legacy_full_input_allowed_but_shape_and_budget_checked(self):
        ident=dict(tiny=False,shape=asdict(Shape()),budget={**asdict(Budget()),'steps':4960},design='v1.4')
        require_benchmark_input(ident)
        ident['budget']['batch']=256
        with self.assertRaises(ValueError): require_benchmark_input(ident)

    def test_benchmark_cannot_run_formal_or_cpu_fixture(self):
        for kwargs in (dict(benchmark=True),dict(benchmark=True,smoke_steps=3815),
                       dict(benchmark=True,smoke_steps=32,test=True)):
            with self.assertRaisesRegex(ValueError,'exactly 32'):
                run({},'unused','unused','unused',**kwargs)

    def test_empirical_statistics_rejected_by_formal_loader(self):
        shape=Shape(n_embd=4,vocab_size=8)
        ident=dict(id='legacy',initialization_tensor_hash='init')
        value=dict(schema='lm-head-K-benchmark-v1',design=VERSION,benchmark_only=True,passed=False,
                   input_id='legacy',initialization_hash='init',code=code_identity()['sha256'],
                   shape=asdict(shape),statistics=dict(B=torch.eye(4),pbar=torch.ones(8)/8,alpha=torch.tensor(1.)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'K-benchmark.pt';save_checkpoint(path,value)
            load_benchmark_statistics(path,ident,shape)
            with self.assertRaisesRegex(ValueError,'must pass'):
                load_statistics(path,ident,shape)
            value['passed']=True;save_checkpoint(path,value)
            with self.assertRaises(ValueError):load_benchmark_statistics(path,ident,shape)


if __name__=='__main__':unittest.main()
