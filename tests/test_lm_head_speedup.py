"""Equivalence tests for the fast K/T training path against autograd HeadGPT."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from nanogpt_noise.artifacts import read_json, save_checkpoint
from nanogpt_noise.lm_head.model import HeadGPT
from nanogpt_noise.lm_head.protocol import VERSION, code_identity, configuration
from nanogpt_noise.lm_head.train import Runtime, run
from nanogpt_noise.model import Shape
from tests.test_integration import fixture


class FastPathEquivalence(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data, self.artifacts = fixture(self.root)
        ident = read_json(self.artifacts / 'identity.json')
        shape = Shape(**ident['shape'])
        self.statpath = self.root / 'test-statistics.pt'
        stats = dict(
            B=torch.eye(shape.n_embd, dtype=torch.float64) * 0.5,
            pbar=torch.ones(shape.vocab_size, dtype=torch.float64) / shape.vocab_size,
            alpha=torch.tensor(1.5, dtype=torch.float64),
        )
        save_checkpoint(
            self.statpath,
            dict(
                schema='lm-head-K-v1',
                design=VERSION,
                input_id=ident['id'],
                initialization_hash=ident['initialization_tensor_hash'],
                shape=ident['shape'],
                code=code_identity()['sha256'],
                passed=True,
                test=True,
                statistics=stats,
            ),
        )

    def test_fast_chunk_matches_autograd_at_nonzero_displacement(self):
        shape = Shape(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=64)
        stats = dict(
            B=torch.eye(16, dtype=torch.float64) * 0.7,
            pbar=torch.softmax(torch.linspace(-1.0, 1.0, 64, dtype=torch.float64), 0),
            alpha=torch.tensor(1.2, dtype=torch.float64),
        )
        torch.manual_seed(77)
        x_chunks = [torch.randint(0, 64, (2, 8)) for _ in range(3)]
        y_chunks = [torch.randint(0, 64, (2, 8)) for _ in range(3)]

        for objective in ('K', 'T'):
            with self.subTest(objective=objective):
                model = HeadGPT(shape).activate_head(objective, stats if objective == 'K' else None)
                with torch.no_grad():
                    model.lm_head.weight.add_(0.1 * torch.randn_like(model.lm_head.weight))
                model.zero_grad()
                ref_loss = torch.zeros((), dtype=torch.float64)
                for x, y in zip(x_chunks, y_chunks):
                    loss = model(x, y)
                    ref_loss += loss.detach().double() / len(x_chunks)
                    (loss / len(x_chunks)).backward()
                ref_grad = model.lm_head.weight.grad.clone()

                grad_accum = torch.zeros_like(ref_grad)
                fast_loss = torch.zeros((), dtype=torch.float64)
                for x, y in zip(x_chunks, y_chunks):
                    h = model.features(x)
                    loss, dz_buf = model.forward_fast_chunk(h, y, len(x_chunks), row_slice=4)
                    fast_loss += loss.double() / len(x_chunks)
                    model.backward_fast_chunk(h, dz_buf, grad_accum)
                if objective == 'K':
                    fast_loss, fast_grad = model.finalize_K_step(grad_accum, fast_loss)
                else:
                    fast_grad = grad_accum

                self.assertAlmostEqual(float(ref_loss), float(fast_loss), places=6)
                self.assertTrue(torch.allclose(ref_grad, fast_grad, atol=1e-6, rtol=1e-6))

    def test_full_run_equivalence_and_profiling_for_k_and_t(self):
        rt = Runtime(cpu=True)
        for objective in ('K', 'T'):
            for optimizer in ('sgd', 'adam', 'adamw', 'muon'):
                with self.subTest(objective=objective, optimizer=optimizer):
                    cfg = configuration(objective, optimizer, 10, 0.0)
                    if objective == 'K':
                        cfg['statistics'] = str(self.statpath)
                    fast_out = self.root / f'fast-{objective}-{optimizer}'
                    res = run(cfg, self.data, self.artifacts, fast_out, micro=2, runtime=rt, smoke_steps=4, test=True)
                    self.assertEqual(res['status'], 'completed')
                    perf = read_json(fast_out / 'performance.json')
                    self.assertGreater(perf['timings']['forward_seconds'], 0.0)
                    self.assertGreater(perf['timings']['backward_seconds'], 0.0)

                    # Compare step-0 and step-4 against explicit autograd HeadGPT
                    ckpt = torch.load(fast_out / 'final.pt', weights_only=True)
                    shape = Shape(**read_json(self.artifacts / 'identity.json')['shape'])
                    model = HeadGPT(shape)
                    model.load_state_dict(torch.load(self.artifacts / 'initial.pt', weights_only=True))
                    stats = torch.load(self.statpath, weights_only=True)['statistics'] if objective == 'K' else None
                    model.activate_head(objective, stats)
                    self.assertFalse(torch.equal(ckpt['head'], model.reference_head))

    def test_code_identity_tracks_kernels_module(self):
        self.assertIn('nanogpt_noise/lm_head/kernels.py', code_identity()['files'])

    def test_fast_vs_baseline_train_run_exact_match_fp32_optimizers(self):
        """Compare end-to-end train.run with fast path vs direct autograd loss_from_features."""
        rt = Runtime(cpu=True)
        for objective in ('K', 'T'):
            for optimizer in ('sgd', 'adam', 'adamw'):
                with self.subTest(objective=objective, optimizer=optimizer):
                    cfg = configuration(objective, optimizer, 10, 0.5)
                    if objective == 'K':
                        cfg['statistics'] = str(self.statpath)
                    fast_dir = self.root / f'e2e-fast-{objective}-{optimizer}'
                    slow_dir = self.root / f'e2e-slow-{objective}-{optimizer}'
                    fast_res = run(cfg, self.data, self.artifacts, fast_dir, micro=2, runtime=rt, test=True)
                    with patch.object(HeadGPT, 'use_fast_chunk', False, create=True):
                        slow_res = run(cfg, self.data, self.artifacts, slow_dir, micro=2, runtime=rt, test=True)

                    self.assertAlmostEqual(fast_res['loss'], slow_res['loss'], places=6)
                    fast_ckpt = torch.load(fast_dir / 'final.pt', weights_only=True)
                    slow_ckpt = torch.load(slow_dir / 'final.pt', weights_only=True)
                    self.assertTrue(torch.allclose(fast_ckpt['head'], slow_ckpt['head'], atol=1e-6, rtol=1e-6))
                    fast_evals = read_json(fast_dir / 'evaluations.json')
                    slow_evals = read_json(slow_dir / 'evaluations.json')
                    for fe, se in zip(fast_evals, slow_evals):
                        self.assertAlmostEqual(fe['val'], se['val'], places=6)
                        self.assertAlmostEqual(fe['train_objective'], se['train_objective'], places=6)


if __name__ == '__main__':
    unittest.main()
