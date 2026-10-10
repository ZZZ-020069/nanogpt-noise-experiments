import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from nanogpt_noise import preflight
from nanogpt_noise import train


class PreflightTests(unittest.TestCase):
    def test_four_gpu_layout(self):
        with patch.multiple(preflight.torch.cuda, is_available=lambda: True,
                            device_count=lambda: 4, device=MagicMock(),
                            is_bf16_supported=lambda: True,
                            get_device_name=lambda i: 'H100'), \
             patch.object(preflight.torch.distributed, 'is_nccl_available', return_value=True):
            self.assertEqual(preflight.check_hardware(4)['accumulation'], 32)
            with self.assertRaises(ValueError):
                preflight.check_hardware(3)
            with patch.object(preflight.torch.cuda, 'device_count', return_value=2):
                with self.assertRaisesRegex(RuntimeError, 'four'):
                    preflight.check_hardware(4)
            with patch.object(preflight.torch.cuda, 'is_bf16_supported', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'BF16'):
                    preflight.check_hardware(4)
            with patch.object(preflight.torch.distributed, 'is_nccl_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'NCCL'):
                    preflight.check_hardware(4)

    def test_cpu_rejected(self):
        with patch.object(preflight.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                preflight.check_hardware(4)

    def test_artifact_validation(self):
        with patch.object(preflight, 'verify', return_value={'tiny': True}):
            with self.assertRaisesRegex(ValueError, 'tiny'):
                preflight.check_inputs('data', 'artifacts')
        identity = dict(tiny=False, budget=train.asdict(train.Budget()),
                        shape=train.asdict(train.Shape()), id='example')
        with patch.object(preflight, 'verify', return_value=identity):
            self.assertEqual(preflight.check_inputs('data', 'artifacts'), 'example')
        identity['budget']['batch'] = 8
        with patch.object(preflight, 'verify', return_value=identity):
            with self.assertRaisesRegex(ValueError, 'protocol'):
                preflight.check_inputs('data', 'artifacts')

    def test_smoke_failure_exit(self):
        argv = ['train', '--data', 'd', '--artifacts', 'a', '--output', 'o']
        for smoke, status, expected in ((True, 'numerical_failure', 2),
                                        (True, 'infrastructure_failure', 75),
                                        (True, 'completed', None),
                                        (False, 'numerical_failure', None)):
            with self.subTest(smoke=smoke, status=status), \
                 patch.object(sys, 'argv', argv + (['--smoke-steps', '8'] if smoke else [])), \
                 patch.object(train, 'Runtime', return_value=MagicMock(rank=0)), \
                 patch.object(train, 'run', return_value=dict(status=status)), \
                 patch.object(train.dist, 'is_initialized', return_value=False):
                if expected is None:
                    train.main()
                else:
                    with self.assertRaises(SystemExit) as caught:
                        train.main()
                    self.assertEqual(caught.exception.code, expected)


BASH = shutil.which('bash') or ('D:/Program Files/Git/bin/bash.exe'
                                if Path('D:/Program Files/Git/bin/bash.exe').exists() else None)


@unittest.skipUnless(BASH, 'Bash is required for launcher integration tests')
class ShellTests(unittest.TestCase):
    def invoke(self, *args, fail_preflight=False):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='launcher paths ') as directory:
            stub = Path(directory)/'python-stub.sh'
            # Test fixture, not a training implementation or CUDA simulation.
            stub.write_text('#!/usr/bin/env bash\n'
                            'printf "ARG=<%s>\\n" "$@"\n'
                            'if [[ "$*" == *nanogpt_noise.preflight* ]]; then\n'
                            '  exit "${FAIL_PREFLIGHT:-0}"\nfi\n', encoding='utf-8')
            stub.chmod(0o755)
            env = dict(os.environ, PYTHON=stub.as_posix(), DATA_DIR='/data path/tokens',
                       ARTIFACT_DIR='/artifact path/frozen', RUN_DIR='/output path/runs',
                       FAIL_PREFLIGHT='9' if fail_preflight else '0')
            return subprocess.run([BASH, str(root/'scripts/run_4gpu.sh'), *args],
                                  env=env, capture_output=True, text=True)

    def test_smoke_dispatch(self):
        result = self.invoke('smoke')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ARG=<--nproc-per-node=4>', result.stdout)
        self.assertIn('ARG=<nanogpt_noise.train>', result.stdout)
        self.assertIn('ARG=<--smoke-steps>\nARG=<8>', result.stdout)
        self.assertIn('ARG=</data path/tokens>', result.stdout)
        self.assertIn('/smoke/4gpu/full-muon-node10', result.stdout)
        self.assertNotIn('ARG=<nanogpt_noise.node>', result.stdout)

    def test_formal_dispatch(self):
        result = self.invoke('node', '839')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ARG=<nanogpt_noise.node>', result.stdout)
        self.assertIn('ARG=<--job-index>\nARG=<839>', result.stdout)
        self.assertNotIn('ARG=<--smoke-steps>', result.stdout)

    def test_invalid_and_preflight_failure(self):
        for args in ((), ('node',), ('node', '840'), ('node', '-1'),
                     ('node', 'xyz'), ('node', '0', 'extra'), ('unknown',)):
            result = self.invoke(*args)
            self.assertNotEqual(result.returncode, 0, args)
            self.assertNotIn('torch.distributed.run', result.stdout)
        result = self.invoke('node', '0', fail_preflight=True)
        self.assertEqual(result.returncode, 9)
        self.assertNotIn('torch.distributed.run', result.stdout)


if __name__ == '__main__':
    unittest.main()
