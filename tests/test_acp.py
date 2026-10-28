"""Shell routing tests; stubbed CUDA/Conda is NOT an H100 acceptance test."""
import os
from pathlib import Path
import unittest
import test_operator

BASH = test_operator.BASH


@unittest.skipUnless(BASH, 'Bash is required')
class AcpTests(unittest.TestCase):
    def setUp(self):
        self.helper = test_operator.OperatorScriptTests()
        self.context = self.helper.fixture()
        self.root, self.results, _, self.env = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)
        self.base = self.results.parent

    def invoke(self, *args):
        return self.helper.invoke_script(self.root, self.env, 'scripts/run_acp.sh', *args)

    def test_explicit_runtime_smoke_and_safe_directory(self):
        self.env.update(REQUIRE_SAFE_DIRECTORY='1',
                        EXPECTED_SAFE_DIRECTORY=self.helper.shell_path(self.root))
        result = self.invoke('smoke')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('ACP ENVIRONMENT READY', result.stdout)
        self.assertIn('SMOKE PASSED', result.stdout)
        self.assertIn('PY=<--no-deps>', result.stderr)

    def test_prepare_does_not_train(self):
        result = self.invoke('prepare')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('PY=<nanogpt_noise.train>', result.stderr)

    def test_inputs_installs_preparation_dependencies_and_registers_once(self):
        input_root = self.helper.shell_path(self.base/'generated inputs')
        self.env['INPUT_ROOT'] = input_root
        result = self.invoke('inputs')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PY=<requirements-prepare.txt>',result.stderr)
        self.assertIn('PY=<nanogpt_noise.prepare>',result.stderr)
        self.assertIn('PY=<--input-root>',result.stderr)
        self.assertIn('INPUT BUNDLE REGISTERED',result.stdout)
        self.assertNotIn('PY=<nanogpt_noise.train>',result.stderr)

    def test_invalid_explicit_runtime_stops(self):
        stub = self.base / 'bad-python'
        stub.write_text('#!/usr/bin/env bash\nexit 3\n', encoding='utf8')
        stub.chmod(0o755)
        self.env['PYTHON'] = self.helper.shell_path(stub)
        result = self.invoke('smoke')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('explicit PYTHON', result.stderr)
        self.assertNotIn('ACP ENVIRONMENT READY', result.stdout)

    def test_removed_migration_entrypoint_fails_before_install_or_training(self):
        result = self.invoke('migrate-inputs')
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('PY=<pip>', output)
        self.assertNotIn('PY=<nanogpt_noise.train>', output)

    def test_install_failure_never_starts_training(self):
        stub = self.base / 'pip-failure'
        stub.write_text('#!/usr/bin/env bash\n'
                        'if [[ "$*" == *pip* ]]; then exit 19; fi\nexit 0\n', encoding='utf8')
        stub.chmod(0o755)
        self.env['PYTHON'] = self.helper.shell_path(stub)
        result = self.invoke('smoke')
        self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
        self.assertNotIn('ACP ENVIRONMENT READY', result.stdout)
        self.assertFalse((self.results / 'smoke').exists())

    def test_old_default_python_creates_and_reuses_environment(self):
        fake_bin = self.base / 'bin'
        old = fake_bin / 'python'
        old.write_text('#!/usr/bin/env bash\necho "Python 3.9 rejected" >&2\nexit 1\n', encoding='utf8')
        old.chmod(0o755)
        conda = fake_bin / 'conda'
        conda.write_text('#!/usr/bin/env bash\nset -eu\n'
                         'echo CONDA-CREATE\n'
                         'while [[ "$1" != --prefix ]]; do shift; done\n'
                         'mkdir -p "$2/bin"\ncp "$TEST_RUNTIME" "$2/bin/python"\n'
                         'chmod +x "$2/bin/python"\n', encoding='utf8')
        conda.chmod(0o755)
        self.env.update(TEST_RUNTIME=self.env.pop('PYTHON'),
                        CONDA_BIN=self.helper.shell_path(conda),
                        ENV_PREFIX=self.helper.shell_path(self.base / 'isolated env'),
                        PATH=str(fake_bin) + os.pathsep + self.env['PATH'])
        first = self.invoke('prepare')
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertIn('CONDA-CREATE', first.stdout)
        self.assertIn('PY=<torch==2.8.0>', first.stderr)
        second = self.invoke('prepare')
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertNotIn('CONDA-CREATE', second.stdout)
        self.assertNotIn('PY=<torch==2.8.0>', second.stderr)

    def test_invalid_scope_stops_before_install(self):
        result = self.invoke('formal', 'invalid')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('PY=<pip>', result.stderr)
