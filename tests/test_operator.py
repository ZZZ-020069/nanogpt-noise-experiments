import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BASH = shutil.which('bash') or ('D:/Program Files/Git/bin/bash.exe'
                                if Path('D:/Program Files/Git/bin/bash.exe').exists() else None)
EXPECTED_ID = '9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e'


@unittest.skipUnless(BASH, 'Bash is required for operator integration tests')
class OperatorScriptTests(unittest.TestCase):
    @staticmethod
    def shell_path(path):
        if os.name != 'nt':
            return path.as_posix()
        result = subprocess.run(
            [BASH, '-lc', 'cygpath -u "$1"', '_', str(path)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=True)
        return result.stdout.strip()

    @contextlib.contextmanager
    def fixture(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='operator wrapper ') as directory:
            base = Path(directory)
            input_root = base / 'inputs'
            data = input_root / 'openwebtext_subset'
            artifacts = input_root / 'frozen-artifacts'
            run_base = base / 'results'
            fake_bin = base / 'bin'
            for path in (data, artifacts, run_base, fake_bin):
                path.mkdir(parents=True, exist_ok=True)
            for name in ('train.bin', 'val.bin', 'meta.json'):
                (data / name).write_bytes(b'fixture')
            for name in ('initial.pt', 'windows.pt', 'identity.json'):
                (artifacts / name).write_bytes(b'fixture')

            python_stub = base / 'python-stub.sh'
            python_stub.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "${REQUIRE_SAFE_DIRECTORY:-0}" == 1 ]]; then\n'
                '  found=0\n'
                '  for ((index=0; index<${GIT_CONFIG_COUNT:-0}; index++)); do\n'
                '    key_name="GIT_CONFIG_KEY_${index}"\n'
                '    value_name="GIT_CONFIG_VALUE_${index}"\n'
                '    if [[ "${!key_name:-}" == safe.directory && '
                '"${!value_name:-}" == "$EXPECTED_SAFE_DIRECTORY" ]]; then found=1; fi\n'
                '  done\n'
                '  (( found )) || exit 97\n'
                'fi\n'
                'printf "PY=<%s>\\n" "$@" >&2\n'
                f'if [[ "$*" == *nanogpt_noise.data* ]]; then echo "{EXPECTED_ID}"; exit 0; fi\n'
                'if [[ "$*" == *nanogpt_noise.pool* && "${FAIL_POOL:-0}" == 1 ]]; then exit 9; fi\n'
                'if [[ "$*" == *nanogpt_noise.train* ]]; then\n'
                '  output=""; take_output=0\n'
                '  for arg in "$@"; do\n'
                '    if (( take_output )); then output="$arg"; take_output=0\n'
                '    elif [[ "$arg" == --output ]]; then take_output=1\n'
                '    fi\n'
                '  done\n'
                '  mkdir -p "$output"\n'
                '  printf \'{"status":"completed","mode":"smoke","step":8}\\n\' > "$output/status.json"\n'
                'fi\n'
                'exit 0\n',
                encoding='utf-8')
            python_stub.chmod(0o755)

            git_stub = fake_bin / 'git'
            git_stub.write_text(
                '#!/usr/bin/env bash\n'
                'if [[ "${REQUIRE_SAFE_DIRECTORY:-0}" == 1 ]]; then\n'
                '  found=0\n'
                '  for ((index=0; index<${GIT_CONFIG_COUNT:-0}; index++)); do\n'
                '    key_name="GIT_CONFIG_KEY_${index}"\n'
                '    value_name="GIT_CONFIG_VALUE_${index}"\n'
                '    if [[ "${!key_name:-}" == safe.directory && '
                '"${!value_name:-}" == "$EXPECTED_SAFE_DIRECTORY" ]]; then\n'
                '      found=1\n'
                '    fi\n'
                '  done\n'
                '  if (( ! found )); then\n'
                '    echo "fatal: detected dubious ownership in repository" >&2\n'
                '    exit 128\n'
                '  fi\n'
                'fi\n'
                'if [[ "${1:-}" == rev-parse ]]; then echo test-commit; exit 0; fi\n'
                'if [[ "${1:-}" == status ]]; then exit 0; fi\n'
                'exit 0\n',
                encoding='utf-8')
            git_stub.chmod(0o755)

            run_base_shell = self.shell_path(run_base)
            env = dict(os.environ,
                       PYTHON=self.shell_path(python_stub),
                       GIT_BIN=self.shell_path(git_stub),
                       INPUT_ROOT=self.shell_path(input_root),
                       RUN_BASE=run_base_shell)
            yield root, run_base, run_base_shell, env

    def invoke(self, root, env, *args, visible='0'):
        return self.invoke_script(root, env, 'scripts/h100.sh', *args, visible=visible)

    def invoke_script(self, root, env, script, *args, visible='0'):
        command_env = dict(env, CUDA_VISIBLE_DEVICES=visible)
        return subprocess.run(
            [BASH, str(root / script), *args],
            cwd=root,
            env=command_env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace')

    @staticmethod
    def combined(result):
        return result.stdout + result.stderr

    def test_check_uses_confirmed_input_layout(self):
        with self.fixture() as (root, _run_base, run_base_shell, env):
            result = self.invoke(root, env, 'check')
            output = self.combined(result)
            self.assertEqual(result.returncode, 0, output)
            self.assertIn('PY=<nanogpt_noise.data>', output)
            self.assertIn('PY=<unittest>', output)
            self.assertIn('/openwebtext_subset', output)
            self.assertIn('/frozen-artifacts', output)
            self.assertIn(run_base_shell, output)

    def test_smoke_then_scope_and_finalize(self):
        with self.fixture() as (root, run_base, run_base_shell, env):
            smoke = self.invoke(root, env, 'smoke', visible='0')
            self.assertEqual(smoke.returncode, 0, self.combined(smoke))
            status = run_base / 'smoke/1gpu/full-muon-node10/status.json'
            marker = run_base / 'smoke/operator_commit.txt'
            self.assertTrue(status.is_file())
            self.assertEqual(marker.read_text(encoding='utf-8').strip(), 'test-commit')

            formal = self.invoke(root, env, 'scope', 'full', visible='0,1,2,3')
            output = self.combined(formal)
            self.assertEqual(formal.returncode, 0, output)
            self.assertIn('PY=<nanogpt_noise.pool>', output)
            self.assertIn(f'{run_base_shell}/full', output)

            finalize = self.invoke(root, env, 'finalize', 'full', visible='0')
            output = self.combined(finalize)
            self.assertEqual(finalize.returncode, 0, output)
            self.assertIn('PY=<nanogpt_noise.scope>', output)
            self.assertIn('PY=<finalize>', output)

    def test_scope_requires_completed_smoke(self):
        with self.fixture() as (root, _run_base, _run_base_shell, env):
            result = self.invoke(root, env, 'scope', 'full', visible='0,1,2,3')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('smoke', self.combined(result).lower())

    def test_invalid_scope_is_rejected(self):
        with self.fixture() as (root, _run_base, _run_base_shell, env):
            result = self.invoke(root, env, 'scope', 'bad', visible='0,1,2,3')
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('PY=<nanogpt_noise.pool>', self.combined(result))

    def test_help_lists_operator_commands(self):
        with self.fixture() as (root, _run_base, _run_base_shell, env):
            result = self.invoke(root, env, '--help')
            self.assertEqual(result.returncode, 0, self.combined(result))
            for command in ('check', 'smoke', 'scope', 'finalize', 'status'):
                self.assertIn(command, result.stdout)
            self.assertIn('INPUT_ROOT=/data/260010171/inputs', result.stdout)
            self.assertNotIn('260010189', result.stdout)

    def test_public_smoke_and_formal_shortcuts(self):
        with self.fixture() as (root, run_base, _run_base_shell, env):
            smoke = self.invoke_script(
                root, env, 'scripts/run_smoke.sh', visible='0')
            self.assertEqual(smoke.returncode, 0, self.combined(smoke))
            self.assertTrue(
                (run_base / 'smoke/1gpu/full-muon-node10/status.json').is_file())

            formal = self.invoke_script(
                root, env, 'scripts/run_formal.sh', visible='0,1,2,3')
            output = self.combined(formal)
            self.assertEqual(formal.returncode, 0, output)
            self.assertIn('PY=<nanogpt_noise.pool>', output)
            self.assertIn('PY=<nanogpt_noise.scope>', output)
            self.assertIn('PY=<finalize>', output)
            self.assertIn('FORMAL SCOPE FINALIZED', output)

    def test_public_smoke_handles_repository_owner_mismatch(self):
        with self.fixture() as (root, _run_base, _run_base_shell, env):
            guarded_env = dict(
                env,
                REQUIRE_SAFE_DIRECTORY='1',
                EXPECTED_SAFE_DIRECTORY=self.shell_path(root),
                GIT_CONFIG_COUNT='1',
                GIT_CONFIG_KEY_0='color.ui',
                GIT_CONFIG_VALUE_0='false')
            smoke = self.invoke_script(
                root, guarded_env, 'scripts/run_smoke.sh', visible='0')
            self.assertEqual(smoke.returncode, 0, self.combined(smoke))
            self.assertIn('SMOKE PASSED', smoke.stdout)

    def test_formal_shortcut_does_not_finalize_after_pool_failure(self):
        with self.fixture() as (root, _run_base, _run_base_shell, env):
            smoke = self.invoke_script(
                root, env, 'scripts/run_smoke.sh', visible='0')
            self.assertEqual(smoke.returncode, 0, self.combined(smoke))

            failed_env = dict(env, FAIL_POOL='1')
            formal = self.invoke_script(
                root, failed_env, 'scripts/run_formal.sh', visible='0,1,2,3')
            output = self.combined(formal)
            self.assertEqual(formal.returncode, 9, output)
            self.assertIn('PY=<nanogpt_noise.pool>', output)
            self.assertNotIn('PY=<finalize>', output)
            self.assertNotIn('FORMAL SCOPE FINALIZED', output)


if __name__ == '__main__':
    unittest.main()
