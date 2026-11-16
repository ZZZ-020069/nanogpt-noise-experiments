"""H100 fixed-layout command routing, without GPU or training."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests.test_lm_head_shell import BASH


@unittest.skipUnless(BASH,'Bash required')
class H100Launch(unittest.TestCase):
    def invoke(self,action,**overrides):
        with tempfile.TemporaryDirectory(prefix='h100 launch ') as directory:
            stub=Path(directory)/'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "CALL\\n"\n'
                            'printf "ARG=<%s>\\n" "$@"\n'
                            'printf "CUDA=<%s>\\n" "$CUDA_VISIBLE_DEVICES"\n'
                            'printf "SWAN=<%s> PROJECT=<%s> CACHE=<%s>\\n" "$NANOGPT_SWANLAB" "$SWANLAB_PROJ_NAME" "$RUN_BASE"\n'
                            'if [[ "${FAIL_PREFLIGHT:-0}" == 1 && "$*" == *"lm_head.telemetry"* ]]; then exit 19; fi\n'
                            'if [[ "${FAIL_T:-0}" == 1 && "$*" == *"--objective T"* ]]; then exit 17; fi\n',
                            encoding='utf8')
            stub.chmod(0o755)
            env={**os.environ,'PYTHON':stub.as_posix(),'INPUT_ROOT':'/formal inputs',
                 'LH_RUN_ROOT':'/new h100 output','GPUS':'','MICRO_BATCH':'','OBJECTIVE':'G',
                 'K_STATISTICS':'','LH_PHASE':'','LH_CURVATURE_DIR':'/new curvature',
                 'NANOGPT_SWANLAB':'0',**overrides}
            return subprocess.run([BASH,'scripts/run_lm_head_h100.sh',action],env=env,
                      cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,encoding='utf8')

    def test_fixed_micro16_formal_alias_and_default_gpu_pool(self):
        result=self.invoke('formal')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ARG=<--micro-batch>\nARG=<16>',result.stdout)
        self.assertIn('ARG=<--phase>\nARG=<clean>',result.stdout)
        self.assertIn('ARG=<--devices>\nARG=<0,1,2,3>',result.stdout)
        self.assertNotIn('ARG=<--statistics>',result.stdout)

    def test_stale_micro32_is_rejected_before_dispatch(self):
        for action in ('smoke','formal','grid','resume','curvature-fit'):
            with self.subTest(action=action):
                result=self.invoke(action,MICRO_BATCH='32')
                self.assertEqual(result.returncode,2)
                self.assertNotIn('CALL',result.stdout)

    def test_objective_smoke_uses_first_gpu_and_requires_k_only_for_k(self):
        for objective in ('G','T'):
            result=self.invoke('smoke',OBJECTIVE=objective,GPUS='2,3')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('CUDA=<2>',result.stdout)
            self.assertIn(f'ARG=<--objective>\nARG=<{objective}>',result.stdout)
            self.assertNotIn('ARG=<run>',result.stdout)
        self.assertNotEqual(self.invoke('smoke',OBJECTIVE='K').returncode,0)
        result=self.invoke('smoke',OBJECTIVE='K',K_STATISTICS='/formal K-approved.pt')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ARG=</formal K-approved.pt>',result.stdout)

    def test_smoke_all_stops_on_failure_and_never_starts_formal(self):
        missing=self.invoke('smoke-all')
        self.assertNotEqual(missing.returncode,0)
        self.assertNotIn('CALL',missing.stdout)
        failed=self.invoke('smoke-all',K_STATISTICS='/approved.pt',FAIL_T='1')
        self.assertEqual(failed.returncode,17)
        self.assertEqual(failed.stdout.count('CALL'),2)
        self.assertNotIn('ARG=<K>',failed.stdout)
        passed=self.invoke('smoke-all',K_STATISTICS='/approved.pt')
        self.assertEqual(passed.returncode,0,passed.stderr)
        self.assertEqual(passed.stdout.count('CALL'),3)
        self.assertNotIn('ARG=<run>',passed.stdout)

    def test_clean_all_and_resume_chain_gtk_and_guard_prerequisites(self):
        for action in ('clean-all', 'formal-all'):
            missing = self.invoke(action)
            self.assertNotEqual(missing.returncode, 0)
            self.assertNotIn('CALL', missing.stdout)
        with tempfile.TemporaryDirectory(prefix='h100 smoke guard ') as run_root:
            (Path(run_root) / 'smoke' / 'G').mkdir(parents=True)
            (Path(run_root) / 'smoke' / 'G' / 'passed.json').write_text('{}', encoding='utf8')
            incomplete = self.invoke('clean-all', LH_RUN_ROOT=run_root, K_STATISTICS='/approved.pt')
            self.assertEqual(incomplete.returncode, 2)
            self.assertIn('Missing T smoke proof', incomplete.stderr)
            self.assertNotIn('CALL', incomplete.stdout)
            for objective in ('T', 'K'):
                (Path(run_root) / 'smoke' / objective).mkdir(parents=True)
                (Path(run_root) / 'smoke' / objective / 'passed.json').write_text('{}', encoding='utf8')
            failed = self.invoke('clean-all', LH_RUN_ROOT=run_root, K_STATISTICS='/approved.pt', FAIL_T='1')
            self.assertEqual(failed.returncode, 17)
            self.assertEqual(failed.stdout.count('CALL'), 2)
            self.assertNotIn('ARG=<K>', failed.stdout)
            passed = self.invoke('formal-all', LH_RUN_ROOT=run_root, K_STATISTICS='/approved.pt')
            self.assertEqual(passed.returncode, 0, passed.stderr)
            self.assertEqual(passed.stdout.count('CALL'), 3)
            self.assertEqual(passed.stdout.count('ARG=<--phase>\nARG=<clean>'), 3)
            self.assertIn('ARG=<--objective>\nARG=<G>', passed.stdout)
            self.assertIn('ARG=<--objective>\nARG=<T>', passed.stdout)
            self.assertIn('ARG=<--objective>\nARG=<K>', passed.stdout)
            self.assertNotIn('ARG=<--phase>\nARG=<grid>', passed.stdout)
            resumed = self.invoke('resume', LH_RUN_ROOT=run_root, LH_PHASE='clean-all', K_STATISTICS='/approved.pt')
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(resumed.stdout.count('CALL'), 3)
            self.assertEqual(resumed.stdout.count('ARG=<--resume>'), 3)
            self.assertEqual(resumed.stdout.count('ARG=<--phase>\nARG=<clean>'), 3)
        self.assertNotEqual(self.invoke('resume', LH_PHASE='clean-all').returncode, 0)

    def test_resume_and_readonly_routes(self):
        self.assertNotEqual(self.invoke('resume').returncode,0)
        result=self.invoke('resume',LH_PHASE='grid',OBJECTIVE='T')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ARG=<--resume>',result.stdout)
        self.assertIn('ARG=<--phase>\nARG=<grid>',result.stdout)
        status=self.invoke('status',INPUT_ROOT='',OBJECTIVE='K')
        self.assertEqual(status.returncode,0,status.stderr)

    def test_curvature_ignores_stale_k_objective_and_keeps_micro16(self):
        result=self.invoke('curvature-fit',OBJECTIVE='K')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('ARG=<curvature>',result.stdout)
        self.assertIn('ARG=<--micro-batch>\nARG=<16>',result.stdout)

    def test_swanlab_default_on_preflight_and_explicit_off(self):
        missing=self.invoke('smoke',NANOGPT_SWANLAB='',SWANLAB_API_KEY='')
        self.assertNotEqual(missing.returncode,0)
        self.assertIn('SWANLAB_API_KEY',missing.stderr)
        self.assertNotIn('CALL',missing.stdout)
        for action in ('check','smoke','formal','grid','resume'):
            result=self.invoke(action,NANOGPT_SWANLAB='',SWANLAB_API_KEY='fake-key',
                               SWANLAB_PROJ_NAME='',LH_PHASE='clean')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.count('CALL'),2)
            self.assertIn('ARG=<nanogpt_noise.lm_head.telemetry>',result.stdout)
            self.assertIn('SWAN=<1> PROJECT=<nanogpt-lm-head> CACHE=</new h100 output>',result.stdout)
            self.assertNotIn('fake-key',result.stdout+result.stderr)
        failed=self.invoke('formal',NANOGPT_SWANLAB='1',SWANLAB_API_KEY='fake-key',FAIL_PREFLIGHT='1')
        self.assertEqual(failed.returncode,19)
        self.assertEqual(failed.stdout.count('CALL'),1)
        self.assertEqual(self.invoke('smoke',NANOGPT_SWANLAB='0',SWANLAB_API_KEY='').returncode,0)
        self.assertNotEqual(self.invoke('smoke',NANOGPT_SWANLAB='bad').returncode,0)

    def test_k_statistics_env_sync_and_drift_protection(self):
        from nanogpt_noise.lm_head.measure import sync_k_statistics_env

        with tempfile.TemporaryDirectory(prefix='h100 env sync ') as directory:
            root = Path(directory)
            env_file = root / 'test.env'
            stats = root / 'init-check-64k' / 'K-approved.pt'
            stats.parent.mkdir()
            stats.write_bytes(b'approved')
            other = root / 'init-check-32k' / 'K-approved.pt'
            other.parent.mkdir()
            other.write_bytes(b'other')
            init_report = dict(passed=True, purpose='initial_proxy_acceptance')

            env_file.write_text('# header\nexport MICRO_BATCH=16\n# export K_STATISTICS=/old/path.pt\n', encoding='utf8')
            # Must ignore late drift checks and failed reports.
            self.assertEqual(sync_k_statistics_env(env_file, stats, checkpoint='/head.pt', report=init_report), 'skipped')
            self.assertEqual(sync_k_statistics_env(env_file, stats, report=dict(passed=False, purpose='initial_proxy_acceptance')), 'skipped')
            self.assertEqual(sync_k_statistics_env(env_file, stats, report=dict(passed=True, purpose='trajectory_drift')), 'skipped')
            self.assertIn('# export K_STATISTICS=/old/path.pt', env_file.read_text(encoding='utf8'))

            # Initial passing check uncomments and populates K_STATISTICS automatically.
            self.assertEqual(sync_k_statistics_env(env_file, stats, report=init_report), 'updated')
            text = env_file.read_text(encoding='utf8')
            self.assertIn(f'export K_STATISTICS={stats.resolve()}', text.replace("'", ''))
            self.assertNotIn('# export K_STATISTICS=', text)
            self.assertEqual(sync_k_statistics_env(env_file, stats, report=init_report), 'unchanged')

            # Existing active K_STATISTICS is preserved unless force=True.
            self.assertEqual(sync_k_statistics_env(env_file, other, report=init_report), 'preserved')
            self.assertIn(stats.name, env_file.read_text(encoding='utf8'))
            self.assertNotIn(other.parent.name, env_file.read_text(encoding='utf8'))
            self.assertEqual(sync_k_statistics_env(env_file, other, report=init_report, force=True), 'updated')
            self.assertIn(other.parent.name, env_file.read_text(encoding='utf8'))

            # Sourcing scripts/lm_head_h100.env.example exports LH_ENV_FILE automatically.
            copied_env = root / 'copied.env'
            copied_env.write_text(Path('scripts/lm_head_h100.env.example').read_text(encoding='utf8'), encoding='utf8')
            sourced = subprocess.run(
                [BASH, '-c', f'source "{copied_env.as_posix()}" && printf "%s" "$LH_ENV_FILE"'],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, encoding='utf8',
            )
            self.assertEqual(sourced.returncode, 0, sourced.stderr)
            self.assertTrue(sourced.stdout.endswith('copied.env'), sourced.stdout)

    def test_readonly_and_curvature_need_no_swanlab_credentials(self):
        for action in ('plan','status','report','pause','curvature-fit'):
            result=self.invoke(action,NANOGPT_SWANLAB='1',SWANLAB_API_KEY='')
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertNotIn('ARG=<nanogpt_noise.lm_head.telemetry>',result.stdout)


if __name__=='__main__':unittest.main()
