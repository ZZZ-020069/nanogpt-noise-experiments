"""Launch routing using an echo stub; never starts training."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BASH = os.environ.get('BASH_EXE') or (
    'D:/Program Files/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash'))


@unittest.skipUnless(BASH,'Bash required')
class HeadShell(unittest.TestCase):
    def invoke(self,action,**overrides):
        with tempfile.TemporaryDirectory(prefix='lm head shell ') as directory:
            stub=Path(directory)/'python-stub.sh'
            stub.write_text('#!/usr/bin/env bash\nprintf "ARG=<%s>\\n" "$@"\n'
                            'printf "CUDA=<%s>\\n" "$CUDA_VISIBLE_DEVICES"\n',encoding='utf-8')
            stub.chmod(0o755)
            env={**os.environ,'PYTHON':stub.as_posix(),'INPUT_ROOT':'/inputs with spaces',
                 'LH_RUN_ROOT':'/new outputs','OBJECTIVE':'G','GPUS':'4,6',
                 'MICRO_BATCH':'','LH_CURVATURE_DIR':'/curvature output',**overrides}
            return subprocess.run([BASH,'scripts/run_lm_head.sh',action],env=env,
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,encoding='utf-8')

    def test_clean_smoke_and_readonly_plan(self):
        clean=self.invoke('clean')
        self.assertEqual(clean.returncode,0,clean.stderr)
        self.assertIn('ARG=<--phase>\nARG=<clean>',clean.stdout)
        self.assertIn('ARG=</inputs with spaces>',clean.stdout)
        smoke=self.invoke('smoke')
        self.assertEqual(smoke.returncode,0,smoke.stderr)
        self.assertIn('CUDA=<4>',smoke.stdout)
        self.assertIn('ARG=<--dry-run>',self.invoke('plan').stdout)

    def test_pause_needs_no_input_and_resume_requires_phase(self):
        paused=self.invoke('pause',INPUT_ROOT='',OBJECTIVE='K',K_STATISTICS='')
        self.assertEqual(paused.returncode,0,paused.stderr)
        self.assertIn('ARG=<pause>',paused.stdout)
        self.assertNotEqual(self.invoke('resume',LH_PHASE='').returncode,0)
        resumed=self.invoke('resume',LH_PHASE='grid')
        self.assertEqual(resumed.returncode,0,resumed.stderr)
        self.assertIn('ARG=<--resume>',resumed.stdout)

    def test_k_requires_statistics(self):
        self.assertNotEqual(self.invoke('clean',OBJECTIVE='K',K_STATISTICS='').returncode,0)
        valid=self.invoke('clean',OBJECTIVE='K',K_STATISTICS='/fixed K.pt')
        self.assertEqual(valid.returncode,0,valid.stderr)
        self.assertIn('ARG=<--statistics>\nARG=</fixed K.pt>',valid.stdout)

    def test_curvature_reference_blocks_and_fit_source(self):
        initial=self.invoke('curvature-fit')
        self.assertEqual(initial.returncode,0,initial.stderr)
        self.assertIn('ARG=<--blocks>\nARG=<5>',initial.stdout)
        later=self.invoke('curvature-check',LH_CHECKPOINT='/head.pt',LH_FIT_DIR='/fit')
        self.assertEqual(later.returncode,0,later.stderr)
        self.assertIn('ARG=<--blocks>\nARG=<2>',later.stdout)
        self.assertIn('ARG=<--fit-dir>\nARG=</fit>',later.stdout)

    def test_micro_default_and_explicit_32_across_launch_actions(self):
        for action,extra in (
                ('check',{}),('smoke',{}),('clean',{}),('grid',{}),('plan',{}),
                ('resume',{'LH_PHASE':'clean'}),('curvature-fit',{}),
                ('curvature-check',{'LH_FIT_DIR':'/fit'})):
            for value,expected in (('',16),('32',32)):
                with self.subTest(action=action,micro=expected):
                    result=self.invoke(action,MICRO_BATCH=value,**extra)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn(f'ARG=<--micro-batch>\nARG=<{expected}>',result.stdout)
