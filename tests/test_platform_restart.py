"""Real Linux process/lock recovery. Windows cannot certify POSIX signals."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from nanogpt_noise.artifacts import lock,write_json,cleanup_atomic_temps


@unittest.skipUnless(os.name=='posix','requires Linux/POSIX kernel locks and signals')
class KernelLeaseTests(unittest.TestCase):
    def test_live_writer_is_exclusive_and_inode_survives_release(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with lock(root):
                inode=(root/'.lease').stat().st_ino
                with self.assertRaisesRegex(RuntimeError,'another writer'):
                    with lock(root): pass
            self.assertFalse((root/'.lock').exists())
            with lock(root): self.assertEqual((root/'.lease').stat().st_ino,inode)

    def test_legacy_or_malformed_markers_are_not_deleted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for value in ('old-container pid=42\n','{broken','{}'):
                (root/'.lock').write_text(value)
                with self.assertRaisesRegex(RuntimeError,'legacy or unknown'):
                    with lock(root): pass
                self.assertEqual((root/'.lock').read_text(),value)

    def test_abandoned_marker_recovered_even_with_different_host(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_json(root/'.lock',dict(format='nanogpt-flock-v1',host='old-container',pid=1))
            with lock(root):
                self.assertNotEqual(json.loads((root/'.lock').read_text())['host'],'old-container')

    def test_unsupported_filesystem_refuses_fallback(self):
        import errno
        with tempfile.TemporaryDirectory() as temp, \
             patch('fcntl.flock',side_effect=OSError(errno.ENOTSUP,'not supported')):
            with self.assertRaisesRegex(RuntimeError,'reliable flock'):
                with lock(temp): pass

    def test_only_recognized_atomic_temps_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ('latest.pt.abcdefgh.tmp','status.json.12345678.tmp','notes.tmp',
                         'user-data.abcdefgh.tmp','latest.pt'):
                (root/name).write_text('keep or orphan')
            with lock(root):
                self.assertFalse((root/'latest.pt.abcdefgh.tmp').exists())
                self.assertFalse((root/'status.json.12345678.tmp').exists())
                self.assertTrue((root/'notes.tmp').exists())
                self.assertTrue((root/'user-data.abcdefgh.tmp').exists())
                self.assertTrue((root/'latest.pt').exists())


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'requires Linux/POSIX Bash')
class PlatformRestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='platform restart ')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        repo=Path(__file__).resolve().parents[1]
        scripts=self.root/'scripts';scripts.mkdir()
        for name in ('run_formal.sh','process_control.sh'):
            shutil.copyfile(repo/'scripts'/name,scripts/name)
        # Use the actual formal Bash supervisor and actual tiny trainer. Only
        # replace H100 preflight/dispatch, which cannot run on CPU CI machines.
        (scripts/'h100.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'if [[ "$1" == finalize ]]; then touch "$CASE_ROOT/finalized"; exit; fi\n'
            'exec "$PYTHON" -m tests.platform_worker --root "$CASE_ROOT"\n')
        self.env=dict(os.environ,CASE_ROOT=str(self.root),PYTHON=sys.executable,
                      RUN_BASE=str(self.root/'results'),OMP_NUM_THREADS='1',PYTHONPATH=str(repo))
        self.command=['bash',str(scripts/'run_formal.sh'),'last1']
        self.processes=[]
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for process in self.processes:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=10)

    def logs(self):
        return "\n".join(p.read_text() for p in self.root.glob("log-*.txt"))

    def start(self):
        log=(self.root/f'log-{len(self.processes)}.txt').open('w')
        self.addCleanup(log.close)
        process=subprocess.Popen(self.command,env=self.env,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True)
        self.processes.append(process)
        return process

    def active(self,process,step=0):
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            marker=self.root/'active'
            if marker.exists() and marker.read_text().strip()==str(step): return
            if process.poll() is not None: break
            time.sleep(.02)
        self.fail('CPU worker did not reach requested training step: '+
                  '\n'.join(p.read_text() for p in self.root.glob('log-*.txt')))

    def compare_reference(self):
        import torch
        completed=subprocess.run([sys.executable,'-m','tests.platform_worker','--root',str(self.root),
                                  '--reference'],env=self.env,capture_output=True,text=True,timeout=40)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        actual=torch.load(self.root/'scope/job_0140/candidate_00/final.pt',weights_only=True)
        expected=torch.load(self.root/'reference/final.pt',weights_only=True)
        def equal(a,b):
            if isinstance(a,torch.Tensor): self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a: equal(a[key],b[key])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for left,right in zip(a,b): equal(left,right)
            else: self.assertEqual(a,b)
        for key in ('model','optimizer','step','tokens','evaluations'): equal(actual[key],expected[key])

    def test_platform_term_reaches_trainer_saves_and_same_command_resumes(self):
        import torch
        process=self.start();self.active(process)
        # Signal only the outer Bash PID, as a platform entrypoint may receive it.
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30),75,self.logs())
        self.assertFalse((self.root/'finalized').exists())
        state=torch.load(self.root/'scope/job_0140/candidate_00/latest.pt',weights_only=True)
        self.assertGreaterEqual(state['step'],1)
        self.assertLess(state['step'],4)
        self.assertEqual(list(self.root.rglob('.lock')),[])
        self.assertEqual(self.start().wait(timeout=40),0,self.logs())
        self.assertTrue((self.root/'finalized').exists())
        self.compare_reference()

    def test_platform_sigkill_leaves_markers_but_releases_leases_for_restart(self):
        process=self.start();self.active(process,step=1)
        # Kill the entire *owned test session*, simulating container destruction.
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        self.assertTrue(list(self.root.rglob('.lock')))
        self.assertFalse((self.root/'finalized').exists())
        self.assertEqual(self.start().wait(timeout=40),0,self.logs())
        self.assertEqual(list(self.root.rglob('.lock')),[])
        self.compare_reference()
