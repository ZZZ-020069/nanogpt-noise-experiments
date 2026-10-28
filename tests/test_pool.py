import threading
import time
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from nanogpt_noise.pool import run_pool, SubprocessRunner, serve, main
from nanogpt_noise.artifacts import lock


class PoolTests(unittest.TestCase):
    def test_external_pause_stops_dispatch_even_when_current_job_completes(self):
        stop=threading.Event()
        seen=[]
        def runner(gpu,job):
            seen.append(job)
            stop.set()
            return 0
        self.assertEqual(run_pool(range(10),['0'],runner,stop_requested=stop.is_set),[])
        self.assertEqual(seen,[0])

    def test_stop_before_spawn_never_launches_child(self):
        args=SimpleNamespace(data='data',artifacts='artifacts',output='runs',micro_batch=4)
        with tempfile.TemporaryDirectory() as temp, patch('nanogpt_noise.pool.subprocess.Popen') as popen:
            runner=SubprocessRunner(args,temp)
            runner.terminate_all()
            self.assertEqual(runner('0',140),75)
            popen.assert_not_called()

    def test_explicit_resume_clears_request_only_after_acquiring_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pause=root/'pause.request';pause.touch()
            argv=['pool','--start','140','--end','279','--gpus','0,1,2,3',
                  '--data','data','--artifacts','artifacts','--output',temp,'--resume-paused']
            with lock(root/'launcher'), patch('sys.argv',argv):
                with self.assertRaisesRegex(RuntimeError,'active or stale lock'): main()
                self.assertTrue(pause.exists())
            with patch('sys.argv',argv), patch('nanogpt_noise.pool.run_pool',return_value=[]):
                main()
            self.assertFalse(pause.exists())
            self.assertFalse((root/'launcher/.lock').exists())

    def test_ordinary_start_refuses_pause_and_paused_exit_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pause=root/'pause.request';pause.touch()
            args=SimpleNamespace(output=temp,resume_paused=False,start=140,end=279)
            with self.assertRaisesRegex(RuntimeError,'scope is paused'): serve(args,['0'])
            pause.unlink()
            def request(*a,**kw):
                pause.touch()
                return []
            with patch('nanogpt_noise.pool.run_pool',side_effect=request):
                with self.assertRaises(SystemExit) as caught: serve(args,['0'])
            self.assertEqual(caught.exception.code,75)

    def test_every_job_runs_once_with_one_job_per_gpu(self):
        active = set()
        seen = []
        lock = threading.Lock()

        def runner(gpu, job):
            with lock:
                self.assertNotIn(gpu, active)
                active.add(gpu)
                seen.append((gpu, job))
            time.sleep(0.002 * (job % 3))
            with lock:
                active.remove(gpu)
            return 0

        failures = run_pool(range(11), ['0', '1', '2', '3'], runner)
        self.assertEqual(failures, [])
        self.assertEqual(sorted(job for _, job in seen), list(range(11)))
        self.assertLessEqual(len(active), 4)

    def test_failure_stops_new_assignments(self):
        barrier = threading.Barrier(4)
        seen = []
        lock = threading.Lock()

        def runner(gpu, job):
            with lock:
                seen.append(job)
            barrier.wait(timeout=2)
            if job == 0:
                return 75
            time.sleep(0.03)
            return 0

        stopped = []
        failures = run_pool(range(12), ['0', '1', '2', '3'], runner,
                            on_failure=lambda: stopped.append(True))
        self.assertEqual(failures, [(0, 75)])
        self.assertEqual(sorted(seen), [0, 1, 2, 3])
        self.assertEqual(stopped, [True])

    def test_gpu_tokens_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            run_pool([0], ['0', '0'], lambda gpu, job: 0)

    def test_subprocess_runner_isolates_gpu_and_logs_job(self):
        args = SimpleNamespace(data='/data', artifacts='/artifacts', output='/runs',
                               micro_batch=4)
        process = MagicMock()
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory, \
             patch('nanogpt_noise.pool.subprocess.Popen', return_value=process) as popen:
            runner = SubprocessRunner(args, directory)
            self.assertEqual(runner('GPU-deadbeef', 17), 0)
            command = popen.call_args.args[0]
            options = popen.call_args.kwargs
            self.assertEqual(command[-2:], ['--job-index', '17'])
            self.assertIn('nanogpt_noise.node', command)
            self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], 'GPU-deadbeef')
            self.assertTrue((Path(directory) / 'job_0017.log').exists())


if __name__ == '__main__':
    unittest.main()
