"""Pilot provenance, real tiny-model recovery, reporting, and queue isolation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import torch

from nanogpt_noise.artifacts import digest, read_json, write_json
from nanogpt_noise.model import GPT
from tests.test_integration import fixture

MODULE = Path(__file__).resolve().parents[1] / 'scripts' / 'joint_pilot.py'
spec = importlib.util.spec_from_file_location('joint_pilot', MODULE)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PlanTests(unittest.TestCase):
    def test_coverage_pairing_and_unchanged_formal_protocol(self):
        from nanogpt_noise.protocol import Budget, INITIAL, grid
        specs = pilot.candidates()
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({s['name'] for s in specs}), 24)
        self.assertEqual(pilot.PREFIX_STEPS * 524288, 300417024)
        for op in pilot.OPS:
            own = [s for s in specs if s['optimizer'] == op]
            self.assertEqual(sorted(s['exponent'] for s in own if s['node'] == 0),
                             [-1., -.5, 0., .5, 1.])
            self.assertEqual([(s['node'], s['exponent']) for s in own if s['node'] != 0], [(1, 0.)])
        self.assertEqual(Budget().steps, 3815)
        self.assertEqual(INITIAL, (-1., -.5, 0., .5, 1.))
        self.assertEqual(len(grid()), 840)

    def test_dry_run_needs_no_inputs_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'unused'
            result = subprocess.run([sys.executable, str(MODULE), '--data', 'missing',
                                     '--artifacts', 'missing', '--output', str(output), '--dry-run'],
                                    capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['total_tokens'], 7210008576)
            self.assertEqual(len(plan['candidates']), 24)
            self.assertFalse(output.exists())

    def test_invalid_gpu_tokens(self):
        for value in ('0', '0,0', '0,1,2', '0,', ' ,1'):
            with self.assertRaises(ValueError):
                pilot.parse_gpus(value)
        self.assertEqual(pilot.parse_gpus('GPU-first, GPU-second'), ['GPU-first', 'GPU-second'])


class PilotIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data, self.artifacts = fixture(self.root)
        self.output = self.root / 'pilot'
        self.output.mkdir()
        self.manifest = pilot.prepare(self.output, self.data, self.artifacts, 2, test=True, steps=3)
        torch.set_num_threads(1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_resume_binding_and_input_tamper(self):
        self.assertEqual(pilot.prepare(self.output, self.data, self.artifacts, 2, test=True, steps=3), self.manifest)
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            pilot.prepare(self.output, self.data, self.artifacts, 1, test=True, steps=3)
        with self.assertRaisesRegex(ValueError, 'real pilot'):
            pilot.prepare(self.output, self.data, self.artifacts, 2)
        with (self.data / 'val.bin').open('ab') as stream:
            stream.write(b'00')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            pilot.prepare(self.output, self.data, self.artifacts, 2, test=True, steps=3)

    def test_nonempty_unowned_directory_rejected(self):
        other = self.root / 'unrelated'
        other.mkdir()
        (other / 'keep.txt').write_text('keep', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            pilot.prepare(other, self.data, self.artifacts, 2, test=True, steps=3)
        self.assertEqual((other / 'keep.txt').read_text(), 'keep')

    def test_exact_resume_and_terminal_compaction_for_all_optimizers(self):
        for op in pilot.OPS:
            item = next(s for s in pilot.candidates() if s['optimizer'] == op and s['node'] == 1)
            paused = pilot.execute_candidate(item, self.manifest, self.data, self.artifacts,
                                             self.output, pause_after=1)
            self.assertEqual(paused['status'], 'infrastructure_failure')
            result = pilot.execute_candidate(item, self.manifest, self.data, self.artifacts, self.output)
            other = self.root / f'uninterrupted-{op}'
            whole = pilot.execute_candidate(item, self.manifest, self.data, self.artifacts, other)
            self.assertEqual(result['loss'], whole['loss'])
            a = torch.load(self.output / item['name'] / 'final.pt', weights_only=True)
            b = torch.load(other / item['name'] / 'final.pt', weights_only=True)
            for name in a['model']:
                self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
            self.assertEqual(result['mode'], 'test')
            self.assertEqual(result['step'], 3)
            self.assertEqual(a['execution']['config']['pilot']['manifest_id'], self.manifest['identity'])
            self.assertTrue((self.output / item['name'] / 'steps.jsonl.gz').is_file())
            self.assertFalse((self.output / item['name'] / 'steps').exists())
            self.assertEqual(pilot.execute_candidate(item, self.manifest, self.data, self.artifacts, self.output), result)

    def test_real_24_candidate_cpu_flow_and_shared_eval_tokens(self):
        for item in pilot.candidates():
            result = pilot.execute_candidate(item, self.manifest, self.data, self.artifacts, self.output)
            self.assertEqual(result['status'], 'completed')
        report = pilot.summarize(self.output, self.manifest)
        self.assertTrue(report['complete'])
        self.assertEqual(report['counts']['completed'], 24)
        self.assertFalse(report['formal_result'])
        self.assertEqual(report['mode'], 'test')
        for op in pilot.OPS:
            self.assertTrue(report['clean_initial_LR_comparison'][op]['all_five_terminal'])
            pair = report['small_noise_center_pairs'][op]
            self.assertEqual([e['tokens'] for e in pair['shared_evaluations']], [0, 64, 96])
            self.assertAlmostEqual(pair['shared_evaluations'][-1]['noisy_minus_clean'],
                                   pair['final_loss_noisy_minus_clean'])
        for file in ('summary.json', 'candidates.csv', 'evaluations.csv', 'report.txt'):
            self.assertTrue((self.output / file).is_file())

    def test_numerical_failure_is_terminal_and_not_a_finite_loss(self):
        forward = GPT.forward
        def broken(model, x, y):
            return forward(model, x, y) * float('nan')
        item = pilot.candidates()[0]
        with patch.object(GPT, 'forward', broken):
            result = pilot.execute_candidate(item, self.manifest, self.data, self.artifacts, self.output)
        self.assertEqual(result['status'], 'numerical_failure')
        self.assertTrue((self.output / item['name'] / 'failure.pt').is_file())
        report = pilot.summarize(self.output, self.manifest)
        self.assertEqual(report['counts']['numerical_failure'], 1)
        self.assertIsNone(report['candidates'][0]['loss'])
        self.assertFalse(report['complete'])
        self.assertIsNone(report['small_noise_center_pairs']['adamw']['final_loss_noisy_minus_clean'])

    def fake_terminal(self, item, loss):
        folder = self.output / item['name']
        execution = pilot.expected_execution(item, self.manifest)
        identity = digest(execution)
        write_json(folder / 'metadata.json', dict(execution=execution, run_id=identity))
        if loss is None:
            write_json(folder / 'status.json', dict(status='numerical_failure', mode='test', run_id=identity,
                       step=1, tokens=32, loss=None, field='gradient', T_diverge=32))
        else:
            evaluations = [dict(step=3, tokens=96, train=loss, val=loss)]
            write_json(folder / 'evaluations.json', evaluations)
            write_json(folder / 'status.json', dict(status='completed', mode='test', run_id=identity,
                       step=3, tokens=96, loss=loss))
            (folder / 'final.pt').write_bytes(b'fixture-marker-not-real-checkpoint')

    def test_summary_missing_midpoint_gain_and_noise_pair(self):
        clean = [s for s in pilot.candidates() if s['optimizer'] == 'adamw' and s['node'] == 0]
        losses = {-1.: 5., -.5: 2., 0.: 4., .5: 3., 1.: 6.}
        for item in clean[:2]:
            self.fake_terminal(item, losses[item['exponent']])
        partial = pilot.summarize(self.output, self.manifest)['clean_initial_LR_comparison']['adamw']
        self.assertFalse(partial['all_five_terminal'])
        self.assertIsNone(partial['five_initial_winner'])
        for item in clean:
            self.fake_terminal(item, losses[item['exponent']])
        noisy = next(s for s in pilot.candidates() if s['optimizer'] == 'adamw' and s['node'] == 1)
        self.fake_terminal(noisy, 4.1)
        report = pilot.summarize(self.output, self.manifest)
        self.assertEqual(report['clean_initial_LR_comparison']['adamw']['three_initial_minus_five_initial_loss'], 2.)
        self.assertAlmostEqual(report['small_noise_center_pairs']['adamw']['final_loss_noisy_minus_clean'], .1)

    def test_coarse_failed_middle_valid_is_explicit(self):
        for item in pilot.candidates():
            if item['optimizer'] == 'muon' and item['node'] == 0:
                self.fake_terminal(item, 3. if abs(item['exponent']) == .5 else None)
        row = pilot.summarize(self.output, self.manifest)['clean_initial_LR_comparison']['muon']
        self.assertTrue(row['coarse_failed_but_middle_valid'])
        self.assertIsNone(row['three_initial_loss'])

    def test_completed_marker_with_wrong_provenance_is_rejected(self):
        item = pilot.candidates()[0]
        self.fake_terminal(item, 3.)
        path = self.output / item['name'] / 'metadata.json'
        meta = read_json(path)
        meta['execution']['micro'] = 1
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            pilot.summarize(self.output, self.manifest)

    def test_supervisor_launch_environment_reports_and_light_bundle(self):
        specs = pilot.candidates()
        for item in specs[:-1]:
            self.fake_terminal(item, 3.)
        def popen(command, **options):
            self.assertEqual(options['env']['CUDA_VISIBLE_DEVICES'], 'GPU-first')
            for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                self.assertNotIn(key, options['env'])
            self.assertEqual(command[command.index('--worker') + 1], specs[-1]['name'])
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(options['cwd'], pilot.ROOT)
            self.fake_terminal(specs[-1], 3.1)
            class Child:
                def poll(self):
                    return 0
            return Child()
        argv = ['joint_pilot.py', '--data', str(self.data), '--artifacts', str(self.artifacts),
                '--output', str(self.output), '--micro-batch', '2', '--gpus', 'GPU-first,GPU-second']
        with patch.object(sys, 'argv', argv), patch.object(pilot, 'prepare', return_value=self.manifest), \
             patch.object(pilot.subprocess, 'Popen', side_effect=popen), \
             patch.object(pilot.time, 'sleep'), \
             patch.dict(os.environ, RANK='7', LOCAL_RANK='7', WORLD_SIZE='1', MASTER_ADDR='private', MASTER_PORT='1234'):
            self.assertEqual(pilot.main(), 0)
        with zipfile.ZipFile(self.output / 'analysis_bundle.zip') as bundle:
            names = bundle.namelist()
            self.assertIn('summary.json', names)
            self.assertIn(f'{specs[-1]["name"]}/metadata.json', names)
            self.assertFalse(any(n.endswith('.pt') for n in names))
            self.assertTrue(json.loads(bundle.read('summary.json'))['complete'])


class QueueTests(unittest.TestCase):
    def test_fresh_children_exactly_once_and_one_child_per_gpu(self):
        active, seen = {}, []
        class Child:
            def __init__(self, gpu, job):
                self.gpu, self.job, self.polls = gpu, job, 0
            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                active.pop(self.gpu, None)
                return 0
        def launch(gpu, job):
            self.assertNotIn(gpu, active)
            active[gpu] = job
            seen.append(job)
            return Child(gpu, job)
        with patch.object(pilot.time, 'sleep'):
            self.assertEqual(pilot.supervise(range(24), ['0', '1'], launch), 0)
        self.assertEqual(seen, list(range(24)))
        self.assertFalse(active)

    def test_infrastructure_failure_stops_new_jobs_and_terminates_other(self):
        launched, terminated = [], []
        class Child:
            def __init__(self, job):
                self.job, self.stopped = job, False
            def poll(self):
                return 75 if self.job == 0 else (75 if self.stopped else None)
            def terminate(self):
                self.stopped = True
                terminated.append(self.job)
            def kill(self):
                self.stopped = True
        def launch(_gpu, job):
            launched.append(job)
            return Child(job)
        with patch.object(pilot.time, 'sleep'):
            self.assertEqual(pilot.supervise(range(24), ['0', '1'], launch), 75)
        self.assertEqual(launched, [0, 1])
        self.assertEqual(terminated, [1])

    def test_preexisting_stop_launches_nothing(self):
        def launch(*_args):
            self.fail('must not launch')
        self.assertEqual(pilot.supervise(range(24), ['0', '1'], launch,
                                         stop_requested=lambda: True), 130)

    def test_stop_during_launch_does_not_start_second_candidate(self):
        stopped, launches = [False], []
        class Child:
            done = False
            def poll(self):
                return 75 if self.done else None
            def terminate(self):
                self.done = True
        def launch(gpu, job):
            launches.append((gpu, job))
            stopped[0] = True
            return Child()
        with patch.object(pilot.time, 'sleep'):
            self.assertEqual(pilot.supervise(range(24), ['0', '1'], launch,
                                             stop_requested=lambda: stopped[0]), 130)
        self.assertEqual(launches, [('0', 0)])


if __name__ == '__main__':
    unittest.main()
