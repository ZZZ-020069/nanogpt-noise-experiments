"""Synthetic selection evidence; no GPU performance claims."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nanogpt_noise.artifacts import write_json, digest

spec = importlib.util.spec_from_file_location('selection', Path(__file__).resolve().parents[1]/'scripts/select_lm_head_micro.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.identity = dict(input_id='fixture', code='code', gpus=[dict(total_bytes=1000)]*2)

    def success(self, micro, objective, seconds=10, peak=500):
        attempt = self.root/f'micro-{micro}'/objective/'attempt-0000'
        write_json(attempt/'execution.json', dict(exit_code=0, command_seconds=seconds*4+10))
        run_ids = []
        for optimizer in selection.CENTERS:
            path = attempt/'smoke'/objective/optimizer
            execution = dict(artifact_id='fixture', code='code', micro=micro, mode='smoke',
                             device_type='cuda', target_steps=32, config=dict(objective=objective, optimizer=optimizer),
                             statistics_sha256='Khash' if objective == 'K' else None)
            run_id = digest(execution)
            run_ids.append(run_id)
            write_json(path/'metadata.json', dict(run_id=run_id, execution=execution))
            write_json(path/'status.json', dict(run_id=run_id, status='completed', mode='smoke', step=32,
                                               tokens=32*524288, loss=10.))
            write_json(path/'performance.json', dict(segment_start_step=0, segment_end_step=32,
                       wall_seconds=seconds, peak_allocated_bytes=peak))
        write_json(attempt/'smoke'/objective/'passed.json', dict(passed=True, run_ids=run_ids,
                   identity=selection.expected_proof(self.identity, micro, objective, 'Khash')))
        return attempt

    def report(self):
        return selection.selection_report(self.root, self.identity, 'Khash')

    def fill(self):
        for m in (16,32):
            for o in ('G','T','K'):
                self.success(m,o,seconds=10 if m==16 else 8)

    def test_gt_only_cannot_select(self):
        for m in (16,32):
            for o in ('G','T'):
                self.success(m,o)
        result = self.report()
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_micro'])

    def test_complete_selects_faster_and_reports_all_24(self):
        self.fill()
        result = self.report()
        self.assertTrue(result['complete'])
        self.assertEqual(result['selected_micro'],32)
        self.assertEqual(sum(len(g['rows']) for g in result['groups']),24)

    def test_allocation_headroom_disqualifies_otherwise_faster_micro(self):
        self.fill()
        self.success(32,'T',seconds=8,peak=950)
        self.assertEqual(self.report()['selected_micro'],16)

    def test_error_is_not_evidence_of_other_micro_winning_but_oom_is(self):
        self.fill()
        attempt = self.root/'micro-32/T/attempt-0001'
        write_json(attempt/'execution.json',dict(exit_code=1))
        (attempt/'console.log').write_text('ConnectionError',encoding='utf8')
        self.assertIsNone(self.report()['selected_micro'])
        (attempt/'console.log').write_text('torch.OutOfMemoryError: CUDA out of memory.',encoding='utf8')
        self.assertEqual(self.report()['selected_micro'],16)

    def test_resumed_partial_timing_cannot_win(self):
        self.fill()
        path=self.root/'micro-32/T/attempt-0000/smoke/T/adam/performance.json'
        write_json(path,dict(segment_start_step=20,segment_end_step=32,wall_seconds=1,peak_allocated_bytes=500))
        self.assertIsNone(self.report()['selected_micro'])

    def test_changed_k_proof_or_missing_optimizer_rejected(self):
        self.fill()
        report=selection.selection_report(self.root,self.identity,'changed')
        self.assertIsNone(report['selected_micro'])
        (self.root/'micro-16/G/attempt-0000/smoke/G/sgd/status.json').unlink()
        self.assertIsNone(self.report()['selected_micro'])

    def test_busy_gpu_and_duplicate_devices_rejected(self):
        inventory='0, GPU-a, RTX, 96000, 200, 0, 580\n1, GPU-b, RTX, 96000, 15, 0, 580\n'
        with patch.object(selection.subprocess,'check_output',side_effect=[inventory,'GPU-a, 123, python, 5000\n']):
            with self.assertRaisesRegex(RuntimeError,'occupied'):
                selection.gpu_inventory(['0','1'])
        with patch.object(selection.subprocess,'check_output',return_value=inventory):
            with self.assertRaises(ValueError):selection.gpu_inventory(['0','0'])

    def test_wrong_input_rejected_before_gpu_query_or_output(self):
        root=self.root/'new'
        with patch.object(selection,'input_identity',side_effect=ValueError('old protocol')), \
             patch.object(selection,'gpu_inventory') as gpu:
            with self.assertRaisesRegex(ValueError,'old protocol'):
                selection.main(['--selection-root',str(root),'--input-root','old','--stage','gt'])
            gpu.assert_not_called()
        self.assertFalse(root.exists())

    def test_failed_lanes_preserve_attempts_and_require_explicit_retry(self):
        root=self.root/'selection'
        gpus=[dict(uuid=f'GPU-{i}',name='RTX',total_bytes=1000,driver='580') for i in (0,1)]
        calls=[]
        def launch(supervisor,command,attempt,gpu):
            calls.append((command,gpu))
            write_json(attempt/'execution.json',dict(exit_code=1))
            (attempt/'console.log').write_text('torch.OutOfMemoryError: CUDA out of memory',encoding='utf8')
        args=['--selection-root',str(root),'--input-root',str(self.root/'inputs'),'--stage','gt']
        with patch.object(selection,'input_identity',return_value=dict(id='fixture')), \
             patch.object(selection,'gpu_inventory',return_value=gpus), \
             patch.object(selection.Supervisor,'launch',new=launch):
            self.assertEqual(selection.main(args),3)
            self.assertEqual(len(calls),4)
            self.assertEqual({gpu for _,gpu in calls},{'GPU-0','GPU-1'})
            self.assertTrue(all(command[3]=='smoke' for command,_ in calls))
            selection.main(args)
            self.assertEqual(len(calls),4)
            selection.main(args+['--retry-failed'])
            self.assertEqual(len(calls),8)
            self.assertEqual(len(list(root.glob('micro-*/*/attempt-*'))),8)

    def test_h100_handoff_requires_same_input_code_and_k(self):
        from nanogpt_noise.artifacts import file_hash
        stats=self.root/'K-approved.pt'
        stats.write_bytes(b'synthetic test artifact')
        identity=dict(input_id='fixture',code=selection.code_identity()['sha256'],
                      supervisor=file_hash(selection.__file__))
        result=dict(identity=identity,complete=True,selected_micro=32,
                    statistics_sha256=file_hash(stats))
        write_json(self.root/'selection.json',result)
        args=['--selection-root',str(self.root),'--input-root','inputs',
              '--statistics',str(stats),'--handoff-micro']
        with patch.object(selection,'input_identity',return_value=dict(id='fixture',shape={})), \
             patch('nanogpt_noise.lm_head.measure.load_statistics'), \
             patch.object(selection,'gpu_inventory') as gpu:
            self.assertEqual(selection.main(args),0)
            stats.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'must match'):
                selection.main(args)
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
