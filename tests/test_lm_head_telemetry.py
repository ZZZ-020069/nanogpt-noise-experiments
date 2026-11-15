"""Head observer integration; fake SDK only, no cloud or GPU access."""
import contextlib
import io
import os
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from nanogpt_noise.artifacts import read_json
from nanogpt_noise.lm_head.protocol import configuration
from nanogpt_noise.lm_head.train import run, Runtime
from nanogpt_noise.lm_head.telemetry import Telemetry, check_environment
from nanogpt_noise.telemetry import worker
from tests.test_integration import fixture


class HeadTelemetryTests(unittest.TestCase):
    def test_preflight_enabled_disabled_invalid_and_secret_free(self):
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'0'}, clear=True), \
                patch('importlib.import_module') as sdk:
            self.assertFalse(check_environment()['enabled'])
            sdk.assert_not_called()
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'yes'}, clear=True):
            with self.assertRaisesRegex(ValueError, '0 or 1'): check_environment()
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'1'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'SWANLAB_API_KEY'): check_environment()
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'1','SWANLAB_API_KEY':'test-secret'}, clear=True):
            with patch('importlib.import_module', side_effect=ImportError('secret inside exception')):
                with self.assertRaisesRegex(RuntimeError, 'Install') as raised: check_environment()
                self.assertNotIn('secret', str(raised.exception))
            with patch('importlib.import_module', return_value=MagicMock(__version__='test')):
                result=check_environment()
                self.assertTrue(result['enabled'])
                self.assertNotIn('test-secret', str(result))

    def test_objective_evaluation_and_state_keep_their_own_steps(self):
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'0'}):
            t=Telemetry('unused', {}, 'id', 'formal', 0)
        t.process=MagicMock();t.messages=queue.Queue()
        metrics={'step':200,'train_loss':1.2}
        t.log(200,1000,metrics,[dict(step=191,train=3.,val=4.,
              train_objective=1.,val_objective=2.)],'infrastructure_failure')
        item=t.messages.get_nowait()
        self.assertEqual(item['diagnostic/objective_eval_step'],191)
        self.assertEqual(item['diagnostic/val_objective'],2.)
        self.assertEqual(item['evaluation/val'],4.)
        self.assertEqual(item['diagnostic/infrastructure_failure'],1)
        self.assertEqual(metrics,{'step':200,'train_loss':1.2})

    def test_tiny_fixture_never_opens_cloud_even_when_enabled(self):
        with patch.dict(os.environ, {'NANOGPT_SWANLAB':'1'}, clear=True):
            t=Telemetry('unused', {}, 'id', 'test', 0)
            self.assertIsNone(t.process)
            t.log(0,0,{},[],'running');t.close()

    def test_training_wires_observer_and_resume_step_without_mutating_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data,artifacts=fixture(root)
            cfg=configuration('T','sgd',0,0.)
            expected=dict(cfg)
            with patch('nanogpt_noise.lm_head.train.Telemetry') as observer:
                rt=Runtime(cpu=True)
                run(cfg,data,artifacts,root/'run',micro=2,runtime=rt,test=True,pause_after=1)
                self.assertEqual(observer.call_args.args[4],0)
                observer.return_value.close.assert_called_once()
                observer.reset_mock()
                result=run(cfg,data,artifacts,root/'run',micro=2,runtime=rt,test=True)
                args=observer.call_args.args
                self.assertEqual(args[4],1)
                self.assertEqual(args[1]['objective'],'T')
                self.assertEqual(args[1]['micro'],2)
                self.assertEqual(args[2],result['run_id'])
                self.assertEqual(args[1]['input_id'],read_json(artifacts/'identity.json')['id'])
                self.assertEqual(observer.return_value.log.call_args.args[-1],'completed')
                observer.return_value.close.assert_called_once()
                self.assertEqual(cfg,expected)

    def test_upload_failure_and_dead_worker_preserve_training(self):
        sdk=MagicMock();sdk.init.side_effect=RuntimeError('do-not-print-secret')
        stream=io.StringIO()
        with patch.dict('sys.modules', {'swanlab':sdk}), contextlib.redirect_stdout(stream):
            worker(queue.Queue(),{},'example','unused')
        self.assertIn('local training records retained',stream.getvalue())
        self.assertNotIn('do-not-print-secret',stream.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data,artifacts=fixture(root)
            with patch.dict(os.environ, {'NANOGPT_SWANLAB':'0'}):
                t=Telemetry('unused',{},'id','formal',0)
            t.process=MagicMock();t.process.is_alive.return_value=False
            t.messages=MagicMock()
            with patch('nanogpt_noise.lm_head.train.Telemetry',return_value=t), \
                    contextlib.redirect_stdout(io.StringIO()):
                result=run(configuration('G','sgd',0,0.),data,artifacts,root/'run',
                           micro=2,runtime=Runtime(cpu=True),test=True)
            self.assertEqual(result['status'],'completed')
            self.assertTrue((root/'run/final.pt').exists())
            self.assertTrue(t.warned)


if __name__=='__main__': unittest.main()
