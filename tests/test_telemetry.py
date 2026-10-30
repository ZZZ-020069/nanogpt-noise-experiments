import os
import unittest
from unittest.mock import patch
from nanogpt_noise.telemetry import Telemetry,payload

class TelemetryTests(unittest.TestCase):
    def test_disabled_never_starts_uploader(self):
        with patch.dict(os.environ,{'NANOGPT_SWANLAB':'0'}):
            t=Telemetry('unused',{},'id','formal',0)
            self.assertIsNone(t.process)
            t.log(1,2,{},[],'running');t.close()
    def test_missing_key_fails_before_training(self):
        with patch.dict(os.environ,{'NANOGPT_SWANLAB':'1'},clear=True):
            with self.assertRaisesRegex(RuntimeError,'SWANLAB_API_KEY'):
                Telemetry('unused',{},'id','smoke',0)
    def test_finite_metrics_include_their_own_steps(self):
        v=payload(200,1000,{'step':200,'lr':.01,'ratio':float('inf')},
                  [{'step':191,'tokens':900,'train':4.,'val':5.}],'completed')
        self.assertEqual(v['evaluation/step'],191)
        self.assertEqual(v['diagnostic/step'],200)
        self.assertNotIn('diagnostic/ratio',v)
        self.assertEqual(v['status/completed'],1)

    def test_worker_sends_points_and_finishes_without_secrets_in_config(self):
        from unittest.mock import MagicMock
        import queue
        from nanogpt_noise.telemetry import worker
        sdk=MagicMock();messages=queue.Queue()
        messages.put({'progress/step':8});messages.put(None)
        with patch.dict('sys.modules',{'swanlab':sdk}), patch.dict(os.environ,{'SWANLAB_API_KEY':'test-only-secret'}):
            worker(messages,{'mode':'smoke'},'example','unused')
        sdk.init.return_value.log.assert_called_once_with({'progress/step':8},step=0)
        sdk.init.return_value.finish.assert_called_once()
        self.assertNotIn('test-only-secret',str(sdk.init.call_args))

    def test_spawned_uploader_delivers_and_exits(self):
        import tempfile,sys
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'swanlab.py').write_text("import os\nfrom pathlib import Path\nclass Run:\n def log(self,value,step): Path(os.environ['TEST_UPLOAD']).write_text(str(value))\n def finish(self): pass\ndef init(**kw): return Run()\n")
            with patch.object(sys,'path',[tmp]+sys.path), patch.dict(os.environ,{
                'NANOGPT_SWANLAB':'1','SWANLAB_API_KEY':'fake-test-key',
                'RUN_BASE':tmp,'TEST_UPLOAD':str(root/'uploaded')}):
                t=Telemetry(root/'candidate_00',{'scope':'last1','optimizer':'adamw','node':0},'id','smoke',0)
                try: t.log(8,42,{},[],'completed')
                finally: t.close()
                self.assertFalse(t.process.is_alive())
                self.assertIn('42',(root/'uploaded').read_text())
