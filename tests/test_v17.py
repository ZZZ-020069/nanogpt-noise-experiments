"""Budget and launch gates; synthetic CPU files are not GPU evidence."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch
from nanogpt_noise.artifacts import code_identity, digest, read_json, write_json
from nanogpt_noise.model import Shape
from nanogpt_noise.protocol import Budget, CENTERS, SCOPES, configuration
from nanogpt_noise.smoke import validate
from nanogpt_noise.train import _run


class DirectTrainingProtocolTests(unittest.TestCase):
    def test_all_scopes_reject_old_frozen_budget_before_model_creation(self):
        identity = dict(tiny=False, design='v1.6', budget=asdict(Budget(steps=4960)),
                        shape=asdict(Shape()))
        with patch('nanogpt_noise.train.verify', return_value=identity), \
             patch('nanogpt_noise.train.GPT') as model:
            for scope in SCOPES:
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, '2B protocol'):
                    _run(configuration(scope,'adamw',0,0),None,None,None,4,
                         MagicMock(),'formal',None,None)
            model.assert_not_called()

    def test_current_inputs_do_not_allow_old_or_tampered_config(self):
        identity = dict(tiny=False, design='v1.8', budget=asdict(Budget()), shape=asdict(Shape()),
                        meta=dict(input_policy='server-fresh-v1', preparation_design='v1.8'))
        for key, value in [('design','v1.6'), ('budget',asdict(Budget(steps=4960))),
                           ('sigma2',.5), ('grad_clip',1)]:
            config = configuration('last1','adamw',0,0)
            config[key] = value
            with patch('nanogpt_noise.train.verify', return_value=identity), \
                 self.subTest(key=key), self.assertRaisesRegex(ValueError, 'candidate config'):
                _run(config,None,None,None,4,MagicMock(),'formal',None,None)


class SmokeGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for optimizer in CENTERS:
            path = self.path(optimizer)
            # argparse --exponent produces a float; JSON hashes distinguish 0.0 from 0.
            execution = dict(config=configuration('last1',optimizer,10,0.0), mode='smoke',
                             artifact_id='test-id', code=code_identity()['sha256'], world=1,
                             micro=4, device_type='cuda', torch=str(torch.__version__),
                             cuda=torch.version.cuda, target_steps=8)
            identity = dict(tiny=False, design='v1.8', budget=asdict(Budget()),
                            shape=asdict(Shape()), id='test-id',
                            meta=dict(input_policy='server-fresh-v1', preparation_design='v1.8'))
            metadata = dict(execution=execution, run_id=digest(execution),
                            artifact_identity=identity, D=7079424)
            write_json(path/'metadata.json', metadata)
            write_json(path/'status.json', dict(status='completed',mode='smoke',step=8,
                       target_steps=8,tokens=8*524288,run_id=metadata['run_id']))
            write_json(path/'evaluations.json', [dict(step=0),dict(step=8)])
            (path/'final.pt').write_bytes(b'synthetic gate fixture, not a real checkpoint')

    def path(self, optimizer):
        return self.root/'smoke/1gpu'/f'last1-{optimizer}-node10'

    def test_all_four_required_and_scope_layout_input_bound(self):
        self.assertEqual(len(validate(self.root,'last1','test-id',4)['optimizers']),4)
        for scope, input_id, micro in [('last1','other',4), ('last1','test-id',8),
                                       ('full','test-id',4)]:
            with self.subTest(scope=scope,input_id=input_id,micro=micro), \
                 self.assertRaises((ValueError, FileNotFoundError)):
                validate(self.root,scope,input_id,micro)
        (self.path('sgd')/'final.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'sgd.*checkpoint'):
            validate(self.root,'last1','test-id',4)

    def test_numerical_failure_is_not_smoke_acceptance(self):
        path = self.path('adam')/'status.json'
        status = read_json(path)
        status['status'] = 'numerical_failure'
        write_json(path,status)
        with self.assertRaisesRegex(ValueError, 'adam.*eight steps'):
            validate(self.root,'last1','test-id',4)

    def test_stale_code_rejected(self):
        with patch('nanogpt_noise.smoke.code_identity', return_value={'sha256':'changed'}):
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate(self.root,'last1','test-id',4)


if __name__ == '__main__':
    unittest.main()
