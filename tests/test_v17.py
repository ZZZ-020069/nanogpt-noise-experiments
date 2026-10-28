"""Budget migration and launch gates; synthetic CPU files are not GPU evidence."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import torch
from nanogpt_noise.artifacts import code_identity, digest, file_hash, read_json, write_json
from nanogpt_noise.data import verify
from nanogpt_noise.input_bundle import verify_bundle
from nanogpt_noise.migrate_inputs import migrate
from nanogpt_noise.model import Shape
from nanogpt_noise.protocol import Budget, CENTERS, SCOPES, configuration
from nanogpt_noise.smoke import validate
from nanogpt_noise.train import _run
from tests.test_integration import fixture


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'old'
        self.source.mkdir()
        data, artifacts = fixture(self.source)
        data.rename(self.source/'openwebtext_subset')
        artifacts.rename(self.source/'frozen-artifacts')
        self.data, self.artifacts = self.source/'openwebtext_subset', self.source/'frozen-artifacts'
        self.old = verify(self.data, self.artifacts)
        self.target = replace(Budget(**self.old['budget']), steps=3)

    def test_preserves_bytes_windows_and_lineage(self):
        original = {p.relative_to(self.source): file_hash(p)
                    for p in self.source.rglob('*') if p.is_file()}
        output = self.root/'new'
        result = migrate(self.source, output, _test_budget=self.target)
        self.assertEqual(result, verify_bundle(output))
        new = verify(output/'openwebtext_subset', output/'frozen-artifacts')
        self.assertNotEqual(result, self.old['id'])
        self.assertEqual(new['migration']['source_input_id'], self.old['id'])
        self.assertEqual(new['data'], self.old['data'])
        self.assertEqual(new['initialization_tensor_hash'], self.old['initialization_tensor_hash'])
        self.assertEqual(new['artifacts']['initial.pt'], self.old['artifacts']['initial.pt'])
        old_windows = torch.load(self.artifacts/'windows.pt', weights_only=True)
        new_windows = torch.load(output/'frozen-artifacts/windows.pt', weights_only=True)
        for key in old_windows:
            expected = old_windows[key][:3] if key == 'train' else old_windows[key]
            self.assertTrue(torch.equal(expected, new_windows[key]), key)
        self.assertEqual(original, {p.relative_to(self.source): file_hash(p)
                                   for p in self.source.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            migrate(self.source, output, _test_budget=self.target)

    def test_rejects_tiny_production_source_and_overlapping_paths(self):
        with self.assertRaisesRegex(ValueError, 'real inputs'):
            migrate(self.source, self.root/'new')
        for output in (self.source, self.source/'nested', self.root):
            with self.assertRaisesRegex(ValueError, 'separate directories'):
                migrate(self.source, output, _test_budget=self.target)

    def test_rejects_changed_data_before_creating_output(self):
        with (self.data/'train.bin').open('ab') as stream:
            stream.write(b'bad')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            migrate(self.source, self.root/'new', _test_budget=self.target)
        self.assertFalse((self.root/'new').exists())

    def test_rejects_longer_or_different_batch_budget(self):
        for target in (replace(self.target, steps=5), replace(self.target, batch=8)):
            with self.assertRaisesRegex(ValueError, 'only shortening'):
                migrate(self.source, self.root/'new', _test_budget=target)


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
        identity = dict(tiny=False, design='v1.7', budget=asdict(Budget()), shape=asdict(Shape()))
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
            identity = dict(tiny=False, design='v1.7', budget=asdict(Budget()),
                            shape=asdict(Shape()), id='test-id')
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
