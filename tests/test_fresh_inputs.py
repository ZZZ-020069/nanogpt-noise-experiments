"""Fresh-input lifecycle checks; mocked downloads never create a real corpus."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from nanogpt_noise.artifacts import lock, read_json, write_json
from nanogpt_noise.data import REVISION
from nanogpt_noise.model import Shape
from nanogpt_noise.prepare import prepare_bundle, prepare_corpus
from nanogpt_noise.protocol import Budget, VERSION, INPUT_POLICY, require_current_input


class FreshInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'inputs'

    def marker(self):
        return dict(schema=1, policy=INPUT_POLICY, design=VERSION,
                    source_revision=REVISION, budget=asdict(Budget()))

    def identity(self):
        return dict(tiny=False, design=VERSION, budget=asdict(Budget()), shape=asdict(Shape()),
                    meta=dict(input_policy=INPUT_POLICY, preparation_design=VERSION))

    def test_empty_root_calls_corpus_then_freeze_then_registration(self):
        calls = []
        with patch('nanogpt_noise.prepare.prepare_corpus', side_effect=lambda *a, **k: calls.append('corpus')), \
             patch('nanogpt_noise.prepare.freeze', side_effect=lambda *a: calls.append('freeze')), \
             patch('nanogpt_noise.prepare.register_bundle', side_effect=lambda *a: calls.append('register') or 'id'):
            self.assertEqual(prepare_bundle(self.root), 'id')
        self.assertEqual(calls, ['corpus','freeze','register'])
        self.assertEqual(read_json(self.root/'preparation.json'), self.marker())
        self.assertFalse((self.root/'.lock').exists())

    def test_existing_unowned_files_are_not_adopted_or_overwritten(self):
        self.root.mkdir()
        old = self.root/'old.bin'
        old.write_bytes(b'preserve')
        with patch('nanogpt_noise.prepare.prepare_corpus') as corpus:
            with self.assertRaisesRegex(ValueError, 'empty INPUT_ROOT'):
                prepare_bundle(self.root, restart_incomplete=True)
            corpus.assert_not_called()
        self.assertEqual(old.read_bytes(), b'preserve')
        self.assertFalse((self.root/'preparation.json').exists())

    def test_interrupted_owned_preparation_can_retry(self):
        with patch('nanogpt_noise.prepare.prepare_corpus', side_effect=OSError('network')):
            with self.assertRaises(OSError):
                prepare_bundle(self.root)
        self.assertTrue((self.root/'preparation.json').exists())
        with patch('nanogpt_noise.prepare.prepare_corpus') as corpus, \
             patch('nanogpt_noise.prepare.freeze'), \
             patch('nanogpt_noise.prepare.register_bundle', return_value='id'):
            self.assertEqual(prepare_bundle(self.root, restart_incomplete=True),'id')
            corpus.assert_called_once_with(self.root/'openwebtext_subset', restart_incomplete=True)

    def test_registered_bundle_only_verifies_without_generating(self):
        write_json(self.root/'preparation.json',self.marker())
        write_json(self.root/'registration.json',{})
        write_json(self.root/'frozen-artifacts/identity.json',self.identity())
        with patch('nanogpt_noise.prepare.verify_bundle', return_value='id') as verify, \
             patch('nanogpt_noise.prepare.prepare_corpus') as corpus, \
             patch('nanogpt_noise.prepare.freeze') as freeze:
            self.assertEqual(prepare_bundle(self.root, restart_incomplete=True),'id')
            verify.assert_called_once_with(self.root)
            corpus.assert_not_called()
            freeze.assert_not_called()

    def test_changed_marker_and_concurrent_preparation_rejected(self):
        write_json(self.root/'preparation.json',{**self.marker(), 'design':'v1.7'})
        with self.assertRaisesRegex(ValueError, 'protocol changed'):
            prepare_bundle(self.root)
        with lock(self.root):
            with self.assertRaisesRegex(RuntimeError, 'active or stale lock'):
                prepare_bundle(self.root)

    def test_old_or_migrated_inputs_do_not_satisfy_fresh_policy(self):
        current = self.identity()
        require_current_input(current)
        for change in ({'design':'v1.7'}, {'meta':{}}, {'migration':{}},
                       {'meta':dict(input_policy=INPUT_POLICY, preparation_design='v1.7')}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'fresh INPUT_ROOT'):
                require_current_input({**current, **change})

    def test_corpus_download_path_pins_source_and_registers_fresh_metadata(self):
        source = MagicMock()
        source.shuffle.return_value = []
        loader = MagicMock(return_value=source)
        encoder = SimpleNamespace(n_vocab=50257, eot_token=50256)
        counts = []
        def small_fill(stream, enc, path, count):
            counts.append(count)
            path.write_bytes(b'\0\0')
            return dict(tokens=count, documents_consumed=1, boundary_document_truncated=True)
        with patch.dict('sys.modules', {
                'datasets':SimpleNamespace(load_dataset=loader),
                'tiktoken':SimpleNamespace(get_encoding=lambda name: encoder)}), \
             patch('nanogpt_noise.prepare.fill', side_effect=small_fill), \
             patch('nanogpt_noise.prepare.importlib.metadata.version', return_value='test'):
            prepare_corpus(self.root)
            prepare_corpus(self.root)  # a completed current corpus is not downloaded twice
        loader.assert_called_once_with('Skylion007/openwebtext',split='train',streaming=True,revision=REVISION)
        source.shuffle.assert_called_once_with(seed=2357,buffer_size=10000)
        self.assertEqual(counts,[10_000_000,1_000_000_000])
        meta=read_json(self.root/'meta.json')
        self.assertEqual(meta['input_policy'],INPUT_POLICY)
        self.assertEqual(meta['preparation_design'],VERSION)
        self.assertTrue((self.root/'train.bin').is_file())
        self.assertFalse((self.root/'train.bin.incomplete').exists())


if __name__ == '__main__':
    unittest.main()
