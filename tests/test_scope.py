import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nanogpt_noise.artifacts import write_json
from nanogpt_noise.scope import (cleanup_scope, estimate_scope, finalize_scope, prepare_scope,
                                 scope_bounds, verify_scope)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scope_bounds_and_estimates(self):
        self.assertEqual(scope_bounds('full'), (0, 139))
        self.assertEqual(scope_bounds('last1'), (140, 279))
        self.assertEqual(scope_bounds('last12'), (700, 839))
        estimates = [estimate_scope(scope) for scope in
                     ('full', 'last1', 'last2', 'last3', 'last6', 'last12')]
        self.assertTrue(all(value['peak_bytes'] < 200_000_000_000 for value in estimates))
        self.assertGreater(estimates[0]['peak_bytes'], estimates[-1]['peak_bytes'])

    def test_directory_is_owned_by_one_scope(self):
        root = self.root/'last2'
        marker = prepare_scope(root, 'last2')
        self.assertEqual(marker['scope'], 'last2')
        self.assertEqual(prepare_scope(root, 'last2'), marker)
        with self.assertRaisesRegex(ValueError, 'named exactly'):
            prepare_scope(root, 'last3')
        (root/'job_0000').mkdir()
        with self.assertRaisesRegex(ValueError, 'outside scope'):
            prepare_scope(root, 'last2')

    def test_finalize_and_verify_detect_copy_tampering(self):
        root = self.root/'full'
        prepare_scope(root, 'full')
        job = root/'job_0000'
        candidate = job/'candidate_00'
        candidate.mkdir(parents=True)
        (candidate/'steps.jsonl.gz').write_bytes(b'archive')
        write_json(candidate/'steps.compaction.json', {'count': 4})
        (candidate/'final.pt').write_bytes(b'checkpoint')
        write_json(job/'search.json', {
            'complete': True,
            'winner': {'candidate': 0},
            'candidates': [{'candidate': 0}],
        })
        with patch('nanogpt_noise.scope.scope_indices', return_value=range(1)):
            manifest = finalize_scope(root, 'full')
            self.assertEqual(manifest['file_count'], len(manifest['files']))
            verified = verify_scope(root)
            self.assertEqual(verified['scope'], 'full')
            (candidate/'final.pt').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash|size'):
                verify_scope(root)

    def test_cleanup_requires_exact_scope_directory_and_confirmation(self):
        scope_root = self.root/'full'
        scope_root.mkdir()
        write_json(scope_root/'scope_manifest.json', {
            'format': 'nanogpt-scope-manifest-v1', 'scope': 'full',
            'files': [], 'file_count': 0,
        })
        with self.assertRaisesRegex(ValueError, 'confirmation'):
            cleanup_scope(scope_root, 'full', 'yes')
        result = cleanup_scope(scope_root, 'full', 'DELETE-full')
        self.assertEqual(result['scope'], 'full')
        self.assertFalse(scope_root.exists())


if __name__ == '__main__':
    unittest.main()
