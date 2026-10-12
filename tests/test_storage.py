import gzip
import json
from pathlib import Path
import tempfile
import unittest

from nanogpt_noise.storage import compact_steps, retain_winner_checkpoint


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_step_compaction_is_lossless_and_idempotent(self):
        candidate = self.root/'candidate_00'
        steps = candidate/'steps'
        steps.mkdir(parents=True)
        expected = []
        for index in range(1, 4):
            value = {'step': index, 'loss': 1/index, 'nested': {'finite': True}}
            expected.append(value)
            (steps/f'{index:06d}.json').write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')

        first = compact_steps(candidate)
        archive = candidate/'steps.jsonl.gz'
        self.assertEqual(first['count'], 3)
        self.assertTrue(archive.exists())
        self.assertFalse(steps.exists())
        with gzip.open(archive, 'rt', encoding='utf8') as source:
            actual = [json.loads(line) for line in source]
        self.assertEqual(actual, expected)
        before = archive.read_bytes()
        second = compact_steps(candidate)
        self.assertEqual(second, first)
        self.assertEqual(archive.read_bytes(), before)

    def test_winner_retains_only_final_checkpoint(self):
        job = self.root/'job_0000'
        for index in range(3):
            candidate = job/f'candidate_{index:02d}'
            candidate.mkdir(parents=True)
            for name in ('latest.pt', 'best.pt', 'final.pt', 'failure.pt'):
                (candidate/name).write_bytes(f'{index}-{name}'.encode())

        retain_winner_checkpoint(job, 1)
        self.assertEqual(
            sorted(path.relative_to(job).as_posix() for path in job.rglob('*.pt')),
            ['candidate_01/final.pt'])

        retain_winner_checkpoint(job, None)
        self.assertEqual(list(job.rglob('*.pt')), [])


if __name__ == '__main__':
    unittest.main()
