"""CPU checks of the fit-only scale decision before held-out measurements."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from nanogpt_noise.artifacts import save_checkpoint
from nanogpt_noise.lm_head.measure import fit_scale_decision, measure
from tests.test_integration import fixture


def moments(std0=.2, std1=.2, mean=1.):
    return {i: {'second_moment': {'mean': mean, 'std': std}}
            for i, std in enumerate((std0, std1))}


class FitScaleGate(unittest.TestCase):
    def test_threshold_both_layouts_and_block_limit(self):
        self.assertTrue(fit_scale_decision(moments(.3, .3), 5, True)['passed'])
        for values in (moments(.31, .2), moments(.2, .31)):
            five = fit_scale_decision(values, 5, True)
            self.assertFalse(five['passed'])
            self.assertEqual(five['action'], 'expand_blocks')
            ten = fit_scale_decision(values, 10, True)
            self.assertFalse(ten['passed'])
            self.assertEqual(ten['action'], 'unresolved_at_block_limit')
        self.assertTrue(fit_scale_decision(moments(), 10, True)['passed'])

    def test_data_stability_precedes_block_expansion(self):
        result = fit_scale_decision(moments(.5, .4), 5, False)
        self.assertFalse(result['passed'])
        self.assertEqual(result['action'], 'expand_tokens')

    def test_undefined_moments_do_not_pass_or_divide_by_zero(self):
        for values in (moments(mean=0.), moments(mean=1e-320), moments(mean=float('nan')),
                       moments(None), moments(float('inf')), moments(-.1)):
            result = fit_scale_decision(values, 5, True)
            self.assertFalse(result['passed'])
            self.assertEqual(result['action'], 'review_undefined_moments')

    def test_cli_failed_scale_returns_failure_and_never_syncs_env(self):
        from nanogpt_noise.lm_head.__main__ import main
        with patch('nanogpt_noise.lm_head.__main__.measure', return_value={
                'passed': False, 'fit_scale': fit_scale_decision(moments(.5), 5, True)}), \
                patch('nanogpt_noise.lm_head.__main__.sync_k_statistics_env') as sync, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['curvature', '--input-root', 'unused',
                                   '--output', 'unused-output', '--mode', 'fit']), 3)
            sync.assert_not_called()

    def test_measured_fit_records_scale_and_heldout_requires_freeze(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            inputs = root / 'inputs'
            inputs.mkdir()
            data, artifacts = fixture(inputs)
            data.rename(inputs / 'openwebtext_subset')
            artifacts.rename(inputs / 'frozen-artifacts')
            fitted = root / 'fit'
            report = measure(inputs, fitted, tokens=256, count=10, micro=2, test=True)
            self.assertEqual(report['fit_scale'], fit_scale_decision(
                report['moments_by_layout'], 10, report['stability_passed']))
            record = torch.load(fitted / 'fit.pt', weights_only=True)
            self.assertEqual(record['fit_scale'], report['fit_scale'])
            self.assertEqual(record['frozen_fit_id'], report['frozen_fit_id'])
            self.assertEqual(record['fit_passed'], report['passed'])
            if not report['fit_scale']['passed']:
                self.assertFalse(report['passed'])
                self.assertIsNone(report['frozen_fit_id'])

            # Synthetic test-only artifact: even a claimed pass cannot bypass scale review.
            record['fit_passed'] = True
            record['fit_scale'] = None
            save_checkpoint(fitted / 'fit.pt', record)
            with patch('nanogpt_noise.lm_head.measure.probabilities') as prob:
                with self.assertRaisesRegex(ValueError, 'block-scale review'):
                    measure(inputs, root / 'blocked', 'check', fit_dir=fitted,
                            tokens=256, count=5, micro=2, test=True)
                prob.assert_not_called()
                self.assertFalse((root / 'blocked').exists())

            # A 10-block frozen fit is accepted with exactly five held-out blocks per layout.
            record['fit_scale'] = fit_scale_decision(moments(), 10, True)
            record['frozen_fit_id'] = report['id']
            save_checkpoint(fitted / 'fit.pt', record)
            checked = measure(inputs, root / 'check', 'check', fit_dir=fitted,
                              tokens=256, count=5, micro=2, test=True)
            self.assertEqual(len(checked['blocks']), 10)
            with self.assertRaisesRegex(ValueError, 'prescribed block count'):
                measure(inputs, root / 'wrong-check', 'check', fit_dir=fitted,
                        tokens=256, count=10, micro=2, test=True)


if __name__ == '__main__':
    unittest.main()
