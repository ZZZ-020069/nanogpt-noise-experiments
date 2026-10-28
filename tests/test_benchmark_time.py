"""CPU checks of benchmark accounting, without pretending to test GPU speed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'benchmark_time', Path(__file__).resolve().parents[1]/'scripts'/'benchmark_time.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class TimingAccountingTests(unittest.TestCase):
    def test_two_billion_rounds_up_and_counts_initial_final_evaluation(self):
        self.assertEqual(benchmark.estimate(1, 2, 3, 2_000_000_000), 3815+21*2+3)

    def test_historical_budget_accounting(self):
        self.assertEqual(benchmark.estimate(1, 2, 3, 2_600_000_000), 4960+27*2+3)

    def test_setup_is_per_candidate_not_per_step(self):
        self.assertEqual(benchmark.estimate(0, 0, 7, 2_000_000_000), 7)


if __name__ == '__main__':
    unittest.main()
