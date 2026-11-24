from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

import cpu_budget as budget


class CpuBudgetTests(unittest.TestCase):
    def test_shared_mask_preserves_narrower_affinity(self):
        with patch.object(budget, 'CPUS', tuple(range(6))):
            process = Mock()
            process.cpu_affinity.side_effect = [list(range(8)), None, list(range(6))]
            self.assertEqual(budget.restrict(process), list(range(6)))
            process.cpu_affinity.assert_any_call(list(range(6)))
            process = Mock()
            process.cpu_affinity.return_value = [2, 3]
            self.assertEqual(budget.restrict(process), [2, 3])
            self.assertTrue(all(not c.args for c in process.cpu_affinity.call_args_list))

    def test_exact_script_scope(self):
        process = Mock()
        process.cwd.return_value = str(budget.ROOT)
        process.cmdline.return_value = ['python', '-u', 'campaign.py']
        self.assertEqual(budget.experiment_script(process), budget.ROOT / 'campaign.py')
        for command in (['-c', str(budget.ROOT / 'campaign.py')], ['-m', 'pytest'], ['other/campaign.py']):
            process.cmdline.return_value = ['python', *command]
            self.assertIsNone(budget.experiment_script(process))

    def test_busy_host_defers_admission(self):
        with patch.object(budget, 'CPUS', tuple(range(6))), patch.object(budget, 'LOGICAL', 8), \
                patch.object(budget, 'restrict'), patch.object(budget, 'FileLock', return_value=nullcontext()), \
                patch.object(budget.psutil, 'cpu_percent', side_effect=[100., 75., 62.5]) as utilization, \
                patch.object(budget.time, 'sleep') as sleep:
            budget.await_capacity()
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(utilization.call_count, 3)


if __name__ == '__main__':
    unittest.main()
