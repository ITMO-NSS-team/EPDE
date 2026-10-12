"""Campaign accounting and content identity regressions (no search needed)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIC))
from epde_bench import campaign, report


class CampaignReportTests(unittest.TestCase):
    def test_distinct_noise_values_have_distinct_run_paths(self):
        self.assertNotEqual(campaign.run_stem('default', .1234561, 0),
                            campaign.run_stem('default', .1234562, 0))

    def test_timeout_only_dataset_and_variant_stay_in_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for ds, var, status in [('ode', 'default', 'ok'), ('ode', 'pysindy', 'timeout'),
                                    ('ks', 'default', 'timeout'), ('ks', 'pysindy', 'timeout')]:
                folder = root / 'runs' / ds; folder.mkdir(parents=True, exist_ok=True)
                rec = dict(dataset=ds, variant=var, noise=0, seed=0, status=status)
                if status == 'ok':
                    rec.update(problem={'kind': 'ode', 'truth': ['law']},
                               metrics={'success_front': True, 'success_selected': True})
                (folder / f'{var}.json').write_text(json.dumps(rec))
            with patch.object(report, '_figures'):
                out = report.make_report(root)
            import pandas as pd
            cells = pd.read_csv(out / 'success_front.csv')
            self.assertEqual(set(cells.dataset), {'ode', 'ks'})
            self.assertEqual(cells[cells.variant == 'default'].success_front.mean(), .5)
            rank = pd.read_csv(out / 'ranking.csv')
            self.assertEqual(set(rank.variant), {'default', 'pysindy'})

    def test_manifest_pending_and_unsupported_do_not_count_as_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planned = [dict(dataset='ode', variant='default', noise=0, seed=s,
                            problem={'kind': 'ode', 'truth_known': True}) for s in range(3)]
            (root / 'campaign.json').write_text(json.dumps({'jobs': planned}))
            folder = root / 'runs' / 'ode'; folder.mkdir(parents=True)
            for seed, status in [(0, 'ok'), (1, 'unsupported')]:
                rec = dict(planned[seed], status=status, metrics={'success_front': True})
                (folder / f'default__noise0__seed{seed}.json').write_text(json.dumps(rec))
            with patch.object(report, '_figures'):
                out = report.make_report(root)
            import pandas as pd
            rows = pd.read_csv(out / 'runs.csv')
            self.assertEqual(set(rows.status), {'ok', 'unsupported', 'pending'})
            cell = pd.read_csv(out / 'success_front.csv').iloc[0]
            self.assertEqual(cell.runs, 1)
            self.assertEqual(cell.success_front, 1.)
            self.assertEqual(cell.pending, 1)
            self.assertEqual(cell.unsupported, 1)

    def test_ranking_uses_same_supported_datasets_for_each_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for dataset, kind in [('ode', 'ode'), ('ns', 'pde_2d')]:
                for variant in ['default', 'pysindy']:
                    plan = dict(dataset=dataset, variant=variant, noise=0, seed=0,
                                problem={'kind': kind, 'truth_known': True})
                    jobs.append(plan)
                    folder = root / 'runs' / dataset; folder.mkdir(parents=True, exist_ok=True)
                    status = 'unsupported' if (dataset, variant) == ('ns', 'pysindy') else 'ok'
                    record = dict(plan, status=status, metrics={'success_front': dataset == 'ode'})
                    (folder / f'{variant}.json').write_text(json.dumps(record))
            (root / 'campaign.json').write_text(json.dumps({'jobs': jobs}))
            with patch.object(report, '_figures'):
                out = report.make_report(root)
            import pandas as pd
            ranking = pd.read_csv(out / 'ranking.csv')
            self.assertEqual(set(ranking.kind), {'ode'})
            self.assertEqual(set(ranking.success_front), {1.})
            self.assertEqual(set(pd.read_csv(out / 'success_front.csv').dataset), {'ode', 'ns'})

    def test_finished_rejects_another_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.json'
            path.write_text(json.dumps({'status': 'ok', 'identity': {'sha256': 'old'}}))
            self.assertFalse(campaign._finished(path, False, {'sha256': 'new'}))
            self.assertTrue(campaign._finished(path, False, {'sha256': 'old'}))

    def test_changed_manifest_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'campaign.json'
            planned = [{'identity': {'sha256': 'old'}}]
            content = json.dumps({'jobs': planned, 'timeout': 30})
            path.write_text(content)
            campaign._validate_manifest(path, planned, 30)
            with self.assertRaisesRegex(SystemExit, 'new campaign name'):
                campaign._validate_manifest(path, [{'identity': {'sha256': 'new'}}], 30)
            self.assertEqual(path.read_text(), content)

    def test_crash_replaces_stale_output_with_planned_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = ('ode', 'default', 0., 0)
            plan = dict(dataset='ode', variant='default', noise=0., seed=0,
                        problem={'kind': 'ode', 'truth_known': True}, identity={'sha256': 'new'})
            path = root / 'runs' / 'ode' / 'default__noise0__seed0.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(plan, status='ok', identity={'sha256': 'old'})))
            with patch.object(campaign.subprocess, 'Popen') as process:
                process.return_value.wait.return_value = 1
                _, status, _ = campaign._run_job(root, job, [], 30, plan)
            self.assertEqual(status, 'error')
            record = json.loads(path.read_text())
            self.assertEqual(record['identity'], plan['identity'])
            self.assertEqual(record['problem'], plan['problem'])

    def test_timeout_keeps_planned_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = ('ks', 'pysindy', 0., 0)
            plan = campaign._planned_job(job, {})
            with patch.object(campaign.subprocess, 'Popen') as process, patch.object(campaign, '_kill_tree'):
                process.return_value.wait.side_effect = campaign.subprocess.TimeoutExpired('bench', 1)
                _, status, _ = campaign._run_job(root, job, [], 1, plan)
            self.assertEqual(status, 'timeout')
            record = json.loads((root / 'runs' / 'ks' / 'pysindy__noise0__seed0.json').read_text())
            self.assertEqual(record['identity'], plan['identity'])
            self.assertTrue(record['problem']['truth_known'])
            self.assertEqual(record['problem']['kind'], 'pde_1d')

    def test_timeout_kills_run_when_process_table_is_refused(self):
        import psutil
        import subprocess
        import sys
        from epde_bench import campaign
        proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                                start_new_session=os.name != 'nt')
        try:
            with patch.object(psutil.Process, 'children', side_effect=psutil.AccessDenied(proc.pid)):
                campaign._kill_tree(proc.pid)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_identity_tracks_source_data_config_and_dependencies(self):
        from epde_bench import identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'epde'; source.mkdir()
            code = source / 'core.py'; code.write_text('value=1\n')
            data = root / 'data'; (data / 'ode').mkdir(parents=True)
            array = data / 'ode' / 'ode_data.npy'; array.write_bytes(b'123')
            with patch.object(identity, 'REPO_ROOT', root), patch.object(identity, 'DATA_DIR', data):
                first = identity.experiment_identity('ode', 'default', 0, 0)
                (root / 'README.md').write_text('docs change')
                self.assertEqual(first, identity.experiment_identity('ode', 'default', 0, 0))
                code.write_text('value=2\n')
                self.assertNotEqual(first, identity.experiment_identity('ode', 'default', 0, 0))
                second = identity.experiment_identity('ode', 'default', 0, 0)
                array.write_bytes(b'124')
                self.assertNotEqual(second, identity.experiment_identity('ode', 'default', 0, 0))
                third = identity.experiment_identity('ode', 'default', 0, 0)
                self.assertNotEqual(third, identity.experiment_identity('ode', 'default', 0, 0,
                                    {'search': {'evolution': {'training_epochs': 7}}}))
                with patch.object(identity, '_dependency_versions', return_value={'numpy': 'other'}):
                    self.assertNotEqual(third, identity.experiment_identity('ode', 'default', 0, 0))
                json.dumps(first, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
