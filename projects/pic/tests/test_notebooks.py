"""Notebook/cache regressions without evolutionary searches."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench import nbcache

class CacheTests(unittest.TestCase):
    def record(self):
        return dict(status='ok', dataset='ode', variant='default', noise=0., seed=0,
                    identity={'code': 'clean', 'data': 'a', 'dependencies': 'b'})

    def test_identity_and_arguments_must_match(self):
        rec = self.record()
        with patch.object(nbcache, 'experiment_identity', return_value=rec['identity']):
            self.assertTrue(nbcache._valid(rec, 'ode', 'default', 0., 0, None))
            for key, value in [('dataset', 'vdp'), ('variant', 'poly'), ('noise', 5.), ('seed', 2)]:
                changed = dict(rec, **{key: value})
                self.assertFalse(nbcache._valid(changed, 'ode', 'default', 0., 0, None))
            for key in ('code', 'data', 'dependencies'):
                changed = dict(rec, identity=dict(rec['identity'], **{key: 'changed'}))
                self.assertFalse(nbcache._valid(changed, 'ode', 'default', 0., 0, None))
            self.assertFalse(nbcache._valid(dict(rec, identity=None), 'ode', 'default', 0., 0, None))

    def test_stale_cache_recomputes_and_stores_new_identity(self):
        import types
        fresh = self.record(); stale = dict(fresh, identity={'code': 'old'})
        fake_runner = types.SimpleNamespace(run_one=lambda *args, **kw: dict(fresh))
        with tempfile.TemporaryDirectory() as tmp, patch.object(nbcache, 'CACHE_DIR', Path(tmp)), \
             patch.object(nbcache, 'experiment_identity', return_value=fresh['identity']), \
             patch.dict(sys.modules, {'epde_bench.runner': fake_runner}):
            out = nbcache.cache_path('ode'); out.parent.mkdir(parents=True)
            out.write_text(json.dumps(stale))
            result = nbcache.cached_run('ode')
            self.assertFalse(result['from_cache'])
            self.assertEqual(json.loads(out.read_text())['identity'], fresh['identity'])
            self.assertTrue(nbcache.cached_run('ode')['from_cache'])

    def test_timeout_does_not_keep_old_success(self):
        import subprocess
        rec = self.record()
        with tempfile.TemporaryDirectory() as tmp, patch.object(nbcache, 'CACHE_DIR', Path(tmp)):
            out = nbcache.cache_path('ode'); out.parent.mkdir(parents=True)
            out.write_text(json.dumps(rec))
            with patch.object(nbcache.subprocess, 'Popen') as proc, \
                 patch('epde_bench.campaign._kill_tree'), \
                 patch.object(nbcache, 'experiment_identity', return_value=rec['identity']):
                proc.return_value.wait.side_effect = subprocess.TimeoutExpired('test', .1)
                _, status, _ = nbcache._compute(('ode', 'default', 0., 0, None), .1)
            self.assertEqual(status, 'timeout')
            self.assertEqual(json.loads(out.read_text())['status'], 'timeout')

class CacheSafetyTests(unittest.TestCase):
    def test_distinct_noise_values_have_distinct_cache_paths(self):
        self.assertNotEqual(nbcache.cache_path('ode', noise=.1234561),
                            nbcache.cache_path('ode', noise=.1234562))

    def test_child_identity_cannot_be_relabelled(self):
        rec = dict(status='ok', dataset='ode', variant='default', noise=0., seed=0,
                   identity={'sha256': 'child-changed'})
        with tempfile.TemporaryDirectory() as tmp, patch.object(nbcache, 'CACHE_DIR', Path(tmp)), \
             patch.object(nbcache, 'experiment_identity', return_value={'sha256': 'parent'}), \
             patch('epde_bench.runner.run_one', return_value=rec):
            with self.assertRaisesRegex(ValueError, 'identity'):
                nbcache.cached_run('ode')
            self.assertFalse(nbcache.cache_path('ode').exists())

    def test_retry_cannot_promote_previous_attempt_output(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(nbcache, 'CACHE_DIR', Path(tmp)), \
             patch.object(nbcache, 'experiment_identity', return_value={'sha256': 'parent'}), \
             patch.object(nbcache.subprocess, 'Popen') as proc, \
             patch('epde_bench.campaign.MEMORY_RETRIES', 1), \
             patch('epde_bench.campaign.MEMORY_WAIT', 0):
            attempts = [0]
            def wait(timeout):
                pending = nbcache.cache_path('ode').with_suffix('.pending.json')
                if attempts[0] == 0:
                    pending.write_text(json.dumps(dict(status='error', identity={'sha256': 'parent'},
                                           traceback='memoryUsageProperties')))
                attempts[0] += 1
            proc.return_value.wait.side_effect = wait
            _, status, _ = nbcache._compute(('ode', 'default', 0., 0, None), .1)
            result = json.loads(nbcache.cache_path('ode').read_text())
            self.assertEqual(status, 'error')
            self.assertIn('no valid record', result.get('error', ''))
            self.assertNotIn('memoryUsageProperties', result.get('traceback', ''))

    def test_pending_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(nbcache, 'CACHE_DIR', Path(tmp)), \
             patch.object(nbcache, 'experiment_identity', return_value={'sha256': 'parent'}), \
             patch.object(nbcache.subprocess, 'Popen') as proc:
            def write_result(timeout):
                out = nbcache.cache_path('ode').with_suffix('.pending.json')
                out.write_text(json.dumps(dict(status='ok', identity={'sha256': 'changed'})))
            proc.return_value.wait.side_effect = write_result
            _, status, _ = nbcache._compute(('ode', 'default', 0., 0, None), .1)
            self.assertEqual(status, 'error')
            self.assertIn('identity', json.loads(nbcache.cache_path('ode').read_text())['error'])

class NotebookTests(unittest.TestCase):
    def test_every_dataset_is_documented(self):
        from epde_bench.datasets import REGISTRY
        notebooks = sorted((PIC / 'notebooks').glob('*.ipynb'))
        self.assertEqual(len(notebooks), 11)
        text = '\n'.join(''.join(c['source']) for p in notebooks
                         for c in json.loads(p.read_text(encoding='utf-8'))['cells'])
        for name, spec in REGISTRY.items():
            missing = not any((PIC / 'data').glob(spec.files.split(',')[0].split(' (')[0].strip()))
            if not missing:
                self.assertIn(f"'{name}'", text)
        self.assertNotIn('_differentiate', text)
        self.assertIn('ballbeam_validation', text)

    def test_prose_has_no_code_names(self):
        """Text for the reader describes; names of branches, files, functions and
        parameters stay in the code cells (they may change)."""
        import re
        pattern = re.compile(r'`[^`]+`|\b\w+\.(?:py|yaml|npy|mat|csv|json)\b|\b\w+\(\)'
                             r'|\b[a-z]+_[a-z0-9_]+\b')
        for path in sorted((PIC / 'notebooks').glob('*.ipynb')):
            for cell in json.loads(path.read_text(encoding='utf-8'))['cells']:
                if cell['cell_type'] == 'markdown':
                    prose = re.sub(r'\$\$.*?\$\$|\$[^$]*\$', '', ''.join(cell['source']), flags=re.S)
                    self.assertEqual(pattern.findall(prose), [], path.name)

if __name__ == '__main__': unittest.main()
