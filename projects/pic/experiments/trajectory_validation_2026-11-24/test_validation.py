import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from settings import ROOT, limit_threads

if __name__ == '__main__':
    limit_threads()
from noise_suite import directory
from compare import read_archive
from kdv_measurement import repeatable_column
from score_archive import ArchiveScorer, load_recording
from validate_trajectories import draw, problems
import numpy as np


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem, cls.trajectories = problems('ode')
        cls.dest = directory('ode', 923701, 'correlated')
        cls.scorer = ArchiveScorer(cls.problem, read_archive(cls.dest))

    def test_frozen_training_replay_and_record_order(self):
        raw = draw(self.problem, 923701, 'correlated')
        noise = draw(self.trajectories[0], 933701, 'correlated')
        s = self.scorer
        np.testing.assert_allclose(s.trajectory_error([raw])[s.valid], s.train[s.valid], rtol=1e-6, atol=2e-9)
        np.testing.assert_allclose(s.trajectory_error([raw, noise]), s.trajectory_error([noise, raw]))
        with self.assertRaises(ValueError):
            s.trajectory_error([])

    def test_raw_fold_separation(self):
        s = self.scorer
        fields = draw(self.problem, 923701, 'correlated')
        original = s.matrix(fields)
        n, = self.problem.shape
        split = (n-14)//2
        for side in ('left','right'):
            changed = {k: v.copy() for k,v in fields.items()}
            first = np.arange(len(original))*s.plan['ss'][0]
            if side == 'left':
                changed['u'][:split] += 10
                kept = first >= split+14
            else:
                changed['u'][split+14:] += 10
                kept = first+2*(s.plan['ms'][0]+s.plan['cut'][0]) < split
            self.assertTrue(kept.any())
            np.testing.assert_allclose(s.matrix(changed)[kept], original[kept], atol=1e-10)
        self.assertIsNone(s.blocked_error(original, n))

    def test_multisample_pareto_tradeoffs_ties_and_empty_pool(self):
        s = ArchiveScorer.__new__(ArchiveScorer)
        s.systems = np.arange(7)[:, None]
        s.valid = np.ones(7, dtype=bool)
        s.stability = np.array([.3,.2,.2,.9,.2,.1,.1])
        errors = np.array([[.1,.4,.8,.9,.4,np.inf,2.1], [.9,.4,.1,.9,.4,.1,.1]])
        selected = s.select_multisample(errors)
        self.assertEqual(selected, {'front':[0,1,2,4]})
        self.assertEqual(s.select(errors.mean(axis=0))['front'], [1,4])
        self.assertEqual(selected, s.select_multisample(errors[::-1]))
        self.assertEqual(selected, s.select_multisample(np.tile(errors,(2,1))))
        self.assertEqual(s.select_multisample(errors[:1])['front'], s.select(errors[0])['front'])
        self.assertEqual(s.select_multisample(np.full_like(errors,2.)), {'front':[]})
        # Every trade-off survives even when the front has more than six members.
        s.stability[:] = .2
        tradeoff = np.linspace(.1,.9,7)
        self.assertEqual(s.select_multisample(np.stack([tradeoff, tradeoff[::-1]])), {'front':list(range(7))})
        # The first two systems trade off across equations even on a single recording.
        s.systems = np.array([[0,2], [1,1], [3,3]])
        self.assertEqual(s.select_multisample(errors[:1])['front'], [0,1])

    def test_cli_frozen_recordings(self):
        s = self.scorer
        with TemporaryDirectory() as temp:
            paths, fields = [], []
            for i in range(2):
                fields.append(draw(self.trajectories[i], 933701+10000*i,'correlated'))
                path = Path(temp)/f'{i}.npz'
                np.savez(path, axis0=s.axes[0], **fields[-1])
                paths.append(path)
            out = subprocess.run([sys.executable,str(ROOT/'score_archive.py'),str(self.dest),*map(str,paths)],
                                 capture_output=True, text=True, check=True)
            result = json.loads(out.stdout)
            by_record = s.recording_errors(fields)
            joint = s.select_multisample(by_record)
            self.assertEqual(set(result),{'front','multisample','mean_error_front','members','recordings'})
            self.assertEqual(result['front'],joint['front'])
            self.assertEqual(result['multisample'],joint)
            self.assertEqual(result['mean_error_front'],s.select(by_record.mean(axis=0))['front'])
            self.assertEqual([item['system_index'] for item in result['members']],joint['front'])
            self.assertEqual(result['recordings'],list(map(str,paths)))
            for item in result['members']:
                np.testing.assert_allclose(item['wape_by_recording'],by_record[:,s.systems[item['system_index']]])
            np.savez(paths[0],axis0=s.axes[0][::-1],**fields[0])
            with self.assertRaises(ValueError):
                load_recording(paths[0],s)

    def test_dispersion_cross_validation(self):
        shape = (17,7)
        u = np.tile(np.linspace(-1,1,shape[1]),shape[0])
        smooth, _, _ = repeatable_column(u, u, shape)
        np.testing.assert_allclose(smooth, u, atol=1e-11)
        v = np.arange(u.size)%5-u
        a, dt, dx = repeatable_column(u,v,shape)
        b, dt2, dx2 = repeatable_column(v,u,shape)
        self.assertEqual((dt,dx),(dt2,dx2))
        np.testing.assert_allclose(a,b)


if __name__ == '__main__':
    unittest.main()
