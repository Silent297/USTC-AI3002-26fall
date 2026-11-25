"""Independent analytic checks for supplementary diagnostics, no formal data."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gmm_anomaly as diagnosis
from gmm_anomaly import align, group_contributions, score_points, variants


class FrozenDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [1., -1.], [2., 2.], [-1., 0.]])
        self.model = dict(weights=np.array([.2, .8]), means=np.array([[0., 0.], [2., 1.]]),
                          variances=np.array([[.5, 2.], [1., .25]]))

    def test_density_and_responsibility_decomposition_against_scipy(self):
        p, _ = variants(self.model)
        original = {k: v.copy() for k, v in self.model.items()}
        out = score_points(self.X, p['original'], batch=3)
        q = np.stack([np.log(w) + multivariate_normal.logpdf(self.X, m, np.diag(v))
                      for w, m, v in zip(*p['original'])], axis=1)
        np.testing.assert_allclose(out['ll'], logsumexp(q, axis=1), atol=1e-12)
        np.testing.assert_array_equal(out['labels'], q.argmax(1))
        np.testing.assert_allclose(out['weight'] + out['normalization'] + out['mahalanobis'] + out['entropy'],
                                   out['ll'], atol=1e-12)
        for key in original:
            np.testing.assert_array_equal(original[key], self.model[key])

    def test_large_common_offset_uses_direct_difference(self):
        p, _ = variants(self.model)
        w, mu, v = p['original']
        small = score_points(self.X, (w, mu, v))
        big = score_points(self.X + 1e8, (w, mu + 1e8, v), batch=1)
        np.testing.assert_allclose(small['ll'], big['ll'], atol=1e-12)

    def test_frozen_sphere_and_nearest_mean_identity(self):
        p, sphere = variants(self.model)
        self.assertAlmostEqual(sphere, (.2 * 2.5 + .8 * 1.25) / 2)
        for name in p:
            np.testing.assert_array_equal(p[name][1], self.model['means'])
        out = score_points(self.X, p['uniform_shared'])
        expected = ((self.X[:, None] - self.model['means']) ** 2).sum(2).argmin(1)
        np.testing.assert_array_equal(out['labels'], expected)

    def test_permutation_and_changed_contributions(self):
        reference = np.array([0, 0, 1, 1, 2, 2])
        candidate = np.array([2, 2, 0, 0, 1, 1])
        aligned, mapping, table = align(reference, candidate, k=3)
        np.testing.assert_array_equal(aligned, reference)
        np.testing.assert_array_equal(table, np.diag([2, 2, 2]))
        p, _ = variants(self.model)
        normal = score_points(self.X, p['original'])
        special = score_points(self.X, p['uniform_shared'])
        groups = group_contributions(normal, special, np.array([True, False, False, True]))
        total = (special['ll'] - normal['ll']).mean()
        self.assertAlmostEqual(groups['all']['mean_delta_ll'], total)
        self.assertAlmostEqual(sum(groups[g]['contribution_to_total_mean'] for g in ['changed', 'unchanged']), total)
        self.assertAlmostEqual(sum(groups['all']['terms'].values()), total)
        self.assertIsNone(group_contributions(normal, special, np.ones(4, bool))['unchanged']['mean_delta_ll'])

    def test_invalid_variances_rejected(self):
        p, _ = variants(self.model)
        w, mu, v = p['original']
        v[0, 0] = 0
        with self.assertRaises(ValueError):
            score_points(self.X, (w, mu, v))

    def test_formal_orchestration_with_tiny_CPU_fixture(self):
        """Mock only allocation/device selection; execute scoring and metrics.

        This is a CPU toy control-flow test, never a GPU/formal-data result.
        """
        import json
        import uuid
        import torch
        from utils import runtime
        from evaluate import evaluate_partition
        from threadpoolctl import threadpool_limits
        root = Path(__file__).resolve().parents[3] / '_tmp/T0023'
        output = root / ('toy_' + uuid.uuid4().hex)
        X = np.array([[-3., 0.], [-2.9, .1], [-3.1, -.1], [3., 0.], [2.9, .1], [3.1, -.1]], np.float32)
        model = dict(weights=np.array([.4, .6]), means=np.array([[-3., 0.], [3., 0.]]),
                     variances=np.array([[.1, .2], [.2, .1]]))
        labels = np.array([0, 0, 0, 1, 1, 1], np.int32)
        sample = np.arange(6)
        real_configure = runtime.configure
        real_as_tensor = torch.as_tensor
        real_configure('cpu', 'float32', 1)
        metric = evaluate_partition(X, labels, sample, diagnosis.load_implementation())
        rows = [dict(key=k, objective=0., best_restart=0, converged=True,
                     **{n: metric[n] for n in ('silhouette', 'calinski_harabasz', 'davies_bouldin')})
                for k in diagnosis.KEYS]

        def cpu_tensor(*args, **kwargs):
            if str(kwargs.get('device', '')).startswith('cuda'):
                kwargs['device'] = 'cpu'
            return real_as_tensor(*args, **kwargs)

        args = SimpleNamespace(command='run', output=output, p2=Path('.'), pca=None)
        with patch.object(diagnosis, 'load_inputs', return_value=(X, [model, model], [labels, labels],
                                                                 rows, sample, None, [])), \
                patch.dict('os.environ', {'SLURM_JOB_ID': 'CPU_TOY_TEST_ONLY'}), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'get_device_name', return_value='MOCK A100 CPU TOY'), \
                patch.object(torch, 'as_tensor', side_effect=cpu_tensor), \
                patch.object(runtime, 'configure', side_effect=lambda *a: real_configure('cpu', 'float32', 1)), \
                threadpool_limits(limits=1):
            diagnosis.execute(args)
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        self.assertTrue(summary['all_metrics_ok'])
        self.assertTrue(summary['nearest_frozen_mean_identity_verified'])
        self.assertEqual(set(summary['interventions']), {'archived_normal', 'archived_special', *diagnosis.VARIANTS})
        self.assertEqual(summary['interventions']['original']['changed_vs_original_direct64_raw_ids'], 0)
        self.assertTrue((output / 'artifact_hashes.json').is_file())


if __name__ == '__main__':
    unittest.main()
