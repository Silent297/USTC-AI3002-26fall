"""未补完和意外失败时也应留下可解释的状态，不能导出最终模型。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import main


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="q2 workflow ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "features.npz"
        np.savez(self.input, X=np.arange(24, dtype=np.float32).reshape(8, 3))
        config = json.loads((PACKAGE / "config.json").read_text())
        config.update(candidate_k=[2, 3], n_init=1, silhouette_sample_size=6, threads=1)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config))
        self.output = self.root / "output"

    def run_experiment(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return main.run(self.config, self.output, "cpu", input_path=self.input,
                            development=True, **options)

    def test_unfinished_submission_saves_status_without_final_output(self):
        # 补全 submission.py 后无法用真实实现复现“未实现”状态；
        # 改用显式抛 NotImplementedError 的临时实现，原断言保持不变。
        implementation = self.root / "unfinished.py"
        implementation.write_text(
            "def init_centers_kmeanspp(*args):\n"
            "    raise NotImplementedError('TODO 1')\n"
            "def assign_clusters(*args):\n"
            "    raise NotImplementedError('TODO 2')\n"
            "def update_centers(*args):\n"
            "    raise NotImplementedError('TODO 3')\n"
            "def gmm_estep(*args):\n"
            "    raise NotImplementedError('TODO 4')\n"
            "def gmm_mstep(*args):\n"
            "    raise NotImplementedError('TODO 5')\n"
            "def silhouette_score(*args):\n"
            "    raise NotImplementedError('TODO E1')\n"
            "def calinski_harabasz_score(*args):\n"
            "    raise NotImplementedError('TODO E2')\n"
            "def davies_bouldin_score(*args):\n"
            "    raise NotImplementedError('TODO E3')\n"
        )
        self.assertEqual(self.run_experiment(implementation_path=implementation), 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "not_implemented")
        self.assertFalse((self.output / "clusters.npy").exists())
        self.assertTrue((self.output / "candidates/kmeans_k2_seed17/fit.json").exists())

    def test_unexpected_student_error_is_recorded(self):
        implementation = self.root / "broken.py"
        implementation.write_text('def init_centers_kmeanspp(*args):\n    raise RuntimeError("student error")\n')
        with self.assertRaisesRegex(RuntimeError, "student error"):
            self.run_experiment(implementation_path=implementation)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "framework_error")
        self.assertIn("RuntimeError", manifest["error"])
        self.assertFalse((self.output / "clusters.npy").exists())

    def test_existing_results_are_not_overwritten(self):
        self.output.mkdir()
        existing = self.output / "keep.txt"
        existing.write_text("previous experiment")
        with self.assertRaisesRegex(ValueError, "output directory must be empty"):
            self.run_experiment()
        self.assertEqual(existing.read_text(), "previous experiment")


if __name__ == "__main__":
    unittest.main()
