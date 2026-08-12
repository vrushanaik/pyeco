import unittest
import importlib.metadata
import numpy as np
import simsimd


class TestSimsimd(unittest.TestCase):

    def test_import(self):
        """Check simsimd can be imported"""
        try:
            import simsimd
        except ImportError:
            self.fail("simsimd is not installed")

    def test_version(self):
        """Verify simsimd version"""
        version = importlib.metadata.version("simsimd")
        assert "6.5.16" in version, f"'6.5.16' not found in version string: {version}"

    def test_cosine_identical_vectors(self):
        """Cosine distance of a vector with itself should be ~0"""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dist = simsimd.cosine(v, v)
        self.assertAlmostEqual(float(dist), 0.0, places=5)

    def test_cosine_orthogonal_vectors(self):
        """Cosine distance of orthogonal vectors should be ~1"""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dist = simsimd.cosine(a, b)
        self.assertAlmostEqual(float(dist), 1.0, places=5)

    def test_sqeuclidean(self):
        """Squared Euclidean distance sanity check"""
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        dist = simsimd.sqeuclidean(a, b)
        self.assertAlmostEqual(float(dist), 25.0, places=4)

    def test_inner_product(self):
        """Inner product matches NumPy dot product"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        expected = float(np.dot(a, b))
        result   = float(simsimd.inner(a, b))
        self.assertAlmostEqual(result, expected, places=3)


if __name__ == "__main__":
    unittest.main()
