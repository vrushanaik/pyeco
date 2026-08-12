import unittest
import importlib.metadata
import numpy as np
from hmmlearn import hmm


class TestHmmlearn(unittest.TestCase):

    def test_import(self):
        """Check hmmlearn can be imported"""
        try:
            from hmmlearn import hmm
        except ImportError:
            self.fail("hmmlearn is not installed")

    def test_version(self):
        """Verify hmmlearn version"""
        version = importlib.metadata.version("hmmlearn")
        assert "0.3.3" in version, f"'0.3.3' not found in version string: {version}"

    def test_gaussian_hmm_fit_and_score(self):
        """Fit a GaussianHMM and verify log-likelihood is finite"""
        np.random.seed(0)
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=10)
        model.startprob_ = np.array([0.6, 0.4])
        model.transmat_  = np.array([[0.7, 0.3], [0.4, 0.6]])
        model.means_     = np.array([[0.0], [5.0]])
        model.covars_    = np.array([[1.0], [1.0]])
        obs, _ = model.sample(50)
        fitted = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=20, random_state=0)
        fitted.fit(obs)
        score = fitted.score(obs)
        self.assertFalse(np.isnan(score))
        self.assertFalse(np.isinf(score))

    def test_viterbi_decode_length(self):
        """Decoded state sequence must match observation length"""
        np.random.seed(1)
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=10)
        model.startprob_ = np.array([0.5, 0.5])
        model.transmat_  = np.array([[0.8, 0.2], [0.3, 0.7]])
        model.means_     = np.array([[1.0], [4.0]])
        model.covars_    = np.array([[0.5], [0.5]])
        obs, _ = model.sample(30)
        _, states = model.decode(obs, algorithm="viterbi")
        self.assertEqual(len(states), 30)
        self.assertTrue(set(states).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
