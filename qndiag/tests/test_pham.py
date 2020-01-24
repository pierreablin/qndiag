import numpy as np
from numpy.testing import assert_array_equal

from qndiag import ajd_pham


def test_ajd():
    """Test approximate joint diagonalization."""
    n, p = 10, 3
    rng = np.random.RandomState(42)
    diagonals = rng.uniform(size=(n, p))
    A = rng.randn(p, p)  # mixing matrix
    C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset
    B, _ = ajd_pham(C)
    BA = np.abs(B.dot(A))  # BA Should be a permutation + scale matrix
    BA /= np.max(BA, axis=1, keepdims=True)
    BA[np.abs(BA) < 1e-8] = 0.
    assert_array_equal(BA[np.lexsort(BA)], np.eye(p))
