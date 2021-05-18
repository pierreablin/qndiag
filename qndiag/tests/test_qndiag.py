import numpy as np
from numpy.testing import assert_allclose

import pytest

from qndiag import qndiag


@pytest.mark.parametrize('weights', [None, True])
@pytest.mark.parametrize('ortho', [False, True])
def test_qndiag(weights, ortho):
    n, p = 10, 3
    rng = np.random.RandomState(42)
    diagonals = rng.uniform(size=(n, p))
    A = rng.randn(p, p)  # mixing matrix
    if ortho:
        Ua,_, Va = np.linalg.svd(A, full_matrices=False)
        A = Ua.dot(Va)
    C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset
    if weights:
        weights = rng.rand(n)
    B, _ = qndiag(C, weights=weights,
                  verbose=True, B0=np.eye(p), ortho=ortho)  # use the algorithm
    BA = np.abs(B.dot(A))  # BA Should be a permutation + scale matrix
    if not ortho:
        BA /= np.max(BA, axis=1, keepdims=True)
    BA[np.abs(BA) < 1e-6] = 0.
    assert_allclose(BA[np.lexsort(BA)], np.eye(p))


def test_errors():
    n, p = 10, 2
    rng = np.random.RandomState(42)
    with pytest.raises(ValueError, match='3 dimensions'):
        x = rng.randn(n, p)
        qndiag(x)
    with pytest.raises(ValueError, match='last two dimensions'):
        x = rng.randn(n, p, p + 1)
        qndiag(x)
    with pytest.raises(ValueError, match='only symmetric'):
        x = rng.randn(n, p, p)
        qndiag(x)
    with pytest.raises(ValueError, match='positive'):
        x = rng.randn(n, p, p)
        x += x.swapaxes(1, 2)
        x[0] = np.array([[0, 1], [1, 0]])
        qndiag(x)
