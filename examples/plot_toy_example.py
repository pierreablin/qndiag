"""
Joint diagonalization on toy data
=================================

"""

# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT

import numpy as np
from qndiag import qndiag

n, p = 10, 3
diagonals = np.random.uniform(size=(n, p))
A = np.random.randn(p, p)  # mixing matrix
C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset
B, _ = qndiag(C)

with np.printoptions(precision=3, suppress=True):
    print(B.dot(A))  # Should be a permutation + scale matrix
