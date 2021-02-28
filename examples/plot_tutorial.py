"""
A simple tutorial on joint diagonalization
==========================================
We generate some independent signals with different powers.
The signals are then mixed, and their covariances are computed.
Joint diagonalization recovers the mixing matrix.
"""
# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT
import numpy as np
import matplotlib.pyplot as plt

from qndiag import qndiag


rng = np.random.RandomState(0)

###############################################################################
# We take 10 different bins, and 5 sources. We generate random powers for
# each source and bin
n_bins = 10
n_sources = 5
powers = rng.rand(n_bins, n_sources)

###############################################################################
# Next, we generate a random minxing matrix A, and for each bin, we generate
# sources s with the powers above, and observe the signals x = A.dot(s).
# We then store the covariances of the signals

n_samples = 100
A = rng.randn(n_sources, n_sources)
covariances = []
for power in powers:
    s = power[:, None] * rng.randn(n_sources, n_samples)
    x = np.dot(A, s)
    covariances.append(np.dot(x, x.T) / n_samples)

covariances = np.array(covariances)
###############################################################################
# We now use qndiag on 'covariances' to recover the unmixing matrix, i.e the
# inverse of A

B, _ = qndiag(covariances)

unmixing_mixing = np.dot(B, A)
plt.matshow(unmixing_mixing)  # Should be ~ a permutation + scale matrix
plt.show()
