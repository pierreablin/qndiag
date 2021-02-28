# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT
from time import time

import numpy as np
from .qndiag import transform_set


def ajd_pham(X, tol=1e-14, max_iter=1000, return_B_list=False, verbose=False):
    """
    This function comes from mne-python/decoding/csp.py
    Approximate joint diagonalization based on Pham's algorithm.

    This is a direct implementation of the PHAM's AJD algorithm [1].

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_channels)
        A set of covariance matrices to diagonalize.
    tol : float, defaults to 1e-6
        The tolerance for stoping criterion.
    max_iter : int, defaults to 1000
        The maximum number of iteration to reach convergence.

    Returns
    -------
    V : ndarray, shape (n_channels, n_channels)
        The diagonalizer.
    D : ndarray, shape (n_epochs, n_channels, n_channels)
        The set of quasi diagonal matrices.

    References
    ----------
    .. [1] Pham, Dinh Tuan. "Joint approximate diagonalization of positive
           definite Hermitian matrices." SIAM Journal on Matrix Analysis and
           Applications 22, no. 4 (2001): 1136-1152.

    """
    # Adapted from http://github.com/alexandrebarachant/pyRiemann
    t0 = time()
    n_epochs = X.shape[0]
    C_mean = np.mean(X, axis=0)
    d, p = np.linalg.eigh(C_mean)
    V = p.T / np.sqrt(d[:, None])
    X = transform_set(V, X)
    # Reshape input matrix
    A = np.concatenate(X, axis=0).T
    # Init variables
    n_times, n_m = A.shape
    epsilon = n_times * (n_times - 1) * tol
    t_list = []
    if return_B_list:
        B_list = []
    for it in range(max_iter):
        t_list.append(time() - t0)
        if return_B_list:
            B_list.append(V.copy())
        decr = 0
        for ii in range(1, n_times):
            for jj in range(ii):
                Ii = np.arange(ii, n_m, n_times)
                Ij = np.arange(jj, n_m, n_times)

                c1 = A[ii, Ii]
                c2 = A[jj, Ij]
                c3 = A[ii, Ij]

                g12 = np.mean(c3 / c1)
                g21 = np.mean(c3 / c2)

                omega21 = np.mean(c1 / c2)
                omega12 = np.mean(c2 / c1)
                omega = np.sqrt(omega12 * omega21)

                tmp = np.sqrt(omega21 / omega12)
                tmp1 = (tmp * g12 + g21) / (omega + 1)
                tmp2 = (tmp * g12 - g21) / max(omega - 1, 1e-9)

                h12 = tmp1 + tmp2
                h21 = np.conj((tmp1 - tmp2) / tmp)

                decr += n_epochs * (g12 * np.conj(h12) + g21 * h21) / 2.0

                tmp = 1 + 1.j * 0.5 * np.imag(h12 * h21)
                tmp = np.real(tmp + np.sqrt(tmp ** 2 - h12 * h21))
                tau = np.array([[1, -h12 / tmp], [-h21 / tmp, 1]])

                A[[ii, jj], :] = np.dot(tau, A[[ii, jj], :])
                tmp = np.c_[A[:, Ii], A[:, Ij]]
                tmp = np.reshape(tmp, (n_times * n_epochs, 2), order='F')
                tmp = np.dot(tmp, tau.T)

                tmp = np.reshape(tmp, (n_times, n_epochs * 2), order='F')
                A[:, Ii] = tmp[:, :n_epochs]
                A[:, Ij] = tmp[:, n_epochs:]
                V[[ii, jj], :] = np.dot(tau, V[[ii, jj], :])
        if verbose:
            print('Iteration %d, decr : %.2e' % (it, decr))
        if decr < epsilon:
            break

    infos = {'t_list': t_list}
    if return_B_list:
        infos['B_list'] = B_list
    return V, infos
