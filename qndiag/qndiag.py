# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT

from time import time

import numpy as np


def qndiag(C, B0=None, weights=None, max_iter=1000, tol=1e-6,
           lambda_min=1e-4, max_ls_tries=10, diag_only=False,
           return_B_list=False, verbose=False):
    """Joint diagonalization of matrices using the quasi-Newton method


    Parameters
    ----------
    C : array-like, shape (n_samples, n_features, n_features)
        Set of matrices to be jointly diagonalized. C[0] is the first matrix,
        etc...

    B0 : None | array-like, shape (n_features, n_features)
        Initial point for the algorithm. If None, a whitener is used.

    weights : None | array-like, shape (n_samples,)
        Weights for each matrix in the loss:
        L = sum(weights * KL(C, C')) / sum(weights).
        No weighting (weights = 1) by default.

    max_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        A positive scalar giving the tolerance at which the
        algorithm is considered to have converged. The algorithm stops when
        |gradient| < tol.

    lambda_min : float, optional
        A positive regularization scalar. Each eigenvalue of the Hessian
        approximation below lambda_min is set to lambda_min.

    max_ls_tries : int, optional
        Maximum number of line-search tries to perform.

    diag_only : bool, optional
        If true, the line search is done by computing only the diagonals of the
        dataset. The dataset is then computed after the line search.
        Taking diag_only = True might be faster than diag_only=False
        when the matrices are large (n_features > 200)

    return_B_list : bool, optional
        Chooses whether or not to return the list of iterates.

    verbose : bool, optional
        Prints informations about the state of the algorithm if True.

    Returns
    -------
    D : array-like, shape (n_samples, n_features, n_features)
        Set of matrices jointly diagonalized

    B : array, shape (n_features, n_features)
        Estimated joint diagonalizer matrix.

    infos : dict
        Dictionnary of monitoring informations, containing the times,
        gradient norms and objective values.

    References
    ----------
    P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm
    for joint diagonalization. Proc. ESANN 2019.
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
    https://hal.archives-ouvertes.fr/hal-01936887v1
    https://arxiv.org/abs/1811.11433
    """
    t0 = time()
    n_samples, n_features, _ = C.shape
    if B0 is None:
        C_mean = np.mean(C, axis=0)
        d, p = np.linalg.eigh(C_mean)
        B = p.T / np.sqrt(d[:, None])
    else:
        B = B0
    if weights is not None:  # normalize
        weights_ = weights / np.mean(weights)
    else:
        weights_ = None

    D = transform_set(B, C)
    current_loss = None

    # Monitoring
    if return_B_list:
        B_list = []
    t_list = []
    gradient_list = []
    loss_list = []
    if verbose:
        print('Running quasi-Newton for joint diagonalization')
        print(' | '.join([name.center(8) for name in
                         ["iter", "obj", "gradient"]]))

    for t in range(max_iter):
        if return_B_list:
            B_list.append(B.copy())
        t_list.append(time() - t0)
        diagonals = np.diagonal(D, axis1=1, axis2=2)
        # Gradient
        G = np.average(D / diagonals[:, :, None], weights=weights_,
                       axis=0) - np.eye(n_features)
        g_norm = np.linalg.norm(G)
        if g_norm < tol * np.sqrt(n_features):  # rescale by identity
            break

        # Hessian coefficients
        h = np.average(diagonals[:, None, :] / diagonals[:, :, None],
                       weights=weights_, axis=0)
        # Quasi-Newton's direction
        det = h * h.T - 1.
        det[det < lambda_min] = lambda_min  # Regularize
        direction = -(G * h.T - G.T) / det

        # Line search
        success, new_D, new_B, new_loss, direction =\
            linesearch(D, B, direction, current_loss, max_ls_tries, diag_only,
                       weights_)
        D = new_D
        B = new_B
        current_loss = new_loss

        # Monitoring
        gradient_list.append(g_norm)
        loss_list.append(current_loss)
        if verbose:
            print(' | '.join([("%d" % (t + 1)).rjust(8),
                              ("%.2e" % current_loss).rjust(8),
                              ("%.2e" % g_norm).rjust(8)]))
    infos = {'t_list': t_list, 'gradient_list': gradient_list,
             'loss_list': loss_list}
    if return_B_list:
        infos['B_list'] = B_list
    return B, infos


def transform_set(M, D, diag_only=False):
    n, p, _ = D.shape
    if not diag_only:
        op = np.zeros((n, p, p))
        for i, d in enumerate(D):
            op[i] = M.dot(d.dot(M.T))
    else:
        op = np.zeros((n, p))
        for i, d in enumerate(D):
            op[i] = np.sum(M * d.dot(M.T), axis=0)
    return op


def loss(B, D, is_diag=False, weights=None):
    n, p = D.shape[:2]
    if not is_diag:
        diagonals = np.diagonal(D, axis1=1, axis2=2)
    else:
        diagonals = D
    logdet = -np.linalg.slogdet(B)[1]
    if weights is None:
        return logdet + 0.5 * np.sum(np.log(diagonals)) / n
    else:
        return logdet + 0.5 * np.sum(weights[:, None] * np.log(diagonals)) / n


def gradient(D, weights=None):
    n, p, _ = D.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    grad = np.average(D / diagonals[:, :, None], weights=weights, axis=0)
    grad.flat[::p + 1] -= 1  # equivalent to - np.eye(p)
    return grad


def linesearch(D, B, direction, current_loss, n_ls_tries, diag_only, weights):
    n, p, _ = D.shape
    step = 1.
    if current_loss is None:
        current_loss = loss(B, D)
    for n in range(n_ls_tries):
        M = np.eye(p) + step * direction
        new_D = transform_set(M, D, diag_only=diag_only)
        new_B = np.dot(M, B)
        new_loss = loss(new_B, new_D, diag_only, weights)
        if new_loss < current_loss:
            success = True
            break
        step /= 2.
    else:
        success = False
    # Compute new value of D if only its diagonal was computed
    if diag_only:
        new_D = transform_set(M, D, diag_only=False)
    return success, new_D, new_B, new_loss, step * direction
