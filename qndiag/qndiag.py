# Authors: Pierre Ablin <pierre.ablin@inria.fr>
#
# License: MIT

from time import time

import numpy as np


def qndiag(C, B0=None, max_iter=10000, tol=1e-10, lambda_min=1e-4,
           max_ls_tries=10, return_B_list=False, verbose=False):
    """Joint diagonalization of matrices using the quasi-Newton method


    Parameters
    ----------
    C : array-like, shape (n_samples, n_features, n_features)
        Set of matrices to be jointly diagonalized. C[0] is the first matrix,
        etc...

    B0 : None | array-like, shape (n_features, n_features)
        Initial point for the algorithm. If None, a whitener is used.

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
    """
    t0 = time()
    n_samples, n_features, _ = C.shape
    if B0 is None:
        C_mean = np.mean(C, axis=0)
        d, p = np.linalg.eigh(C_mean)
        B = p.T / np.sqrt(d[:, None])
    else:
        B = B0
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
        G = np.mean(D / diagonals[:, :, None], axis=0) - np.eye(n_features)
        g_norm = np.linalg.norm(G)
        if g_norm < tol:
            break
        # Hessian coefficients
        h = np.mean(diagonals[:, None, :] / diagonals[:, :, None], axis=0)
        # Quasi-Newton's direction
        det = h * h.T - 1.
        det[det < lambda_min] = lambda_min  # Regularize
        direction = -(G * h.T - G.T) / det
        # Line search
        success, new_D, new_B, new_loss, direction =\
            linesearch(D, B, direction, current_loss, max_ls_tries)
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


def transform_set(M, D):
    K, N, _ = D.shape
    op = np.zeros((K, N, N))
    for i, d in enumerate(D):
        op[i] = M.dot(d.dot(M.T))
    return op


def loss(B, D):
    n, p, _ = D.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    logdet = -np.linalg.slogdet(B)[1]
    return logdet + 0.5 * np.sum(np.log(diagonals)) / n


def gradient(D):
    n, p, _ = D.shape
    diagonals = np.diagonal(D, axis1=1, axis2=2)
    return np.mean(D / diagonals[:, :, None], axis=0) - np.eye(p)


def linesearch(D, B, direction, current_loss, n_ls_tries):
    n, p, _ = D.shape
    step = 1.
    if current_loss is None:
        current_loss = loss(B, D)
    for n in range(n_ls_tries):
        M = np.eye(p) + step * direction
        new_D = transform_set(M, D)
        new_B = np.dot(M, B)
        new_loss = loss(new_B, new_D)
        if new_loss < current_loss:
            success = True
            break
        step /= 2.
    else:
        success = False
    return success, new_D, new_B, new_loss, step * direction
