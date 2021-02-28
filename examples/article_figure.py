# Authors: Pierre Ablin <pierreablin@gmail.com>
#
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

import mne
from mne.datasets import sample

from qndiag import qndiag, ajd_pham, gradient, transform_set

rng = np.random.RandomState(0)

fontsize = 5
params = {
      'axes.titlesize': 10,
      'axes.labelsize': 10,
      'font.size': 7,
      'legend.fontsize': 8,
      'xtick.labelsize': fontsize,
      'ytick.labelsize': fontsize,
      'text.usetex': True,
      'ytick.major.pad': '0',
      'ytick.minor.pad': '0'}
plt.rcParams.update(params)


def loss(D):
    n, p, _ = D.shape
    output = 0
    for i in range(n):
        Di = D[i]
        output += np.sum(np.log(np.diagonal(Di))) - np.linalg.slogdet(Di)[1]
    return output / (2 * n)


n, p = 100, 40


f, axes = plt.subplots(2, 3, figsize=(7, 3.04), sharex='col')
expe_str = ['(a)', '(b)', '(c)']
axes = axes.T
for j, (sigma, axe) in enumerate(zip([0., 0.1, 0], axes)):
    if j != 2:  # Synthetic data
        # Generate diagonal matrices
        D = rng.uniform(size=(n, p))
        # Generate a random mixing matrix
        A = rng.randn(p, p)
        C = np.zeros((n, p, p))
        # Generate the dataset
        for i in range(n):
            R = rng.randn(p, p)
            C[i] = np.dot(A, D[i, :, None] * A.T) + sigma ** 2 * R.dot(R.T)
    else:  # Real data
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        X = raw.get_data()
        # Reduce dimension of X by PCA:
        U, D, V = np.linalg.svd(X, full_matrices=False)
        X = V[-p:, :]
        C = np.array([np.dot(x, x.T) for x in np.split(X, n, axis=1)])

    for algo in [qndiag, ajd_pham]:
        _, infos = algo(C, return_B_list=True)
        # For Pham, compute metrics after the algorithm is run
        B_list = infos['B_list']
        infos['gradient_list'] =\
            [np.linalg.norm(gradient(transform_set(B, C))) for B in B_list]
        infos['loss_list'] =\
            [loss(transform_set(B, C)) for B in B_list]
        for i, (to_plot, name, ax) in enumerate(
                                      zip(['loss_list', 'gradient_list'],
                                          ['Objective function',
                                           'Gradient norm'],
                                          axe)):
            ax.loglog(infos['t_list'], infos[to_plot], linewidth=2)
            if i == 1 and j == 1:
                ax.set_xlabel('Time (sec.)')
            if j == 0:
                ax.set_ylabel(name)
            if i == 1:
                art = ax.annotate(expe_str[j], (0, 0), (50, -30),
                                  xycoords='axes fraction',
                                  textcoords='offset points', va='top')
            ax.grid(True)
            ax.yaxis.set_major_locator(LogLocator(numticks=4, subs=(1.,)))
            ax.minorticks_off()

lgd = plt.figlegend(ax.lines, ['Quasi-Newton (proposed)', 'Pham 01'],
                    loc=(0.32, .9), ncol=2, labelspacing=0.)
plt.savefig('expe.pdf', bbox_extra_artists=(art, lgd), bbox_inches='tight')
