QNDIAG
======

This is a library to run the QNDIAG algorithm [1].
This algorithm exploits a state-of-the-art quasi-Newton strategy for
approximate joint diagonalization of a list of matrices.

Installation
------------

To install qndiag::

	$ pip install qndiag

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import qndiag'

and it should not give any error message.

Quickstart
----------

The easiest way to get started is to copy the following lines of code
in your script:

.. code:: python

    >>> import numpy as np
    >>> from qndiag import qndiag
    >>> n, p = 10, 3
    >>> diagonals = np.random.uniform(size=(n, p))
    >>> A = np.random.randn(p, p)  # mixing matrix
    >>> C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset
    >>> B, _ = qndiag(C)
    >>> print(B.dot(A))  # Should be a permutation + scale matrix  # doctest:+ELLIPSIS

Bug reports
-----------

Use the `github issue tracker <https://github.com/pierreablin/qndiag/issues>`_ to report bugs.

Cite
----

   [1] P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham's algorithm
   for joint diagonalization. Proc. ESANN 2019.
   https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
   https://hal.archives-ouvertes.fr/hal-01936887v1
   https://arxiv.org/abs/1811.11433

API
---

.. toctree::
    :maxdepth: 1

    api.rst
    whats_new.rst
