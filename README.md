# Quasi-Newton algorithm for joint-diagonalization


![Build](https://github.com/pierreablin/qndiag/workflows/tests/badge.svg)
![Codecov](https://codecov.io/gh/pierreablin/qndiag/branch/master/graph/badge.svg)

## Doc and website

See here for the documentation and examples: https://pierreablin.github.io/qndiag/

## Summary

This Python package contains code for fast joint-diagonalization of a set of
positive definite symmetric matrices. The main function is `qndiag`,
which takes as input a set of matrices of size `(p, p)`, stored as a `(n, p, p)`
array, `C`. It outputs a `(p, p)` array, `B`, such that the matrices
`B @ C[i] @ B.T` (python), i.e. `B * C(i,:,:) * B'` (matlab/octave)
are as diagonal as possible.

## Installation of Python package

To install the package, simply do:

  `$ pip install qndiag`

You can also simply clone it, and then do:

  `$ pip install -e .`

To check that everything worked, the command

  `$ python -c 'import qndiag'`

should not return any error.

## Use with Python

Here is a toy example (also available at `examples/toy_example.py`)

```python
import numpy as np
from qndiag import qndiag

n, p = 10, 3
diagonals = np.random.uniform(size=(n, p))
A = np.random.randn(p, p)  # mixing matrix
C = np.array([A.dot(d[:, None] * A.T) for d in diagonals])  # dataset


B, _ = qndiag(C)  # use the algorithm

print(B.dot(A))  # Should be a permutation + scale matrix
```

## Use with Matlab or Octave

See `qndiag.m` and `toy_example.m` in the folder `matlab_octave`.

## Cite

If you use this code please cite:

    P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham’s algorithm
    for joint diagonalization. Proc. ESANN 2019.
    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
    https://hal.archives-ouvertes.fr/hal-01936887v1
    https://arxiv.org/abs/1811.11433
