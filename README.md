# Quasi-Newton algorithm for joint-diagonalization

## Summary

This Python package contains code for fast joint-diagonalization of a set of positive definite symmetric matrices. The main function is `qndiag.qndiag()`, which takes as input a set of matrices of size `(p, p)`, stored as a `(n, p, p)` array, `C`. It outputs a `(p, p)` array, `B`, such that the matrices `B.dot(C[i]).dot(B.T)` are as diagonal as possible.


## Installation
To install the package, simply clone it, and then do:

  `$ pip install -e.`

To check that everything worked, the command

  `$ python -c 'import tlnmf'`

should not return any error.

## Use
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
