% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%
% License: MIT

clc; clear

rand('seed', 42);
randn('seed', 42);

n = 10;
p = 3;

diagonals = rand(n, p);
A = randn(p, p);  % mixing matrix

C = zeros(n, p, p);
for k=1:n
    C(k, :, :) = A * diag(diagonals(k, :)) * A';
end

[D, B] = qndiag(C, 'max_iter', 10);

B * A  % Should be a permutation + scale matrix
