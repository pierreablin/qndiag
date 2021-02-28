% Authors: Pierre Ablin <pierreablin@gmail.com>
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

[D, B] = qndiag(C, 'max_iter', 100);

B * A  % Should be a permutation + scale matrix

weights = rand(n, 1);

[D, B] = qndiag(C, 'max_iter', 100, 'weights', weights);

B * A  % Should be a permutation + scale matrix
