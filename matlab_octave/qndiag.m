function [D, B, infos] = qndiag(C, varargin)
% Joint diagonalization of matrices using the quasi-Newton method
%
% The algorithm is detailed in:
%
%    P. Ablin, J.F. Cardoso and A. Gramfort. Beyond Pham’s algorithm
%    for joint diagonalization. Proc. ESANN 2019.
%    https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2019-119.pdf
%    https://hal.archives-ouvertes.fr/hal-01936887v1
%    https://arxiv.org/abs/1811.11433
%
% The function takes as input a set of matrices of size `(p, p)`, stored as
% a `(n, p, p)` array, `C`. It outputs a `(p, p)` array, `B`, such that the
% matrices `B * C(i,:,:) * B'` are as diagonal as possible.
%
% There are several optional parameters which can be provided in the
% varargin variable.
%
% Optional parameters:
% --------------------
% 'm'                         Initial point for the algorithm.
%                             If absent, a whitener is used.
%
% 'maxiter'                   (int) Maximum number of iterations to perform.
%                             Default : 100
%
% 'tol'                       (float) A positive scalar giving the tolerance at
%                             which the algorithm is considered to have converged.
%                             The algorithm stops when  |gradient| < tol.
%                             Default : 1e-10
%
% lambda_min                  (float) A positive regularization scalar. Each
%                             eigenvalue of the Hessian approximation below
%                             lambda_min is set to lambda_min.
%
% max_ls_tries                (int), Maximum number of line-search tries to
%                             perform.
%
% return_B_list               (bool) Chooses whether or not to return the list
%                              of iterates.
%
% verbose                     (bool) Prints informations about the state of the
%                             algorithm if True.
%
% Returns
% -------
% D : Set of matrices jointly diagonalized
% B : Estimated joint diagonalizer matrix.
% infos : structure containing monitoring informations, containing the times,
%     gradient norms and objective values.
%
% Example:
% --------
%
%  [D, B] = qndiag(C, 'maxiter', 100, 'tol', 1e-5)
%
% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%
% License: MIT

% First tests

if nargin == 0,
    error('No signal provided');
end

if length(size(C)) ~= 3,
    error('Input C should be 3 dimensional');
end

if ~isa (C, 'double'),
  fprintf ('Converting input data to double...');
  X = double(X);
end

% Default parameters

C_mean = squeeze(mean(C, 1));
[p, d] = eigs(C_mean);
p = fliplr(p);
d = flip(diag(d));
B = p' ./ repmat(sqrt(d), 1, size(p, 1));

max_iter = 100;
tol = 1e-10;
lambda_min = 1e-4;
max_ls_tries = 10;
return_B_list = false;
verbose = false;

% Read varargin

if mod(length(varargin), 2) == 1,
    error('There should be an even number of optional parameters');
end

for i = 1:2:length(varargin)
    param = lower(varargin{i});
    value = varargin{i + 1};
    switch param
        case 'B0'
            B = value;
        case 'max_iter'
            max_iter = value;
        case 'tol'
            tol = value;
        case 'lambda_min'
            lambda_min = value;
        case 'max_ls_tries'
            max_ls_tries = value;
        case 'return_B_list'
            return_B_list = value;
        case 'verbose'
            verbose = value;
        otherwise
            error(['Parameter ''' param ''' unknown'])
    end
end

[n_samples, n_features, ~] = size(C);

D = transform_set(B, C, false);
current_loss = NaN;

% Monitoring
if return_B_list
    B_list = []
end

t_list = [];
gradient_list = [];
loss_list = [];

if verbose
    print('Running quasi-Newton for joint diagonalization');
    print('iter | obj | gradient');
end

for t=1:max_iter
    if return_B_list
        B_list(k) = B;
    end

    diagonals = zeros(n_samples, n_features);
    for k=1:n_samples
        diagonals(k, :) = diag(squeeze(D(k, :, :)));
    end

    % Gradient
    G = squeeze(mean(bsxfun(@rdivide, D, ...
                            reshape(diagonals, n_samples, n_features, 1)), ...
                     1)) - eye(n_features);
    g_norm = norm(G);
    if g_norm < tol
        break
    end

    % Hessian coefficients
    h = mean(bsxfun(@rdivide, ...
                    reshape(diagonals, n_samples, 1, n_features), ...
                    reshape(diagonals, n_samples, n_features, 1)), 1);
    h = squeeze(h);

    % Quasi-Newton's direction
    dt = h .* h' - 1.;
    dt(dt < lambda_min) = lambda_min;  % Regularize
    direction = -(G .* h' - G') ./ dt;

    % Line search
    [success, new_D, new_B, new_loss, direction] = ...
        linesearch(D, B, direction, current_loss, max_ls_tries);
    D = new_D;
    B = new_B;
    current_loss = new_loss;

    % Monitoring
    gradient_list(t) = g_norm;
    loss_list(t) = current_loss;
    if verbose
        print(sprintf('%d  - %.2e - %.2e', t, current_loss, g_norm))
    end
end

infos = struct();
infos.t_list = t_list;
infos.gradient_list = gradient_list;
infos.loss_list = loss_list;

if return_B_list
    infos.B_list = B_list
end

end

function [op] = transform_set(M, D, diag_only)
    [n, p, ~] = size(D);
    if ~diag_only
        op = zeros(n, p, p);
        for k=1:length(D)
            op(k, :, :) = M * squeeze(D(k, :, :)) * M';
        end
    else
        op = zeros(n, p);
        for k=1:length(D)
            op(k, :) = sum(M .* (squeeze(D(k, :, :)) * M'), 1);
        end
    end
end

function [v] = slogdet(A)
    v = log(abs(det(A)));
end

function [out] = loss(B, D, is_diag)
    [n, p, ~] = size(D);
    if ~is_diag
        diagonals = zeros(n, p);
        for k=1:n
            diagonals(k, :) = diag(squeeze(D(k, :, :)));
        end
    else
        diagonals = D
    end
    logdet = -slogdet(B);
    out = logdet + 0.5 * sum(log(diagonals(:))) / n;
end

function [success, new_D, new_B, new_loss, delta] = linesearch(D, B, direction, current_loss, n_ls_tries)
    [n, p, ~] = size(D);
    step = 1.;
    if current_loss == NaN
        current_loss = loss(B, D, false);
    end
    success = false;
    for n=1:n_ls_tries
        M = eye(p) + step * direction;
        new_D = transform_set(M, D, true);
        new_B = M * B;
        new_loss = loss(new_B, new_D, true);
        
        if new_loss < current_loss
            success = true;
            break
        end
        step = step / 2;
    end
    new_D = transform_set(M, D, false);
    delta = step * direction;
end
