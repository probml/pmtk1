function [beta, C] = logregFitIRLS(y,X, lambda)
% Iteratively reweighted least squares for logistic regression
% Based on code by David Martin 
%
% Rows of X contain data
% y(i) = 0 or 1
% lambda is optional strenght of L2 regularizer
%
% Returns beta, a row vector
% and C, the asymptotic covariance matrix

if nargin < 3, lambda = 0; end
[N,p] = size(X);
beta = zeros(p,1); % initial guess for beta: all zeros
iter = 0;
tol = 1e-4; % termination criterion based on loglik
nll = 0;
while 1
  iter = iter + 1;
  nll_prev = nll;
  [nll, g, H] = logregNLLgradHess(beta, X, y, lambda);
  beta = beta - H\g; % more stable than beta - inv(hess)*deriv
  if abs((nll-nll_prev)/(nll+nll_prev)) < tol, break; end;
end;
[nll, g, H] = logregNLLgradHess(beta, X, y, lambda); % Hessian of neg log lik
C = inv(H);
