function [beta, C, nll] = logregFitFminunc(y, X, lambda)
% MLE for logistic regression using fminunc
% Based on code by Mark Schmidt
%
% Rows of X contain data
% y(i) = 0 or 1
% lambda is strenght of L2 regularizer
% Returns beta, a row vector, and the asympototic covariance matrix


if nargin < 3, lambda = 0; end
[N p] = size(X);
beta = zeros(p,1);
options = optimset('Display','none','Diagnostics','off','GradObj','on','Hessian','on');
[beta, err] = fminunc(@logregNLLgradHess, beta, options, X, y, lambda);
[nll, g, H] = logregNLLgradHess(beta, X, y, lambda); % H = hessian of neg log lik
C = inv(H); 
