function [beta,iter] = LassoShooting(X, y, lambda,varargin)
% min_w ||Xw-y||_2^2 + lambda ||w||_1
% Coordinate descent method  ("Shooting"), [Fu, 1998]

%#url http://www.cs.ubc.ca/~schmidtm/Software/lasso.html
%#author Mark Schmidt
%#modified  Kevin Murphy

[n p] = size(X);
[maxIter, optTol, verbose, beta] = ...
    process_options(varargin, 'maxIter',10000, 'optTol',1e-5, 'verbose', 0,...
		    'w0', []);
if isempty(beta)
  beta = (X'*X + sqrt(lambda)*eye(p))\(X'*y);
end
iter = 0;
XX2 = X'*X*2;
Xy2 = X'*y*2;
converged = 0;
while ~converged & (iter < maxIter)
  beta_old = beta;
  for j = 1:p
    cj = Xy2(j) - sum(XX2(j,:)*beta) + XX2(j,j)*beta(j);
    aj = XX2(j,j);
    if cj < -lambda
      beta(j,1) = (cj + lambda)/aj;
    elseif cj > lambda
      beta(j,1) = (cj  - lambda)/aj;
    else
      beta(j,1) = 0;
    end
  end
  iter = iter + 1;
  converged = (sum(abs(beta-beta_old)) < optTol);
end

