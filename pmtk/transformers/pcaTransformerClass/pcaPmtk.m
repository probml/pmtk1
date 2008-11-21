function [B, Z, evals, Xrecon, mu] = pcaPmtk(X, K, method, centerX)
% Principal components analysis
%
% INPUT
% X is n*d - rows are examples, columns are features
% If K is not specified, we use the maximum possible value (min(n,d))
% method - one of {'default', 'eigCov', 'eigGram', 'svd'}
% centerX - true if data should be centered
%
% OUTPUT
% B is d*K (the basis vectors)
% Z is  n*K (the low dimensional representation)
% evals(1:K) is a vector of all eigenvalues 
% Xrecon is n*d - reconstructed from first K
% mu is d*1

if(nargin < 4)
   centerX = true; 
end

[n d] = size(X);
if nargin < 2
  %K = min(n,d); 
  K = rank(X);
end
methodNames = {'eigCov', 'eigGram', 'svd', 'svds'};
if nargin < 3 || isempty(method) || strcmp(method, 'default')
  cost = [d^3 n^3 min(n*d^2, d*n^2)];
  method = methodNames{argmin(cost)};
end


if(centerX)
    mu = mean(X);
    %X = X - repmat(mu, n, 1);
    X = bsxfun(@minus,X,mu);
else
    mu = zeros(1,size(X,2));
end
switch method
 case 'eigCov',
  %[evec, evals] = eig(cov(X));
  [evec, evals] = eig(X'*X/n);
  [evals, perm] = sort(diag(evals), 'descend');
  B = evec(:, perm(1:K));
 case 'eigGram',
  [evecs, evals] = eig(X*X');
  [evals, perm] = sort(diag(evals), 'descend');
  V = evecs(:, perm);
  B = (X'*V)*diag(1./sqrt(evals));
  B = B(:, 1:K);
  evals = evals / n;
 case 'svd',
  [U,S,V] = svd(X,0);
  B = V(:,1:K);
  evals = (1/n)*diag(S).^2;
 case 'svds',
  [U,S,V] = svds(X,K); % slow
  B = V(:,1:K);
  evals = (1/n)*diag(S).^2;
end
Z = X*B;
%Xrecon = Z*B' + repmat(mu, n, 1);
if(nargout > 3)
    Xrecon = bsxfun(@plus,Z*B',mu);
end
