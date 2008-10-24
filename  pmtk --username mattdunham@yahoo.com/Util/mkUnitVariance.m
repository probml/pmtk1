function [X, s] = mkUnitVariance(X, s)
% Make each column of X be variance 1
% ie., sum_i x(i,j)^2 = n (so var(X(:,j))=1)
% If s is omitted, it computed from X and returned for use at test time

if nargin < 2, s = []; end
[n d] = size(X);
if isempty(s)
  s = std(X);
  s(find(s<eps))=1;
end
X = X./repmat(s, [n 1]);
