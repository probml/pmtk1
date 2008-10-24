function [mu, assign, errHist] = kmeansSimple(data, K, varargin)
% data(i,j) = case i, feature j (n * d)
% mu(k,:) = k'th centroid (K * d)
% assign(i) = number in 1:K
% errHist(t) = error vs iteration

[maxIter, thresh, fn] = process_options(...
    varargin, 'maxIter', 100, 'thresh', 1e-3, 'fn', []);

[N D] = size(data);
% initialization by picking random points
perm = randperm(N);
mu = data(perm(1:K),:);

converged = 0;
iter = 1;
err = 0;
while ~converged & (iter <= maxIter)
  % dist(i,k) = squared distance from pixel i to center k
  dist = sqdist(data',mu');
  [junk, assign] = min(dist,[],2);
  if ~isempty(fn), feval(fn, data, mu, assign, err, iter, converged); end
  olderr = err;
  err = 0;
  for k=1:K
    members = find(assign==k);
    mu(k,:) = mean(data(members,:), 1);
    err = err + sum(dist(members,k));
  end
  err  = err/N;
  converged =  convergenceTest(err, olderr, thresh);
  errHist(iter) = err;
  iter = iter + 1;
end
if ~converged
  sprintf('warning: did not converge within %d iterations', maxIter)
end


