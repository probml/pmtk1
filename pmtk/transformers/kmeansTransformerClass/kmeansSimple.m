function [mu, assign, errHist] = kmeansSimple(data, K, varargin)
% data(i,j) = case i, feature j (n * d)
% mu(k,:) = k'th centroid (K * d)
% assign(i) = number in 1:K
% errHist(t) = error vs iteration

[maxIter, thresh, fn] = process_options(...
    varargin, 'maxIter', 100, 'thresh', 1e-3, 'progressFn', []);

[N D] = size(data);
% initialization by picking random points
perm = randperm(N);
mu = data(perm(1:K),:);

converged = false;
iter = 1;
err = inf;
while ~converged && (iter <= maxIter)
  % E step
  % dist(i,k) = squared distance from pixel i to center k
  dist = sqdist(data',mu');
  [junk, assign] = min(dist,[],2);
  olderr = err;
  err = 0;
  for k=1:K
    members = find(assign==k);
    err = err + sum(dist(members,k));
  end
  err  = err/N;
  if ~isempty(fn), feval(fn, data, mu, assign, err, iter); end
  % M step
  for k=1:K
    members = find(assign==k);
    mu(k,:) = mean(data(members,:), 1);
  end
  % Convergence
  converged =  convergenceTest(err, olderr, thresh);
  errHist(iter) = err; %#ok
  iter = iter + 1;
end
if ~converged
  %sprintf('warning: did not converge within %d iterations', maxIter)
end


