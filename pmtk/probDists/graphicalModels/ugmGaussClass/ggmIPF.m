function [precMat, covMat] = ggmIPF(S, G, varargin)
% Iterative proportional fitting algorithm for Gaussian graphical models
% S is the empirical covariance matrix (S=cov(data))
% G is the adjacency matrix of the graph

[maxIter, tol, verbose] = process_options(...
  varargin, 'maxIter', 100, 'tol', 1e-5, 'verbose', false);

d = length(S);
precMat = eye(d);
covMat = eye(d);
tic;
clqs = maximalCliques(G);
if  verbose
  fprintf('finding max cliques took %5.4f seconds\n', toc)
end
nclqs = length(clqs);

% precompute for speed
for c=1:nclqs
  clqc = clqs{c};
  Sinv{c} = inv(S(clqc, clqc));
end

done = false;
iter = 1;
while ~done
  for c=1:nclqs
    clqc = clqs{c};
    E = Sinv{c} - inv(covMat(clqc, clqc));
    oldPrecMat = precMat;
    precMat(clqc, clqc) = precMat(clqc, clqc) + E;
    covMat = inv(precMat);
    change = norm(precMat - oldPrecMat, inf);
    if (change < tol) | (iter > maxIter), done = true; end
  end
  if verbose, fprintf('iter %d, change %5.4f\n', iter, change); end
  iter = iter + 1;
end
end



