%% Learn GGM structure using graphical lasso
% HTF 2e p637


X = load('sachsCtsHTF.txt'); % 7466 x 11
%X = standardize(X);
%X = normalize(X);
lambdas = [36 27 7 0];
lambdas = [0];
N = size(X,1);
S = cov(X);
debug = true;
 
for i=1:length(lambdas)
  lambda = lambdas(i);
  [P] = ggmLassoHtf(S, lambda);
  [PR] = ggmLassoR(S, lambda);
  A = precmatToAdjmat(P);
  Graphlayout('adjMatrix', A, 'undirected', true);
  title(sprintf('lambda=%3.2f', lambda));  
  figure; imagesc(P); title(sprintf('lambda=%3.2f', lambda)); colorbar
  precMat{i} = P;
  
  if debug
    [PR] = ggmLassoR(S, lambda);
    assert(max(P(:)- PR(:)) < 0.01);
  end
end
