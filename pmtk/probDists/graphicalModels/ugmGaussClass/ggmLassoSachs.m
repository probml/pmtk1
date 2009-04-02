%% Learn GGM structure using graphical lasso on flow cytometry data
% HTF 2e p637

load('sachsCtsHTF.mat'); % 7466 x 11
%X = standardize(X);
%X = normalize(X);
%lambdas = [5000 1000 500 100 0];
lambdas = [logspace(5,0,5) 0]
S = cov(X);
debug = true;
folder = 'C:\kmurphy\PML\pdfFigures';
doPrint = false;

for i=1:length(lambdas)
  lambda = lambdas(i);
  [P] = ggmLassoHtf(S, lambda);
  A = precmatToAdjmat(P, 1e-9);
  %M = fitStructure(UgmGaussDist, 'data', X, 'lambda', lambda);
  %A = M.G.adjMat;
  nnzeros(i) = sum(A(:));
  ttl=sprintf('lambda=%3.2f, nnz=%d', lambda, nnzeros(i))
  Graphlayout('adjMatrix', A, 'undirected', true, ...
    'nodeLabels', labels, 'currentLayout', CircleLayout());
  title(ttl)
  %figure; imagesc(P); colorbar; title(ttl);
  precMat{i} = P;
  if doPrint
    fname = fullfile(folder, sprintf('glassoSachs%d.pdf',nnzeros(i))); 
    pdfcrop; print(gcf, '-dpdf', fname);
  end
  
  if debug
    tol = 1e-3;
    [PR] = ggmLassoR(S, lambda);
    assert(max(P(:)- PR(:)) < tol);
    [PM] = ggmLassoCoordDescQP(S, lambda);
    assert(max(P(:)- PM(:)) < tol);
  end
end
