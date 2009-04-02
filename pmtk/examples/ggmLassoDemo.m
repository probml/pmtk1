%% Learn GGM structure using graphical lasso
% HTF 2e p637

load('sachsCtsHTF.mat'); % X: 7466 x 11, labels
%X = standardize(X);
%X = normalize(X);
%lambdas = [36 27];
lambdas = [logspace(5,1,5) 0]
C = [];
debug = false;
folder = 'C:\kmurphy\PML\pdfFigures';
doPrint = true;

for i=1:length(lambdas)
  lambda = lambdas(i);
 
  tic
  M = fitStructure(UgmGaussDist, 'data', X, 'lambda', lambda, ...
    'warmStartCov', C, 'method', 'glasso');
  timM(i) = toc;
  %C = M.Sigma; % warm starting gives discrepnacies with R
  P = M.precMat;
  A = M.G.adjMat;
  %drawGraph(M);
  %A = precmatToAdjmat(P, 1e-9);
  Graphlayout('adjMatrix', A, 'undirected', true, 'currentLayout', CircleLayout());
  nzeros(i) = sum(A(:));
  ttl=sprintf('lambda=%3.2f, nnz=%d', lambda, nzeros(i))
  title(ttl)
  %title(sprintf('log(lambda)=%5.3f', log(lambda))); 
  %figure; imagesc(P); title(sprintf('lambda=%3.2f', lambda)); colorbar
  precMat{i} = P;
  if doPrint
    fname = fullfile(folder, sprintf('glassoSachs%d.pdf',nzeros(i))); 
    pdfcrop; print(gcf, '-dpdf', fname);
  end
  
  if debug
  tic
  MR = fitStructure(UgmGaussDist, 'data', X, 'lambda', lambda, ...
    'method', 'glassoR');
  timR(i) = toc;
  PR = MR.precMat;
  AR = MR.G.adjMat;  
  %assert(approxeq(A, AR)) % may be false due to rounding of precmat to 0,1
  assert(max(P(:)- PR(:)) < 0.01);
  end
end
