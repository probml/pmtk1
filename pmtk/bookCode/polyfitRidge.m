% based on code code by Romain Thibaux
% (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)

makePolyData;
deg = 14;

if 1
  xtrain1 = rescaleData(xtrain);
  xtest1 = rescaleData(xtest);
  Xtrain = polyBasis(xtrain1, deg);
  Xtest = polyBasis(xtest1, deg);
else
  [Xtrain,m,s] = polyBasis(xtrain, deg);
  Xtest = polyBasis(xtest1, deg, m, s);
end

lambdas = [0 0.00001 0.001];
for lambdai=1:length(lambdas)
  lambda = lambdas(lambdai);
  w = ridgeQR(Xtrain,ytrain,[],[],lambda,0);
  fprintf('%5.2f, ', w); fprintf('\n');
  ypred = Xtest*w;
  figure(2);clf
  folder = 'C:\kmurphy\PML\Figures';
  scatter(xtrain,ytrain,'b','filled');
  hold on;
  plot(xtest, ypred, 'k', 'linewidth', 3);
  hold off
  title(sprintf('degree %d, lambda %d', deg, lambda))
  fname = sprintf('%s/polyfitDemoDeg%dRidge%d.eps', folder, deg, lambda)
  pause
end

