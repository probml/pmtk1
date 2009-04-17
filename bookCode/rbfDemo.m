
makePolyData;
xtrain1 = rescaleData(xtrain);
xtest1 = rescaleData(xtest);

K = 10; sigma = 1;
centers = linspace(min(xtrain1), max(xtrain1), K)';

lambda = 0.001; % just for numerical stability
sigmas = [0.05 0.5 50];
doPrint = 0;
folder = 'C:\kmurphy\PML\Figures';
for i=1:length(sigmas)
  sigma = sigmas(i);
  Xtrain = [ones(length(xtrain1),1) rbfKernel(xtrain1, centers, sigma)];
  Xtest = [ones(length(xtest1),1) rbfKernel(xtest1, centers, sigma)];
  
  w = ridgeQR(Xtrain,ytrain,[],[],lambda);
  ypred = Xtest*w;
  figure(2);clf
  scatter(xtrain1,ytrain,'b','filled');
  hold on;
  plot(xtest1, ypred, 'k', 'linewidth', 3);
  title(sprintf('RBF, sigma %f', sigma))
  fname = sprintf('%s/rbfDemoFitSigma%5.3f.eps', folder, sigma)
  if doPrint, print(gcf, '-depsc', fname); end
  
  figure(1);clf
  % visualize the kernel centers
  %for j=1:K, plot(xtest1,Xtest(:,j+1)*2 -10); end
  for j=1:K, plot(xtest1,Xtest(:,j+1)); hold on; end
  fname = sprintf('%s/rbfDemoBasisSigma%5.3f.eps', folder, sigma)
  if doPrint, print(gcf, '-depsc', fname); end
  
  figure(3);clf;
  %imagesc(Xtest(:,2:end)); colormap('gray')
  imagesc(Xtrain(:,2:end)); colormap('gray')
  fname = sprintf('%s/rbfDemoSpySigma%5.3f.eps', folder, sigma)
  if doPrint, print(gcf, '-depsc', fname); end
  pause
end
