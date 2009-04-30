%% Plot performance of linear regression vs sample size
clear;
degrees = [1,2,30];

for d=1:numel(degrees)
  deg = degrees(d);
  lambda = 1e-3;
  ns = linspace(10,200,10);
  Nns = length(ns);
  testMse = zeros(1, Nns); trainMse = zeros(1, Nns);
  for i=1:length(ns)
    n=ns(i);
    %[xtrain, ytrain, xtest, ytest] = makePolyData(n);
    [xtrain, ytrain, xtest, ytestNoiseFree, ytest,sigma2] = polyDataMake('n', n, 'sampling', 'thibaux');
    %m = LinregDist();
    m = LinregL2('-lambda', lambda);
    m.transformer =  ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg)});
    %m = fit(m, 'X', xtrain, 'y', ytrain, 'prior', 'L2', 'lambda', lambda);
    Dtrain  = DataTable(xtrain, ytrain);
    Dtest = DataTable(xtest, ytest);
    m = fit(m, Dtrain);
    testMse(i) = mean(squaredErr(m, Dtest));
    trainMse(i) = mean(squaredErr(m, Dtrain));
  end
  figure;
  hold on
  ndx = ns;
  plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
  plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
  legend('train', 'test')
  ylabel('mse')
  xlabel('size of training set')
  title(sprintf('truth=degree 2, model = degree %d', deg));
  set(gca,'ylim',[0 22]);
  line([0 max(ns)],[sigma2 sigma2],'color','k','linewidth',3);
end


