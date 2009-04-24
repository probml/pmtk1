% A simple demo of the spherical, diagonal, and full covariance models for an multivariate normal
%#testPMK
%#author Cody Severinski

setSeed(0); doPlot = true; doPrint = false;
N = 500; d = 3; K = 2;
X = zeros(N,d);
mu = sample(MvnDist(zeros(d,1),5*eye(d)),1); Sigma = sample(InvWishartDist(d+1,50*eye(d)));
X = sample(MvnDist(mu, Sigma), N);
m = MvnDist();
m.plotData(X);
suptitle('Plot of Data');

covstr = {'full', 'diagonal', 'spherical'};
priorstr = {'niw', 'nig', 'nig'};
ncases = length(covstr);
prior = cell(ncases,1);
for j=1:ncases
  prior{j} = mkPrior(MvnDist(), '-data', X, '-prior', priorstr{j}, '-covtype', covstr{j});
  model{j} = MvnDist('-mu', mean(X)', '-Sigma', cov(X), '-prior', prior{j}, '-covtype', covstr{j});
  fitted{j} = fit(model{j}, 'data', X);
  plotDist(fitted{j});
  suptitle(sprintf('Model: %s',covstr{j}));
end


