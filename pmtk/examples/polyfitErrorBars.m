%% Illustrate that predictive interval is larger if sigma is unknown
%% Linear Regression with a Polynomial Basis Expansion and Error Bars

function linregPolyFitErrorBars()
for deg=1:2
  helper('mvnIG',deg);
  helper('mvn',deg);
end
end

function helper(prior, deg)
[xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
  'sampling', 'thibaux'); %#ok
T =  ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg, false)});
m = LinregConjugate('-transformer', T, '-lambda', 1e-3);
if strcmp(prior, 'mvn')
  m.sigma2 = sigma2;
end
m = fit(m, DataTable(xtrain, ytrain));
ypredTest = predict(m, xtest);
figure;
hold on;
h = plot(xtest, mean(ypredTest),  'k-', 'linewidth', 3);
%scatter(xtrain,ytrain,'r','filled');
h = plot(xtrain,ytrain,'ro','markersize',14,'linewidth',3);
NN = length(xtest);
ndx = 1:20:NN; % plot subset of errorbars
sigma = sqrt(var(ypredTest));
mu = mean(ypredTest);
[lo,hi] = credibleInterval(ypredTest);
if strcmp(prior, 'mvn')
  % predictive distribution is a Gaussian
  assert(approxeq(2*1.96*sigma, hi-lo))
end
hh=errorbar(xtest(ndx), mu(ndx), mu(ndx)-lo(ndx), hi(ndx)-mu(ndx));
%set(gca,'ylim',[-10 15]);
set(gca,'xlim',[-1 21]);
title(sprintf('degree %d,  prior %s', deg, prior))
end
