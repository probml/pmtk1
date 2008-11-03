%% Estimate mu/sigma by cross validation over a small grid
% See also crossValidation class
mu = 0; sigma = 1;
mtrue = gaussDist(mu, sigma^2);
ntrain = 100;
Xtrain = sample(mtrue, ntrain);
mus = [-10 0 10];
sigmas = [1 1 1];
for i=1:length(sigmas)
    models{i} = gaussDist(mus(i), sigmas(i)^2);
    models{i}.clampedMu = true;
    models{i}.clampedSigma = true;
end
[mestCV, cvMean, cvStdErr] = exhaustiveSearch(models, @(m) cvScore(m, Xtrain))
mestMLE = fit(mtrue, 'data', Xtrain)