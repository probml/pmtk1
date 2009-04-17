function [bestParams, bestHyperParams, err] = ...
    crossvalidate(X, y, K, hyperParams, trainFn, predFn, errFn)
% K-fold Cross validation

if nargin < 6, predFn = @linregPred; end
if nargin < 7, errFn = @meanSquaredError; end

[n d] = size(X);
% rp = randperm(n);
rp = 1:n;
kappa = floor(n/K);
[trainfolds, testfolds] = Kfold(x, K);
for k = 1:K
  %testidx = rp((k-1)*kappa + 1:k*kappa);
  %trainidx = setdiff(rp(1:K*kappa), testidx);
  trainidx = trainfolds{k};
  testidx = testfolds{k};
  Xtest = X(testidx,:);  ytest = y(testidx);
  Xtrain = X(trainidx, :);  ytrain= y(trainidx);
  for i=1:length(hyperParams)
    params{i} = trainFn(Xtrain, ytrain, hyperParams{i});
    ypred = predFn(Xtest, params{i});
    err(k,i) = errFn(ytest, ypred);
  end
end

errMean = mean(err); % average across rows (cv folds)
errSE = std(err)/sqrt(K);
bestHyperParamsNdx = oneStdErrorRule(errMean, errSE);
bestHyperParams = hyperParams{bestHyperParamsNdx};
bestParams = trainFn(X, y, bestHyperParams);

