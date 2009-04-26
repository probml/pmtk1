function [bestModel, LLmean, LLse] = selectCV(models, X, y, Nfolds)
Nx = size(X,1);
randomizeOrder = true;
[trainfolds, testfolds] = Kfold(Nx, Nfolds, randomizeOrder);
Nm = length(models);
LL = zeros(Nx, Nm);
complexity = zeros(1, Nm);
for m=1:Nm % for every model
  complexity(m) = nparams(models{m});
  for f=1:Nfolds % for every fold
    Xtrain = X(trainfolds{f},:);
    Xtest = X(testfolds{f},:);
    if isempty(y)
      tmp = fit(models{m}, 'data', Xtrain);
      ll = logprob(tmp, Xtest);
    else
      ytrain = y(trainfolds{f},:);
      ytest = y(testfolds{f},:);
      tmp = fit(models{m}, 'X', Xtrain, 'y', ytrain);
      ll = logprob(tmp, Xtest, ytest);
    end
    LL(testfolds{f},m) = ll;
  end
end
LLmean = mean(LL,1);
LLse = std(LL,0,1)/Nx;
bestNdx = oneStdErrorRule(-LLmean, LLse, complexity);
%bestNdx = argmax(LLmean);
% Now fit chosen model to all the data
if isempty(y)
  bestModel = fit(models{bestNdx}, X);
else
  bestModel = fit(models{bestNdx}, X, y);
end
end
