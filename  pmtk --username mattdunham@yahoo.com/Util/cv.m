function [trainErr, testErr] = cv(X, y, K, f, varargin)
% K-fold Cross validation
% trainErr(k) = mse on training fold k of the data
% testErr(k) = mse on testing fold k of the data
% For each fold, we call the training/ testing function f as follows:
%  [params, mseTrain, mseTest] = f(Xtrain, ytrain, Xtest, ytest, varargin{:})

[n d] = size(X);
% rp = randperm(n);
rp = 1:n;
kappa = floor(n/K);
for k = 1:K
  testidx = rp((k-1)*kappa + 1:k*kappa);
  trainidx = setdiff(rp(1:K*kappa), testidx);
  Xtest = X(testidx,:);  ytest = y(testidx);
  Xtrain = X(trainidx, :);  ytrain= y(trainidx);
  [params,trainErr(k),testErr(k)] = f(Xtrain, ytrain, Xtest, ytest, varargin{:});
end
