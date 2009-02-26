function [w, mseTrain, mseTest, df, gcv] = ridgeSVD(Xtrain, ytrain, Xtest, ytest, lambdas, doStandardize)
% Ridge regression using SVD
% [w, mseTrain, mseTest, df, gcv] = ridgeSVD(Xtrain, ytrain, Xtest, ytest, lambdas)
% The offset term (w0) will be the first element of w.
% If the first column of Xtrain/ Xtest is all 1s, it will be removed. 
% The data will be standardized unless doStandardize = 0
% w(:,i) is the estimate for lambdas(i); first element is w0
% mseTrain(i) = mean squared error on training for lambdas(i) - column
% mseTest(i) = mean squared error on testing
% df(i) is degrees of freedom for lambdas(i)
% gcv(i) is the generalized cross validation estimate


if nargin < 6, doStandardize = 1; end

% we don't want to apply the ridge penalty to the offset term
if all(Xtrain(:,1)==1)
  fprintf('removing column of 1s from Xtrain\n');
  Xtrain = Xtrain(:,2:end);
end
if ~isempty(Xtest) & all(Xtest(:,1)==1)
  fprintf('removing column of 1s from Xtest\n');
  Xtest = Xtest(:,2:end);
end
[n,d] = size(Xtrain);
if doStandardize
  [Xtrain, mu]  = center(Xtrain);
  [Xtrain, s]  = mkUnitVariance(Xtrain);
  if ~isempty(Xtest)
    Xtest = center(Xtest, mu);
    Xtest = mkUnitVariance(Xtest, s);
  end
end

% center input and output, so we can estimate w0 separately
xbar = mean(Xtrain);
XtrainC = Xtrain - repmat(xbar,size(Xtrain,1),1);
ybar = mean(ytrain); 
ytrainC = ytrain-ybar;

[U,D,V] = svd(XtrainC,'econ');
D2 = diag(D.^2);
w = zeros(d+1, length(lambdas));
for i=1:length(lambdas)
  lambda = lambdas(i);
  if lambda==0
    ww = pinv(XtrainC)*ytrainC;
  else
    ww  = V*diag(1./(D2 + lambda))*D*U'*ytrainC;
  end
  w0 = ybar - xbar*ww;
  w(1,i) = w0;
  w(2:d+1,i) = ww;
  df(i) = sum(D2./(D2+lambda));
  
  ypredTrain = [ones(n,1) Xtrain]*w(:,i);
  denom = 1-df(i)/n;
  RSS = sum((ytrain-ypredTrain).^2);
  gcv(i) = RSS/(n-df(i));
  mseTrain(i) = RSS/n;
  if ~isempty(Xtest)
    ntest = size(Xtest, 1);
    ypredTest = [ones(ntest,1) Xtest]*w(:,i);
    mseTest(i) = mean((ypredTest-ytest).^2);
  else
    mseTest = [];
  end
end



