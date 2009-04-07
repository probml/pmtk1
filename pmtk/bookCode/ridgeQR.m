function [w, mseTrain, mseTest, ypredTrain, ypredTest] = ...
    ridgeQR(Xtrain, ytrain, Xtest, ytest, Lambda, doStandardize)
% Ridge regression using QR decomposition
% The offset term (w0) will be the first element of w.
% If the first column of Xtrain/ Xtest is all 1s, it will be removed. 
% [w, mseTrain] = ridgeQR(Xtrain, ytrain, [], [], Lambda)
%   trains using Lambda, which can be a scalar or a matrix.
%   It standardizes the train and test data data.
%   It returns the weight vector and mean squared error on the training set.
% [w, mseTrain] = ridgeQR(Xtrain, ytrain, [], [], Lambda, 0)
%    is same as above, but does not standardize the data.
% [w, mseTrain, mseTest] = ridgeQR(Xtrain, ytrain, Xtest, ytest, Lambda)
%     standardizes and also returns the MSE on the test set.

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

if isscalar(Lambda)
  if Lambda==0
    w = XtrainC \ ytrainC; % least squares
  else
    XX  = [XtrainC; sqrt(Lambda)*eye(d)];
    yy = [ytrainC; zeros(d,1)];
    w  = XX \ yy; % ridge
  end
else 
  XX  = [XtrainC; Lambda];
  yy = [ytrainC; zeros(size(Lambda,1),1)];
  w  = XX \ yy; % generalized ridge
end
w0 = ybar - xbar*w;
w = [w0; w];

ntrain = size(Xtrain, 1);
ypredTrain = [ones(ntrain,1) Xtrain] * w;
mseTrain = mean((ypredTrain-ytrain).^2);

if ~isempty(Xtest)
  ntest = size(Xtest, 1);
  ypredTest = [ones(ntest,1) Xtest]*w;
  mseTest = mean((ypredTest-ytest).^2);
end
