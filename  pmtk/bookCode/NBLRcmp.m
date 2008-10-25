function [LRerrorRate,NBerrorRate] = NBLRcmp(X,Y,dataSetName)
% Compare Naive Bayes vs Logistic Regression as we vary the number of
% training examples. 
%
% X is binary matrix s.t. X(i,j) = 1 iff feature j is 'on' in example i.
% Y is numeric matrix of size nexamples-by-one holding the class labels for 
%   for the corresponding examples in X. 
%   dataSetName is a string name for the dataset, used in the plotted figure.
% 
    
setSeed(0);
nexperiments = 5;
lambda = 1e-3;      % regularizer applied to Logistic Regression
alpha = 2;          % regularizer applied to Naive Bayes
ntrain = floor(0.8*size(X,1)); % equal sized train and test splits
trainsize = floor(linspace(0.1*ntrain,ntrain,10));
nsizes = numel(trainsize);
nclasses = numel(unique(Y));
[nexamples, nfeatures] = size(X);

options.Method = 'bb';     % options for LR optimization
options.MaxIter = 1000;
options.MaxFunEvals = 1000;
options.Display = 'off';
w0 = zeros(nfeatures*(nclasses-1),1);

LRerrors = zeros(nexperiments,nsizes);
NBerrors = zeros(nexperiments,nsizes);
for i=1:nexperiments
  fprintf('exp %d of %d\n', i, nexperiments);
  perm = randperm(nexamples);

  xtrain = X(perm(1:ntrain),:); xtest = X(perm(ntrain+1:end),:);
  ytrain = Y(perm(1:ntrain),:); ytest = Y(perm(ntrain+1:end),:);
  ytrainOOK = oneOfK(ytrain,nclasses);
  counter = 1;
  for j = 1:numel(trainsize)
    sz = trainsize(j);
    [theta,classprior] = NBtrainMulticlass(xtrain(1:sz,:),ytrain(1:sz,:),alpha);
    yhatNB = NBapplyMulticlass(xtest,theta,classprior);
    NBerrors(i,counter) = sum(yhatNB ~= ytest);

    wLR = minFunc(@multinomLogregNLLGradHessL2,w0,options,xtrain(1:sz,:),ytrainOOK(1:sz,:),lambda);
    [val,yhatLR] = max(multiSigmoid(xtest,wLR),[],2);
    LRerrors(i,counter) = sum(yhatLR ~=ytest);
    counter = counter + 1;
  end
end
ntest = nexamples-ntrain;
LRerrorRate = mean(LRerrors,1)/ntest;
NBerrorRate = mean(NBerrors,1)/ntest;

LRse = std(LRerrors)/sqrt(ntest);
NBse = std(NBerrors)/sqrt(ntest);

figure; hold on;
ndx = trainsize;
plot(ndx,NBerrorRate,'--ro','LineWidth',2);
plot(ndx,LRerrorRate,'-bo','LineWidth',2);

%figure; hold on
%errorbar(ndx, NBerrorRate, NBse, '--ro');
%errorbar(ndx, LRerrorRate, LRse, '-bo');

title(sprintf('NB vs LR\n%s data set',dataSetName));
xlabel('size of training set');
ylabel('error rate');
legend({'Naive Bayes','Logistic Regression'});


end




