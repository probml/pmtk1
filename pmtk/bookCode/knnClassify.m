function [ypred, ypredProb] = knnClassify(Xtrain, ytrain, Xtest, K)
% function [ypred, ypredProb] = knnClassify(Xtrain, ytrain, Xtest, K)
% Xtrain(n,:) = n'th example (d-dimensional)
% ytrain(n) in {1,2,...,C} where C is the number of classes
% Xtest(m,:)
% ypred(m) in {1,2..,C} is most likely class (ties broken by picking lowest class)
% ypredProb(m,:) is the empirical distribution over classes

Ntest = size(Xtest, 1);
Ntrain = size(Xtrain, 1);
Nclasses = max(ytrain);
if K>Ntrain
  fprintf('reducing K = %d to Ntrain = %d\n', K, Ntrain-1);
  K = Ntrain-1;
end
if K==1
  dst = sqdist(Xtrain', Xtest'); % dst(n,m) = || Xtrain(n) - Xtest(m) || ^2
  [junk, closest] = min(dst,[],1);
  ypred = ytrain(closest);
  ypredProb = oneOfK(ypred, Nclasses);
else
  ypredProb = zeros(Ntest, Nclasses);
  ypred = zeros(Ntest, 1);
  for m=1:Ntest
    dst = sqdist(Xtrain', Xtest(m,:)');
    %[vals, closest] = sort(dst(:,m));
    [vals, closest] = sort(dst);
    labels = ytrain(closest(1:K));
    votes = hist(labels, 1:Nclasses);
    [junk, ypred(m)] = max(votes);
    ypredProb(m,:) = normalize(votes);
  end
end

