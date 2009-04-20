symbols = {'r+', 'b*',  'gx', 'mx', 'r.', 'gs', 'c*'};
errrate = zeros(1,2);
K = 3; d = 2;
setSeed(3);

doPlot = true;

mu = [+0.0, 0.6;
      -0.3, -0.4;
      +1.0, -0.5]';

var = [0.6,0.4,0.6];

trainSize = [2, 5, 8];
testSize = 20*trainSize;
Xtrain = zeros(sum(trainSize),d); Xtest = zeros(sum(testSize),d);
Ytrain = zeros(sum(trainSize),1); Ytest = zeros(sum(testSize),1);

dist = cell(K,1);

for k=1:K
  dist{k} = MvnDist(mu(:,k), var(k));
end

ntrain = 1; ntest = 1;
for k=1:K
  Xtrain(ntrain:(ntrain + trainSize(k) -1),:) = sample(dist{k}, trainSize(k));
  Xtest(ntest:(ntest + testSize(k) - 1),:) = sample(dist{k}, testSize(k));
  Ytrain(ntrain:(ntrain + trainSize(k) -1)) = k;
  Ytest(ntest:(ntest + testSize(k) - 1)) = k;
  ntrain = ntrain + trainSize(k); ntest = ntest + testSize(k);
end

%idxtrain = randperm(sum(trainSize));
%Xtrain = Xtrain(idxtrain,:);
%Ytrain = Ytrain(idxtrain);
%idxtest = randperm(sum(testSize));
%Xtest = Xtest(idxtest,:);
%Ytest = Ytest(idxtest);

if(doPlot)
  subplot(2,2,1);
  for k = 1:K
    plot(Xtrain(Ytrain == k,1), Xtrain(Ytrain == k,2), symbols{k});
    hold on
  end
  title('training data')
  subplot(2,2,2)
  for k = 1:K
    plot(Xtest(Ytest == k,1), Xtest(Ytest == k,2), symbols{k});
    hold on
  end
  title('test data')
end

method = {'plugin', 'bayes'};

nigPrior = MvnInvGammaDist('mu', zeros(d,1), 'Sigma', 0, 'a', 0.1, 'b', 0.1);
classConditionals = copy(MvnDist('mu', zeros(d,1), 'Sigma', diag(ones(1,d)),'prior', nigPrior, 'covtype', 'diagonal'),1,K);
classPrior = DiscreteDist('-T',normalize(ones(3,1)),'-support',1:K);
baseClassifier = GenerativeClassifierDist('classConditionals',classConditionals,'classPrior',classPrior);
classifier = fit(baseClassifier, 'X', Xtrain, 'y', Ytrain);
% plugin estimate

post{1} = predict(classifier, Xtest);
Ypred{1} = mode(post{1});

% integrate out mu, Sigma
ll = zeros(size(Xtest,1),K);
for k=1:K
    bayesConditional{k} = Mvn_MvnInvGammaDist(classifier.classConditionals{k}.prior);
    bayesConditional{k} = fit(bayesConditional{k}, 'data', Xtrain(Ytrain == k,:));
    bayesMarginal{k} = marginal(bayesConditional{k});
    ll(:,k) = logprob(bayesMarginal{k}, Xtest);
end

mixingWeights = classifier.classPrior.T;
logjoint = ll + repmat(log(mixingWeights'), size(Xtest,1), 1);
post{2} = exp(normalizeLogspace(logjoint));
[junk, Ypred{2}] = max(post{2},[],2);



for i=1:length(method)
  errors = Ypred{i} ~= Ytest;
  nerr = sum(errors);
  errrate(i) = mean(errors);
  if (doPlot)
    subplot(2,2,2+i)
    for j = 1:K
      plot(Xtest(Ypred{i} == j, 1), Xtest(Ypred{i} == j, 2), symbols{j});
      hold on
    end
    plot(Xtest(errors, 1), Xtest(errors, 2), 'ko');
    title(sprintf('%s %d errors', method{i}, nerr))
  end

end

