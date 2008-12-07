%% BIC Demo
%#broken
setSeed(0);
d = 4;
G = UndirectedGraph('type', 'loop', 'nnodes', d);
obj = GgmDist(G, [], []);
obj = mkRndParams(obj);
n = 100;
X = sample(obj, n);
modelL1 = fitStructure(GgmDist, 'data', X, 'lambda', 1e-3);
adjL1 = modelL1.G.adjMat;
Gs = mkAllUG(UndirectedGraph(), d);
for i=1:length(Gs)
    if isequal(Gs{i}, G), truendx = i; end
    if isequal(Gs{i}, modelL1.G), L1ndx = i; end
    models{i} = fit(GgmDist(Gs{i}), 'data', X);
    BIC(i) = bicScore(models{i}, X);
end
logZ = logsumexp(BIC(:));
postG = exp(BIC - logZ);

figure;
h=bar(postG);
title(sprintf('p(G|D), true model is red, L1 is green'))
Nmodels = length(BIC);
set(gca,'xtick',0:5:Nmodels)
% Find all models with 10% of maximum (could truncate this)
[pbest, ndxbest] = max(postG);
ndxWindow = find(postG >= 0.1*pbest);
line([0 64],[0.1*pbest 0.1*pbest]);
colorbars(h,truendx,'r');
colorbars(h,L1ndx,'g');
%colorbars(h,ndxWindow,'g');

% Compute test set log likelihood for different mehtods
Xtest = sample(obj, 1000);
loglikBest = sum(logprob(models{ndxbest}, Xtest));
loglikL1 = sum(logprob(modelL1, Xtest));
loglikBMA = 0;
for i=ndxWindow(:)'
    loglikBMA = loglikBMA + sum(logprob(models{i}, Xtest))*postG(i);
end
fprintf('NLL best %5.3f, L1 %5.3f, BMA over %d = %5.3f\n', ...
    -loglikBest, -loglikL1, length(ndxWindow), -loglikBMA);