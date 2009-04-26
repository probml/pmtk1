%% Model selection for GGMs on 4 nodes 

setSeed(0);
d = 4;
Gtrue = UndirectedGraph('type', 'loop', 'nnodes', d);
obj = UgmGaussDist(Gtrue);
obj = mkRndParams(obj);
n = 100;
X = sample(obj, n);
Xtest = sample(obj, 1000);

Gs = mkAllUG(UndirectedGraph(), d);
models = cellfuncell(@(G) UgmGaussDist(G), Gs);
ML = ModelList(models, 'bic');
ML = fit(ML, X);
nM = length(ML.models);


modelL1 = fitStructure(UgmGaussDist(), X, '-lambda', 1e-3);

% figure out model indexes of chosen model
for i=1:length(Gs)
    if isequal(Gs{i}, Gtrue), truendx = i; end
    if isequal(Gs{i}, modelL1.G), L1ndx = i; end
    if isequal(Gs{i}, ML.bestModel.G), bestNdx = i; end
end

postG = ML.posterior;
Nmodels = length(postG);
figure;
h=bar(postG);
title(sprintf('p(G|D), true is red, L1 is green'))
set(gca,'xtick',0:5:Nmodels)
colorbars(h,truendx,'r');
colorbars(h,L1ndx,'g');
%colorbars(h,bestNdx,'k');

% Find all models with 25% of maximum (could truncate this)
[pbest, ndxbest] = max(postG);
ndxWindow = find(postG >= 0.25*pbest);
line([0 64],[0.1*pbest 0.1*pbest]);


% Compute test set log likelihood for different methods
loglikL1 = sum(logprob(modelL1, Xtest), 1);
ML.predMethod = 'plugin';
loglikBest = sum(logprob(ML, Xtest), 1);
ML.predMethod = 'bma';
loglikBMA = sum(logprob(ML, Xtest), 1);


fprintf('NLL on test: L1 %5.3f, best %5.3f,  BMA %5.3f\n', ...
   -loglikL1, -loglikBest,  -loglikBMA);
  
  