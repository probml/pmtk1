%% Penalized linear regression on prostate cancer data
% Reproduce fig 3.6  on p58 of Elements 1st ed

clear all
load('prostate.mat') % from prostateDataMake
lambdas = [logspace(4, 0, 20) 0];
ndxTrain = find(istrain);
ndxTest = find(~istrain);
y = y(ndxTrain); X = X(ndxTrain,:);

%T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
T = StandardizeTransformer(false);

ML = LinregL2ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
ML = fit(ML, X, y);
[W, w0, sigma2, df] = getParamsForAllModels(ML);

figure;
errorbar(df, ML.LLmean, ML.LLse);
title('CV loglik vs df(lambda)')
hold on
bestNdx = find(dof(ML.bestModel)==df);
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');





