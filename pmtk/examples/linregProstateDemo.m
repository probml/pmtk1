%% Penalized linear regression on prostate cancer data
% Reproduce fig 3.6  on p58 of Elements 1st ed
% and table 3.3 on p57

clear all
load('prostate.mat') % from prostateDataMake
ndxTrain = find(istrain);
ndxTest = find(~istrain);
Dtrain = DataTable(X(ndxTrain,:), y(ndxTrain), names);
Dtest = DataTable(X(ndxTest,:), y(ndxTest));

% It is very important to shuffle the cases to reproduce the figure
% since there seems to be an ordering effect in the data
setSeed(0);
n = length(ndxTrain);
perm = randperm(n);
Dtrain  = Dtrain(perm);


%T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
T = StandardizeTransformer(true);

%{
%lambdas = [logspace(4, 0, 20) 0];
ML = LinregL2ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true, ...
  '-costFnForCV', @(M,D) squaredErr(M,D) );
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
mseTestRidge = mean(squaredErr(ML.bestModel, Dtest));
wRidge =W(:,bestNdx)';

figure;
errorbar(df, ML.costMean, ML.costSe);
title('Ridge on prostate')
xlabel('df(lambda)'); ylabel('5-CV MSE')
hold on
bestNdx = find(ML.bestModel.lambda==lambdas);
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');

%}

ML = LinregL1ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true, ...
  '-costFnForCV', @(M,D) squaredErr(M,D) );
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
mseTestL1 = mean(squaredErr(ML.bestModel, Dtest));
wL1 =W(:,bestNdx)';

figure;
errorbar(df, ML.costMean, ML.costSe);
title('Ridge on prostate')
xlabel('df(lambda)'); ylabel('5-CV MSE')
hold on
bestNdx = find(ML.bestModel.lambda==lambdas);
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');



