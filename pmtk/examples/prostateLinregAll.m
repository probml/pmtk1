%% Penalized linear regression on prostate cancer data
% Reproduce fig 3.6  on p58 of Elements 1st ed
% and table 3.3 on p57

%#testPMTK

clear all
load('prostate.mat') % from prostateDataMake
ndxTrain = find(istrain);
ndxTest = find(~istrain);
ndx = 1:8;
Dtrain = DataTable(X(ndxTrain,ndx), y(ndxTrain), names);
Dtest = DataTable(X(ndxTest,ndx), y(ndxTest), names);

% It is very important to shuffle the cases to reproduce the figure
% since there seems to be an ordering effect in the data
setSeed(0);
n = length(ndxTrain);
perm = randperm(n);
Dtrain  = Dtrain(perm);


%T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
T = StandardizeTransformer(true);
Nfolds = 2; % for speed


%% least squares
M = fit(Linreg, Dtrain);
mseLS = squaredErr(M, Dtest);
mseLSmean = mean(mseLS); mseLSse = stderr(mseLS);
wLS = M.w';
w0LS = M.w0;

%% Ridge
%lambdas = [logspace(4, 0, 20) 0];
ML = LinregL2ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', Nfolds, '-verbose', true, ...
  '-costFnForCV', @(M,D) squaredErr(M,D) );
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
mseRidge = squaredErr(ML.models{ML.bestNdx}, Dtest);
mseRidgeMean = mean(mseRidge); mseRidgeSe = stderr(mseRidge);
bestNdx = ML.bestNdx;
wRidge = W(:,bestNdx)';
w0Ridge = w0(bestNdx);

figure;
errorbar(df, ML.costMean, ML.costSe);
title('Ridge on prostate')
xlabel('df(lambda)'); ylabel('5-CV MSE')
hold on
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');


%% Lasso
ML = LinregL1ModelList('-nlambdas',20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', Nfolds, '-verbose', true, ...
  '-costFnForCV', @(M,D) squaredErr(M,D) );
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas, shrinkage] = getParamsForAllModels(ML);
mseL1 = squaredErr(ML.models{ML.bestNdx}, Dtest);
mseLassoMean = mean(mseL1); mseLassoSe = stderr(mseL1);
bestNdx = ML.bestNdx;
wLasso = W(:,ML.bestNdx)';
w0Lasso = w0(ML.bestNdx);

figure;
errorbar(shrinkage, ML.costMean, ML.costSe);
title('Lasso on prostate')
xlabel('shrinkage(lambda)'); ylabel('5-CV MSE')
hold on
ylim = get(gca, 'ylim');
h = line([shrinkage(bestNdx) shrinkage(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');



%% Subsets
% 2-fold CV for speed
ML = LinregAllSubsetsModelList('-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 2, '-verbose', true, ...
  '-costFnForCV', @(M,D) squaredErr(M,D) );
ML = fit(ML, Dtrain);
[W, w0, sigma2, nnz] = getParamsForAllModels(ML);
mseSS = squaredErr(ML.models{ML.bestNdx}, Dtest);
mseSSmean = mean(mseSS); mseSSse = stderr(mseSS);
bestNdx = ML.bestNdx;
wSS = W(:,bestNdx)';
w0SS = w0(bestNdx);

% Lower envelope
d = ndimensions(Dtrain);
for i=0:d
  ndx = find(nnz==i);
  costs = ML.costMean(ndx);
  costsSe = ML.costSe(ndx);
  j = argmin(costs);
  bestCost(i+1) = costs(j);
  bestCostSe(i+1) = costsSe(j);
  sz(i+1) = i;
end

figure;
%errorbar(nnz, ML.costMean, ML.costSe);
errorbar(sz, bestCost, bestCostSe)
title('Subset regression on prostate')
xlabel('num non zeros'); ylabel('MSE')
hold on
ylim = get(gca, 'ylim');
h = line([sz(bestNdx) sz(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');


%% Summary  (table 3.3)
d = ndimensions(Dtrain);
%w0SS = 0; wSS = zeros(d,1); mseSSmean = 0; mseSSse = 0;

fprintf('%10s %7s %7s %7s %7s\n',...
	'Term', 'LS', 'Subset', 'Ridge', 'Lasso');
fprintf('%10s %7.3f %7.3f %7.3f %7.3f\n',...
	'intercept', w0LS(1), w0SS(1), w0Ridge(1), w0Lasso(1));
for i=1:d
  fprintf('%10s %7.3f %7.3f %7.3f %7.3f\n',...
	  names{i}, wLS(i), wSS(i), wRidge(i), wLasso(i));
end  
fprintf('\n%10s %7.3f %7.3f %7.3f %7.3f\n',...
	'Test MSE', mseLSmean, mseSSmean, mseRidgeMean, mseLassoMean);
fprintf('%10s %7.3f %7.3f %7.3f %7.3f\n',...
	'Std err', mseLSse, mseSSse, mseRidgeSe, mseLassoSe);

