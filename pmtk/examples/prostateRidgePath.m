%% Ridge regression path on prostate cancer data
%#broken
% Reproduce fig 3.7  on p61 of Elements 1st ed

load('prostate.mat') % from prostateDataMake

ndxTrain = find(istrain);
ndxTest = find(~istrain);
Dtrain = DataTable(X(ndxTrain,:), y(ndxTrain), names);
Dtest = DataTable(X(ndxTest,:), y(ndxTest), names);

% It is very important to shuffle the cases to reproduce the figure
% since there seems to be an ordering effect in the data
setSeed(0);
n = length(ndxTrain);
perm = randperm(n);
Dtrain  = Dtrain(perm);

%T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
T = StandardizeTransformer(false);

%% Reg path 
ML = LinregL2ModelList('-nlambdas', 20, '-transformer', T);
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
figW=figure;
plot(df, W, 'o-');
legend(names(1:8))

%% BIC
bestNdx = find(ML.bestModel.lambda==lambdas);
h = line(df([bestNdx bestNdx]), [min(W(:)), max(W(:))]);
set(h, 'color', 'r', 'linestyle', '--');
title('ridge path on prostate, red vertical line = BIC')
xlabel('dof')
ylabel('regression coef.')
drawnow

figure;
plot(df, ML.penloglik, 'o-')
title('BIC vs df(lambda)')
hold on
%plot(df(bestNdx), ML.penloglik(bestNdx), 'ro', 'markersize', 12)
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'r', 'linestyle', '-.');
drawnow

%% CV on loglik
MLcv = LinregL2ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
 % '-costFnForCV', @(M,D) squaredErr(M,D));
MLcv = fit(MLcv, Dtrain);
[Wcv, w0cv, sigma2cv, dfcv] = getParamsForAllModels(MLcv);
% Fitted models same as in BIC, only bestModel ndx differs...
assert(approxeq(Wcv, W))
assert(approxeq(w0cv, w0))
assert(approxeq(sigma2cv, sigma2))
assert(approxeq(dfcv, df))

figure(figW);
bestNdx = find(MLcv.bestModel.lambda == lambdas);
h = line(dfcv([bestNdx bestNdx]), [min(Wcv(:)), max(Wcv(:))]);
set(h, 'color', 'k', 'linestyle', ':');
title('ridge path on prostate, red = BIC, black = 5-CV')


figure;
yvec = -MLcv.costMean; % yvec = loglik
errorbar(df, yvec, MLcv.costSe);
title('CV loglik vs df(lambda)')
hold on
ylim = get(gca, 'ylim');
h = line([df(bestNdx) df(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');






