%% Ridge regression path on prostate cancer data
% Reproduce fig 3.7  on p61 of Elements 1st ed

clear all
load('prostate.mat') % from prostateDataMake
lambdas = [logspace(4, 0, 20) 0];
ndx = find(istrain);
y = y(ndx); X = X(ndx,:);
T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});


% Check reasonableness of auto-lambda
%mm = LinregL2ModelList.maxLambda(X);
%ML = LinregL2ModelList(lambdas, '-transformer', T);
ML = LinregL2ModelList('-nlambdas', 20, '-X', X, '-transformer', T);
ML = fit(ML, X, y);
[W, sigma2, df] = getParamsForAllModels(ML);
bestNdx = find(dof(ML.bestModel)==df);

figure;
plot(df, ML.penloglik, 'o-')
title('BIC vs df(lambda)')
hold on
plot(df(bestNdx), ML.penloglik(bestNdx), 'ro', 'markersize', 12)

figW=figure;
plot(df, W(2:end,:), 'o-');
legend(names(1:8))

ww = W(2:end,:); ww = ww(:);
h = line(df([bestNdx bestNdx]), [min(ww), max(ww)]);
set(h, 'color', 'r', 'linestyle', '--');
title('ridge path on prostate, red vertical line = BIC')
xlabel('dof')
ylabel('regression coef.')
drawnow

% Now do CV
MLcv = LinregL2ModelList('-nlambdas', 20, '-X', X, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
MLcv = fit(MLcv, X, y);
[Wcv, sigma2cv, dfcv] = getParamsForAllModels(MLcv);
% Fitted models same as in BIC, only bestModel ndx differs...
assert(approxeq(Wcv, W))
assert(approxeq(sigma2, sigma2cv))
assert(approxeq(dfcv, df))

figure(figW);
bestNdx = find(dof(MLcv.bestModel)==dfcv);
h = line(dfcv([bestNdx bestNdx]), [min(ww), max(ww)]);
set(h, 'color', 'k', 'linestyle', ':');
title('ridge path on prostate, red = BIC, black = 5-CV')


figure;
errorbar(df, MLcv.LLmean, MLcv.LLse);
title('CV loglik vs df(lambda)')
hold on
plot(df(bestNdx), MLcv.LLmean(bestNdx), 'ko', 'markersize', 12)




