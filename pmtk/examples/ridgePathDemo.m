%% Ridge regression path on prostate cancer data
% Reproduce fig 3.7  on p61 of Elements 1st ed

clear all
load('prostate.mat') % from prostateDataMake
lambdas = [logspace(4, 0, 20) 0];
ndx = find(istrain);
y = y(ndx); X = X(ndx,:);
%T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
T = StandardizeTransformer(false);

ML = LinregL2ModelList('-nlambdas', 20, '-transformer', T);
ML = fit(ML, X, y);
[W, w0, sigma2, df] = getParamsForAllModels(ML);
figW=figure;
plot(df, W, 'o-');
legend(names(1:8))

bestNdx = find(dof(ML.bestModel)==df);
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
plot(df(bestNdx), ML.penloglik(bestNdx), 'ro', 'markersize', 12)



% Now do CV
MLcv = LinregL2ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
MLcv = fit(MLcv, X, y);
[Wcv, w0cv, sigma2cv, dfcv] = getParamsForAllModels(MLcv);
% Fitted models same as in BIC, only bestModel ndx differs...
assert(approxeq(Wcv, W))
assert(approxeq(w0cv, w0))
assert(approxeq(sigma2cv, sigma2))
assert(approxeq(dfcv, df))

figure(figW);
bestNdx = find(dof(MLcv.bestModel)==dfcv);
h = line(dfcv([bestNdx bestNdx]), [min(Wcv(:)), max(Wcv(:))]);
set(h, 'color', 'k', 'linestyle', ':');
title('ridge path on prostate, red = BIC, black = 5-CV')


figure;
errorbar(df, MLcv.LLmean, MLcv.LLse);
title('CV loglik vs df(lambda)')
hold on

plot(df(bestNdx), MLcv.LLmean(bestNdx), 'ko', 'markersize', 12)




