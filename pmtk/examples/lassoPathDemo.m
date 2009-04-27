%% Lasso path on prostate cancer data
% Reproduce fig 3.9  on p65 of Elements 1st ed

clear all
load('prostate.mat') % from prostateDataMake
ndx = find(istrain);
y = y(ndx); X = X(ndx,:);
T = StandardizeTransformer(false);


% Compute solutions at discontinuities using lars
ML = LinregL1ModelList('-lambdas', 'all', '-transformer', T);
ML = fit(ML, X, y);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
figure;
plot(df, W, 'o-');
legend(names(1:8), 'location', 'northwest')
title('lasso path on prostate')

% Now compute solutions on dense path using lars + interpolation
%lambdas = [logspace(4, 0, 20) 0];
ML = LinregL1ModelList('-nlambdas', 50, '-transformer', T);
ML = fit(ML, X, y);
[W, w0, sigma2, df, lambdas, shrinkage] = getParamsForAllModels(ML, X, y);
figW = figure;
Nm = size(W,2);
xvec = shrinkage;
plot(xvec, W, 'o-');
legend(names(1:8),'location','northwest')

bestNdx = find(ML.bestModel.lambda==lambdas);
h = line(xvec([bestNdx bestNdx]), [min(W(:)), max(W(:))]);
set(h, 'color', 'r', 'linestyle', '--');
title('lasso path on prostate, red vertical line = BIC')
xlabel('norm(w(:,m))/max(norm(w))')
ylabel('regression coef.')
drawnow

figure;
plot(xvec, ML.penloglik, 'o-')
title('BIC vs df(lambda)')
hold on
plot(xvec(bestNdx), ML.penloglik(bestNdx), 'ro', 'markersize', 12)



% Now do CV
MLcv = LinregL1ModelList('-nlambdas', 50, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
MLcv = fit(MLcv, X, y);
[Wcv, w0cv, sigma2cv, dfcv, lambdascv, shrinkagecv] = getParamsForAllModels(MLcv, X, y);


% Fitted models same as in BIC, only bestModel ndx differs...
assert(approxeq(Wcv, W))
assert(approxeq(w0cv, w0))
assert(approxeq(sigma2cv, sigma2))
assert(approxeq(dfcv, df))


figure(figW);
bestNdx = find(MLcv.bestModel.lambda==lambdascv);
h = line(xvec([bestNdx bestNdx]), [min(Wcv(:)), max(Wcv(:))]);
set(h, 'color', 'k', 'linestyle', ':');
title('lasso path on prostate, red = BIC, black = 5-CV')


figure;
errorbar(xvec, MLcv.LLmean, MLcv.LLse);
title('CV loglik vs df(lambda)')
hold on
plot(xvec(bestNdx), MLcv.LLmean(bestNdx), 'ko', 'markersize', 12)
