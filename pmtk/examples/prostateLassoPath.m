%% Lasso path on prostate cancer data
%#broken
% Reproduce fig 3.9  on p65 of Elements 1st ed

clear all
load('prostate.mat') % from prostateDataMake
ndxTrain = find(istrain);
ytrain = y(ndxTrain); Xtrain = X(ndxTrain,:);
Dtrain = DataTable(Xtrain, ytrain, names);

% It is very important to shuffle the cases to reproduce the figure
% since there seems to be an ordering effect in the data
setSeed(0);
n = length(ndxTrain);
perm = randperm(n);
Dtrain  = Dtrain(perm);


T = StandardizeTransformer(false);


% Compute solutions at discontinuities using lars
ML = LinregL1ModelList('-lambdas', 'all', '-transformer', T);
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas] = getParamsForAllModels(ML);
figure;
plot(df, W, 'o-');
legend(names(1:8), 'location', 'northwest')
title('lasso path on prostate')

% Now compute solutions on dense path using lars + interpolation
%lambdas = [logspace(4, 0, 20) 0];
ML = LinregL1ModelList('-nlambdas', 20, '-transformer', T);
ML = fit(ML, Dtrain);
[W, w0, sigma2, df, lambdas, shrinkage] = getParamsForAllModels(ML);
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
%plot(xvec(bestNdx), ML.penloglik(bestNdx), 'ro', 'markersize', 12)
ylim = get(gca, 'ylim');
h = line([xvec(bestNdx) xvec(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'r', 'linestyle', '-.');


% Now do CV
MLcv = LinregL1ModelList('-nlambdas', 20, '-transformer', T, ...
  '-selMethod', 'cv', '-nfolds', 5, '-verbose', true);
MLcv = fit(MLcv, Dtrain);
[Wcv, w0cv, sigma2cv, dfcv, lambdascv, shrinkagecv] = getParamsForAllModels(MLcv);


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
yvec = -MLcv.costMean; %loglik
errorbar(xvec, yvec, MLcv.costSe);
title('CV loglik vs df(lambda)')
hold on
ylim = get(gca, 'ylim');
h = line([xvec(bestNdx) xvec(bestNdx)], [ylim(1), ylim(2)]);
set(h, 'color', 'k', 'linestyle', ':');
%plot(xvec(bestNdx), yvec, 'ko', 'markersize', 12)
