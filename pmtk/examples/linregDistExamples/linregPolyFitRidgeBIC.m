%% Ridge Regression using BIC
n = 100;
if n==21, nfolds = -1; else   nfolds =5; end
%[xtrain, ytrain, xtest, ytest] = makePolyData(n);
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux', 'n', n);
deg = 14;
m = LinregDist();
m.transformer =  ChainTransformer({RescaleTransformer, ...
    PolyBasisTransformer(deg)});
lambdas = logspace(-10,1.2,15);
for i=1:length(lambdas)
    lambda = lambdas(i);
    m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
    testLogprob(i) = mean(logprob(m, xtest, ytest));
    trainLogprob(i) = mean(logprob(m, xtrain, ytrain));
    [CVmeanLogprob(i), CVstdErrLogprob(i)] = cvScore(m, xtrain, ytrain, ...
        'objective', 'logprob', 'nfolds', nfolds);
    bic(i) = bicScore(m, xtrain, ytrain, lambda);
    aic(i) = aicScore(m, xtrain, ytrain, lambda);
    nparams(i) = length(m.w);
end
figure;
hold on
ndx = log(lambdas);
n = size(xtrain,1);
plot(ndx, CVmeanLogprob, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(ndx, trainLogprob, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(ndx, testLogprob, 'rx-', 'linewidth', 2, 'markersize', 12);
plot(ndx, bic/n, 'g-^', 'linewidth', 2, 'markersize', 12);
plot(ndx, aic/n, 'm-v', 'linewidth', 2, 'markersize', 12);
legend(sprintf('CV(%d)', nfolds), 'train', 'test', 'bic', 'aic')
errorbar(ndx, CVmeanLogprob, CVstdErrLogprob, 'k');
xlabel('log(lambda)')
ylabel('1/n * log p(D)')
title(sprintf('poly degree %d, ntrain  %d', deg, n))

% draw vertical line at best value
ylim = get(gca, 'ylim');
best = [argmax(CVmeanLogprob) argmax(bic) argmax(aic)];
colors = {'k','g','m'};
for k=1:3
    xjitter = ndx(best(k))+0.5*randn;
    h=line([xjitter xjitter], ylim);
    set(h, 'color', colors{k}, 'linewidth', 2);
end
if n==21, set(gca,'ylim',[-7 0]); end