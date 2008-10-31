%% Ridge Regression Using Cross Validation
n = 21;
if n==21, nfolds = -1; else   nfolds =5; end
%[xtrain, ytrain, xtest, ytest] = makePolyData(n);
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux', 'n', n);
deg = 14;
m = linregDist;
m.transformer =  chainTransformer({rescaleTransformer,polyBasisTransformer(deg)});
lambdas = logspace(-10,1.2,15);

for i=1:length(lambdas)
    lambda = lambdas(i);
    m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
    testMse(i) = mean(squaredErr(m, xtest, ytest));
    trainMse(i) = mean(squaredErr(m, xtrain, ytrain));
    [CVmeanMse(i), CVstdErrMse(i)] = cvScore(m, xtrain, ytrain, ...
        'objective', 'squaredErr', 'nfolds', nfolds);
    nparams(i) = length(m.w);
end

figure;
hold on
ndx = log(lambdas);
plot(ndx, CVmeanMse, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
legend(sprintf('CV(%d)', nfolds), 'train', 'test')
errorbar(ndx, CVmeanMse, CVstdErrMse, 'k');
xlabel('log(lambda)')
ylabel('mse')
title(sprintf('poly degree %d, ntrain  %d', deg, n))

df = dofRidge(m, xtrain, lambdas);
ylim = get(gca, 'ylim');
% illustrate 1 SE rule
[bestCV bestCVndx] = min(CVmeanMse);
% vertical line at bestCVndx
h=line([ndx(bestCVndx) ndx(bestCVndx)], ylim); set(h, 'color', 'k');
% horizontal line at height of bestCV
h=line([min(ndx) max(ndx)], [bestCV bestCV]); set(h,'color','k');
% vertical line at bestCVndx1SE
idx_opt = oneStdErrorRule(CVmeanMse, CVstdErrMse, df);
h=line([ndx(idx_opt) ndx(idx_opt)], ylim); set(h, 'color', 'k');
set(gca,'ylim',[0 10]);