%% Select lambda for ridge Regression on polynomial regression using
%% various methods (CV, BIC, AIC)

n = 100;
if n==21, nfolds = -1; else   nfolds =5; end

[xtrain, ytrain, xtest, ytest] = polyDataMake('n', n, 'sampling', 'thibaux');
Dtrain = DataTable(xtrain, ytrain);
Dtest = DataTable(xtest, ytest);
deg = 14;
T =  ChainTransformer({RescaleTransformer,  PolyBasisTransformer(deg,false)});
lambdas = logspace(-10,1.2,15);
ML = LinregL2ModelList('-transformer', T, '-lambdas', lambdas, ...
  '-selMethod', 'cv', '-nfolds', nfolds);
ML = fit(ML, Dtrain);
for k=1:length(lambdas)
    m = ML.models{k};
    df(k) = m.df;
    testLL(k) = mean(logprob(m, Dtest)); %#ok
    trainLL(k) = sum(logprob(m, Dtrain)); %#ok
    bic(k) = (trainLL(k) - df(k)*log(n)/2)/n;
    aic(k) = (trainLL(k) - df(k))/n;
    cvLL(k) = -ML.costMean(k); % NLL averaged over training cases
    trainLL(k) = trainLL(k)/n;
end


figure;
hold on
ndx = log(lambdas);
n = size(xtrain,1);
plot(ndx, cvLL, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(ndx, trainLL, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(ndx, testLL, 'rx-', 'linewidth', 2, 'markersize', 12);
plot(ndx, bic, 'g-^', 'linewidth', 2, 'markersize', 12);
plot(ndx, aic, 'm-v', 'linewidth', 2, 'markersize', 12);
legend(sprintf('CV(%d)', nfolds), 'train', 'test', 'bic', 'aic')
xlabel('log(lambda)')
title(sprintf('poly degree %d, ntrain  %d', deg, n))

%{
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
%}
