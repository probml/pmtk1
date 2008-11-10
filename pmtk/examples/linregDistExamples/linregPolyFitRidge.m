%% Ridge Regression with Polynomial Basis Expansion
% based on code by Romain Thibaux
%(Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)
%makePolyData;
n = 21;
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux','n',n);
deg = 14;
m = LinregDist();
m.transformer =  ChainTransformer({RescaleTransformer, ...
    PolyBasisTransformer(deg)});
lambdas = [0 0.00001 0.001];
for k=1:length(lambdas)
    lambda = lambdas(k);
    m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
    format bank
    m.w.point(:)'
    ypred = mean(predict(m, xtest));
    figure;
    scatter(xtrain,ytrain,'b','filled');
    hold on;
    plot(xtest, ypred, 'k', 'linewidth', 3);
    hold off
    title(sprintf('degree %d, lambda %10.8f', deg, lambda))
end