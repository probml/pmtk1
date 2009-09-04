%% Ridge Regression with Polynomial Basis Expansion
% based on code by Romain Thibaux
%(Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)
%makePolyData;
n = 21;
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux','n',n);
deg = 14;
T =  ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg,false)});
[xtrain, T] = train(T, xtrain);
[xtest, T] = test(T, xest);

model = LinregL2;
lambdas = [0 0.00001 0.001];
for k=1:length(lambdas)
    lambda = lambdas(k);
    model = fit;
    %m = fit(m, 'X', xtrain, 'y', ytrain, 'prior', 'L2', 'lambda', lambda);
    m = fit(m, DataTable(xtrain, ytrain));
    format bank
    m.w
    ypredTest = predict(m, xtest);
    figure;
    scatter(xtrain,ytrain,'b','filled');
    hold on;
    plot(xtest, ypredTest, 'k', 'linewidth', 3);
    hold off
    title(sprintf('degree %d, lambda %10.8f', deg, lambda))
end