%% Ridge Regression 
n = 21;
[xtrain, ytrain, xtest, ytest] = polyDataMake('n', n, 'sampling', 'thibaux');
deg = 14;
m = linregDist();
m.transformer =  chainTransformer({rescaleTransformer,  polyBasisTransformer(deg)});
lambdas = logspace(-10,1.2,15);
for k=1:length(lambdas)
    lambda = lambdas(k);
    m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
    testMse(k) = mean(squaredErr(m, xtest, ytest));
    trainMse(k) = mean(squaredErr(m, xtrain, ytrain));
end
figure;
hold on
ndx = log(lambdas);
%ndx  = lambdas;
if 1
    plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
else
    semilogx(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    semilogx(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
end
legend('train', 'test')
%xlabel('dof')
xlabel('log(lambda)')
ylabel('mse')
title(sprintf('poly degree %d, ntrain  %d', deg, n))

%xx = train(m.transformer, xtrain);
df = dofRidge(m, xtrain, lambdas);
figure; hold on;
%ndx = log(df);
ndx = df;
if 1
    plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
else
    semilogx(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    semilogx(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
end
legend('train', 'test')
xlabel('dof')
ylabel('mse')
title(sprintf('poly degree %d, ntrain  %d', deg, n))