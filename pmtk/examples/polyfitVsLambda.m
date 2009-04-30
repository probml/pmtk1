%% Example of U shape test error using ridge Regression on polynomial regression 
function polyfitRidgeU()

n = 21;
[xtrain, ytrain, xtest, ytest] = polyDataMake('n', n, 'sampling', 'thibaux');
Dtrain = DataTable(xtrain, ytrain);
Dtest = DataTable(xtest, ytest);
deg = 14;
T =  ChainTransformer({RescaleTransformer,  PolyBasisTransformer(deg,false)});
lambdas = logspace(-10,1.2,15);
ML = LinregL2ModelList('-transformer', T, '-lambdas', lambdas);
ML = fit(ML, Dtrain);
for k=1:length(lambdas)
    m = ML.models{k};
    df(k) = m.df;
    testMse(k) = mean(squaredErr(m, Dtest)); %#ok
    trainMse(k) = mean(squaredErr(m, Dtrain)); %#ok
end
%doPlot(log(lambdas), trainMse, testMse);
%xlabel('log(lambda)')

doPlot(df, trainMse, testMse);
xlabel('dof')
end

function doPlot(ndx, trainMse, testMse)
figure;
hold on
if 1
    plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
else
    semilogx(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    semilogx(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
end
legend('train', 'test')
ylabel('mse')
%title(sprintf('poly degree %d, ntrain  %d', deg, n))
end