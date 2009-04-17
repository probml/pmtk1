%% Fit Linear Regression Models with Various Polynomial Basis Expansions
clear;
degrees = [1,2,30];

for i=1:numel(degrees)
    deg = degrees(i);
    lambda = 1e-3;
    ns = linspace(10,200,10);
    for i=1:length(ns)
        n=ns(i);
        %[xtrain, ytrain, xtest, ytest] = makePolyData(n);
        [xtrain, ytrain, xtest, ytestNoiseFree, ytest,sigma2] = polyDataMake('n', n, 'sampling', 'thibaux');
        m = LinregDist();
        m.transformer =  ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg)});
        m = fit(m, 'X', xtrain, 'y', ytrain, 'prior', 'L2', 'lambda', lambda);
        testMse(i) = mean(squaredErr(m, xtest, ytest));
        trainMse(i) = mean(squaredErr(m, xtrain, ytrain));
    end
    figure;
    hold on
    ndx = ns;
    plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
    plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
    legend('train', 'test')
    ylabel('mse')
    xlabel('size of training set')
    title(sprintf('truth=degree 2, model = degree %d', deg));
    set(gca,'ylim',[0 22]);
    line([0 max(ns)],[sigma2 sigma2],'color','k','linewidth',3);

end


